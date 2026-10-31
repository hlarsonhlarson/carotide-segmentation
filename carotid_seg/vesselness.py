import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.filters import frangi
from skimage.graph import route_through_array
from skimage.morphology import remove_small_objects, ball, closing


def adaptive_lumen_hu_band(ct: np.ndarray, seed_mask: np.ndarray, pad: float = 40.0,
                            min_core_voxels: int = 200):
    """Per-scan contrast HU range, learned from the voxels TotalSegmentator
    already labeled as carotid lumen. Avoids hardcoding a bolus-timing-
    dependent HU threshold that would not generalize across scanners/protocols.

    Uses only the *interior* of `seed_mask` (eroded), not its raw boundary.
    On real high-resolution CTA, TotalSegmentator's mask edge sits right on
    top of a partial-volume transition to fat/muscle/thyroid — on one real
    case ~7% of the raw seed's voxels were <100 HU, which is not plausible
    contrast-enhanced lumen. Those low-HU boundary voxels dragged the 1st
    percentile down to the point where the resulting band (with padding)
    started at roughly the soft-tissue floor, essentially disabling the HU
    filter and letting non-vascular tissue (e.g. thyroid gland) into the
    "lumen" mask — which then breaks skeletonization downstream (a blobby,
    non-tube region skeletonizes into a tangle of spurious branch points
    instead of a clean centerline). Eroding first keeps the stats
    representative of actual lumen. Falls back to less erosion (then none)
    if the mask is too small/thin to survive it.
    """
    core = seed_mask
    for iterations in (2, 1, 0):
        eroded = ndi.binary_erosion(seed_mask, iterations=iterations) if iterations else seed_mask
        if eroded.sum() >= min_core_voxels:
            core = eroded
            break
    hu = ct[core]
    lo, hi = np.percentile(hu, [1, 99])
    return lo - pad, hi + pad


def vessel_lumen_mask(ct: np.ndarray, spacing_mm, seed_mask: np.ndarray,
                       frangi_thresh: float = 0.05, min_object_voxels: int = 30) -> np.ndarray:
    """Binary mask of all bright tubular structures in `ct` that look like
    arteries, guaranteed to include `seed_mask` itself. Meant to be run on a
    small cropped ROI, not a whole-body volume (Frangi is expensive).
    """
    lo, hi = adaptive_lumen_hu_band(ct, seed_mask)
    intensity_candidate = (ct >= lo) & (ct <= hi)

    # vessel radii of interest ~1-5mm; convert to voxel sigmas per axis avg spacing
    voxel_mm = float(np.mean(spacing_mm))
    sigmas = np.array([1.0, 2.0, 3.0, 4.0]) / voxel_mm
    sigmas = sigmas[sigmas > 0]

    vesselness = frangi(ct.astype(np.float32), sigmas=sigmas, black_ridges=False)
    vesselness /= max(vesselness.max(), 1e-6)

    lumen = (vesselness > frangi_thresh) & intensity_candidate
    lumen |= seed_mask  # never lose the trusted seed voxels

    lumen = closing(lumen, ball(1))
    lumen = remove_small_objects(lumen, max_size=min_object_voxels)
    lumen |= seed_mask
    return lumen


def _dominant_component(labeled: np.ndarray, seed: np.ndarray):
    vals = labeled[seed]
    vals = vals[vals != 0]
    if vals.size == 0:
        return None
    ids, counts = np.unique(vals, return_counts=True)
    return ids[np.argmax(counts)]


def bridge_gap(lumen_mask: np.ndarray, ct: np.ndarray, spacing_mm, seed_a: np.ndarray,
               seed_b: np.ndarray, tube_radius_mm: float = 2.5,
               search_margin_mm: float = 15.0, max_gap_mm: float = 40.0) -> np.ndarray:
    """TotalSegmentator sometimes leaves a real gap between the CCA and ICA
    masks right at the bifurcation, where Frangi's tubularity assumption
    also legitimately breaks down (the bulb is a blob, not a tube) — so
    `lumen_mask` can come out with the two seeds in separate components.

    Finds the minimum-cost path (low cost = HU inside the adaptive lumen
    band) between the nearest points of the two components and adds a thin
    tube around it. A first version of this grew both regions isotropically
    within the HU band until they touched — since that band alone is very
    permissive (it matches most contrast-enhanced tissue, not just this
    vessel), it produced a blob 10-50x the size of a real bridge, which then
    made the skeleton graph huge and bifurcation search pathologically slow.
    A geodesic shortest path stays thin by construction.

    `max_gap_mm` guards a different failure mode: if the nearest points of
    the two *lumen components* (not the raw seed masks, which can be much
    closer — one real case: 8mm between raw masks, 38mm between the lumen
    components actually touching them, because Frangi didn't reach either
    seed's own edge) are implausibly far apart, that's not a small
    under-segmentation at the bifurcation — it means Frangi missed most of
    the actual vessel course (weak contrast, motion) and "bridging" would
    draw an artificial tube through whatever tissue happens to be
    HU-plausible along a straight-ish line, which is not the real anatomy.
    Refuse rather than fabricate it. Calibrated from two real cases: 38mm
    bridged into a correct, valid bifurcation once allowed through; a
    separate 52mm case was confirmed bad. 40mm sits just above the good
    case and well under the bad one.
    """
    labeled, _ = ndi.label(lumen_mask, structure=np.ones((3, 3, 3)))
    id_a, id_b = _dominant_component(labeled, seed_a), _dominant_component(labeled, seed_b)
    if id_a is None or id_b is None or id_a == id_b:
        return lumen_mask
    comp_a, comp_b = labeled == id_a, labeled == id_b

    pts_a, pts_b = np.argwhere(comp_a), np.argwhere(comp_b)
    tree_b = cKDTree(pts_b * np.asarray(spacing_mm))
    dists, idxs = tree_b.query(pts_a * np.asarray(spacing_mm))
    i_a = int(np.argmin(dists))
    if dists[i_a] > max_gap_mm:
        return lumen_mask
    start_vox, end_vox = pts_a[i_a], pts_b[idxs[i_a]]

    lo, hi = adaptive_lumen_hu_band(ct, seed_a | seed_b)

    margin_vox = np.ceil(search_margin_mm / np.asarray(spacing_mm)).astype(int)
    lo_idx = np.maximum(np.minimum(start_vox, end_vox) - margin_vox, 0)
    hi_idx = np.minimum(np.maximum(start_vox, end_vox) + margin_vox + 1, np.asarray(ct.shape))
    sub = tuple(slice(int(a), int(b)) for a, b in zip(lo_idx, hi_idx))

    intensity_sub = (ct[sub] >= lo) & (ct[sub] <= hi)
    cost = np.where(intensity_sub, 1.0, 25.0).astype(np.float32)
    start_local = tuple(int(v) for v in (start_vox - lo_idx))
    end_local = tuple(int(v) for v in (end_vox - lo_idx))
    indices, _ = route_through_array(cost, start_local, end_local, fully_connected=True)

    path_mask_sub = np.zeros(cost.shape, dtype=bool)
    for idx in indices:
        path_mask_sub[tuple(idx)] = True
    radius_vox = max(1, int(round(tube_radius_mm / float(np.mean(spacing_mm)))))
    path_mask_sub = ndi.binary_dilation(path_mask_sub, structure=np.ones((3, 3, 3)), iterations=radius_vox)

    # The gap is often exactly where a real branch (e.g. the ECA takeoff)
    # sits, since a non-tubular bulb/trifurcation defeats both
    # TotalSegmentator and Frangi similarly. Let the tube fatten into any
    # HU-plausible tissue right around it, but only keep what's actually
    # connected to the path itself — not unrelated bright blobs elsewhere
    # in this local sub-crop.
    #
    # This helps when the branch is real but faint; but in some patients
    # the surrounding tissue is intensity-plausible over a wide area (e.g.
    # a broad venous/nodal blush near the bifurcation) and "connected to
    # the path" isn't a tight enough constraint — the fattened region can
    # balloon to 10-50x the thin tube. That doesn't just look wrong: the
    # resulting skeleton gets thousands of small interior loops, and
    # collapsing them in treeanalysis is O(cycles x graph size) — one real
    # case took 60+ seconds just for that step, on a tube that should be
    # milliseconds. Cap the growth and fall back to the thin tube instead
    # of trusting a blob.
    thin_count = path_mask_sub.sum()
    fattened = ndi.binary_dilation(path_mask_sub, iterations=radius_vox) & intensity_sub
    fattened |= path_mask_sub
    labeled_local, _ = ndi.label(fattened, structure=np.ones((3, 3, 3)))
    keep_id = _dominant_component(labeled_local, path_mask_sub)
    if keep_id is not None:
        candidate = labeled_local == keep_id
        # A volume-ratio cap alone scales *with* the gap: a long gap already
        # has a large thin_count, so the same multiplier lets proportionally
        # more unrelated tissue in for exactly the gaps where that's most
        # likely (real case: a 31mm gap over the bifurcation let in ~700
        # voxels/slice of what measured out to a >7mm-radius blob — the
        # adjacent thyroid lobe, not a vessel). A real vessel, even widened
        # at a bulb, stays within a plausible caliber everywhere along its
        # length, so also cap on local radius via the distance transform,
        # which catches this regardless of how long the gap is.
        max_radius_mm = ndi.distance_transform_edt(candidate, sampling=spacing_mm).max()
        plausible_radius_mm = tube_radius_mm * 2.5
        if candidate.sum() <= 4 * thin_count and max_radius_mm <= plausible_radius_mm:
            path_mask_sub = candidate

    bridge = np.zeros_like(lumen_mask)
    bridge[sub] = path_mask_sub
    return lumen_mask | bridge


def largest_component_touching(mask: np.ndarray, *seed_masks: np.ndarray) -> np.ndarray:
    """Connected component of `mask` that overlaps every mask in `seed_masks`."""
    labeled, n = ndi.label(mask, structure=np.ones((3, 3, 3)))
    for comp_id in range(1, n + 1):
        comp = labeled == comp_id
        if all(np.any(comp & s) for s in seed_masks):
            return comp
    return np.zeros_like(mask, dtype=bool)
