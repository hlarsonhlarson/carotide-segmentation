"""Real 2D image segmentation for one oblique slice at a path point —
SimVascular's Threshold and Level Set tools, reimplemented with real
open-source algorithms (not a heuristic): threshold + connected-component
for Threshold, and skimage's morphological geodesic active contour (a
published level-set method, Marquez-Neila et al.) for Level Set.

Calibration mirrors centerline_path.py's point-seeded approach (no seed
mask exists yet here either, same reasoning as there): CT has a real HU
scale to threshold against, sampled from a small neighborhood at the
center; TOF-MR doesn't, so a local percentile stands in instead.
"""
import math

import numpy as np
from scipy import ndimage as ndi
from skimage import feature, measure, segmentation

CENTER_SAMPLE_RADIUS_PX = 2
# A contrast-opacified lumen vs. the muscle/fat/air around it is a huge,
# qualitative HU drop (e.g. ~150-400 HU lumen vs. ~0-80 HU everything
# nearby) — not a narrow band around one sampled value. Grow aggressively
# from the seed, keeping anything not an order of magnitude darker, and
# let the *real* wall (the actual cliff) stop it, rather than an
# artificially narrow band stopping short of the wall on ordinary
# in-lumen HU variation/noise. CT_MIN_ABSOLUTE_FLOOR_HU guards the low
# end for a weak/borderline seed (where the ratio alone would barely
# exclude anything) and keeps air/fat out regardless.
CT_RELATIVE_FLOOR_RATIO = 0.4
CT_MIN_ABSOLUTE_FLOOR_HU = 100.0
CT_CALCIUM_OVER_LUMEN = 1.3
CT_CALCIUM_MIN_HU = 500.0
# VERIFIED against a synthetic lumen/muscle test: a lower floor (0.15x
# seed / 20 HU absolute) sounds right for "an order of magnitude lower"
# in the abstract, but real neck muscle sits at ~40-60 HU — only ~5-7x
# below a typical ~300 HU contrast lumen, not 10x+. A floor that low
# landed *at* muscle's own HU (noise straddling the threshold both ways)
# and leaked through it instead of stopping. 100 HU sits comfortably
# above muscle while still well below any realistically contrasted lumen.
MR_LOCAL_PERCENTILE = 70.0
LEVELSET_ITERATIONS = 70
LEVELSET_SMOOTHING = 1
LEVELSET_BALLOON = 0.6
LEVELSET_GRADIENT_SIGMA = 1.0
# VERIFIED on real CT data: the *original* wide calibration (7x7 center
# sample, 40 HU pad, balloon=1.0) leaked across into an adjacent bright
# structure (jugular vein/bone) — fitted radius 9-11mm for a real ~2-3mm
# carotid. The fix at the time (CENTER_SAMPLE_RADIUS_PX=1, a +/-5 HU band,
# balloon=0.3) swung too far the other way and started stopping *short* of
# the real wall on ordinary lumen heterogeneity. This calibration aims
# for the actual failure mode in both directions: a real cliff (order-of-
# magnitude HU drop) stops it; ordinary in-lumen noise/partial volume at
# the seed's own scale does not.


def _center_component(binary: np.ndarray, center_px):
    labeled, _ = ndi.label(binary, structure=np.ones((3, 3)))
    label_id = labeled[int(round(center_px[0])), int(round(center_px[1]))]
    return None if label_id == 0 else labeled == label_id


def _center_contour(binary: np.ndarray, center_px):
    """Picks the connected component containing center_px and returns its
    outer boundary as ordered [row, col] points, or None if the center
    pixel isn't inside any foreground component."""
    labeled, _ = ndi.label(binary, structure=np.ones((3, 3)))
    label_id = labeled[int(round(center_px[0])), int(round(center_px[1]))]
    if label_id == 0:
        return None
    component = labeled == label_id
    contours = measure.find_contours(component.astype(np.float32), level=0.5)
    if not contours:
        return None
    # find_contours can return multiple rings for one component (e.g. a
    # hole's inner boundary) — keep the longest, which is the outer wall.
    return max(contours, key=len)


def threshold_contour(slice_2d: np.ndarray, center_px, modality: str, max_radius_px: float):
    cy, cx = int(round(center_px[0])), int(round(center_px[1]))
    lo = [max(0, cy - CENTER_SAMPLE_RADIUS_PX), max(0, cx - CENTER_SAMPLE_RADIUS_PX)]
    hi = [min(slice_2d.shape[0], cy + CENTER_SAMPLE_RADIUS_PX + 1),
          min(slice_2d.shape[1], cx + CENTER_SAMPLE_RADIUS_PX + 1)]
    sample = slice_2d[lo[0]:hi[0], lo[1]:hi[1]]

    if modality == "CT":
        seed_hu = float(np.median(sample))
        floor = max(seed_hu * CT_RELATIVE_FLOOR_RATIO, CT_MIN_ABSOLUTE_FLOOR_HU) if seed_hu > 0 else CT_MIN_ABSOLUTE_FLOOR_HU
        binary = slice_2d >= floor
        # Wall calcium is brighter than the opacified lumen (~1.5x and more on
        # real CTA, lumen p99 ~1.05x its median). Including it made bulb
        # contours swallow calcified plaque and overstated lumen area. The
        # reference is the median of the whole first-pass lumen, not the
        # small centre sample, which can sit in a darker mixing or oblique
        # region and would then cut real lumen away.
        first = _center_component(binary & (np.hypot(*np.mgrid[0:slice_2d.shape[0], 0:slice_2d.shape[1]]
                                                     - np.array([cy, cx])[:, None, None]) <= max_radius_px), (cy, cx))
        if first is not None:
            reference = max(seed_hu, float(np.median(slice_2d[first])))
            if reference > 0:
                # Obliquely cut or poorly opacified sections have a dim
                # lumen; a purely relative cut would remove real lumen there.
                binary &= slice_2d < max(reference * CT_CALCIUM_OVER_LUMEN, CT_CALCIUM_MIN_HU)
    else:
        thresh = float(np.percentile(slice_2d, MR_LOCAL_PERCENTILE))
        binary = slice_2d > thresh

    # Safety cap (same spirit as centerline_path/bridge_gap's max-gap caps):
    # never trust a region whose radius from center exceeds max_radius_px —
    # a real vessel lumen stays within a plausible caliber; an unbounded
    # threshold leak into adjacent tissue should not silently pass as a
    # "segmentation."
    yy, xx = np.mgrid[0:slice_2d.shape[0], 0:slice_2d.shape[1]]
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    binary = binary & (dist <= max_radius_px)

    binary = _cut_thin_bridges(binary, (cy, cx))
    binary = _split_at_waist(binary, (cy, cx))
    contour = _center_contour(binary, (cy, cx))
    return contour  # list of [row, col] float pixel coords, or None


# A neighbouring bright vessel (jugular, ECA branch, small vein) often touches
# the lumen through a partial-volume bridge only a few pixels wide; plain
# connected-component growth then swallows it. Necks narrower than this
# fraction of the lumen's inscribed diameter are cut.
BRIDGE_CUT_FRACTION = 0.7
BRIDGE_CENTER_SEARCH_PX = 3


def _disk(radius: int) -> np.ndarray:
    yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    return (yy ** 2 + xx ** 2) <= radius ** 2


def _cut_thin_bridges(binary: np.ndarray, center_px):
    """Keep the centre component without appendages joined by a thin neck.

    Opening with a disk of half the lumen's inscribed radius removes necks
    narrower than that disk; the surviving core is grown back by the same
    disk inside the original mask, so the lumen's own outline is preserved.
    """
    cy, cx = center_px
    labeled, _ = ndi.label(binary, structure=np.ones((3, 3)))
    label_id = labeled[cy, cx]
    if label_id == 0:
        return binary
    component = labeled == label_id
    edt = ndi.distance_transform_edt(component)
    r = BRIDGE_CENTER_SEARCH_PX
    window = edt[max(0, cy - r):cy + r + 1, max(0, cx - r):cx + r + 1]
    radius = int(math.floor(float(window.max()) * BRIDGE_CUT_FRACTION))
    if radius < 1:
        return component
    footprint = _disk(radius)
    core = ndi.binary_opening(component, structure=footprint)
    core_labels, _ = ndi.label(core, structure=np.ones((3, 3)))
    near = core_labels[max(0, cy - r):cy + r + 1, max(0, cx - r):cx + r + 1]
    ids = near[near > 0]
    if ids.size == 0:
        return component
    core = core_labels == np.bincount(ids).argmax()
    return component & ndi.binary_dilation(core, structure=footprint)


# Two touching vessels can share a contact wider than a "thin bridge". A
# distance-transform watershed separates them at the waist; neighbouring
# basins are merged back unless the waist is clearly narrower than both.
WAIST_RATIO = 0.8
WAIST_MIN_PEAK_DISTANCE_PX = 3


def _split_at_waist(binary: np.ndarray, center_px):
    cy, cx = center_px
    labeled, _ = ndi.label(binary, structure=np.ones((3, 3)))
    label_id = labeled[cy, cx]
    if label_id == 0:
        return binary
    component = labeled == label_id
    edt = ndi.distance_transform_edt(component)
    peaks = feature.peak_local_max(edt, min_distance=WAIST_MIN_PEAK_DISTANCE_PX,
                                   labels=component.astype(np.int32), exclude_border=False)
    if len(peaks) < 2:
        return component
    markers = np.zeros(edt.shape, dtype=np.int32)
    markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
    basins = segmentation.watershed(-edt, markers, mask=component)
    peak_of = ndi.maximum(edt, basins, index=np.arange(1, len(peaks) + 1))
    peak_of = dict(zip(range(1, len(peaks) + 1), peak_of))
    keep = {int(basins[cy, cx])}
    if 0 in keep:
        return component
    frontier = list(keep)
    while frontier:
        region = frontier.pop()
        inside = basins == region
        ring = ndi.binary_dilation(inside, structure=np.ones((3, 3))) & ~inside & component
        for other in np.unique(basins[ring]):
            other = int(other)
            if other == 0 or other in keep:
                continue
            touching = ring & (basins == other)
            waist = float(edt[touching].max())
            if waist >= WAIST_RATIO * min(peak_of[region], peak_of[other]):
                keep.add(other)
                frontier.append(other)
    return np.isin(basins, list(keep))


def calcium_notch(slice_2d: np.ndarray, contour_px) -> bool:
    """True when the concavity of a CT lumen contour is filled by calcium.

    Excluding wall calcium leaves a notch that lowers the contour's
    compactness although the lumen itself is well defined. The region
    between the contour and its convex hull is calcium when its upper
    quartile reaches the calcium cut used by threshold_contour.
    """
    from scipy.spatial import ConvexHull
    from skimage.draw import polygon
    contour = np.asarray(contour_px, dtype=float)
    if len(contour) < 3:
        return False
    hull = contour[ConvexHull(contour).vertices]
    inside, hull_mask = (np.zeros(slice_2d.shape, dtype=bool) for _ in range(2))
    inside[polygon(contour[:, 0], contour[:, 1], slice_2d.shape)] = True
    hull_mask[polygon(hull[:, 0], hull[:, 1], slice_2d.shape)] = True
    notch = hull_mask & ~inside
    if notch.sum() < 4 or not inside.any():
        return False
    cut = max(float(np.median(slice_2d[inside])) * CT_CALCIUM_OVER_LUMEN, CT_CALCIUM_MIN_HU)
    return float(np.percentile(slice_2d[notch], 75)) >= cut


def levelset_contour(slice_2d: np.ndarray, center_px, initial_radius_px: float,
                      max_radius_px: float, iterations: int = LEVELSET_ITERATIONS):
    cy, cx = center_px
    gradient_image = segmentation.inverse_gaussian_gradient(slice_2d, sigma=LEVELSET_GRADIENT_SIGMA)
    init_level_set = segmentation.disk_level_set(
        slice_2d.shape, center=(cy, cx), radius=max(initial_radius_px, 2.0)
    )
    evolved = segmentation.morphological_geodesic_active_contour(
        gradient_image, num_iter=iterations, init_level_set=init_level_set,
        smoothing=LEVELSET_SMOOTHING, balloon=LEVELSET_BALLOON,
    )

    yy, xx = np.mgrid[0:slice_2d.shape[0], 0:slice_2d.shape[1]]
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    evolved = evolved.astype(bool) & (dist <= max_radius_px)

    contour = _center_contour(evolved, (int(round(cy)), int(round(cx))))
    return contour


def circle_contour(center_px, radius_px: float, n_points: int = 48):
    """A perfect circle, not a detected wall — used to seed every path
    point with a "maximally round" starting contour right after a Series
    is prepared, so there's a complete (if rough) tube immediately rather
    than an empty one until each point is segmented by hand. Same [row,
    col] convention as threshold_contour/levelset_contour's output, so
    callers can't tell the difference."""
    cy, cx = center_px
    return [
        (cy + radius_px * math.sin(theta), cx + radius_px * math.cos(theta))
        for theta in (2 * math.pi * i / n_points for i in range(n_points))
    ]
