"""Point-to-point geodesic "auto-dotyazhka" (auto-stitch) for manual
centerline tracing, when no seed mask exists yet to calibrate against —
generalizes carotid_seg.vesselness.bridge_gap's proven approach (minimum-
cost path via skimage.graph.route_through_array through an intensity-
plausibility cost field) from "bridge a gap between two mask components" to
"connect two arbitrary clicked points," since a manual trace starts from
nothing but clicks.

Calibration source differs by modality for the same reason the rest of this
project splits CT/MR: CT has a real Hounsfield scale to threshold against
(sampled from the user's own clicked points' neighborhoods, same spirit as
vesselness.adaptive_lumen_hu_band but seeded from points instead of a
mask); TOF-MR doesn't, so a local intensity percentile within the search
crop stands in instead (same idea as mra_pipeline._regrow_locally).
"""
import weakref

import numpy as np
from scipy import ndimage as ndi
from skimage.graph import route_through_array

SEARCH_MARGIN_MM = 15.0
CT_SAMPLE_RADIUS_VOX = 2
CT_BAND_PAD_HU = 40.0
CT_BAND_UPPER_RATIO = 1.5
MR_LOCAL_PERCENTILE = 70.0


# Inside the lumen the cost rises toward the wall, so the geodesic follows the
# vessel axis instead of the shortest route hugging the inner curve. With a
# flat in-lumen cost, a real CCA trace ran ~2.9 mm off the lumen centre.
CENTERING_WEIGHT = 2.0


def _centered(good: np.ndarray, spacing_mm):
    depth_mm = ndi.distance_transform_edt(good, sampling=spacing_mm)
    inside = 1.0 + CENTERING_WEIGHT / (depth_mm + 0.5)
    return np.where(good, inside, 25.0).astype(np.float32)


def _ct_cost_field(crop: np.ndarray, anchor_points_local, spacing_mm=(1.0, 1.0, 1.0)):
    samples = []
    for p in anchor_points_local:
        lo = [max(0, int(c) - CT_SAMPLE_RADIUS_VOX) for c in p]
        hi = [min(s, int(c) + CT_SAMPLE_RADIUS_VOX + 1) for c, s in zip(p, crop.shape)]
        sl = tuple(slice(a, b) for a, b in zip(lo, hi))
        samples.append(crop[sl].ravel())
    vals = np.concatenate(samples) if samples else crop.ravel()
    lo_hu, hi_hu = np.percentile(vals, [5, 95])
    lo_hu -= CT_BAND_PAD_HU
    # Contrast is brightest at the lumen centre and varies along the vessel,
    # so endpoint samples underestimate it (a real CCA centre was ~600 HU
    # against a 570 HU band). Capping just above them excluded the vessel
    # axis and pushed the path to the blurred wall. Keep only calcium and
    # cortical bone out.
    hi_hu = max(hi_hu + CT_BAND_PAD_HU, hi_hu * CT_BAND_UPPER_RATIO)
    good = (crop >= lo_hu) & (crop <= hi_hu)
    return _centered(good, spacing_mm)


def _mr_cost_field(crop: np.ndarray, spacing_mm=(1.0, 1.0, 1.0)):
    thresh = float(np.percentile(crop, MR_LOCAL_PERCENTILE))
    good = crop > thresh
    return _centered(good, spacing_mm)


def stitch(array: np.ndarray, spacing_mm, modality: str, point_a_vox, point_b_vox,
           search_margin_mm: float = SEARCH_MARGIN_MM):
    """Returns a list of [z,y,x] voxel points (inclusive of both endpoints)
    forming a plausible minimum-cost path between the two points, cropped
    to a local bounding box (padded by search_margin_mm) around them for
    speed — consistent with bridge_gap's approach for the same reason: a
    thin geodesic path stays local by construction, no need to search the
    whole volume.
    """
    point_a = np.asarray(point_a_vox, dtype=np.int64)
    point_b = np.asarray(point_b_vox, dtype=np.int64)
    spacing = np.asarray(spacing_mm)

    margin_vox = np.ceil(search_margin_mm / spacing).astype(int)
    lo = np.maximum(np.minimum(point_a, point_b) - margin_vox, 0)
    hi = np.minimum(np.maximum(point_a, point_b) + margin_vox + 1, np.asarray(array.shape))
    sub = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    crop = array[sub]

    a_local = tuple(int(v) for v in (point_a - lo))
    b_local = tuple(int(v) for v in (point_b - lo))

    if modality == "CT":
        cost = _ct_cost_field(crop, [a_local, b_local], spacing)
    else:
        cost = _mr_cost_field(crop, spacing)

    indices, _ = route_through_array(cost, a_local, b_local, fully_connected=True)
    return [[int(idx[i] + lo[i]) for i in range(3)] for idx in indices]


# Moving or adding one control point only changes its two neighbouring hops,
# but every edit restitches the whole path (~0.3 s per hop on a real CTA).
# Hops are memoized per source volume; the volume of a loaded case is never
# modified in place.
_HOP_CACHE: dict = {}
_HOP_CACHE_MAX = 64


def _cached_stitch(array, spacing_mm, modality, a, b, search_margin_mm):
    ref = _HOP_CACHE.get(id(array))
    if ref is None or ref[0]() is not array:
        for key in [k for k, (w, _) in _HOP_CACHE.items() if w() is None]:
            del _HOP_CACHE[key]
        ref = (weakref.ref(array), {})
        _HOP_CACHE[id(array)] = ref
    hops = ref[1]
    key = (tuple(int(v) for v in a), tuple(int(v) for v in b), tuple(float(v) for v in spacing_mm),
           modality, float(search_margin_mm))
    if key not in hops:
        if len(hops) >= _HOP_CACHE_MAX:
            hops.pop(next(iter(hops)))
        hops[key] = stitch(array, spacing_mm, modality, a, b, search_margin_mm=search_margin_mm)
    return [list(p) for p in hops[key]]


def stitch_full_path(array: np.ndarray, spacing_mm, modality: str, control_points,
                      search_margin_mm: float = SEARCH_MARGIN_MM):
    """Rebuilds the full dense path from scratch given the ordered sparse
    control points (what the user actually clicked/placed) — stitches each
    consecutive pair and concatenates, dropping the duplicate shared point
    at each join. Used whenever a control point is added, moved, or
    deleted, since any of those can change the stitched route on both
    sides of that point. `search_margin_mm` is shared across every hop —
    the default suits the short hops between manually-guided points; a
    single-hop auto-trace between two far-apart endpoints (see app.py's
    /auto_trace) passes a much larger one, since real vessel tortuosity
    can swing tens of mm off the straight line between a distant start
    and end."""
    if len(control_points) < 2:
        return [list(p) for p in control_points]
    path = [list(control_points[0])]
    for a, b in zip(control_points, control_points[1:]):
        stitched = _cached_stitch(array, spacing_mm, modality, a, b, search_margin_mm)
        path.extend(stitched[1:])
    return path
