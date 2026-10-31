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
from skimage import measure, segmentation

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

    contour = _center_contour(binary, (cy, cx))
    return contour  # list of [row, col] float pixel coords, or None


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
