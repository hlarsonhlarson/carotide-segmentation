"""Experimental sectional plaque composition on CTA.

Within one MPR section, the region between the lumen contour and the outer
wall (plus any calcium the lumen contour swallowed) is split by attenuation:

- calcified: >= the effective calcium threshold, connected components of at
  least ``min_component_mm2``. The effective threshold is the larger of
  ``calcium_hu`` and ``CALCIUM_OVER_LUMEN`` x the median lumen HU: on a
  bright CTA (lumen ~600 HU) a fixed threshold would call the lumen calcium. Calcium inside the lumen contour is counted here
  and reported separately, because a threshold lumen contour on CTA often
  includes bright wall calcium.
- atheromatous (low-attenuation, lipid-rich/necrotic candidate):
  ``fat_hu`` <= HU < ``lipid_hu``, components of at least
  ``min_component_mm2``, at least ``edge_margin_mm`` inside the outer wall
  so that partial volume with perivascular fat is not called lipid core.
- fibrous: the remaining wall/plaque pixels from ``lipid_hu`` up to
  ``fibrous_max_hu``.
- between ``fibrous_max_hu`` and the calcium threshold: reported as contrast
  partial volume (the blurred lumen edge), never as plaque.
- below ``fat_hu``: reported as excluded (fat or air partial volume, or a
  wall contour drawn outside the vessel), never as plaque.

HU thresholds on contrast CTA are approximations: calcium blooms, contrast
leaks into low-attenuation ranges by partial volume, and published cut-offs
differ between studies. Results are review proposals, not histology.
"""
import numpy as np
from scipy import ndimage as ndi

ALGORITHM_VERSION = "hu-composition-1"
DEFAULTS = {"lipid_hu": 50.0, "fat_hu": -50.0, "fibrous_max_hu": 150.0, "edge_margin_mm": 0.5}
# Contrast lumen HU varies little within a section (p99 ~1.05x median on real
# CTA); wall calcium is typically >= 1.5x. 1.3x separates them with margin.
CALCIUM_OVER_LUMEN = 1.3


def _components(mask, min_pixels):
    labels, _ = ndi.label(mask, structure=np.ones((3, 3)))
    counts = np.bincount(labels.ravel())
    keep = counts >= min_pixels
    keep[0] = False
    return keep[labels]


def classify_plaque(image, lumen_mask, wall_mask, pixel_mm, calcium_hu, min_component_mm2,
                    lipid_hu=DEFAULTS["lipid_hu"], fat_hu=DEFAULTS["fat_hu"],
                    fibrous_max_hu=DEFAULTS["fibrous_max_hu"], edge_margin_mm=DEFAULTS["edge_margin_mm"]):
    image = np.asarray(image, dtype=float)
    lumen_mask = np.asarray(lumen_mask, dtype=bool)
    wall_mask = np.asarray(wall_mask, dtype=bool)
    if not (np.isfinite([pixel_mm, calcium_hu, min_component_mm2, lipid_hu, fat_hu, fibrous_max_hu,
                         edge_margin_mm]).all()
            and pixel_mm > 0 and min_component_mm2 > 0 and edge_margin_mm >= 0
            and fat_hu < lipid_hu < fibrous_max_hu < calcium_hu):
        raise ValueError("Composition thresholds must satisfy fat < lipid < fibrous max < calcium HU")
    if image.shape != lumen_mask.shape or image.shape != wall_mask.shape:
        raise ValueError("Image and masks must share one MPR grid")

    min_pixels = max(1, int(np.ceil(min_component_mm2 / pixel_mm ** 2)))
    pixel_area = pixel_mm ** 2
    annulus = wall_mask & ~lumen_mask

    lumen_reference = float(np.median(image[lumen_mask])) if lumen_mask.any() else None
    effective_calcium = float(calcium_hu)
    if lumen_reference is not None:
        effective_calcium = max(effective_calcium, CALCIUM_OVER_LUMEN * lumen_reference)
    calcified = _components(wall_mask & (image >= effective_calcium), min_pixels)
    calcium_in_lumen = calcified & lumen_mask
    plaque = annulus | calcified

    depth_mm = ndi.distance_transform_edt(wall_mask) * pixel_mm
    lipid_candidates = (annulus & ~calcified & (image >= fat_hu) & (image < lipid_hu)
                        & (depth_mm >= edge_margin_mm))
    lipid = _components(lipid_candidates, min_pixels)
    excluded = annulus & ~calcified & (image < fat_hu)
    contrast = annulus & ~calcified & (image >= fibrous_max_hu)
    fibrous = plaque & ~calcified & ~lipid & ~excluded & ~contrast

    def area(mask):
        return float(mask.sum() * pixel_area)

    plaque_area = area(plaque & ~excluded & ~contrast)
    def fraction(mask):
        return float(area(mask) / plaque_area) if plaque_area > 0 else None

    return {
        "plaque_area_mm2": plaque_area,
        "calcified_area_mm2": area(calcified),
        "lipid_area_mm2": area(lipid),
        "fibrous_area_mm2": area(fibrous),
        "excluded_low_hu_area_mm2": area(excluded),
        "contrast_partial_volume_area_mm2": area(contrast),
        "calcium_in_lumen_area_mm2": area(calcium_in_lumen),
        "calcified_fraction": fraction(calcified),
        "lipid_fraction": fraction(lipid),
        "fibrous_fraction": fraction(fibrous),
        "calcified_pixels": np.argwhere(calcified).tolist(),
        "lipid_pixels": np.argwhere(lipid).tolist(),
        "fibrous_pixels": np.argwhere(fibrous).tolist(),
        "lumen_reference_hu": lumen_reference,
        "calcium_hu_effective": effective_calcium,
        "parameters": {"calcium_hu": float(calcium_hu), "lipid_hu": float(lipid_hu), "fat_hu": float(fat_hu), "fibrous_max_hu": float(fibrous_max_hu),
                       "min_component_mm2": float(min_component_mm2), "edge_margin_mm": float(edge_margin_mm)},
        "algorithm_version": ALGORITHM_VERSION,
        "validated": False,
    }
