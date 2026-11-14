"""Conservative CTA lumen estimate from radial half-height edge crossings.

Only measured crossings are returned. Missing edges and bright outer peaks
are review failures, never replaced by a circle or interpolated through plaque.
"""
import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter1d


def fwhm_contour(image, center_px, pixel_mm, max_radius_mm=6., n_angles=96):
    image = np.asarray(image, dtype=float)
    if pixel_mm <= 0 or max_radius_mm <= pixel_mm or n_angles < 8:
        raise ValueError("Invalid radial sampling parameters")
    if image.ndim != 2 or not np.isfinite(image).all():
        raise ValueError("Expected a finite 2D HU image")
    center = np.asarray(center_px, dtype=float)
    rows, cols = np.indices(image.shape)
    core = image[np.hypot(rows - center[0], cols - center[1]) * pixel_mm <= .75]
    if not core.size:
        return None, {"reason": "center_outside_image"}
    lumen_hu = float(np.median(core))
    # Do not infer a vessel from air or unenhanced soft tissue at the center.
    if lumen_hu < 100:
        return None, {"reason": "low_center_enhancement", "lumen_hu": lumen_hu}
    theta = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    radii = np.linspace(0, max_radius_mm, int(np.ceil(max_radius_mm / (pixel_mm / 2))) + 1)
    rr = center[0] + np.sin(theta)[:, None] * radii / pixel_mm
    cc = center[1] + np.cos(theta)[:, None] * radii / pixel_mm
    profiles = map_coordinates(image, [rr, cc], order=1, mode="constant", cval=np.nan)
    profiles = gaussian_filter1d(profiles, .6, axis=1)
    boundary, strengths = [], []
    for profile in profiles:
        background = float(np.median(profile[-max(3, len(radii) // 6):]))
        contrast = lumen_hu - background
        if not np.isfinite(background) or contrast < 60:
            return None, {"reason": "weak_or_missing_edge", "lumen_hu": lumen_hu}
        level = background + .5 * contrast
        crossing = np.flatnonzero(
            (profile[:-1] >= level) & (profile[1:] < level) & (radii[1:] >= .3))
        if not crossing.size:
            return None, {"reason": "open_radial_profile", "lumen_hu": lumen_hu}
        i = int(crossing[0])
        # An outer high-density peak may be calcium or bone. Do not silently
        # substitute an adjacent radius and erase a possible stenosis.
        if np.max(profile[:i + 1]) > lumen_hu + max(150., .5 * contrast):
            return None, {"reason": "bright_peak_requires_review", "lumen_hu": lumen_hu}
        fraction = (profile[i] - level) / max(profile[i] - profile[i + 1], 1e-9)
        boundary.append(radii[i] + fraction * (radii[i + 1] - radii[i]))
        strengths.append(contrast)
    boundary = np.asarray(boundary)
    contour = np.column_stack((center[0] + np.sin(theta) * boundary / pixel_mm,
                               center[1] + np.cos(theta) * boundary / pixel_mm))
    return contour, {
        "reason": None, "lumen_hu": lumen_hu,
        "min_edge_contrast_hu": float(min(strengths)),
        "radii_mm": boundary.tolist(),
        "review_required": True,
    }
