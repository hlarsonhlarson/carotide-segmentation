"""Experimental CTA outer-wall proposal and sectional high-density analysis.

The output is a review proposal, not a plaque diagnosis. Areas are measured
in this MPR plane; they are deliberately not presented as 3D volumes.
"""
import numpy as np
from scipy import ndimage as ndi
from skimage.draw import polygon
from skimage.measure import points_in_poly
from .lumen_measurements import contour_metrics


def _ray_radii(contour, center, theta, pixel_mm):
    points = np.asarray(contour, dtype=float) - center
    a, edge = points, np.roll(points, -1, axis=0) - points
    def cross(x, y):
        return x[..., 0] * y[..., 1] - x[..., 1] * y[..., 0]
    radii = []
    for angle in theta:
        direction = np.array([np.sin(angle), np.cos(angle)])
        denominator = cross(direction, edge)
        valid = abs(denominator) > 1e-10
        t = np.divide(cross(a, edge), denominator, out=np.zeros(len(a)), where=valid)
        u = np.divide(cross(a, direction), denominator, out=np.zeros(len(a)), where=valid)
        hits = t[valid & (t > 0) & (u >= -1e-8) & (u < 1 - 1e-8)]
        if len(hits) != 1:
            raise ValueError("Lumen must be star-shaped around the path center")
        radii.append(float(hits[0]) * pixel_mm)
    return np.asarray(radii)


def _closed_boundary(cost, radii, max_jump_mm=.75, smoothness=.5):
    """Exact cyclic dynamic programming, enumerating every initial radius."""
    angles, candidates = cost.shape
    best_cost, best_path = np.inf, None
    for start in range(candidates):
        score = np.full(candidates, np.inf)
        score[start] = cost[0, start]
        parents = []
        for i in range(1, angles):
            difference = abs(radii[i - 1, :, None] - radii[i, None, :])
            transition = np.where(difference <= max_jump_mm, smoothness * difference**2, np.inf)
            total = score[:, None] + transition
            parent = np.argmin(total, axis=0)
            score = total[parent, np.arange(candidates)] + cost[i]
            parents.append(parent)
        gap = abs(radii[-1] - radii[0, start])
        closed = score + np.where(gap <= max_jump_mm, smoothness * gap**2, np.inf)
        end = int(np.argmin(closed))
        if closed[end] < best_cost:
            best_cost = closed[end]
            path = [end]
            for parent in reversed(parents):
                path.append(int(parent[path[-1]]))
            best_path = np.asarray(path[::-1])
    if best_path is None:
        raise ValueError("No smooth closed wall boundary within the search limits")
    return best_path


def measure_wall_section(image, lumen_px, wall_px, pixel_mm, calcium_hu, min_component_mm2=.5):
    """Measure a pair of nested, nonintersecting rings, independent of detection."""
    image, lumen, wall = (np.asarray(value, dtype=float) for value in (image, lumen_px, wall_px))
    if (image.ndim != 2 or not np.isfinite(image).all()
            or not np.isfinite([pixel_mm, calcium_hu, min_component_mm2]).all()
            or pixel_mm <= 0 or min_component_mm2 <= 0):
        raise ValueError("Invalid sectional measurement input")
    for ring in (lumen, wall):
        if ring.ndim != 2 or ring.shape[1] != 2:
            raise ValueError("Expected row/column contour vertices")
        contour_metrics(np.column_stack((np.zeros(len(ring)), ring * pixel_mm)))
        if np.any(ring < 0) or np.any(ring > np.asarray(image.shape) - 1):
            raise ValueError("Contours must remain inside the MPR image")
    if not points_in_poly(lumen, wall).all():
        raise ValueError("Outer wall must enclose the entire lumen")
    # Vertex inclusion alone misses a lumen edge crossing a concave wall.
    def cross(a, b):
        return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    c, d = wall, np.roll(wall, -1, axis=0)
    for a, b in zip(lumen, np.roll(lumen, -1, axis=0)):
        overlap = np.all(np.maximum(np.minimum(a, b), np.minimum(c, d)) <=
                         np.minimum(np.maximum(a, b), np.maximum(c, d)) + 1e-9, axis=1)
        intersects = ((cross(b-a, c-a) * cross(b-a, d-a) <= 1e-12)
                      & (cross(d-c, a-c) * cross(d-c, b-c) <= 1e-12))
        if np.any(overlap & intersects):
            raise ValueError("Outer wall and lumen boundaries intersect or touch")
    lumen_mask, wall_mask = np.zeros(image.shape, bool), np.zeros(image.shape, bool)
    lumen_mask[polygon(lumen[:, 0], lumen[:, 1], image.shape)] = True
    wall_mask[polygon(wall[:, 0], wall[:, 1], image.shape)] = True
    annulus = wall_mask & ~lumen_mask
    labels, _ = ndi.label(annulus & (image >= calcium_hu))
    counts = np.bincount(labels.ravel())
    keep = counts * pixel_mm**2 >= min_component_mm2
    keep[0] = False
    high_density = keep[labels]
    return {
        "wall_contour_px": wall.tolist(),
        "high_density_pixels": np.argwhere(high_density).tolist(),
        "annulus_area_mm2": float(annulus.sum() * pixel_mm**2),
        "high_density_area_mm2": float(high_density.sum() * pixel_mm**2),
    }


def analyze_wall(image, lumen_px, pixel_mm, calcium_hu, min_component_mm2=.5,
                 min_thickness_mm=.3, max_thickness_mm=6.):
    image = np.asarray(image, dtype=float)
    lumen = np.asarray(lumen_px, dtype=float)
    if (image.ndim != 2 or not np.isfinite(image).all() or
            not np.isfinite([pixel_mm, min_component_mm2, min_thickness_mm, max_thickness_mm]).all() or pixel_mm <= 0 or
            not 0 < min_thickness_mm < max_thickness_mm <= 6 or
            not np.isfinite(calcium_hu) or min_component_mm2 <= 0):
        raise ValueError("Invalid wall-analysis input")
    if lumen.ndim != 2 or lumen.shape[1] != 2:
        raise ValueError("Expected row/column lumen vertices")
    contour_metrics(np.column_stack((np.zeros(len(lumen)), lumen * pixel_mm)))
    center = (np.asarray(image.shape) - 1) / 2
    theta = np.linspace(0, 2 * np.pi, 96, endpoint=False)
    inner = _ray_radii(lumen, center, theta, pixel_mm)
    thickness = np.arange(min_thickness_mm, max_thickness_mm + 1e-8, .2)
    radii = inner[:, None] + thickness
    def sample(r):
        rows = center[0] + np.sin(theta)[:, None] * r / pixel_mm
        cols = center[1] + np.cos(theta)[:, None] * r / pixel_mm
        return ndi.map_coordinates(image, [rows, cols], order=1, mode="constant", cval=np.nan)
    inside, outside = sample(radii - .25), sample(radii + .25)
    gradient = (inside - outside) / .5
    # A bright structure is not an outer-wall landmark. Exclude it from the
    # boundary reward rather than following a bone/calcium edge.
    valid = np.isfinite(gradient) & (inside < calcium_hu) & (outside < calcium_hu)
    cost = np.where(valid, -np.clip(gradient / 100., -2., 2.), np.inf)
    chosen = _closed_boundary(cost, radii)
    strengths = gradient[np.arange(len(theta)), chosen]
    confidence = float(np.mean(strengths >= 30))
    if confidence < .65:
        raise ValueError("Outer-wall edge is not visible on enough angles; manual review required")
    if np.mean((chosen == 0) | (chosen == len(thickness) - 1)) > .25:
        raise ValueError("Wall boundary reaches search limits; review ROI and contour")
    outer = radii[np.arange(len(theta)), chosen]
    wall = np.column_stack((center[0] + np.sin(theta) * outer / pixel_mm,
                            center[1] + np.cos(theta) * outer / pixel_mm))
    return {
        **measure_wall_section(image, lumen, wall, pixel_mm, calcium_hu, min_component_mm2),
        "max_radial_thickness_mm": float((outer - inner).max()),
        "edge_support_fraction": confidence,
        "status": "proposal", "validated": False,
        "parameters": {"calcium_hu": calcium_hu, "min_component_mm2": min_component_mm2,
                       "min_thickness_mm": min_thickness_mm, "max_thickness_mm": max_thickness_mm},
        "algorithm_version": "cyclic-radial-wall-1",
    }
