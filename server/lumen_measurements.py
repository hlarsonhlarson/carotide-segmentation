"""Geometric lumen measurements; no automatic anatomical reference selection."""
import numpy as np
from scipy.spatial import ConvexHull, distance


def contour_metrics(contour_mm):
    points = np.asarray(contour_mm, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3 or not np.isfinite(points).all():
        raise ValueError("A finite, closed planar contour with at least three points is required")
    if np.linalg.norm(points[0] - points[-1]) < 1e-8:
        points = points[:-1]
    if len(points) < 3 or np.any(np.linalg.norm(points - np.roll(points, 1, axis=0), axis=1) < 1e-8):
        raise ValueError("Contour has duplicate or insufficient vertices")
    centered = points - points.mean(axis=0)
    _, singular, axes = np.linalg.svd(centered, full_matrices=False)
    if singular[1] < 1e-8:
        raise ValueError("Contour is degenerate")
    if np.max(np.abs(centered @ axes[2])) > .01:
        raise ValueError("Contour is not planar within 0.01 mm")
    xy = centered @ axes[:2].T
    # Reject crossing and touching nonadjacent segments rather than reporting
    # an apparently plausible area for a self-intersecting manual contour.
    def orient(a, b, c):
        d, e = b - a, c - a
        return d[0] * e[1] - d[1] * e[0]
    n = len(xy)
    for i in range(n):
        a, b = xy[i], xy[(i + 1) % n]
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            c, d = xy[j], xy[(j + 1) % n]
            if (np.all(np.maximum(np.minimum(a, b), np.minimum(c, d)) <=
                       np.minimum(np.maximum(a, b), np.maximum(c, d)) + 1e-10)
                    and orient(a, b, c) * orient(a, b, d) <= 1e-12
                    and orient(c, d, a) * orient(c, d, b) <= 1e-12):
                raise ValueError("Contour self-intersects")
    area = abs(float(np.dot(xy[:, 0], np.roll(xy[:, 1], -1)) -
                     np.dot(xy[:, 1], np.roll(xy[:, 0], -1)))) / 2
    if area < 1e-8:
        raise ValueError("Contour has zero area")
    perimeter = float(np.linalg.norm(xy - np.roll(xy, 1, axis=0), axis=1).sum())
    hull = xy[ConvexHull(xy).vertices]
    edges = np.roll(hull, -1, axis=0) - hull
    normals = np.column_stack((-edges[:, 1], edges[:, 0]))
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    widths = np.ptp(hull @ normals.T, axis=0)
    return {
        "area_mm2": area,
        "perimeter_mm": perimeter,
        "equivalent_diameter_mm": float(2 * np.sqrt(area / np.pi)),
        "min_feret_mm": float(widths.min()),
        "max_feret_mm": float(distance.pdist(hull).max()),
        "compactness": float(4 * np.pi * area / perimeter**2),
    }


def nascet_estimate(lesion, reference):
    """Unvalidated diameter-ratio estimate using the same caliper definition."""
    dmin, dref = lesion["min_feret_mm"], reference["min_feret_mm"]
    if not np.isfinite([dmin, dref]).all() or dmin <= 0 or dref <= 0:
        raise ValueError("NASCET requires two nonzero measurable lumens")
    if dmin > dref:
        raise ValueError("Selected narrowing is wider than the reference; check section selection")
    return {
        "percent": float(100 * (1 - dmin / dref)),
        "diameter_definition": "minimum_feret",
        "validated": False,
        "reference_selection": "operator",
        "limitations": "Not applicable to near-occlusion or occlusion; normal distal ICA reference required.",
    }
