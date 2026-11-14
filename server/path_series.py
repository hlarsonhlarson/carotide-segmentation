"""Per-branch Series (SimVascular's ContourGroup): resamples the clicked +
auto-stitched path (dense, ~1 voxel between points) to even arc-length
spacing suitable as segmentation control points, with a tangent at each —
ports simvascular_build_mesh.py's resample_by_arclength (already written
and reasoned about for this same purpose), operating on world-mm points
instead of the SimVascular-specific path object.
"""
from .coordinates import voxel_to_world

import numpy as np
from scipy.ndimage import gaussian_filter1d

CONTROL_POINT_SPACING_MM = 2.0
SMOOTHING_SIGMA_POINTS = 3.0


def voxel_path_to_world_mm(path_voxel, spacing_mm, origin_mm):
    return voxel_to_world(path_voxel, spacing_mm, origin_mm).tolist()


def smooth_path(points_mm, sigma_points: float = SMOOTHING_SIGMA_POINTS):
    """The clicked+auto-stitched path moves one voxel at a time (26-
    connectivity geodesic stitching — see centerline_path.stitch), which
    gives a visibly staircased polyline, and noisy tangent directions for
    whatever control point lands on a staircase step. A light Gaussian
    smooth along each coordinate (mode="nearest" so the endpoints don't
    get pulled inward) fixes both — purely a display/tangent-quality
    improvement, doesn't change which vessel the path follows at this
    density of points."""
    pts = np.asarray(points_mm, dtype=np.float64)
    if len(pts) < 5:
        return points_mm
    smoothed = gaussian_filter1d(pts, sigma=sigma_points, axis=0, mode="nearest")
    return smoothed.tolist()


def resample_by_arclength(points_mm, spacing_mm: float = CONTROL_POINT_SPACING_MM):
    if not np.isfinite(spacing_mm) or spacing_mm <= 0:
        raise ValueError("Path sampling spacing must be positive")
    if len(points_mm) < 2:
        return list(points_mm)
    pts = np.asarray(points_mm, dtype=float)
    distance = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    pts = pts[np.r_[True, distance > 1e-9]]
    if len(pts) < 2:
        return pts.tolist()
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    samples = np.r_[np.arange(0, arc[-1], spacing_mm), arc[-1]]
    return np.column_stack([np.interp(samples, arc, pts[:, axis]) for axis in range(3)]).tolist()


def parallel_transport_frames(tangents):
    """Rotation-minimizing frames, with no arbitrary up-axis switches."""
    normals, binormals = [], []
    for tangent in np.asarray(tangents, dtype=float):
        t = tangent / max(np.linalg.norm(tangent), 1e-12)
        if normals:
            n = normals[-1] - np.dot(normals[-1], t) * t
        else:
            seed = np.eye(3)[np.argmin(np.abs(t))]
            n = seed - np.dot(seed, t) * t
        if np.linalg.norm(n) < 1e-8:
            seed = np.eye(3)[np.argmin(np.abs(t))]
            n = seed - np.dot(seed, t) * t
        n /= np.linalg.norm(n)
        normals.append(n)
        binormals.append(np.cross(t, n))
    return np.asarray(normals), np.asarray(binormals)


def tangents_for(points_mm):
    """Unit tangent at each point via central differences (forward/backward
    at the endpoints)."""
    pts = np.asarray(points_mm, dtype=np.float64)
    n = len(pts)
    tangents = np.zeros_like(pts)
    for i in range(n):
        if i == 0:
            d = pts[min(1, n - 1)] - pts[0]
        elif i == n - 1:
            d = pts[-1] - pts[-2]
        else:
            d = pts[i + 1] - pts[i - 1]
        norm = np.linalg.norm(d)
        tangents[i] = d / norm if norm > 1e-9 else np.array([1.0, 0.0, 0.0])
    return tangents.tolist()


def build_series(path_voxel, spacing_mm, origin_mm,
                  control_spacing_mm: float = CONTROL_POINT_SPACING_MM):
    """Returns [{point_mm, tangent_mm}, ...] — the resampled control points
    a Series will hold one segmentation per, in order."""
    world_pts = voxel_path_to_world_mm(path_voxel, spacing_mm, origin_mm)
    world_pts = smooth_path(world_pts)
    control_pts = resample_by_arclength(world_pts, control_spacing_mm)
    tangents = tangents_for(control_pts)
    normals, binormals = parallel_transport_frames(tangents)
    return [{"point_mm": p, "tangent_mm": t, "u_vec": u.tolist(), "v_vec": v.tolist()}
            for p, t, u, v in zip(control_pts, tangents, normals, binormals)]
