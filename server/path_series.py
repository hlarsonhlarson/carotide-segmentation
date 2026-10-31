"""Per-branch Series (SimVascular's ContourGroup): resamples the clicked +
auto-stitched path (dense, ~1 voxel between points) to even arc-length
spacing suitable as segmentation control points, with a tangent at each —
ports simvascular_build_mesh.py's resample_by_arclength (already written
and reasoned about for this same purpose), operating on world-mm points
instead of the SimVascular-specific path object.
"""
import math

import numpy as np
from scipy.ndimage import gaussian_filter1d

CONTROL_POINT_SPACING_MM = 2.0
SMOOTHING_SIGMA_POINTS = 3.0


def voxel_path_to_world_mm(path_voxel, spacing_mm, origin_mm):
    spacing = np.asarray(spacing_mm)
    origin = np.asarray(origin_mm)
    return [list(origin + np.asarray(p) * spacing) for p in path_voxel]


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
    if len(points_mm) < 2:
        return list(points_mm)
    kept = [points_mm[0]]
    acc = 0.0
    for a, b in zip(points_mm, points_mm[1:]):
        acc += math.dist(a, b)
        if acc >= spacing_mm:
            kept.append(b)
            acc = 0.0
    if kept[-1] != points_mm[-1]:
        kept.append(points_mm[-1])
    return kept


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
    return [{"point_mm": p, "tangent_mm": t} for p, t in zip(control_pts, tangents)]
