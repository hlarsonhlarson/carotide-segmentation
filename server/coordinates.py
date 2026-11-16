"""Internal image coordinates are z,y,x; surfaces use x,y,z. Units: millimetres."""
import numpy as np


def voxel_to_world(points, spacing_mm, origin_mm):
    return np.asarray(origin_mm) + np.asarray(points) * np.asarray(spacing_mm)


def world_to_voxel(points, spacing_mm, origin_mm):
    return (np.asarray(points) - np.asarray(origin_mm)) / np.asarray(spacing_mm)


def surface_xyz(point_zyx):
    return list(point_zyx[::-1])


def voxel_to_lps(points_zyx, spacing_zyx, origin_zyx, direction_lps):
    """Convert stored voxel indices to patient LPS millimetres in XYZ order."""
    xyz = np.asarray(points_zyx, dtype=float)[..., ::-1]
    return (xyz * np.asarray(spacing_zyx)[::-1]) @ np.asarray(direction_lps).T + np.asarray(origin_zyx)[::-1]


def lps_to_voxel(points_lps, spacing_zyx, origin_zyx, direction_lps):
    """Inverse of voxel_to_lps; supports arrays of points."""
    local = (np.asarray(points_lps) - np.asarray(origin_zyx)[::-1]) @ np.linalg.inv(np.asarray(direction_lps)).T
    return (local / np.asarray(spacing_zyx)[::-1])[..., ::-1]
