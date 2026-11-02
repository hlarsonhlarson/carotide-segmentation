"""Internal image coordinates are z,y,x; surfaces use x,y,z. Units: millimetres."""
import numpy as np


def voxel_to_world(points, spacing_mm, origin_mm):
    return np.asarray(origin_mm) + np.asarray(points) * np.asarray(spacing_mm)


def world_to_voxel(points, spacing_mm, origin_mm):
    return (np.asarray(points) - np.asarray(origin_mm)) / np.asarray(spacing_mm)


def surface_xyz(point_zyx):
    return list(point_zyx[::-1])
