"""Oblique plane extraction — the one genuinely new geometric primitive the
SimVascular-style per-point segmentation step needs. Everything else in
this codebase only ever reads axis-aligned planes (dicom_io/app.py's
axial/coronal/sagittal slices); segmenting along a path means sampling the
plane perpendicular to the path's tangent at each point, which isn't
axis-aligned in general.
"""
import numpy as np
from scipy import ndimage as ndi
from .coordinates import world_to_voxel


def _orthonormal_in_plane_axes(tangent_mm):
    """Gram-Schmidt: given a unit tangent, pick two unit vectors (u, v)
    spanning the plane perpendicular to it. The "up" seed just needs to be
    non-parallel to the tangent; which one is picked doesn't matter for
    segmentation (the contour is rotationally arbitrary either way)."""
    t = np.asarray(tangent_mm, dtype=np.float64)
    t = t / max(np.linalg.norm(t), 1e-9)
    up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(up, t)) > 0.9:
        up = np.array([0.0, 1.0, 0.0])
    u = up - np.dot(up, t) * t
    u = u / max(np.linalg.norm(u), 1e-9)
    v = np.cross(t, u)
    return u, v


def sample_oblique_slice(array: np.ndarray, spacing_mm, origin_mm, point_mm, tangent_mm,
                          size_px: int = 160, px_spacing_mm: float = 0.25,
                          background_value: float | None = None, frame=None):
    """Samples a size_px x size_px plane perpendicular to tangent_mm,
    centered at point_mm (both world-mm, (z,y,x)-axis-ordered like the rest
    of this codebase), via trilinear interpolation.

    Returns (slice_2d, u_vec, v_vec, grid_origin_world_mm) where slice_2d[j,i]
    is the sample at grid_origin_world_mm + i*px_spacing_mm*u_vec +
    j*px_spacing_mm*v_vec — i.e. i indexes u (columns/x-like), j indexes v
    (rows/y-like), matching how the rest of this project treats 2D arrays
    for PNG export (row-major, row 0 = one edge of the plane).
    """
    point = np.asarray(point_mm, dtype=np.float64)
    u, v = _orthonormal_in_plane_axes(tangent_mm) if frame is None else np.asarray(frame, dtype=float)

    half = size_px / 2.0
    offsets = (np.arange(size_px) - half + 0.5) * px_spacing_mm  # pixel-center offsets
    grid_origin_world_mm = point + offsets[0] * u + offsets[0] * v

    # World-mm position of every grid cell: point + i*px*u + j*px*v
    ii, jj = np.meshgrid(offsets, offsets)  # jj varies along rows (v), ii along cols (u)
    world = point[None, None, :] + ii[..., None] * u[None, None, :] + jj[..., None] * v[None, None, :]

    # World mm -> voxel index, same convention used throughout this codebase
    # (meshing.py, mra_pipeline.py): voxel = (world - origin) / spacing, no
    # rotation component.
    voxel = world_to_voxel(world, spacing_mm, origin_mm)
    coords = [voxel[..., 0], voxel[..., 1], voxel[..., 2]]  # z, y, x

    if background_value is None:
        background_value = float(array.min())
    slice_2d = ndi.map_coordinates(array, coords, order=1, mode="constant", cval=background_value)
    return slice_2d.astype(np.float32), u, v, grid_origin_world_mm
