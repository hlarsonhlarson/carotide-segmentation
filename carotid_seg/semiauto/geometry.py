"""Physical coordinates of a SimpleITK image.

Index arrays from ``GetArrayFromImage`` are ordered (k, j, i) = (z, y, x).
SimpleITK physical points and ``GetSpacing`` / ``GetSize`` are (x, y, z).
Distances use the direction matrix, so a rotated acquisition is not treated
as an identity axis stack.
"""
import numpy as np
import SimpleITK as sitk


def spacing_zyx(image: sitk.Image) -> np.ndarray:
    sx, sy, sz = image.GetSpacing()
    return np.array([sz, sy, sx], dtype=float)


def voxel_volume_mm3(image: sitk.Image) -> float:
    sx, sy, sz = image.GetSpacing()
    return float(sx * sy * sz)


def index_to_physical(image: sitk.Image, zyx) -> np.ndarray:
    z, y, x = (float(v) for v in zyx)
    return np.asarray(image.TransformContinuousIndexToPhysicalPoint((x, y, z)), dtype=float)


def physical_to_continuous_zyx(image: sitk.Image, point_xyz) -> np.ndarray:
    x, y, z = image.TransformPhysicalPointToContinuousIndex(tuple(float(v) for v in point_xyz))
    return np.array([z, y, x], dtype=float)


def physical_to_voxel(image: sitk.Image, point_xyz):
    """Nearest voxel inside the image, or None if the point is outside."""
    try:
        continuous = physical_to_continuous_zyx(image, point_xyz)
    except RuntimeError:
        return None
    zyx = tuple(int(np.rint(v)) for v in continuous)
    shape = sitk.GetArrayViewFromImage(image).shape
    if any(i < 0 or i >= n for i, n in zip(zyx, shape)):
        return None
    return zyx


def physical_points(image: sitk.Image, voxels_zyx: np.ndarray) -> np.ndarray:
    if len(voxels_zyx) == 0:
        return np.zeros((0, 3), dtype=float)
    out = np.empty((len(voxels_zyx), 3), dtype=float)
    for i, zyx in enumerate(voxels_zyx):
        out[i] = index_to_physical(image, zyx)
    return out


def offset_mm(image: sitk.Image, dz: int, dy: int, dx: int) -> float:
    sx, sy, sz = image.GetSpacing()
    direction = np.array(image.GetDirection(), dtype=float).reshape(3, 3)
    step = direction @ np.array([dx * sx, dy * sy, dz * sz], dtype=float)
    return float(np.linalg.norm(step))


def image_from_array(array: np.ndarray, reference: sitk.Image) -> sitk.Image:
    image = sitk.GetImageFromArray(array)
    image.CopyInformation(reference)
    return image
