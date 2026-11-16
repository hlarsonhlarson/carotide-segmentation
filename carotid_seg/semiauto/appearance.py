"""Local lumen appearance from the neighbourhood of an operator seed.

The seed is a sample of contrast, not a segmentation boundary. Soft tissue is
the low tail of a shell that has actually left the lumen. Calcium is whatever
is brighter than the local lumen by a fraction of that same contrast.
"""
from dataclasses import dataclass, field

import numpy as np

from .geometry import index_to_physical, physical_to_voxel, spacing_zyx
from .params import LumenParams


@dataclass
class SeedModel:
    name: str
    point_xyz: np.ndarray
    voxel_zyx: tuple
    lumen: float
    soft: float
    reliable: bool
    message: str = ""
    core_voxels: list = field(default_factory=list)

    @property
    def contrast(self) -> float:
        return float(self.lumen - self.soft)

    def calcium_limit(self, params: LumenParams, lumen: float | None = None) -> float:
        level = self.lumen if lumen is None else lumen
        return float(level + params.calcium_contrast_k * self.contrast)

    def core_limit(self, params: LumenParams, lumen: float | None = None) -> float:
        level = self.lumen if lumen is None else lumen
        return float(level + params.calcium_core_k * self.contrast)


def _ball_voxels(image, voxel_zyx, radius_mm: float):
    array_shape = sitk_shape(image)
    step = spacing_zyx(image)
    radii = np.maximum(1, np.ceil(radius_mm / step).astype(int))
    z0, y0, x0 = voxel_zyx
    found = []
    for z in range(max(0, z0 - radii[0]), min(array_shape[0], z0 + radii[0] + 1)):
        for y in range(max(0, y0 - radii[1]), min(array_shape[1], y0 + radii[1] + 1)):
            for x in range(max(0, x0 - radii[2]), min(array_shape[2], x0 + radii[2] + 1)):
                dist = np.linalg.norm(index_to_physical(image, (z, y, x)) - index_to_physical(image, voxel_zyx))
                if dist <= radius_mm:
                    found.append((dist, (z, y, x)))
    return found


def sitk_shape(image):
    return image.GetSize()[::-1]


def fit_seed_model(image, array: np.ndarray, point_xyz, name: str, params: LumenParams) -> SeedModel:
    point = np.asarray(point_xyz, dtype=float)
    voxel = physical_to_voxel(image, point)
    if voxel is None:
        return SeedModel(name, point, (-1, -1, -1), 0.0, 0.0, False, f"Точка {name} вне изображения.")

    core = _ball_voxels(image, voxel, params.seed_radius_mm)
    if not core:
        return SeedModel(name, point, voxel, 0.0, 0.0, False, f"Пустая окрестность точки {name}.")
    core_voxels = [zyx for _, zyx in core]
    lumen = float(np.median([array[zyx] for zyx in core_voxels]))

    # Walk outward. The first shell that is only slightly darker than the
    # core is still contrast or partial volume, so it is not soft tissue.
    # Keep the darkest low-tail whose shell median has actually left the
    # lumen, and stop before a discontinuous drop into air.
    soft = None
    previous = None
    outer = params.seed_radius_mm
    origin = index_to_physical(image, voxel)
    while outer < params.max_shell_mm - 1e-6:
        inner = outer
        outer = min(params.max_shell_mm, outer + params.shell_step_mm)
        shell = []
        for dist, zyx in _ball_voxels(image, voxel, outer):
            if dist <= inner:
                continue
            shell.append(float(array[zyx]))
        if len(shell) < params.min_shell_voxels:
            continue
        values = np.asarray(shell, dtype=np.float64)
        candidate = float(np.percentile(values, params.shell_percentile))
        median = float(np.median(values))
        left_lumen = (
            lumen - candidate >= params.min_separation
            and median <= 0.5 * (lumen + candidate)
        )
        # The first large drop is the vessel wall. A later one, after a
        # tissue plateau exists, is air entering the shell.
        if previous is not None and previous - candidate > params.shell_discontinuity:
            if soft is not None:
                break
            if left_lumen:
                soft = candidate
                previous = candidate
            continue
        previous = candidate
        if left_lumen:
            soft = candidate
    if soft is None:
        return SeedModel(
            name, point, voxel, lumen, lumen, False,
            f"Вокруг точки {name} нет локального контраста просвет/мягкие ткани. "
            "Точка, возможно, в кальции, кости или неконтрастированном участке.",
            core_voxels,
        )

    model = SeedModel(name, origin, voxel, lumen, soft, True, "", core_voxels)
    accepted_core = []
    for zyx in core_voxels:
        value = float(array[zyx])
        if value <= model.calcium_limit(params) and value >= soft + params.membership_min * model.contrast:
            accepted_core.append(zyx)
    if voxel not in accepted_core:
        return SeedModel(
            name, origin, voxel, lumen, soft, False,
            f"Точка {name} не похожа на локальный просвет (кальций, край или мягкие ткани).",
            core_voxels,
        )
    model.core_voxels = accepted_core
    return model


def nearest_model(models: list[SeedModel], point_xyz) -> SeedModel:
    point = np.asarray(point_xyz, dtype=float)
    return min(models, key=lambda model: float(np.linalg.norm(model.point_xyz - point)))
