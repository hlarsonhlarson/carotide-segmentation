"""Connected growth from lumen seeds.

A voxel is added only from a face-neighbour already in the lumen, and only
when its intensity sits between local soft tissue and local calcium. Nothing
is closed, dilated, or bridged across a rejected gap.
"""
from dataclasses import dataclass, field
import heapq

import numpy as np

from .appearance import SeedModel, nearest_model
from .geometry import index_to_physical, voxel_volume_mm3
from .params import LumenParams

_FACE = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))


@dataclass
class GrowResult:
    accepted: set
    rejected: dict
    scores: dict
    fov_exit: bool = False
    volume_capped: bool = False
    notes: list = field(default_factory=list)


def _in_bounds(shape, zyx):
    z, y, x = zyx
    return 0 <= z < shape[0] and 0 <= y < shape[1] and 0 <= x < shape[2]


class _LocalLumen:
    def __init__(self, image, params: LumenParams):
        self.image = image
        self.params = params
        self.samples: dict[tuple, list] = {}

    def _cell(self, zyx):
        physical = index_to_physical(self.image, zyx)
        return tuple(np.floor(physical / self.params.context_cell_mm).astype(int))

    def add(self, zyx, value: float, score: float):
        if score < self.params.high_score:
            return
        bucket = self.samples.setdefault(self._cell(zyx), [])
        if len(bucket) < 80:
            bucket.append(float(value))

    def level(self, zyx, fallback: float) -> float:
        cells = [self._cell(zyx)]
        total = []
        for cell in cells:
            total.extend(self.samples.get(cell, ()))
        if len(total) < self.params.min_local_samples:
            base = self._cell(zyx)
            for offset in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
                total.extend(self.samples.get(tuple(np.add(base, offset)), ()))
        if len(total) < self.params.min_local_samples:
            return float(fallback)
        return float(np.median(total))


def grow_lumen(image, array: np.ndarray, models: list[SeedModel], params: LumenParams) -> GrowResult:
    shape = array.shape
    local = _LocalLumen(image, params)
    accepted: set[tuple] = set()
    rejected: dict[tuple, str] = {}
    scores: dict[tuple, float] = {}
    heap: list = []
    token = 0
    volume = 0.0
    voxel_mm3 = voxel_volume_mm3(image)
    fov_exit = False
    capped = False

    def push_neighbor(zyx):
        nonlocal token
        if zyx in accepted or zyx in rejected:
            return
        physical = index_to_physical(image, zyx)
        model = nearest_model(models, physical)
        contrast = model.contrast
        if contrast < params.min_separation:
            rejected[zyx] = "low"
            return
        slack = params.local_lumen_slack * contrast
        lumen = float(np.clip(
            local.level(zyx, model.lumen),
            model.lumen - slack,
            model.lumen + slack,
        ))
        value = float(array[zyx])
        if value > lumen + params.calcium_contrast_k * contrast:
            rejected[zyx] = "core" if value > lumen + params.calcium_core_k * contrast else "calcium"
            return
        # The floor stays on the seed contrast. A drifting local level must
        # not redefine soft tissue as lumen.
        membership = (value - model.soft) / contrast
        if membership < params.membership_min:
            rejected[zyx] = "low"
            return
        score = 1.0 - abs(value - lumen) / contrast
        token += 1
        heapq.heappush(heap, (-score, token, zyx, membership))

    for model in models:
        for zyx in model.core_voxels:
            if zyx in accepted:
                continue
            accepted.add(zyx)
            value = float(array[zyx])
            scores[zyx] = 1.0
            local.add(zyx, value, 1.0)
            volume += voxel_mm3

    for zyx in list(accepted):
        z, y, x = zyx
        for dz, dy, dx in _FACE:
            nxt = (z + dz, y + dy, x + dx)
            if not _in_bounds(shape, nxt):
                fov_exit = True
                continue
            push_neighbor(nxt)

    while heap:
        if volume >= params.max_volume_mm3:
            capped = True
            break
        neg_score, _, zyx, membership = heapq.heappop(heap)
        if zyx in accepted or zyx in rejected:
            continue
        accepted.add(zyx)
        scores[zyx] = membership
        local.add(zyx, float(array[zyx]), -neg_score)
        volume += voxel_mm3
        z, y, x = zyx
        for dz, dy, dx in _FACE:
            nxt = (z + dz, y + dy, x + dx)
            if not _in_bounds(shape, nxt):
                fov_exit = True
                continue
            push_neighbor(nxt)

    notes = []
    if capped:
        notes.append("Рост остановлен предохранителем объёма. Маска может быть неполной, нужна проверка.")
    if fov_exit:
        notes.append("Просвет дошёл до края поля обзора. Конец сегмента там не анатомический.")
    return GrowResult(accepted, rejected, scores, fov_exit, capped, notes)


def component_contains_both(accepted, voxel_a, voxel_b) -> bool:
    """6-connected membership without building a full label image twice."""
    if voxel_a not in accepted or voxel_b not in accepted:
        return False
    seen = {voxel_a}
    stack = [voxel_a]
    while stack:
        zyx = stack.pop()
        if zyx == voxel_b:
            return True
        z, y, x = zyx
        for dz, dy, dx in _FACE:
            nxt = (z + dz, y + dy, x + dx)
            if nxt in accepted and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return False


def paint(shape, voxels, dtype=np.uint8) -> np.ndarray:
    out = np.zeros(shape, dtype=dtype)
    for z, y, x in voxels:
        out[z, y, x] = 1
    return out
