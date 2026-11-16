"""Keep the lumen that actually passes through the four operator points.

Intensity growth can enter a parallel vein of the same brightness. The vein
does not lie on the shortest path from the CCA through the bifurcation to the
ICA and the ECA. The kept mask is the cross-section around that path, plus
the proximal CCA and distal ICA continuations. Radius comes from rays in the
plane across the path, capped by the tubular CCA, so a bulb and a stenosis
stay and a neighbouring lumen does not.
"""
import heapq

import numpy as np
from scipy.spatial import cKDTree

from .appearance import nearest_model
from .geometry import index_to_physical, offset_mm, physical_to_voxel

_FACE = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))


def restrict_to_route(image, array, accepted, models, params, cca, bifurcation, ica, eca):
    """Return the voxel set on the four-point route, or None if a leg is missing."""
    cca = _snap(image, accepted, cca)
    bifurcation = _snap(image, accepted, bifurcation)
    ica = _snap(image, accepted, ica)
    eca = _snap(image, accepted, eca)
    if None in (cca, bifurcation, ica, eca):
        return None
    trunk = _geodesic(image, accepted, cca, bifurcation)
    ica_leg = _geodesic(image, accepted, bifurcation, ica) if trunk else None
    eca_leg = _geodesic(image, accepted, bifurcation, eca) if trunk else None
    if not trunk or not ica_leg or not eca_leg:
        return None

    samples = _samples(image, trunk + ica_leg[1:] + eca_leg[1:])
    reference = _reference_radius(image, array, samples, cca, models, params)
    cap = params.max_radius_factor * max(reference, 0.8)
    radii = np.array([
        min(_section_radius(image, array, point, tangent, models, params, cap), cap)
        for point, tangent in samples
    ], dtype=float)

    proximal = _extend(image, array, accepted, models, params, cap, trunk, forward=False)
    distal = _extend(image, array, accepted, models, params, cap, trunk[:-1] + ica_leg, forward=True)
    extra = _samples(image, proximal + distal)
    if extra:
        extra_radii = np.array([
            min(_section_radius(image, array, point, tangent, models, params, cap), cap)
            for point, tangent in extra
        ], dtype=float)
        samples = samples + extra
        radii = np.concatenate([radii, extra_radii])

    return _within_radius(image, accepted, samples, radii)


def _snap(image, accepted, voxel, radius_mm=2.5):
    if voxel in accepted:
        return voxel
    origin = index_to_physical(image, voxel)
    best, best_d = None, radius_mm
    for candidate in accepted:
        dist = float(np.linalg.norm(index_to_physical(image, candidate) - origin))
        if dist <= best_d:
            best, best_d = candidate, dist
    return best


def _geodesic(image, accepted, start, goal):
    if start not in accepted or goal not in accepted:
        return None
    dist = {start: 0.0}
    prev = {start: None}
    heap = [(0.0, start)]
    while heap:
        travelled, node = heapq.heappop(heap)
        if travelled > dist[node] + 1e-9:
            continue
        if node == goal:
            break
        z, y, x = node
        for dz, dy, dx in _FACE:
            nxt = (z + dz, y + dy, x + dx)
            if nxt not in accepted:
                continue
            step = abs(offset_mm(image, dz, dy, dx))
            candidate = travelled + step
            if candidate < dist.get(nxt, 1e100):
                dist[nxt] = candidate
                prev[nxt] = node
                heapq.heappush(heap, (candidate, nxt))
    if goal not in prev:
        return None
    path = [goal]
    while path[-1] != start:
        path.append(prev[path[-1]])
    path.reverse()
    return path


def _samples(image, voxels, stride_mm=1.5):
    """Physical points and tangents, about one sample per stride."""
    if len(voxels) < 2:
        point = index_to_physical(image, voxels[0])
        return [(point, np.array([0.0, 0.0, 1.0]))]
    chosen = [voxels[0]]
    previous = index_to_physical(image, voxels[0])
    for voxel in voxels[1:]:
        point = index_to_physical(image, voxel)
        if float(np.linalg.norm(point - previous)) >= stride_mm:
            chosen.append(voxel)
            previous = point
    if chosen[-1] != voxels[-1]:
        chosen.append(voxels[-1])
    samples = []
    for index, voxel in enumerate(chosen):
        point = index_to_physical(image, voxel)
        nxt = index_to_physical(image, chosen[min(len(chosen) - 1, index + 1)])
        prv = index_to_physical(image, chosen[max(0, index - 1)])
        tangent = nxt - prv
        norm = float(np.linalg.norm(tangent))
        tangent = np.array([0.0, 0.0, 1.0]) if norm < 1e-6 else tangent / norm
        samples.append((point, tangent))
    return samples


def _basis(tangent):
    helper = np.array([1.0, 0.0, 0.0]) if abs(float(tangent[0])) < 0.9 else np.array([0.0, 1.0, 0.0])
    first = np.cross(tangent, helper)
    first /= np.linalg.norm(first)
    second = np.cross(tangent, first)
    return first, second


def _section_radius(image, array, origin, tangent, models, params, cap):
    """Median distance to the lumen wall in the plane across the path."""
    model = nearest_model(models, origin)
    contrast = model.contrast
    if contrast < params.min_separation:
        return 0.8
    first, second = _basis(tangent)
    lengths = []
    step = 0.45
    for angle in np.linspace(0.0, np.pi, 8, endpoint=False):
        direction = np.cos(angle) * first + np.sin(angle) * second
        length = 0.0
        while length + step <= cap:
            probe = origin + direction * (length + step)
            voxel = physical_to_voxel(image, probe)
            if voxel is None:
                break
            value = float(array[voxel])
            membership = (value - model.soft) / contrast
            if membership < params.membership_min or value > model.calcium_limit(params):
                break
            length += step
        lengths.append(max(length, step))
    return float(np.median(lengths))


def _reference_radius(image, array, samples, cca, models, params):
    origin = index_to_physical(image, cca)
    near = [
        _section_radius(image, array, point, tangent, models, params, 12.0)
        for point, tangent in samples
        if float(np.linalg.norm(point - origin)) <= params.radius_sample_mm
    ]
    if not near:
        near = [_section_radius(image, array, samples[0][0], samples[0][1], models, params, 12.0)]
    return float(np.median(near))


def _extend(image, array, accepted, models, params, cap, path, forward):
    """Walk one end away from the bifurcation while the section stays tubular."""
    if len(path) < 2:
        return []
    tip = path[-1] if forward else path[0]
    anchor = index_to_physical(image, path[0] if forward else path[-1])
    seen = set(path)
    extra = []
    for _ in range(500):
        best = None
        best_gain = 0.3
        tip_point = index_to_physical(image, tip)
        z, y, x = tip
        for dz, dy, dx in _FACE:
            nxt = (z + dz, y + dy, x + dx)
            if nxt not in accepted or nxt in seen:
                continue
            gain = float(np.linalg.norm(index_to_physical(image, nxt) - anchor) - np.linalg.norm(tip_point - anchor))
            if gain > best_gain:
                best, best_gain = nxt, gain
        if best is None:
            break
        point = index_to_physical(image, best)
        tangent = point - tip_point
        norm = float(np.linalg.norm(tangent))
        tangent = np.array([0.0, 0.0, 1.0]) if norm < 1e-6 else tangent / norm
        if _section_radius(image, array, point, tangent, models, params, cap) > cap * 0.98:
            break
        extra.append(best)
        seen.add(best)
        tip = best
    return extra


def _within_radius(image, accepted, samples, radii):
    points = np.vstack([point for point, _ in samples])
    tree = cKDTree(points)
    coords = np.array(list(accepted), dtype=int)
    physical = np.vstack([index_to_physical(image, voxel) for voxel in coords])
    distance, nearest = tree.query(physical)
    margin = 0.6 * float(np.min(np.abs(image.GetSpacing())))
    keep = distance <= radii[nearest] + margin
    return {tuple(int(v) for v in voxel) for voxel in coords[keep]}
