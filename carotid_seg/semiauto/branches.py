"""Centerline and branch labels after a lumen mask already exists.

The mask boundary is not rewritten. Skeleton voxels are sources for a
geodesic labeling inside the mask, and the places where a segment ends:

* CCA, proximally — the last centerline point before the local radius exceeds
  ``max_radius_factor`` times the tubular radius sampled beside the CCA seed.
* ICA, distally — the start of a sustained bone-like ring outside the lumen
  (entrance of the carotid canal). The distal seed only names the branch.
* ECA — the other bifurcation branch, only until its own first split or
  ``eca_max_length_mm``. A third point picks the branch when two are similar;
  it is not the cutting plane.

The junction voxel is labeled CCA.
"""
from dataclasses import dataclass, field
import heapq

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

from .geometry import index_to_physical, offset_mm
from .params import LumenParams

_FACES = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
_NEIGH = tuple(
    (dz, dy, dx)
    for dz in (-1, 0, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
    if (dz, dy, dx) != (0, 0, 0)
)


@dataclass
class BranchResult:
    labels: np.ndarray
    uncertainty: np.ndarray
    requests: list
    samples: list
    reference_radius_mm: float
    proximal_index: tuple | None
    distal_index: tuple | None
    eca_labeled: bool
    stats: dict = field(default_factory=dict)


def _nodes_from_mask(mask: np.ndarray):
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return []
    z0, y0, x0 = coords.min(axis=0)
    z1, y1, x1 = coords.max(axis=0) + 1
    skeleton = skeletonize(mask[z0:z1, y0:y1, x0:x1])
    return [(int(z + z0), int(y + y0), int(x + x0)) for z, y, x in np.argwhere(skeleton)]


def _graph(nodes, image):
    present = set(nodes)
    adj = {p: [] for p in present}
    for z, y, x in present:
        for dz, dy, dx in _NEIGH:
            nxt = (z + dz, y + dy, x + dx)
            if nxt in present and nxt > (z, y, x):
                weight = offset_mm(image, dz, dy, dx)
                adj[(z, y, x)].append((nxt, weight))
                adj[nxt].append(((z, y, x), weight))
    return adj


def _drop_edge(adj, a, b):
    adj[a] = [(m, w) for m, w in adj[a] if m != b]
    adj[b] = [(m, w) for m, w in adj[b] if m != a]


def _prune_spurs(adj, min_mm: float):
    changed = True
    while changed:
        changed = False
        for leaf in [n for n, edges in adj.items() if len(edges) == 1]:
            if len(adj.get(leaf, [])) != 1:
                continue
            length = 0.0
            prev, cur = leaf, adj[leaf][0][0]
            length += adj[leaf][0][1]
            chain = [leaf]
            while len(adj[cur]) == 2 and length < min_mm:
                nxt, weight = next((m, w) for m, w in adj[cur] if m != prev)
                chain.append(cur)
                length += weight
                prev, cur = cur, nxt
            if length < min_mm and len(adj[cur]) != 2:
                chain.append(cur)
                for a, b in zip(chain, chain[1:]):
                    _drop_edge(adj, a, b)
                changed = True
                break
    return adj


def _distances(adj, start, blocked=None, goal=None):
    dist = {start: 0.0}
    prev = {start: None}
    heap = [(0.0, start)]
    while heap:
        d, node = heapq.heappop(heap)
        if d != dist.get(node):
            continue
        if goal is not None and node == goal:
            break
        for nxt, weight in adj.get(node, []):
            if nxt == blocked:
                continue
            nd = d + weight
            if nd < dist.get(nxt, 1e100):
                dist[nxt] = nd
                prev[nxt] = node
                heapq.heappush(heap, (nd, nxt))
    return dist, prev


def _path(prev, start, goal):
    if goal not in prev:
        return None
    nodes = [goal]
    while nodes[-1] != start:
        nodes.append(prev[nodes[-1]])
    nodes.reverse()
    return nodes


def _downstream(adj, start, blocked):
    dist, _ = _distances(adj, start, blocked=blocked)
    return max(dist.values()) if dist else 0.0


def _walk_away(adj, start, coming_from):
    """Follow the centerline away from ``coming_from``. At a fork, take the longest arm."""
    path = []
    prev = coming_from
    cur = start
    seen = set()
    if coming_from is not None:
        seen.add(coming_from)
    while cur not in seen:
        seen.add(cur)
        path.append(cur)
        options = [(m, w) for m, w in adj.get(cur, []) if m != prev]
        if not options:
            break
        nxt = options[0][0] if len(options) == 1 else max(options, key=lambda item: _downstream(adj, item[0], cur))[0]
        prev, cur = cur, nxt
    return path


def _snap(nodes, image, voxel, radius_mm):
    target = index_to_physical(image, voxel)
    best, best_d = None, 1e100
    for node in nodes:
        dist = float(np.linalg.norm(index_to_physical(image, node) - target))
        if dist < best_d:
            best, best_d = node, dist
    if best is None or best_d > radius_mm:
        return None
    return best


def _bone_fraction(array, mask, image, voxel, radius_mm, threshold, params: LumenParams, tangent):
    """Bone-like fraction of a ring in the plane across the lumen, not ahead of it."""
    inner = max(0.0, radius_mm + params.bone_ring_gap_mm)
    outer = inner + params.bone_ring_width_mm
    origin = index_to_physical(image, voxel)
    tangent = np.asarray(tangent, dtype=float)
    step = np.array([abs(offset_mm(image, *delta)) for delta in ((1, 0, 0), (0, 1, 0), (0, 0, 1))])
    reach = np.maximum(1, np.ceil(outer / np.maximum(step, 1e-3)).astype(int))
    thickness = float(np.max(step))
    z0, y0, x0 = voxel
    bright = total = 0
    for z in range(max(0, z0 - reach[0]), min(array.shape[0], z0 + reach[0] + 1)):
        for y in range(max(0, y0 - reach[1]), min(array.shape[1], y0 + reach[1] + 1)):
            for x in range(max(0, x0 - reach[2]), min(array.shape[2], x0 + reach[2] + 1)):
                if mask[z, y, x]:
                    continue
                delta = index_to_physical(image, (z, y, x)) - origin
                if abs(float(np.dot(delta, tangent))) > thickness:
                    continue
                dist = float(np.linalg.norm(delta))
                if inner <= dist <= outer:
                    total += 1
                    if float(array[z, y, x]) >= threshold:
                        bright += 1
    if total < 8:
        return 0.0
    return bright / total


def _sustained(points, flags, min_length_mm, image):
    start = None
    length = 0.0
    for index, point in enumerate(points):
        if not flags[index]:
            start = None
            length = 0.0
            continue
        if start is None:
            start = index
            length = 0.0
        elif index > 0:
            length += float(np.linalg.norm(
                index_to_physical(image, point) - index_to_physical(image, points[index - 1])
            ))
        if length >= min_length_mm:
            return start
    return None


def _between(path, a, b):
    ia, ib = path.index(a), path.index(b)
    lo, hi = sorted((ia, ib))
    return path[lo:hi + 1]


def _collect_distances(adj, start, blocked):
    dist, _ = _distances(adj, start, blocked=blocked)
    return dist


def _median_radius(dt, dist, limit_mm):
    values = [float(dt[node]) for node, travelled in dist.items() if travelled <= limit_mm]
    return float(np.median(values)) if values else 0.0


def _eca_stop(adj, start, blocked, branch_dist, params: LumenParams):
    for node, travelled in sorted(branch_dist.items(), key=lambda item: item[1]):
        if travelled > params.eca_max_length_mm:
            return params.eca_max_length_mm
        long_children = 0
        for nxt, _ in adj.get(node, []):
            if branch_dist.get(nxt, -1) <= travelled:
                continue
            if _downstream(adj, nxt, node) >= params.min_branch_mm:
                long_children += 1
        # The carotid bifurcation itself is not the ECA's first division.
        if long_children >= 2 and travelled >= params.min_branch_mm:
            return travelled
    if not branch_dist:
        return 0.0
    return min(params.eca_max_length_mm, max(branch_dist.values()))


def _geodesic_labels(mask, image, skeleton_labels):
    labels = np.zeros(mask.shape, dtype=np.uint8)
    best = {}
    heap = []
    for node, label in skeleton_labels.items():
        if not (0 <= node[0] < mask.shape[0] and 0 <= node[1] < mask.shape[1] and 0 <= node[2] < mask.shape[2]):
            continue
        if not mask[node]:
            continue
        best[node] = (0.0, int(label))
        heapq.heappush(heap, (0.0, node, int(label)))
    while heap:
        dist, node, label = heapq.heappop(heap)
        stored = best.get(node)
        if stored is None or dist != stored[0] or label != stored[1]:
            continue
        labels[node] = label
        z, y, x = node
        for dz, dy, dx in _FACES:
            nxt = (z + dz, y + dy, x + dx)
            if nxt[0] < 0 or nxt[1] < 0 or nxt[2] < 0 or nxt[0] >= mask.shape[0] or nxt[1] >= mask.shape[1] or nxt[2] >= mask.shape[2]:
                continue
            if not mask[nxt]:
                continue
            nd = dist + offset_mm(image, dz, dy, dx)
            if nd < best.get(nxt, (1e100, -1))[0]:
                best[nxt] = (nd, label)
                heapq.heappush(heap, (nd, nxt, label))
    return labels


def _section_area(labels, image, voxel, tangent, reach_mm):
    step = np.array([abs(offset_mm(image, *delta)) for delta in ((1, 0, 0), (0, 1, 0), (0, 0, 1))])
    half = np.maximum(1, np.ceil(reach_mm / step).astype(int))
    z, y, x = voxel
    origin = index_to_physical(image, voxel)
    thickness = float(np.max(step))
    count = 0
    for zz in range(max(0, z - half[0]), min(labels.shape[0], z + half[0] + 1)):
        for yy in range(max(0, y - half[1]), min(labels.shape[1], y + half[1] + 1)):
            for xx in range(max(0, x - half[2]), min(labels.shape[2], x + half[2] + 1)):
                if labels[zz, yy, xx] == 0:
                    continue
                delta = index_to_physical(image, (zz, yy, xx)) - origin
                along = float(np.dot(delta, tangent))
                if abs(along) > thickness / 2:
                    continue
                if float(np.linalg.norm(delta - along * tangent)) <= reach_mm:
                    count += 1
    return float(count * float(np.prod(step)) / thickness)


def _samples(labels, image, centerline, dt, params: LumenParams):
    if len(centerline) < 3:
        return []
    travelled = [0.0]
    for a, b in zip(centerline, centerline[1:]):
        travelled.append(travelled[-1] + float(np.linalg.norm(
            index_to_physical(image, b) - index_to_physical(image, a)
        )))
    total = travelled[-1]
    samples = []
    for index, node in enumerate(centerline):
        tangent = index_to_physical(image, centerline[min(len(centerline) - 1, index + 1)])
        tangent = tangent - index_to_physical(image, centerline[max(0, index - 1)])
        norm = float(np.linalg.norm(tangent))
        if norm < 1e-6:
            continue
        tangent /= norm
        samples.append({
            "voxel": node,
            "point_xyz": index_to_physical(image, node).tolist(),
            "tangent": tangent.tolist(),
            "area_mm2": _section_area(labels, image, node, tangent, float(dt[node]) + 1.5),
            "radius_mm": float(dt[node]),
            "label": int(labels[node]),
            "from_end_mm": float(min(travelled[index], total - travelled[index])),
        })
    return samples


def _mark_narrow(uncertainty, image, sample):
    tangent = np.asarray(sample["tangent"], dtype=float)
    step = np.array([abs(offset_mm(image, *delta)) for delta in ((1, 0, 0), (0, 1, 0), (0, 0, 1))])
    half = np.maximum(1, np.ceil(3.0 / step).astype(int))
    z, y, x = sample["voxel"]
    origin = index_to_physical(image, sample["voxel"])
    thickness = float(np.max(step))
    for zz in range(max(0, z - half[0]), min(uncertainty.shape[0], z + half[0] + 1)):
        for yy in range(max(0, y - half[1]), min(uncertainty.shape[1], y + half[1] + 1)):
            for xx in range(max(0, x - half[2]), min(uncertainty.shape[2], x + half[2] + 1)):
                delta = index_to_physical(image, (zz, yy, xx)) - origin
                if abs(float(np.dot(delta, tangent))) <= thickness:
                    uncertainty[zz, yy, xx] |= 64


def label_branches(image, array, mask, cca_voxel, ica_voxel, eca_point, models, params: LumenParams) -> BranchResult:
    requests = []
    uncertainty = np.zeros(mask.shape, dtype=np.uint16)
    empty = np.zeros(mask.shape, dtype=np.uint8)
    nodes = _nodes_from_mask(mask)
    if not nodes:
        requests.append("Скелет маски пуст. Ветви не размечены.")
        return BranchResult(empty, uncertainty, requests, [], 0.0, None, None, False)

    spacing = np.array([abs(offset_mm(image, *delta)) for delta in ((1, 0, 0), (0, 1, 0), (0, 0, 1))])
    dt = ndi.distance_transform_edt(mask, sampling=spacing)
    adj = _prune_spurs(_graph(nodes, image), params.min_spur_mm)
    alive = [n for n, edges in adj.items() if edges]
    if cca_voxel in adj and not adj[cca_voxel]:
        alive.append(cca_voxel)
    snap_r = max(8.0, float(dt[cca_voxel]) + 4.0, float(dt[ica_voxel]) + 4.0)
    cca_node = _snap(alive, image, cca_voxel, snap_r)
    ica_node = _snap(alive, image, ica_voxel, snap_r)
    if cca_node is None or ica_node is None:
        requests.append("Точка ОСА или ВСА не легла на центр маски. Обе точки должны быть внутри одного просвета.")
        return BranchResult(empty, uncertainty, requests, [], 0.0, None, None, False)

    _, prev = _distances(adj, cca_node, goal=ica_node)
    trunk = _path(prev, cca_node, ica_node)
    if not trunk:
        requests.append("Скелет между ОСА и ВСА разорван. Разрыв не заполняется.")
        return BranchResult(empty, uncertainty, requests, [], 0.0, None, None, False)

    coming_ica = trunk[1] if len(trunk) > 1 else None
    coming_cca = trunk[-2] if len(trunk) > 1 else None
    proximal = _walk_away(adj, cca_node, coming_ica)
    distal = _walk_away(adj, ica_node, coming_cca)
    full = list(reversed(proximal[1:])) + trunk + distal[1:]

    near = [
        float(dt[node]) for node in trunk
        if float(np.linalg.norm(index_to_physical(image, node) - index_to_physical(image, cca_voxel))) <= params.radius_sample_mm
    ]
    reference = float(np.median(near)) if near else float(dt[cca_node])
    radius_limit = params.max_radius_factor * max(reference, 0.5)
    proximal_stop = proximal[-1]
    for node in proximal:
        if float(dt[node]) > radius_limit:
            break
        proximal_stop = node
    if float(dt[proximal[0]]) > radius_limit:
        requests.append("Точка ОСА уже в широком просвете. Поставьте её в трубчатую ОСА, не в луковицу и не в дугу.")

    ica_model = next(model for model in models if model.name == "ica")
    flags = []
    for index, node in enumerate(distal):
        tangent = index_to_physical(image, distal[min(len(distal) - 1, index + 1)])
        tangent = tangent - index_to_physical(image, distal[max(0, index - 1)])
        norm = float(np.linalg.norm(tangent))
        tangent = np.array([0.0, 0.0, 1.0]) if norm < 1e-6 else tangent / norm
        flags.append(
            _bone_fraction(array, mask, image, node, float(dt[node]), ica_model.calcium_limit(params), params, tangent)
            >= params.bone_fraction
        )
    canal = _sustained(distal, flags, params.encasement_length_mm, image)
    if canal is None:
        distal_stop = distal[-1]
        requests.append("Вход в сонный канал по костному кольцу не найден. Дистальный конец ВСА нужно проверить.")
        uncertainty[distal[-1]] |= 8
    else:
        distal_stop = distal[max(0, canal - 1)]
        if canal == 0:
            requests.append("Дистальная точка уже у костного канала. Сдвиньте её проксимальнее, в шейную ВСА.")

    centerline = set(full)
    candidates = []
    seen_starts = set()
    for node in trunk:
        for nxt, _ in adj.get(node, []):
            if nxt in centerline or nxt in seen_starts:
                continue
            length = _downstream(adj, nxt, node)
            if length < params.min_branch_mm:
                continue
            seen_starts.add(nxt)
            candidates.append({
                "attach": node,
                "start": nxt,
                "length": length,
                "radius": _median_radius(dt, _collect_distances(adj, nxt, node), params.min_branch_mm),
            })

    eca = None
    ambiguous = False
    if eca_point is not None:
        target = np.asarray(eca_point, dtype=float)
        best, best_d = None, 1e100
        for item in candidates:
            dist = _collect_distances(adj, item["start"], item["attach"])
            nearest = min(float(np.linalg.norm(index_to_physical(image, n) - target)) for n in dist)
            if nearest < best_d:
                best, best_d = item, nearest
        allowance = params.eca_seed_max_distance_mm + (best["radius"] if best else 0.0)
        if best is None or best_d > allowance:
            requests.append("Третья точка не попала в ветвь бифуркации. НСА не размечена.")
        else:
            eca = best
    elif len(candidates) == 1:
        eca = candidates[0]
    elif len(candidates) >= 2:
        ranked = sorted(candidates, key=lambda item: item["radius"], reverse=True)
        ratio = ranked[1]["radius"] / ranked[0]["radius"] if ranked[0]["radius"] else 1.0
        if ratio >= params.eca_ambiguity_ratio:
            ambiguous = True
            requests.append("Начальный отдел НСА неоднозначен. Нужна третья точка внутри просвета НСА.")
            for item in ranked[:2]:
                uncertainty[item["attach"]] |= 16
        else:
            eca = ranked[0]
    else:
        requests.append("Отдельная ветвь НСА не найдена.")

    if eca is not None:
        bifurcation = eca["attach"]
        branch_dist = _collect_distances(adj, eca["start"], eca["attach"])
        stop = _eca_stop(adj, eca["start"], eca["attach"], branch_dist, params)
        eca_nodes = [node for node, travelled in branch_dist.items() if travelled <= stop + 1e-6]
    else:
        segment = _between(full, cca_node, ica_node)
        bifurcation = max(segment, key=lambda node: float(dt[node]))
        eca_nodes = []
        if not ambiguous:
            uncertainty[bifurcation] |= 16

    on_path = set(full)
    cca_nodes = set(_between(full, proximal_stop, bifurcation))
    ica_nodes = set(_between(full, bifurcation, distal_stop))
    skeleton_labels = {node: 0 for node in adj}
    for node in cca_nodes:
        skeleton_labels[node] = 1
    for node in ica_nodes:
        skeleton_labels[node] = 2
    skeleton_labels[bifurcation] = 1
    for node in eca_nodes:
        if node in on_path:
            continue
        skeleton_labels[node] = 3

    labels = _geodesic_labels(mask, image, skeleton_labels)
    uncertainty[mask & (labels == 0)] |= 32
    trunk_line = [node for node in full if skeleton_labels.get(node) in (1, 2)]
    samples = _samples(labels, image, trunk_line, dt, params)
    if eca is not None and eca_nodes:
        ordered = sorted(branch_dist, key=branch_dist.get)
        eca_line = [bifurcation] + [node for node in ordered if node in set(eca_nodes)]
        samples.extend(_samples(labels, image, eca_line, dt, params))
    interior = [s for s in samples if s["from_end_mm"] >= params.endpoint_ignore_mm and s["label"] in (1, 2)]
    if interior and min(s["area_mm2"] for s in interior) < params.subocclusion_area_mm2:
        narrow = min(interior, key=lambda s: s["area_mm2"])
        requests.append(
            "Очень узкий просвет (возможная субокклюзия). Он оставлен как есть, без сглаживания; нужна локальная проверка."
        )
        _mark_narrow(uncertainty, image, narrow)

    if labels[cca_voxel] != 1 or labels[ica_voxel] != 2:
        requests.append("Одна из исходных точек не попала в свой сегмент. Проверьте положение точек и концы маски.")

    return BranchResult(
        labels,
        uncertainty,
        requests,
        samples,
        reference,
        proximal_stop,
        distal_stop,
        bool(eca_nodes),
        {
            "reference_radius_mm": reference,
            "radius_limit_mm": radius_limit,
            "n_skeleton": len(alive),
            "n_candidates": len(candidates),
            "ambiguous_eca": ambiguous,
            "bifurcation_xyz": index_to_physical(image, bifurcation).tolist(),
        },
    )
