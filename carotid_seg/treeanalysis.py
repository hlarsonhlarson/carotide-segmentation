import numpy as np
import networkx as nx
from scipy.spatial import cKDTree

# Neighbor offsets in raw voxel-index space (axis0, axis1, axis2) — NOT
# necessarily x/y/z; we always carry the matching per-axis spacing alongside
# any array so distances stay physically correct regardless of NIfTI axis order.
_NEIGHBOR_OFFSETS = [
    (a, b, c)
    for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)
    if not (a == 0 and b == 0 and c == 0)
]


def _build_graph(skeleton: np.ndarray, spacing_mm) -> nx.Graph:
    coords = set(map(tuple, np.argwhere(skeleton)))
    s0, s1, s2 = spacing_mm
    g = nx.Graph()
    g.add_nodes_from(coords)
    for p in coords:
        for da, db, dc in _NEIGHBOR_OFFSETS:
            n = (p[0] + da, p[1] + db, p[2] + dc)
            if n in coords and n > p:
                dist = np.sqrt((da * s0) ** 2 + (db * s1) ** 2 + (dc * s2) ** 2)
                g.add_edge(p, n, weight=dist)
    return g


def _collapse_small_cycles(g: nx.Graph, max_cycle_length_mm: float) -> nx.Graph:
    """Thick vessels meeting at a junction often skeletonize into a small
    loop instead of one clean branch node (a known thinning artifact, not a
    real anatomical loop — the carotid tree has none). Merge any short cycle
    into its first node so the real trifurcation point becomes visible.
    """
    g = g.copy()
    changed = True
    while changed:
        changed = False
        for cycle in nx.cycle_basis(g):
            if len(cycle) < 3:
                continue
            edges = list(zip(cycle, cycle[1:] + cycle[:1]))
            if not all(g.has_edge(a, b) for a, b in edges):
                continue
            length = sum(g[a][b]["weight"] for a, b in edges)
            if length > max_cycle_length_mm:
                continue
            keep = cycle[0]
            external = []
            for n in cycle[1:]:
                for nb in g.neighbors(n):
                    if nb not in cycle:
                        external.append((nb, g[n][nb]["weight"]))
            g.remove_nodes_from(cycle[1:])
            for nb, w in external:
                if g.has_edge(keep, nb):
                    g[keep][nb]["weight"] = min(g[keep][nb]["weight"], w)
                else:
                    g.add_edge(keep, nb, weight=w)
            changed = True
            break
    return g


def _collapse_dense_hubs(g: nx.Graph, spacing_mm, max_hub_diameter_mm: float) -> nx.Graph:
    """`_collapse_small_cycles` only removes actual graph cycles (a closed
    loop, which can have every member node at degree 2 if it has no other
    connections). A non-tubular blob of mask (real anatomy: a bulb, or —
    seen on a real case — an adjacent gland/node that leaked into the lumen
    mask despite the HU/geometry guards elsewhere) skeletonizes into
    something worse: a locally clustered *tangle* of many genuinely
    degree>=3 nodes, which isn't one clean cycle and isn't a spur either
    (`_prune_spurs`'s walk stops the instant it hits a degree>=3 node, so it
    can't reach into a tangle at all). Left alone, this defeats
    `locate_bifurcation`'s per-node splitting test everywhere inside the
    tangle (removing any single node rarely isolates the ICA seed into
    exactly one component when there are many redundant paths through the
    mess) — so the search can end up picking some unrelated, trivially-clean
    single node far away (e.g. a minor irregularity right at the CCA's
    origin) as "the bifurcation" instead of failing loudly or finding the
    real one. Collapse each spatially-compact cluster of degree>=3 nodes
    into one representative node, same as a cycle collapse.
    """
    g = g.copy()
    spacing_mm = np.asarray(spacing_mm, dtype=float)
    while True:
        hub_nodes = [n for n in g.nodes if g.degree(n) >= 3]
        if len(hub_nodes) < 2:
            break
        # Cluster by *physical proximity*, not graph adjacency: on a real
        # case, the true trifurcation and e.g. the first ECA side-branch
        # (superior thyroid a.) taking off a few mm downstream showed up as
        # two separate degree>=3 nodes connected only via a short chain of
        # plain degree-2 path nodes between them — not a graph edge to each
        # other — so a "collapse graph-adjacent hub nodes" pass left both
        # standing, and the per-node bifurcation search downstream picked
        # neither correctly. Single-linkage clustering on distance catches
        # this regardless of how many plain nodes sit in between.
        pts = np.array(hub_nodes, dtype=float) * spacing_mm
        tree = cKDTree(pts)
        pairs = tree.query_pairs(r=max_hub_diameter_mm)
        if not pairs:
            break
        parent = list(range(len(hub_nodes)))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for a, b in pairs:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        clusters = {}
        for i in range(len(hub_nodes)):
            clusters.setdefault(find(i), []).append(hub_nodes[i])

        collapsed_any = False
        for members in clusters.values():
            if len(members) < 2:
                continue
            comp_set = set(members)
            keep = members[0]
            external = []
            for n in members:
                if n == keep:
                    continue
                for nb in g.neighbors(n):
                    if nb not in comp_set:
                        external.append((nb, g[n][nb]["weight"]))
            g.remove_nodes_from([n for n in members if n != keep])
            for nb, w in external:
                if g.has_edge(keep, nb):
                    g[keep][nb]["weight"] = min(g[keep][nb]["weight"], w)
                else:
                    g.add_edge(keep, nb, weight=w)
            collapsed_any = True
        if not collapsed_any:
            break
    return g


def _prune_spurs(g: nx.Graph, min_spur_length_mm: float) -> nx.Graph:
    g = g.copy()
    changed = True
    while changed:
        changed = False
        for ep in [n for n in g.nodes if g.degree(n) == 1]:
            if ep not in g:
                continue
            path, length, cur, prev = [ep], 0.0, ep, None
            while g.degree(cur) <= 2:
                nbrs = [n for n in g.neighbors(cur) if n != prev]
                if not nbrs:
                    break
                nxt = nbrs[0]
                length += g[cur][nxt]["weight"]
                prev, cur = cur, nxt
                path.append(cur)
                if g.degree(cur) != 2:
                    break
            if length < min_spur_length_mm and g.degree(cur) >= 3 and g.number_of_nodes() > len(path):
                g.remove_nodes_from(path[:-1])
                changed = True
    return g


def _clean_graph(skeleton, spacing_mm, min_spur_length_mm, max_cycle_length_mm, max_hub_diameter_mm):
    g = _build_graph(skeleton, spacing_mm)
    g = _collapse_small_cycles(g, max_cycle_length_mm)
    g = _collapse_dense_hubs(g, spacing_mm, max_hub_diameter_mm)
    g = _prune_spurs(g, min_spur_length_mm)
    return g


def _seed_extreme_points(cca_seed, ica_seed, spacing_mm):
    """The point in each seed mask farthest from the other seed — reference
    points guaranteed to be deep in each vessel's own unambiguous territory,
    used to identify "which side of a cut" a component is on regardless of
    any TotalSegmentator label overlap near the bifurcation. See
    locate_bifurcation's docstring for why this matters."""
    cca_pts = np.argwhere(cca_seed)
    ica_pts = np.argwhere(ica_seed)
    cca_tree, ica_tree = cKDTree(cca_pts * spacing_mm), cKDTree(ica_pts * spacing_mm)
    cca_ica_dist = ica_tree.query(cca_pts * spacing_mm)[0]
    ica_cca_dist = cca_tree.query(ica_pts * spacing_mm)[0]
    return (
        cca_pts[np.argmin(cca_ica_dist)], ica_pts[np.argmin(ica_cca_dist)],  # near (facing) points
        cca_pts[np.argmax(cca_ica_dist)], ica_pts[np.argmax(ica_cca_dist)],  # far (extreme) points
    )


def resolve_cca_ica_path(skeleton: np.ndarray, spacing_mm, cca_seed: np.ndarray, ica_seed: np.ndarray,
                          min_spur_length_mm: float = 4.0, max_cycle_length_mm: float = 15.0,
                          max_hub_diameter_mm: float = 6.0):
    """Fallback for when locate_bifurcation finds no valid 3-way split —
    seen on a real case where the ECA's own vesselness signal was too
    fragmented for any branch point to survive graph cleanup at all (0
    degree>=3 nodes even with spur-pruning disabled entirely: a genuine
    per-scan contrast/data limit on that branch, not a threshold to tune
    away). Returns just a plain CCA<->ICA path with no branch — 2 outlets
    (CCA inlet, ICA outlet) instead of 3 — or None if even that path
    doesn't exist. Same graph and endpoint-anchoring approach as
    locate_bifurcation, minus the branch requirement.
    """
    g = _clean_graph(skeleton, spacing_mm, min_spur_length_mm, max_cycle_length_mm, max_hub_diameter_mm)
    cca_near_pt, ica_near_pt, cca_far_pt, ica_far_pt = _seed_extreme_points(cca_seed, ica_seed, spacing_mm)
    if not g.number_of_nodes():
        return None
    node_tree = cKDTree(np.asarray(g.nodes) * spacing_mm)
    graph_nodes = list(g.nodes)
    cca_ref = graph_nodes[node_tree.query(cca_far_pt * spacing_mm)[1]]
    ica_ref = graph_nodes[node_tree.query(ica_far_pt * spacing_mm)[1]]
    try:
        path = nx.shortest_path(g, cca_ref, ica_ref, weight="weight")
    except nx.NetworkXNoPath:
        return None
    # No real branch point exists (that's why locate_bifurcation failed) —
    # use the path's own midpoint as a stand-in "bn" so downstream code
    # (which cuts each outlet by walking outward from bn) still has
    # somewhere to walk outward *from* on both sides.
    bn = path[len(path) // 2]
    return {"graph": g, "bifurcation": bn,
            "cca_comp": set(path[:len(path) // 2 + 1]),
            "ica_comp": set(path[len(path) // 2:]),
            "eca_comps": []}


def locate_bifurcation_from_origin(skeleton: np.ndarray, spacing_mm, origin_point_vox,
                                    min_spur_length_mm: float = 4.0,
                                    max_cycle_length_mm: float = 15.0,
                                    max_hub_diameter_mm: float = 6.0,
                                    min_eca_length_mm: float = 3.0):
    """Same two-centerline idea as locate_bifurcation, for inputs with no
    TotalSegmentator seed masks at all (MR angiography — TotalSegmentator
    has no carotid classes for MR, so there's no CCA/ICA label to anchor
    on). Takes a single known point instead: the CCA's own origin (for a
    neck MRA, its most inferior point — found geometrically, not from a
    label). From there, a single Dijkstra tree gives the CCA->ICA
    centerline as the path to the single farthest reachable node (ICA runs
    the longest uninterrupted course, typically to/through the skull base
    — farther than any ECA branch within a neck+lower-head FOV) and the
    CCA->ECA centerline as the path to the farthest node *not* on that
    first path. Their last common node is the bifurcation. See
    locate_bifurcation's docstring for why this shortest-path-tree
    construction is more robust than searching for branch points directly.
    """
    g = _clean_graph(skeleton, spacing_mm, min_spur_length_mm, max_cycle_length_mm, max_hub_diameter_mm)
    if not g.number_of_nodes():
        return None
    node_tree = cKDTree(np.asarray(g.nodes) * spacing_mm)
    graph_nodes = list(g.nodes)
    origin = graph_nodes[node_tree.query(np.asarray(origin_point_vox) * spacing_mm)[1]]

    lengths, paths = nx.single_source_dijkstra(g, origin, weight="weight")
    if len(lengths) < 2:
        return None
    ica_far_node = max(lengths, key=lengths.get)
    path_to_ica = paths[ica_far_node]
    on_ica_path = set(path_to_ica)

    off_path_candidates = [(dist, node) for node, dist in lengths.items() if node not in on_ica_path]
    bn = origin
    eca_comps = []
    if off_path_candidates:
        eca_far_node = max(off_path_candidates, key=lambda t: t[0])[1]
        path_to_eca = paths[eca_far_node]
        for a, b in zip(path_to_ica, path_to_eca):
            if a != b:
                break
            bn = a
        eca_length_mm = lengths[eca_far_node] - lengths[bn]
        if eca_length_mm >= min_eca_length_mm:
            j = path_to_eca.index(bn)
            eca_comps = [set(path_to_eca[j:])]

    i = path_to_ica.index(bn)
    return {"graph": g, "bifurcation": bn,
            "cca_comp": set(path_to_ica[:i + 1]),
            "ica_comp": set(path_to_ica[i:]),
            "eca_comps": eca_comps}


def locate_bifurcation(skeleton: np.ndarray, spacing_mm,
                        cca_seed: np.ndarray, ica_seed: np.ndarray,
                        min_spur_length_mm: float = 4.0,
                        max_cycle_length_mm: float = 15.0,
                        max_hub_diameter_mm: float = 6.0,
                        min_eca_length_mm: float = 3.0):
    """Finds the bifurcation as the meeting point of two centerlines from
    the same CCA origin: CCA->ICA (both ends are real TotalSegmentator
    seeds) and CCA->ECA. ECA has no seed of its own — TotalSegmentator
    doesn't label it at all — so its "far end" is defined as whichever
    reachable point ends up farthest from the CCA origin among everything
    *not* already on the CCA->ICA path. One Dijkstra tree from the CCA
    origin gives both centerlines and their meeting point at once: the
    tree-path to ICA's far point is CCA->ICA, the tree-path to that
    farthest other point is CCA->ECA, and the bifurcation is just the last
    node the two tree-paths still agree on — no branch-point search or
    per-candidate split validation needed.

    This replaced an earlier version that searched every degree>=3 node in
    the graph and validated each as a 3-way split — workable, but each
    fragile edge case (a false-positive branch near CCA's origin passing a
    naive split test; TotalSegmentator's CCA/ICA masks overlapping over a
    long stretch on one real scan, degenerating "nearest touching points"
    to ~0mm; that overlap's ambiguous labels then failing a stricter
    exclusivity check even on the *correct* candidate) needed its own
    patch, and the fixes interacted. Anchoring on a single shortest-path
    tree from a real, unambiguous point sidesteps all three at once: seed
    *label* overlap near the bifurcation doesn't matter because sides are
    defined by tree topology, not by which mask a voxel happens to carry;
    a spurious near-origin branch doesn't matter because nothing about
    finding the CCA->ICA path involves searching for branch points at all.

    Returns a dict with the cleaned graph and the three node-sets (each a
    plain set of (a,b,c) voxel-index tuples: cca_comp is CCA-origin->bn,
    ica_comp is bn->ICA, eca_comps is [bn->ECA-candidate]), or None if no
    branch at least `min_eca_length_mm` long exists — callers should fall
    back to resolve_cca_ica_path (plain 2-outlet, no ECA) in that case.
    """
    g = _clean_graph(skeleton, spacing_mm, min_spur_length_mm, max_cycle_length_mm, max_hub_diameter_mm)
    if not g.number_of_nodes():
        return None

    cca_near_pt, ica_near_pt, cca_far_pt, ica_far_pt = _seed_extreme_points(cca_seed, ica_seed, spacing_mm)
    node_tree = cKDTree(np.asarray(g.nodes) * spacing_mm)
    graph_nodes = list(g.nodes)

    def nearest_graph_node(pt):
        return graph_nodes[node_tree.query(pt * spacing_mm)[1]]

    cca_ref, ica_ref = nearest_graph_node(cca_far_pt), nearest_graph_node(ica_far_pt)

    lengths, paths = nx.single_source_dijkstra(g, cca_ref, weight="weight")
    if ica_ref not in paths:
        return None
    path_to_ica = paths[ica_ref]
    on_ica_path = set(path_to_ica)

    off_path_candidates = [(dist, node) for node, dist in lengths.items() if node not in on_ica_path]
    if not off_path_candidates:
        return None
    eca_far_node = max(off_path_candidates, key=lambda t: t[0])[1]
    path_to_eca = paths[eca_far_node]

    bn = cca_ref
    for a, b in zip(path_to_ica, path_to_eca):
        if a != b:
            break
        bn = a

    eca_length_mm = lengths[eca_far_node] - lengths[bn]
    if eca_length_mm < min_eca_length_mm:
        return None

    i = path_to_ica.index(bn)
    j = path_to_eca.index(bn)
    return {"graph": g, "bifurcation": bn,
            "cca_comp": set(path_to_ica[:i + 1]),
            "ica_comp": set(path_to_ica[i:]),
            "eca_comps": [set(path_to_eca[j:])]}


def find_bifurcation_and_label(skeleton: np.ndarray, spacing_mm,
                                cca_seed: np.ndarray, ica_seed: np.ndarray,
                                min_spur_length_mm: float = 4.0,
                                max_cycle_length_mm: float = 15.0):
    """Labels skeleton voxels 1=cca / 2=ica / 3=eca. Returns (label_volume,
    found_bool). Thin wrapper around locate_bifurcation() for callers that
    just want a labeled volume (see meshing.py for the branch-trimmed,
    single-outlet-per-branch variant used to build CFD-ready surfaces).
    """
    label_vol = np.zeros(skeleton.shape, dtype=np.uint8)
    result = locate_bifurcation(skeleton, spacing_mm, cca_seed, ica_seed,
                                 min_spur_length_mm, max_cycle_length_mm)
    if result is None:
        return label_vol, False

    for n in result["cca_comp"]:
        label_vol[n] = 1
    for n in result["ica_comp"]:
        label_vol[n] = 2
    for c in result["eca_comps"]:
        for n in c:
            label_vol[n] = 3
    label_vol[result["bifurcation"]] = 1  # bifurcation apex ~ end of common trunk (convention)
    return label_vol, True
