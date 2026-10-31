import logging

import networkx as nx

log = logging.getLogger("carotid_seg")


def trim_to_single_path(g: nx.Graph, root, max_length_mm: float):
    """Reduce `g` to the single longest path from `root`, truncated at
    `max_length_mm`.

    The real anatomical continuation (CCA trunk, or the surviving ECA branch
    after a real side branch like the superior thyroid a. takes off) is
    assumed to be the longest one reachable from `root` — a tributary is
    shorter within a tight neck-CTA ROI than the main vessel.

    Uses Dijkstra's shortest-path tree from `root` and picks the farthest
    reachable node, rather than a greedy node-by-node walk that re-ranks
    subtrees at each branch point. That greedy walk assumed a clean tree and
    broke on real data twice: (1) a residual cycle within the induced
    subgraph — surviving graph-wide cleanup upstream — sent it circling
    forever since it only tracked the immediately-previous node; (2) even
    once fixed to track every visited node, a branch that meets `root`
    through *more than one* edge (measured on a real case: the bulb-adjacent
    ECA entry touched the bifurcation node at several points at once, not a
    single clean joint) made the walk dead-end after just one real step,
    because the second entry point's only other neighbor was `root` itself
    (already visited). Dijkstra distances aren't sensitive to either failure
    mode — they're well-defined on any weighted graph, cycles or multi-edges
    into the same node included.

    Returns (kept_nodes, cut_node, discarded_nodes). `cut_node` is the
    last kept node on the path — the outlet cut plane goes through it.
    """
    lengths, paths = nx.single_source_dijkstra(g, root, weight="weight")
    far_node = max(lengths, key=lengths.get)
    full_path = paths[far_node]

    kept = {root}
    cut_node = root
    dist = 0.0
    for prev, nxt in zip(full_path, full_path[1:]):
        step = g[prev][nxt]["weight"]
        if dist + step > max_length_mm:
            break
        kept.add(nxt)
        cut_node = nxt
        dist += step

    discarded = set(g.nodes) - kept
    log.debug("trim_to_single_path: kept %d/%d nodes, stump length %.1fmm (cap %.1fmm)",
              len(kept), g.number_of_nodes(), dist, max_length_mm)
    return kept, cut_node, discarded
