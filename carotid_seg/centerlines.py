"""Centerline + radius extraction for one carotid bifurcation side, meant to
feed SimVascular's own path/segmentation/lofting workflow (sv.pathplanning.Path
+ sv.segmentation.Circle) instead of exporting a marching-cubes STL.

Reuses the exact same lumen-mask / skeleton / bifurcation-resolution pipeline
as meshing.build_side_surface (see that module for the reasoning behind each
step and the real-data failure modes that shaped it) — this module stops
short of the VTK surface-building half, returning ordered 3D centerline
points and a local radius estimate at each one instead.
"""
import logging

import networkx as nx
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

from .vesselness import vessel_lumen_mask, largest_component_touching, bridge_gap
from .treeanalysis import locate_bifurcation
from .branchtrim import trim_to_single_path
from .meshing import _bbox_with_margin, _vox_to_world, MESH_ROI_MARGIN_MM, ECA_STUMP_LENGTH_MM

log = logging.getLogger("carotid_seg")


def _ordered_path_points_with_radius(g, bn, kept_nodes, cut_node, spacing_mm,
                                      origin_world, radius_field_mm):
    """`kept_nodes` (from branchtrim.trim_to_single_path) is a set, not a
    sequence — recover point order by walking the shortest path from `bn` to
    `cut_node` within the induced subgraph (a simple path by construction,
    so this is exact, not an approximation)."""
    sub = g.subgraph(kept_nodes | {bn})
    ordered = nx.shortest_path(sub, bn, cut_node, weight="weight")
    points_world = [_vox_to_world(n, spacing_mm, origin_world).tolist() for n in ordered]
    radii_mm = [float(radius_field_mm[n]) for n in ordered]
    return points_world, radii_mm


def extract_branch_centerlines(ct, affine, spacing_mm, cca_mask, ica_mask, side: str,
                                eca_stump_length_mm: float = ECA_STUMP_LENGTH_MM):
    """Returns (diag, branches) where branches = {"cca": (points_world_mm, radii_mm),
    "ica": (...), "eca": (...)}, each ordered from the bifurcation outward, plus
    diag["bifurcation_world_mm"]. Returns (diag, None) on failure — diag["error"]
    explains why, same failure modes as meshing.build_side_surface since this
    is the same upstream pipeline.
    """
    diag = {"side": side, "ok": False}
    if not cca_mask.any() or not ica_mask.any():
        diag["error"] = "missing CCA or ICA seed mask"
        return diag, None

    roi, roi_lo = _bbox_with_margin(cca_mask | ica_mask, spacing_mm, MESH_ROI_MARGIN_MM, ct.shape)
    ct_roi, cca_roi, ica_roi = ct[roi], cca_mask[roi], ica_mask[roi]
    origin_world = affine[:3, 3] + np.asarray(roi_lo) * np.asarray(spacing_mm)

    lumen = vessel_lumen_mask(ct_roi, spacing_mm, cca_roi | ica_roi)
    tree_mask = largest_component_touching(lumen, cca_roi, ica_roi)
    if not tree_mask.any():
        lumen = bridge_gap(lumen, ct_roi, spacing_mm, cca_roi, ica_roi)
        tree_mask = largest_component_touching(lumen, cca_roi, ica_roi)
        diag["gap_bridged"] = bool(tree_mask.any())
    if not tree_mask.any():
        diag["error"] = "CCA and ICA seeds did not join into one component"
        return diag, None

    skeleton = skeletonize(tree_mask)
    result = locate_bifurcation(skeleton, spacing_mm, cca_roi, ica_roi)
    if result is None:
        diag["error"] = "bifurcation not resolved"
        return diag, None

    g, bn = result["graph"], result["bifurcation"]
    huge = 1e6
    kept_cca, cut_cca, _ = trim_to_single_path(g.subgraph(result["cca_comp"] | {bn}), bn, huge)
    kept_ica, cut_ica, _ = trim_to_single_path(g.subgraph(result["ica_comp"] | {bn}), bn, huge)
    eca_all = set().union(*result["eca_comps"])
    kept_eca, cut_eca, _ = trim_to_single_path(
        g.subgraph(eca_all | {bn}), bn, eca_stump_length_mm)
    kept_ica.discard(bn)
    kept_eca.discard(bn)
    if len(kept_eca) == 0:
        diag["error"] = "no ECA branch survived trimming"
        return diag, None

    # Local radius at every tree_mask voxel: distance to the nearest
    # background voxel, in mm (the transform is already isotropic-aware via
    # `sampling`). At a centerline voxel this is a good proxy for the vessel's
    # true lumen radius there — same quantity SimVascular's own radius-based
    # meshing would estimate, just computed from the mask we already trust.
    radius_field = ndi.distance_transform_edt(tree_mask, sampling=spacing_mm)

    branches = {}
    for name, kept, cut in (("cca", kept_cca, cut_cca), ("ica", kept_ica, cut_ica),
                             ("eca", kept_eca, cut_eca)):
        pts, radii = _ordered_path_points_with_radius(
            g, bn, kept, cut, spacing_mm, origin_world, radius_field)
        branches[name] = (pts, radii)
        diag[f"{name}_n_points"] = len(pts)
        diag[f"{name}_length_mm"] = nx.shortest_path_length(
            g.subgraph(kept | {bn}), bn, cut, weight="weight")

    diag["bifurcation_world_mm"] = _vox_to_world(bn, spacing_mm, origin_world).tolist()
    diag["ok"] = True
    return diag, branches
