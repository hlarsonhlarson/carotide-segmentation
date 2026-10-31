import logging
import time

import networkx as nx
import numpy as np
import vtk
from scipy import ndimage as ndi
from vtk.util import numpy_support
from skimage.morphology import skeletonize

from .vesselness import vessel_lumen_mask, largest_component_touching, bridge_gap
from .treeanalysis import locate_bifurcation, resolve_cca_ica_path
from .volumize import propagate_labels
from .branchtrim import trim_to_single_path

log = logging.getLogger("carotid_seg")

MESH_ROI_MARGIN_MM = 35.0
ECA_STUMP_LENGTH_MM = 25.0
CUT_INSET_MM = 8.0          # how far in from each natural/trimmed end to place the flat cut
TANGENT_SAMPLE_MM = 5.0     # how far back along the path to estimate the local direction
SMOOTHING_ITERATIONS = 150
SMOOTHING_PASSBAND = 0.01


def _bbox_with_margin(mask, spacing_mm, margin_mm, shape):
    idx = np.argwhere(mask)
    lo = idx.min(axis=0)
    hi = idx.max(axis=0) + 1
    margin_vox = np.ceil(margin_mm / np.asarray(spacing_mm)).astype(int)
    lo = np.maximum(lo - margin_vox, 0)
    hi = np.minimum(hi + margin_vox, np.asarray(shape))
    return tuple(slice(a, b) for a, b in zip(lo, hi)), lo


def _walk_back(g, path_nodes, start, target_mm, spacing_mm):
    """From `start`, walk along nodes in `path_nodes` toward increasing
    distance, returning the first node at least target_mm away (or the
    furthest reachable one)."""
    seen = {start}
    frontier = [(start, 0.0)]
    best = start
    while frontier:
        n, d = frontier.pop()
        if d >= target_mm:
            return n
        best = n
        for nb in g.neighbors(n):
            if nb in path_nodes and nb not in seen:
                seen.add(nb)
                frontier.append((nb, d + g[n][nb]["weight"]))
    return best


def _inset_cut_node(g, kept_nodes, bn, end_node, inset_mm, spacing_mm):
    """Move `inset_mm` back from `end_node` toward `bn`, staying within
    `kept_nodes`, so the cut sits inside clean, well-segmented tissue
    rather than right at the ragged edge of the mask."""
    # distances from bn along the kept path (BFS since it's a simple path/tree)
    dist_from_bn = {bn: 0.0}
    frontier = [bn]
    parent = {bn: None}
    while frontier:
        n = frontier.pop()
        for nb in g.neighbors(n):
            if nb in kept_nodes and nb not in dist_from_bn:
                dist_from_bn[nb] = dist_from_bn[n] + g[n][nb]["weight"]
                parent[nb] = n
                frontier.append(nb)
    if end_node not in dist_from_bn:
        return end_node
    total_length = dist_from_bn[end_node]
    # a branch shorter than the requested inset (a short real ECA stub, or a
    # thin sliver of a branch) would otherwise collapse the cut onto bn
    # itself — clamp so some stump always survives on both sides of the cut.
    effective_inset = min(inset_mm, total_length / 2.0)
    target = max(total_length - effective_inset, 0.0)
    node = end_node
    while parent.get(node) is not None and dist_from_bn[node] > target:
        node = parent[node]
    return node


def _cut_plane(g, kept_nodes, bn, cut_node, spacing_mm):
    origin_vox = np.asarray(cut_node, dtype=float)
    back_node = _walk_back(g, kept_nodes, cut_node, TANGENT_SAMPLE_MM, spacing_mm) \
        if cut_node != bn else bn
    # tangent should point from bn-side toward the free end (the direction we cut away)
    dist_from_bn = {bn: 0.0}
    frontier = [bn]
    while frontier:
        n = frontier.pop()
        for nb in g.neighbors(n):
            if nb in kept_nodes and nb not in dist_from_bn:
                dist_from_bn[nb] = dist_from_bn[n] + g[n][nb]["weight"]
                frontier.append(nb)
    ref_node = back_node if dist_from_bn.get(back_node, 0) < dist_from_bn.get(cut_node, 0) else bn
    direction_vox = origin_vox - np.asarray(ref_node, dtype=float)
    direction_mm = direction_vox * np.asarray(spacing_mm)
    norm = np.linalg.norm(direction_mm)
    if norm < 1e-6:
        direction_mm = np.array([0.0, 0.0, 1.0])
    else:
        direction_mm = direction_mm / norm
    return origin_vox, direction_mm


def _resample_isotropic(volume_uint8, spacing_mm):
    """CT slice thickness is usually much coarser than in-plane resolution
    (e.g. 1.25mm vs 0.625mm here — a real case). Marching cubes directly on
    that anisotropic grid produces a visible corkscrew/terracing artifact on
    any vessel running obliquely to the slice axis — the surface steps in
    coarse slice-thick increments along that direction. Upsample to
    isotropic spacing before extracting the surface.

    VERIFIED this staircasing survives linear-interpolate-then-threshold
    upsampling of the raw binary mask: halving a 1.25mm step to 0.625mm
    still leaves a visible step, just a smaller one — the mask only ever
    flips exactly at original-voxel boundaries either way, since the
    interpolated *binary* field crosses 0.5 at very nearly the same place
    regardless of the new sampling rate. Resampling a signed distance field
    instead (continuous, not binary) and thresholding at 0 doesn't have
    that limitation: cubic interpolation of a smooth scalar field genuinely
    estimates intermediate values between original slices, so the zero
    level set can fall anywhere, not just where the old voxel grid put it.
    """
    spacing_mm = np.asarray(spacing_mm, dtype=float)
    target = spacing_mm.min()
    zoom_factors = spacing_mm / target
    inside = ndi.distance_transform_edt(volume_uint8, sampling=spacing_mm)
    outside = ndi.distance_transform_edt(1 - volume_uint8, sampling=spacing_mm)
    sdf = outside.astype(np.float32) - inside.astype(np.float32)  # <0 inside, >0 outside
    sdf_resampled = ndi.zoom(sdf, zoom_factors, order=3)
    resampled = (sdf_resampled <= 0).astype(np.uint8)
    return resampled, (target, target, target)


def _numpy_to_vtk_image(volume_uint8, spacing_mm, origin_mm):
    # numpy array is (axis0, axis1, axis2) C-contiguous, so axis2 already
    # varies fastest when raveled — exactly what VTK wants for x=axis2,
    # y=axis1, z=axis0 with dims set as shape[::-1]. No transpose needed
    # (a transpose here silently scrambles the volume).
    img = vtk.vtkImageData()
    img.SetDimensions(volume_uint8.shape[::-1])
    img.SetSpacing(spacing_mm[::-1])
    img.SetOrigin(origin_mm[::-1])
    flat = np.ascontiguousarray(volume_uint8).ravel()
    arr = numpy_support.numpy_to_vtk(flat, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
    img.GetPointData().SetScalars(arr)
    return img


def _vox_to_world(vox_ijk, spacing_mm, origin_mm):
    return np.asarray(origin_mm) + np.asarray(vox_ijk) * np.asarray(spacing_mm)


def _clean(poly):
    """vtkClipPolyData (and marching cubes) leave unused/orphan points in
    the point buffer, which silently corrupts GetBounds() and any point
    locator built on the output. Always compact after those filters."""
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(poly)
    cleaner.Update()
    out = vtk.vtkPolyData()
    out.DeepCopy(cleaner.GetOutput())
    return out


def _xyz(world_natural):
    """_numpy_to_vtk_image maps vtk (x,y,z) to array (axis2,axis1,axis0) —
    reversed relative to our natural (axis0,axis1,axis2) world tuples. Any
    raw coordinate handed to a vtk Set*/FindClosestPoint call (evaluated
    against the mesh's actual point coordinates) must go through this, or
    it silently lands somewhere else in space."""
    return tuple(float(v) for v in reversed(list(world_natural)))


def _clip_away(poly, plane_origin_world, plane_normal_world, keep_towards):
    """Clip `poly` with a plane, keeping the larger of the two resulting
    pieces.

    `keep_towards` is unused for the decision itself (kept as a parameter
    for now since callers still compute and pass it) — a real case showed
    why "keep whichever side is closer to a reference point" isn't safe
    here: the cut plane is *infinite*, and this ROI can span 400+mm of a
    real, curving vessel. A plane positioned and oriented from a short
    local tangent near the bifurcation cut through a distant, unrelated
    part of the same curving CCA/ICA body that happened to fall on its far
    side — a mesh with 44918 points on one side collapsed to 1811 on a
    "closer to bn" verdict, having amputated most of the actual vessel
    instead of the small distal cap being trimmed. Every one of these three
    cuts (CCA/ICA/ECA) is meant to only shave off a small distal cap and
    keep everything else, so the larger piece is always the correct side
    regardless of how the plane's infinite extent interacts with distant
    geometry.

    (A local-radius-bounded variant of this clip was tried to handle a
    tightly coiled vessel where the infinite plane also sliced an unrelated
    nearby loop of the same vessel — VERIFIED worse: it measurably shrank
    output on multiple already-working cases, e.g. one side's point count
    dropped by more than half on a case that was previously clean. Reverted;
    the self-proximity problem on extremely tortuous cases needs a different
    fix than bounding this clip.)"""
    plane = vtk.vtkPlane()
    plane.SetOrigin(*_xyz(plane_origin_world))
    plane.SetNormal(*_xyz(plane_normal_world))

    clipper = vtk.vtkClipPolyData()
    clipper.SetInputData(poly)
    clipper.SetClipFunction(plane)
    clipper.GenerateClippedOutputOn()
    clipper.Update()

    side_a = _clean(clipper.GetOutputDataObject(0))
    side_b = _clean(clipper.GetOutputDataObject(1))

    if side_a.GetNumberOfPoints() == 0:
        return side_b
    if side_b.GetNumberOfPoints() == 0:
        return side_a
    return side_a if side_a.GetNumberOfPoints() >= side_b.GetNumberOfPoints() else side_b


def _largest_component(poly):
    conn = vtk.vtkPolyDataConnectivityFilter()
    conn.SetInputData(poly)
    conn.SetExtractionModeToLargestRegion()
    conn.Update()
    return _clean(conn.GetOutput())


def _smooth(poly):
    smoother = vtk.vtkWindowedSincPolyDataFilter()
    smoother.SetInputData(poly)
    smoother.SetNumberOfIterations(SMOOTHING_ITERATIONS)
    smoother.SetPassBand(SMOOTHING_PASSBAND)
    smoother.BoundarySmoothingOn()
    smoother.NonManifoldSmoothingOn()
    smoother.NormalizeCoordinatesOn()
    smoother.Update()
    out = vtk.vtkPolyData()
    out.DeepCopy(smoother.GetOutput())
    return out


def count_boundary_loops(poly):
    edges = vtk.vtkFeatureEdges()
    edges.SetInputData(poly)
    edges.BoundaryEdgesOn()
    edges.FeatureEdgesOff()
    edges.NonManifoldEdgesOff()
    edges.ManifoldEdgesOff()
    edges.Update()
    if edges.GetOutput().GetNumberOfPoints() == 0:
        return 0, 0
    conn = vtk.vtkPolyDataConnectivityFilter()
    conn.SetInputData(edges.GetOutput())
    conn.SetExtractionModeToAllRegions()
    conn.Update()
    n_loops = conn.GetNumberOfExtractedRegions()

    nonmanifold = vtk.vtkFeatureEdges()
    nonmanifold.SetInputData(poly)
    nonmanifold.BoundaryEdgesOff()
    nonmanifold.FeatureEdgesOff()
    nonmanifold.NonManifoldEdgesOn()
    nonmanifold.ManifoldEdgesOff()
    nonmanifold.Update()
    n_nonmanifold_edges = nonmanifold.GetOutput().GetNumberOfCells()
    return n_loops, n_nonmanifold_edges


def build_side_surface(ct, affine, spacing_mm, cca_mask, ica_mask, side: str,
                        eca_stump_length_mm: float = ECA_STUMP_LENGTH_MM):
    """Builds a single open (uncapped) surface for one carotid bifurcation:
    one CCA inlet, one ICA outlet, one ECA outlet — real side branches off
    the ECA (e.g. superior thyroid a.) are identified via the centerline
    graph and surgically discarded rather than left dangling.

    Returns (vtkPolyData or None, diagnostics dict).
    """
    diag = {"side": side, "ok": False}
    if not cca_mask.any() or not ica_mask.any():
        diag["error"] = "missing CCA or ICA seed mask"
        return None, diag

    roi, roi_lo = _bbox_with_margin(cca_mask | ica_mask, spacing_mm, MESH_ROI_MARGIN_MM, ct.shape)
    ct_roi, cca_roi, ica_roi = ct[roi], cca_mask[roi], ica_mask[roi]
    log.info("[%s] roi shape=%s", side, ct_roi.shape)

    t0 = time.monotonic()
    lumen = vessel_lumen_mask(ct_roi, spacing_mm, cca_roi | ica_roi)
    log.info("[%s] vessel_lumen_mask done in %.1fs (%d voxels)", side, time.monotonic() - t0, lumen.sum())
    tree_mask = largest_component_touching(lumen, cca_roi, ica_roi)
    if not tree_mask.any():
        t0 = time.monotonic()
        lumen = bridge_gap(lumen, ct_roi, spacing_mm, cca_roi, ica_roi)
        tree_mask = largest_component_touching(lumen, cca_roi, ica_roi)
        diag["gap_bridged"] = tree_mask.any()
        log.info("[%s] bridge_gap done in %.1fs (tree_mask=%d voxels)", side, time.monotonic() - t0, tree_mask.sum())
    if not tree_mask.any():
        diag["error"] = "CCA and ICA seeds did not join into one component"
        return None, diag

    t0 = time.monotonic()
    skeleton = skeletonize(tree_mask)
    log.info("[%s] skeletonize done in %.1fs (%d voxels)", side, time.monotonic() - t0, skeleton.sum())
    t0 = time.monotonic()
    result = locate_bifurcation(skeleton, spacing_mm, cca_roi, ica_roi)
    log.info("[%s] locate_bifurcation done in %.1fs (found=%s)", side, time.monotonic() - t0, result is not None)
    if result is None:
        # Fallback: on a real case, the ECA's vesselness signal was too
        # fragmented for any branch point to survive graph cleanup at all
        # (0 degree>=3 nodes, confirmed independent of spur-pruning — a
        # genuine per-scan contrast/data limit on that one branch, not a
        # threshold to tune away). Rather than fail the whole side, fall
        # back to a plain CCA<->ICA tube with no ECA outlet — still useful,
        # clearly flagged as degraded rather than silently short a branch.
        t0 = time.monotonic()
        result = resolve_cca_ica_path(skeleton, spacing_mm, cca_roi, ica_roi)
        log.info("[%s] resolve_cca_ica_path fallback done in %.1fs (found=%s)",
                  side, time.monotonic() - t0, result is not None)
        if result is None:
            diag["error"] = "bifurcation not resolved (CCA/ICA fallback path also failed)"
            return None, diag
        diag["eca_missing"] = True

    g, bn = result["graph"], result["bifurcation"]
    has_eca = bool(result["eca_comps"])
    log.info("[%s] bn=%s cca_comp=%d ica_comp=%d eca_comps=%s", side, bn,
              len(result["cca_comp"]), len(result["ica_comp"]), [len(c) for c in result["eca_comps"]])
    t0 = time.monotonic()
    huge = 1e6
    kept_cca, cut_cca, disc_cca = trim_to_single_path(g.subgraph(result["cca_comp"] | {bn}), bn, huge)
    kept_ica, cut_ica, disc_ica = trim_to_single_path(g.subgraph(result["ica_comp"] | {bn}), bn, huge)
    kept_ica.discard(bn)
    if has_eca:
        eca_all = set().union(*result["eca_comps"])
        kept_eca, cut_eca, disc_eca = trim_to_single_path(
            g.subgraph(eca_all | {bn}), bn, eca_stump_length_mm)
        kept_eca.discard(bn)
        if len(kept_eca) == 0:
            diag["error"] = "no ECA branch survived trimming"
            return None, diag
    else:
        kept_eca, cut_eca, disc_eca = set(), None, set()
    log.info("[%s] trim_to_single_path done in %.1fs", side, time.monotonic() - t0)

    if has_eca:
        eca_length_mm = nx.shortest_path_length(
            g.subgraph(kept_eca | {bn}), bn, cut_eca, weight="weight")
        diag["eca_stump_length_mm"] = eca_length_mm
        if eca_length_mm < 2 * CUT_INSET_MM:
            # A stump this short means the branch fades out (weak contrast /
            # segmentation gap) before it visibly separates from the bulb —
            # the same real-data limitation that can also defeat
            # TotalSegmentator itself right at the bifurcation. The cut plane
            # ends up slicing through tissue still fused with the main vessel,
            # so it may not open a clean third boundary loop. Flagged for
            # manual review rather than silently reported as clean.
            diag["eca_stump_short"] = True

    label_skel = np.zeros(tree_mask.shape, dtype=np.uint8)
    for n in kept_cca:
        label_skel[n] = 1
    for n in kept_ica:
        label_skel[n] = 2
    for n in kept_eca:
        label_skel[n] = 3
    for n in (disc_cca | disc_ica | disc_eca):
        label_skel[n] = 9

    t0 = time.monotonic()
    partition = propagate_labels(label_skel, tree_mask, spacing_mm)
    kept_volume = np.isin(partition, [1, 2, 3]).astype(np.uint8)
    kept_volume = np.pad(kept_volume, 1)  # guarantee no contact with array border
    log.info("[%s] propagate_labels done in %.1fs (kept_volume=%d voxels)",
              side, time.monotonic() - t0, kept_volume.sum())

    pad_origin_world = affine[:3, 3] + np.asarray(roi_lo) * np.asarray(spacing_mm) \
        - np.asarray(spacing_mm)  # account for the 1-voxel pad above, in original spacing
    t0 = time.monotonic()
    kept_volume, iso_spacing = _resample_isotropic(kept_volume, spacing_mm)
    log.info("[%s] _resample_isotropic done in %.1fs (shape=%s, iso_spacing=%s)",
              side, time.monotonic() - t0, kept_volume.shape, iso_spacing)
    img = _numpy_to_vtk_image(kept_volume, iso_spacing, pad_origin_world)

    t0 = time.monotonic()
    mc = vtk.vtkDiscreteFlyingEdges3D()
    mc.SetInputData(img)
    mc.SetValue(0, 1)
    mc.Update()
    poly = vtk.vtkPolyData()
    poly.DeepCopy(mc.GetOutput())
    log.info("[%s] marching cubes done in %.1fs (%d points, %d cells)",
              side, time.monotonic() - t0, poly.GetNumberOfPoints(), poly.GetNumberOfCells())
    t0 = time.monotonic()
    poly = _largest_component(poly)
    log.info("[%s] _largest_component done in %.1fs (%d points)", side, time.monotonic() - t0, poly.GetNumberOfPoints())
    t0 = time.monotonic()
    poly = _smooth(poly)
    log.info("[%s] _smooth done in %.1fs", side, time.monotonic() - t0)

    bn_world = _vox_to_world(bn, spacing_mm, np.asarray(roi_lo) * np.asarray(spacing_mm) + affine[:3, 3])

    branches = [("cca", kept_cca, cut_cca, None), ("ica", kept_ica, cut_ica, None)]
    if has_eca:
        branches.append(("eca", kept_eca, cut_eca, None))
    for name, g_kept, cut_end, max_len in branches:
        path_nodes = g_kept | {bn}
        inset_node = _inset_cut_node(g, path_nodes, bn, cut_end, CUT_INSET_MM, spacing_mm)
        origin_vox, direction_mm = _cut_plane(g, path_nodes, bn, inset_node, spacing_mm)
        origin_world = _vox_to_world(origin_vox, spacing_mm,
                                      np.asarray(roi_lo) * np.asarray(spacing_mm) + affine[:3, 3])
        log.info("[%s] %s: bn=%s cut_end=%s inset_node=%s origin_vox=%s direction_mm=%s path_nodes=%d",
                  side, name, bn, cut_end, inset_node, origin_vox, direction_mm, len(path_nodes))
        t0 = time.monotonic()
        poly = _clip_away(poly, origin_world, direction_mm, bn_world)
        log.info("[%s] clip %s done in %.1fs (%d points left)",
                  side, name, time.monotonic() - t0, poly.GetNumberOfPoints())
        diag[f"{name}_cut_world_mm"] = origin_world.tolist()

    poly = _largest_component(poly)
    n_loops, n_nonmanifold = count_boundary_loops(poly)
    diag.update({
        "ok": True,
        "n_points": poly.GetNumberOfPoints(),
        "n_cells": poly.GetNumberOfCells(),
        "n_boundary_loops": n_loops,
        "n_nonmanifold_edges": n_nonmanifold,
        "clean_topology": (n_loops == (3 if has_eca else 2) and n_nonmanifold == 0),
        "eca_side_branches_removed": len(disc_cca) + len(disc_ica) + len(disc_eca) > 0,
    })
    return poly, diag


def write_stl(poly, path):
    writer = vtk.vtkSTLWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(poly)
    writer.SetFileTypeToBinary()
    writer.Write()
