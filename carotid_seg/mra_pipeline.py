"""Carotid (CCA/ICA/ECA) segmentation from TOF-MR angiography, for cases
with no TotalSegmentator seed masks available at all — TotalSegmentator's
MR task has no carotid/neck-vessel classes (checked directly against the
installed package: total_mr's only vessel classes are the iliac arteries).

TOF makes flowing blood bright against a suppressed background by design,
so this reaches for a plain intensity threshold instead of the CT
pipeline's Frangi vesselness + HU-band calibration (vesselness.py) — there
is no Hounsfield scale on MR to calibrate against, and on a real case Otsu
badly over-included background tissue (~48% of the volume) since TOF
intensity isn't bimodal the way CT soft-tissue-vs-contrast is; a high
percentile threshold (vessels are a small bright minority) worked cleanly
instead.

Everything downstream of getting a per-side binary vessel tree — skeleton
graph, two-centerline bifurcation search, branch trimming, surface
building, cutting — reuses the CT pipeline's generic (not CT-specific)
machinery directly: treeanalysis.locate_bifurcation_from_origin (the CT
pipeline's locate_bifurcation, adapted for "a known CCA origin point"
instead of "two seed masks" — same shortest-path-tree idea), and most of
meshing.py's geometry/VTK helpers.
"""
import logging
import time

import networkx as nx
import numpy as np
import SimpleITK as sitk
import vtk
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

from .treeanalysis import locate_bifurcation_from_origin
from .branchtrim import trim_to_single_path
from .volumize import propagate_labels
from .meshing import (
    _bbox_with_margin, _inset_cut_node, _cut_plane, _resample_isotropic,
    _numpy_to_vtk_image, _vox_to_world, _largest_component, _smooth, _clip_away,
    count_boundary_loops, write_stl,
    MESH_ROI_MARGIN_MM, ECA_STUMP_LENGTH_MM, CUT_INSET_MM,
)

log = logging.getLogger("carotid_seg")

VESSEL_PERCENTILE = 99.0
MIN_COMPONENT_VOXELS = 20
N_ELONGATED_CANDIDATES = 12   # how many top-by-bbox-diagonal components to consider
SKIN_MARGIN_MM = 12.0         # exclude anything this close to the body surface (fat/skin artifact)
# Unlike the CT pipeline (where TotalSegmentator's own ICA class mask
# naturally stops around the skull base), this FOV runs straight through
# into the intracranial circle of Willis — VERIFIED on a real case: an
# unbounded ICA trim produced 6 boundary loops instead of 3, because the
# kept path extended into cerebral branching (MCA/ACA/PCoA all close
# together there) and only the one single longest branch got a cut; every
# other small intracranial branch stayed attached, uncut, each opening its
# own extra hole. The user's own ask was "ICA at least to skull base entry"
# — cap at a length that comfortably covers cervical + petrous/cavernous
# ICA (real cervical ICA alone runs ~70-120mm) without reaching the COW.
ICA_MAX_LENGTH_MM = 110.0


def load_tof_volume(dicom_file_path):
    """Reads one Siemens-style enhanced (single-file, multi-frame) DICOM
    TOF-MRA series via SimpleITK, which handles that layout natively
    (confirmed on real data — pydicom's per-frame geometry tags were in an
    unexpected nesting; SimpleITK's reader didn't need any of that).

    Returns (array, spacing_mm, origin_mm) where array is (axis0,axis1,axis2)
    matching SimpleITK's (z,y,x) GetArrayFromImage convention, and
    spacing_mm/origin_mm are reordered to match that same axis order (so
    they plug directly into meshing.py's helpers, which — like the rest of
    this codebase — assume spacing/origin share the array's axis order and
    ignore any rotation component of the image's direction matrix; this
    scan's rotation is small (~2.7 degrees) so treating it as axis-aligned
    is a minor, visually negligible approximation, not a new limitation
    this module introduces).
    """
    img = sitk.ReadImage(str(dicom_file_path))
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    spacing_mm = np.asarray(img.GetSpacing())[::-1]
    origin_mm = np.asarray(img.GetOrigin())[::-1]
    return arr, spacing_mm, origin_mm


COHERENCE_OPEN_ITERATIONS = 2
COHERENCE_MIN_RATIO = 0.5


def _elongated_components(mask, spacing_mm, min_voxels=MIN_COMPONENT_VOXELS):
    """Connected components ranked by bounding-box diagonal (mm). Vessels
    threaded through a 3D volume span tens of mm; bright fat/skin/artifact
    blobs that survive the intensity threshold don't — this is what
    actually separates them (VERIFIED on real data: a plain size or
    min-size+distance-from-skin filter still left large fat-pad blobs
    beside the carotids; bounding-box diagonal cleanly isolated the 4
    major neck vessels — 2 carotid + 2 vertebral — from everything else).

    Also computes a `coherence` score per candidate: the fraction of its
    voxels that survive a small local morphological opening (applied only
    to that candidate's own crop, as a shape test — never to the mask
    actually used downstream). VERIFIED on a real case: a genuine vessel
    (solid tube, even when thin) keeps ~0.75-0.95 of its volume under this
    test, while a bright but grainy/porous non-vascular blob (fat, motion
    artifact) that happened to have a larger raw voxel count and an
    elongated bounding box — enough to otherwise look like a better
    candidate than the real vessel on its side — collapsed to <0.3,
    usually <0.1. Picking candidates by raw size alone selected that blob
    outright on one real patient, producing a surface that looked like a
    perforated sheet instead of a vessel; this is what actually separates
    them.

    Returns [(diagonal_mm, size, component_id, centroid_vox, coherence), ...]
    sorted descending by diagonal.
    """
    labeled, n = ndi.label(mask, structure=np.ones((3, 3, 3)))
    objs = ndi.find_objects(labeled)
    results = []
    for i, sl in enumerate(objs):
        if sl is None:
            continue
        cid = i + 1
        size = int((labeled[sl] == cid).sum())
        if size < min_voxels:
            continue
        extent_mm = np.array([(s.stop - s.start) for s in sl]) * spacing_mm
        diagonal = float(np.linalg.norm(extent_mm))
        centroid = np.array([(s.start + s.stop) / 2.0 for s in sl])
        pad = tuple(slice(max(0, s.start - 3), s.stop + 3) for s in sl)
        comp_crop = labeled[pad] == cid
        opened_size = int(ndi.binary_opening(comp_crop, iterations=COHERENCE_OPEN_ITERATIONS).sum())
        coherence = opened_size / size
        results.append((diagonal, size, cid, centroid, coherence))
    results.sort(key=lambda t: t[0], reverse=True)
    return results, labeled


def extract_carotid_trees(arr, spacing_mm, percentile=VESSEL_PERCENTILE):
    """Returns {"left": mask, "right": mask} — each a binary tree_mask
    isolating one side's carotid tree (not vertebral — carotid caliber is
    consistently larger; VERIFIED on real data: on each side, the carotid
    component came out 3-4x the voxel count of the vertebral component
    found at a similar bounding-box diagonal and more medial position).
    Also returns diag with the raw candidate list for troubleshooting.
    """
    diag = {}
    thresh = float(np.percentile(arr, percentile))
    mask = arr > thresh
    candidates, labeled = _elongated_components(mask, spacing_mm)
    diag["threshold"] = thresh
    diag["n_candidates_considered"] = min(len(candidates), N_ELONGATED_CANDIDATES)
    top = candidates[:N_ELONGATED_CANDIDATES]

    mid_x = arr.shape[2] / 2.0
    left_side = [c for c in top if c[3][2] < mid_x]   # smaller x index = one side
    right_side = [c for c in top if c[3][2] >= mid_x]
    if not left_side or not right_side:
        diag["error"] = f"could not find elongated vessel candidates on both sides " \
                         f"(left={len(left_side)}, right={len(right_side)})"
        return None, diag

    # Prefer coherent (solid-tube-like) candidates over raw size — a bright
    # but grainy/porous non-vascular blob can otherwise outrank the real,
    # thinner vessel by voxel count alone (see _elongated_components). Falls
    # back to the full side list if nothing clears the coherence bar, rather
    # than failing outright.
    def pick(side_list, side_name):
        coherent = [c for c in side_list if c[4] >= COHERENCE_MIN_RATIO]
        pool = coherent if coherent else side_list
        if not coherent:
            diag[f"{side_name}_no_coherent_candidate"] = True
        pool = sorted(pool, key=lambda t: t[1], reverse=True)
        return pool[0]

    left_best = pick(left_side, "left")
    right_best = pick(right_side, "right")
    left_id = left_best[2]
    right_id = right_best[2]
    diag["left_component_voxels"] = left_best[1]
    diag["right_component_voxels"] = right_best[1]
    diag["left_component_coherence"] = left_best[4]
    diag["right_component_coherence"] = right_best[4]

    trees = {"left": labeled == left_id, "right": labeled == right_id}
    for side, tree in trees.items():
        trees[side], regrow_n = _regrow_locally(arr, tree, spacing_mm)
        diag[f"{side}_regrow_added_voxels"] = regrow_n
    return trees, diag


def _regrow_locally(arr, tree_mask, spacing_mm, dilate_mm=8.0, local_percentile=90.0,
                     open_mm=1.2):
    """TOF signal strength isn't uniform across the slab — VERIFIED on a
    real case: the global p99 threshold that cleanly finds the tortuous
    (and apparently higher-signal) part of each carotid left the CCA trunk
    a thin, barely-connected thread, visibly too thin for a real vessel in
    the rendered STL, even though the vessel is plainly present and wider
    in the source images there. A single global threshold can't serve both
    regions at once. Fix: once the vessel's general course is known (from
    the strict global threshold), regrow it using a *local* percentile
    computed only within a thin dilated shell around that known course —
    local background there is almost entirely perivascular tissue, not a
    whole head's worth of noise, so a much more permissive local threshold
    stays safe.

    The permissive local threshold has a side effect though — VERIFIED on a
    real case: it also picked up a thin, independent, curling thread (most
    likely a small adjacent vein, not the carotid) wherever that thread
    happened to brush against the 8mm shell, and since the component-keep
    step below accepts a whole component the instant any part of it touches
    tree_mask, the entire stray thread got pulled in with it, breaking the
    rendered surface's topology (6 boundary loops instead of 3). The real
    vessel lumen is several mm across; a stray thread like that is ~1 voxel
    wide. A small morphological opening on the grown mask before accepting
    components strips thin threads like that while leaving the actual lumen
    intact, and is unioned back with the original tree_mask so already-
    trusted voxels are never removed by it.
    """
    dilate_vox = max(1, int(round(dilate_mm / float(np.mean(spacing_mm)))))
    shell = ndi.binary_dilation(tree_mask, iterations=dilate_vox)
    local_vals = arr[shell]
    local_thresh = float(np.percentile(local_vals, local_percentile))
    local_thresh = min(local_thresh, float(np.percentile(arr[tree_mask], 1)))  # never looser than the weakest already-kept voxel
    grown = (arr > local_thresh) & shell
    open_vox = max(1, int(round(open_mm / float(np.mean(spacing_mm)))))
    grown_opened = ndi.binary_opening(grown, iterations=open_vox) | tree_mask
    labeled, n = ndi.label(grown_opened, structure=np.ones((3, 3, 3)))
    keep_ids = set(np.unique(labeled[tree_mask])) - {0}
    regrown = np.isin(labeled, list(keep_ids))
    return regrown, int(regrown.sum() - tree_mask.sum())


def build_mra_side_surface(arr, spacing_mm, origin_mm, tree_mask_full, side: str,
                            eca_stump_length_mm: float = ECA_STUMP_LENGTH_MM,
                            ica_max_length_mm: float = ICA_MAX_LENGTH_MM):
    """Mirrors meshing.build_side_surface's surface-building half exactly
    (propagate_labels -> resample -> marching cubes -> smooth -> cut per
    branch) — only the seeding/bifurcation-finding half differs, since
    there's no CCA/ICA label to start from. Returns (vtkPolyData or None,
    diagnostics dict).
    """
    diag = {"side": side, "ok": False}
    if not tree_mask_full.any():
        diag["error"] = "empty tree mask for this side"
        return None, diag

    roi, roi_lo = _bbox_with_margin(tree_mask_full, spacing_mm, MESH_ROI_MARGIN_MM, arr.shape)
    tree_mask = tree_mask_full[roi]
    log.info("[%s] roi shape=%s, tree_mask voxels=%d", side, tree_mask.shape, tree_mask.sum())

    t0 = time.monotonic()
    skeleton = skeletonize(tree_mask)
    log.info("[%s] skeletonize done in %.1fs (%d voxels)", side, time.monotonic() - t0, skeleton.sum())

    # CCA origin = the skeleton's most inferior point. This scan's array
    # axis0 increases toward superior (DICOM LPS: +Z is superior, and this
    # series' direction matrix is close enough to identity that array axis0
    # tracks physical Z) — so the origin is the *minimum* axis0 index.
    skel_pts = np.argwhere(skeleton)
    origin_vox = skel_pts[np.argmin(skel_pts[:, 0])]
    diag["origin_vox_in_roi"] = origin_vox.tolist()

    t0 = time.monotonic()
    result = locate_bifurcation_from_origin(skeleton, spacing_mm, origin_vox)
    log.info("[%s] locate_bifurcation_from_origin done in %.1fs (found=%s)",
              side, time.monotonic() - t0, result is not None)
    if result is None:
        diag["error"] = "no CCA->ICA path found from the detected origin"
        return None, diag

    g, bn = result["graph"], result["bifurcation"]
    has_eca = bool(result["eca_comps"])
    if not has_eca:
        diag["eca_missing"] = True
    log.info("[%s] bn=%s cca_comp=%d ica_comp=%d eca_comps=%s", side, bn,
              len(result["cca_comp"]), len(result["ica_comp"]), [len(c) for c in result["eca_comps"]])

    t0 = time.monotonic()
    huge = 1e6
    kept_cca, cut_cca, disc_cca = trim_to_single_path(g.subgraph(result["cca_comp"] | {bn}), bn, huge)
    kept_ica, cut_ica, disc_ica = trim_to_single_path(
        g.subgraph(result["ica_comp"] | {bn}), bn, ica_max_length_mm)
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
    kept_volume = np.pad(kept_volume, 1)
    log.info("[%s] propagate_labels done in %.1fs (kept_volume=%d voxels)",
              side, time.monotonic() - t0, kept_volume.sum())

    pad_origin_world = origin_mm + np.asarray(roi_lo) * spacing_mm - spacing_mm
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
    poly = _largest_component(poly)
    poly = _smooth(poly)

    bn_world = _vox_to_world(bn, spacing_mm, np.asarray(roi_lo) * spacing_mm + origin_mm)

    branches = [("cca", kept_cca, cut_cca, None), ("ica", kept_ica, cut_ica, None)]
    if has_eca:
        branches.append(("eca", kept_eca, cut_eca, None))
    for name, g_kept, cut_end, max_len in branches:
        path_nodes = g_kept | {bn}
        inset_node = _inset_cut_node(g, path_nodes, bn, cut_end, CUT_INSET_MM, spacing_mm)
        origin_vox_cut, direction_mm = _cut_plane(g, path_nodes, bn, inset_node, spacing_mm)
        origin_world = _vox_to_world(origin_vox_cut, spacing_mm, np.asarray(roi_lo) * spacing_mm + origin_mm)
        poly = _clip_away(poly, origin_world, direction_mm, bn_world)
        diag[f"{name}_cut_world_mm"] = origin_world.tolist()

    poly = _largest_component(poly)

    # Extreme coiling (what these scans are selected for) winds the vessel
    # tightly enough that adjacent coil turns pass very close to each
    # other — VERIFIED on a real case: this left extra tiny boundary loops
    # beyond the real outlet cuts, i.e. small self-proximity gaps in the
    # coiled wall, not missed branches. A *fixed* hole-size threshold isn't
    # safe though — VERIFIED on the same case: a real outlet's loop (a
    # narrow ECA, 77 boundary points) measured only 2.68mm radius, smaller
    # than an initially-reasonable-looking 3mm cutoff, so that cutoff
    # closed real anatomy while a genuine defect (32 points balled up into
    # a 0.2mm-radius knot) also got closed — no single mm value is
    # guaranteed to fall between "real outlet" and "defect" sizes on every
    # case. What's actually known is the *count*: there should be exactly
    # 3 open loops (2 if no ECA was found at all). Search for the smallest
    # hole-size that reaches that count, so only as many (and only the
    # smallest) spurious loops get closed as needed.
    n_expected = 3 if has_eca else 2
    n_loops, n_nonmanifold = count_boundary_loops(poly)
    if n_loops > n_expected:
        for hole_size in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0):
            fill = vtk.vtkFillHolesFilter()
            fill.SetInputData(poly)
            fill.SetHoleSize(hole_size)
            fill.Update()
            candidate = vtk.vtkPolyData()
            candidate.DeepCopy(fill.GetOutput())
            cand_loops, cand_nonmanifold = count_boundary_loops(candidate)
            if cand_loops <= n_expected:
                poly, n_loops, n_nonmanifold = candidate, cand_loops, cand_nonmanifold
                diag["hole_fill_size_mm"] = hole_size
                break
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
