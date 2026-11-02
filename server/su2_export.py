"""Produces a CFD-ready volume mesh (.vtu) + boundary surface (.vtp, with
a ModelFaceID cell array marking wall vs. each inlet/outlet cap) from a
build_surface.py result — the same (.vtu, .vtp-with-ModelFaceID) pair the
user's separate su2-project pipeline expects.

Two steps, both pure Python/VTK, no SimVascular:
  1. Face tagging (this module): caps the three open ends by
     triangulating their boundary loops and tags every cell with
     ModelFaceID (1=wall, 2=cca_inlet, 3=ica_outlet, 4=eca_outlet).
  2. Volume meshing (this module, via the `tetgen` package — a direct
     Python binding to the TetGen library, no SimVascular involved):
     tetrahedralizes the tagged surface directly and recovers each
     output boundary triangle's ModelFaceID from TetGen's own facet
     markers.

     This replaces an earlier version that shelled out to SimVascular's
     own bundled Python for `sv.meshing.TetGen`, which also runs a
     boundary-layer extrusion. CONFIRMED (on a real patient's own
     surface, already watertight and manifold by every check both this
     module's tag_surface_faces() and a separate pyvista
     is_manifold check agree on) that SimVascular's required
     surface_mesh_flag=True preprocessing step — an internal MMG remesh
     it always runs before TetGen, regardless of boundary-layer settings
     or target edge size, and which can't be skipped without crashing
     (TetGen's boundary-layer code depends on an array MMG's pass
     creates) — reliably introduces 3 non-manifold edges on this
     geometry's sharpest concave curve (the bifurcation crotch), which
     then makes TetGen's own later boundary-recovery step fail outright
     with a self-intersecting-PLC error. Plain TetGen, run directly on
     the same already-clean surface with no forced remesh in front of
     it, tetrahedralizes it in under 2 seconds with no such failure —
     so the fix is removing the unnecessary (and, here, actively
     harmful) remeshing step, not tuning it further.

     Deliberately doesn't attempt a prism boundary layer here (plain
     TetGen has no native equivalent to SimVascular's
     set_boundary_layer_options) — this produces a uniform-density
     volume mesh at GLOBAL_EDGE_SIZE_MM throughout, a real working
     mesh now rather than a blocked one; graded/boundary-layer near-wall
     refinement is a follow-up, not a prerequisite for a usable mesh.
"""
import logging
from .coordinates import surface_xyz
import tempfile
from pathlib import Path

import numpy as np
import tetgen
import vtk
from vtk.util import numpy_support

log = logging.getLogger("carotid_server")

GLOBAL_EDGE_SIZE_MM = 0.4

FACE_ID_WALL = 1
FACE_ID_FOR = {"inlet": 2, "ica_outlet": 3, "eca_outlet": 4}
LABEL_FOR = {"inlet": "cca_inlet", "ica_outlet": "ica_outlet", "eca_outlet": "eca_outlet"}


def _xyz(point_mm):
    """build_surface()'s output surface's vertex coordinates are in VTK
    (x,y,z) order, the reverse of this app's own [z,y,x] point_mm
    convention (carotid_seg/meshing.py's _numpy_to_vtk_image builds the
    marching-cubes input image with SetSpacing/SetOrigin reversed) — any
    world point has to go through this same flip to land in the surface's
    own coordinate space."""
    return surface_xyz(point_mm)


def _extract_boundary_loops(poly):
    edges = vtk.vtkFeatureEdges()
    edges.SetInputData(poly)
    edges.BoundaryEdgesOn()
    edges.FeatureEdgesOff()
    edges.NonManifoldEdgesOff()
    edges.ManifoldEdgesOff()
    edges.Update()
    edge_poly = edges.GetOutput()

    conn = vtk.vtkPolyDataConnectivityFilter()
    conn.SetInputData(edge_poly)
    conn.SetExtractionModeToAllRegions()
    conn.ColorRegionsOn()
    conn.Update()
    labeled = conn.GetOutput()
    region_ids = numpy_support.vtk_to_numpy(labeled.GetPointData().GetArray("RegionId"))
    n_loops = int(region_ids.max()) + 1 if labeled.GetNumberOfPoints() else 0

    loops = []
    for region in range(n_loops):
        pt_indices = np.nonzero(region_ids == region)[0]
        adjacency = {i: [] for i in pt_indices}
        lines = labeled.GetLines()
        lines.InitTraversal()
        id_list = vtk.vtkIdList()
        while lines.GetNextCell(id_list):
            a, b = id_list.GetId(0), id_list.GetId(1)
            if a in adjacency and b in adjacency:
                adjacency[a].append(b)
                adjacency[b].append(a)
        start = pt_indices[0]
        ordered = [start]
        prev, cur = None, start
        while True:
            nbrs = [n for n in adjacency[cur] if n != prev]
            if not nbrs:
                break
            nxt = nbrs[0]
            if nxt == start:
                break
            ordered.append(nxt)
            prev, cur = cur, nxt
        loops.append([labeled.GetPoint(i) for i in ordered])
    return loops


def _fan_cap(loop_points_world):
    """Triangulates a cap loop with vtkContourTriangulator rather than a
    single-point fan from the loop's own centroid. A real cut cross-
    section this close to the bifurcation isn't always a neat circle —
    CONFIRMED (on a real patient's own inlet loop) it can be irregular
    enough that the arithmetic centroid sits off-center relative to the
    loop's own shape, which a fan from it turns into very thin, near-
    degenerate sliver triangles at that single, very-high-valence center
    vertex. SimVascular's own TetGen run (surface_mesh_flag=True, which
    remeshes the surface with MMG before volume meshing) then fails on
    exactly that surface with "bad edges" even though it's already a
    manifold, watertight, individually-clean mesh on this module's own
    checks — MMG itself is what chokes on the singular fan apex. A proper
    polygon triangulation of the same loop has no singular point and no
    thin slivers from it, and survived the same MMG remesh pass without
    error."""
    pts = vtk.vtkPoints()
    n = len(loop_points_world)
    for p in loop_points_world:
        pts.InsertNextPoint(*p)
    lines = vtk.vtkCellArray()
    lines.InsertNextCell(n + 1)
    for i in range(n):
        lines.InsertCellPoint(i)
    lines.InsertCellPoint(0)
    contour = vtk.vtkPolyData()
    contour.SetPoints(pts)
    contour.SetLines(lines)

    triangulator = vtk.vtkContourTriangulator()
    triangulator.SetInputData(contour)
    triangulator.Update()
    cap = vtk.vtkPolyData()
    cap.DeepCopy(triangulator.GetOutput())
    return cap


def _centroid_fan_cap(loop_points_world):
    """The single-point-fan capping _fan_cap() replaced. CONFIRMED (on a
    different real patient's own inlet loop — one with ~0.1-0.4mm of
    genuine out-of-plane waviness, not perfectly flat) that
    vtkContourTriangulator isn't universally safer either: it can leave 2
    boundary-loop vertices non-manifold once the cap is merged with the
    wall, on a loop shape where the plain centroid fan merges in cleanly.
    Each method has a real failure mode the other doesn't, on different
    loop shapes — see tag_surface_faces for the fallback this exists
    for."""
    pts = vtk.vtkPoints()
    centroid = np.mean(loop_points_world, axis=0)
    pts.InsertNextPoint(*centroid)
    for p in loop_points_world:
        pts.InsertNextPoint(*p)
    n = len(loop_points_world)
    cells = vtk.vtkCellArray()
    for i in range(n):
        tri = vtk.vtkTriangle()
        tri.GetPointIds().SetId(0, 0)
        tri.GetPointIds().SetId(1, 1 + i)
        tri.GetPointIds().SetId(2, 1 + (i + 1) % n)
        cells.InsertNextCell(tri)
    cap = vtk.vtkPolyData()
    cap.SetPoints(pts)
    cap.SetPolys(cells)
    return cap


def _count_nonmanifold_edges(poly) -> int:
    fe = vtk.vtkFeatureEdges()
    fe.SetInputData(poly)
    fe.NonManifoldEdgesOn()
    fe.BoundaryEdgesOff()
    fe.FeatureEdgesOff()
    fe.ManifoldEdgesOff()
    fe.Update()
    return fe.GetOutput().GetNumberOfPoints()


def _nonmanifold_points(poly):
    fe = vtk.vtkFeatureEdges()
    fe.SetInputData(poly)
    fe.NonManifoldEdgesOn()
    fe.BoundaryEdgesOff()
    fe.FeatureEdgesOff()
    fe.ManifoldEdgesOff()
    fe.Update()
    out = fe.GetOutput()
    if out.GetNumberOfPoints() == 0:
        return np.zeros((0, 3))
    return numpy_support.vtk_to_numpy(out.GetPoints().GetData())


def _count_boundary_edges(poly) -> int:
    """CONFIRMED (on a real patient's inlet loop, 62 boundary points) that
    vtkContourTriangulator can silently triangulate a non-planar simple
    polygon *incompletely* — it produced 58 triangles instead of the 60 a
    simple n-2 ear-cut of 62 points must yield, leaving a small hole in
    the cap itself. That hole shows up as a handful of leftover
    *boundary* edges (each used by only 1 triangle), not as a non-
    manifold edge (used by 3+) — so a check that only looks for non-
    manifold edges (as this function's sibling below did before) misses
    it entirely. The "capped" surface then isn't actually watertight,
    which makes TetGen's inside/outside classification fail over the
    *entire* domain (it reported removing all input vertices as
    exterior) rather than failing locally near the hole — hence the
    opaque "Failed to tetrahedralize" with no indication of where the
    real defect was."""
    fe = vtk.vtkFeatureEdges()
    fe.SetInputData(poly)
    fe.BoundaryEdgesOn()
    fe.NonManifoldEdgesOff()
    fe.FeatureEdgesOff()
    fe.ManifoldEdgesOff()
    fe.Update()
    return fe.GetOutput().GetNumberOfPoints()


def _boundary_points(poly):
    fe = vtk.vtkFeatureEdges()
    fe.SetInputData(poly)
    fe.BoundaryEdgesOn()
    fe.NonManifoldEdgesOff()
    fe.FeatureEdgesOff()
    fe.ManifoldEdgesOff()
    fe.Update()
    out = fe.GetOutput()
    if out.GetNumberOfPoints() == 0:
        return np.zeros((0, 3))
    return numpy_support.vtk_to_numpy(out.GetPoints().GetData())


def _tag_cells(poly, face_id):
    n = poly.GetNumberOfCells()
    arr = numpy_support.numpy_to_vtk(np.full(n, face_id, dtype=np.int32), deep=True, array_type=vtk.VTK_INT)
    arr.SetName("ModelFaceID")
    poly.GetCellData().AddArray(arr)
    return poly


def tag_surface_faces(poly, series_by_branch: dict):
    """poly: the uncapped surface from build_surface.build_surface(), open
    at exactly one loop per path present (the shared CCA inlet; the
    cca_ica path's ICA-tip outlet; the cca_eca path's ECA-tip outlet).
    Returns (tagged_capped_polydata, face_map) —
    face_map: {face_id(int): "wall"|"cca_inlet"|"ica_outlet"|"eca_outlet"}.
    """
    inlet_series = series_by_branch.get("cca_ica") or series_by_branch.get("cca_eca")
    far_points = {}
    if inlet_series:
        far_points["inlet"] = _xyz(inlet_series[0]["point_mm"])
    if series_by_branch.get("cca_ica"):
        far_points["ica_outlet"] = _xyz(series_by_branch["cca_ica"][-1]["point_mm"])
    if series_by_branch.get("cca_eca"):
        far_points["eca_outlet"] = _xyz(series_by_branch["cca_eca"][-1]["point_mm"])

    loops = _extract_boundary_loops(poly)
    expected = len(far_points)
    if len(loops) != expected:
        raise ValueError(
            f"expected exactly {expected} open boundary loop(s) ({list(far_points)}), "
            f"found {len(loops)} — surface topology isn't clean enough to tag automatically"
        )

    loop_centroids = [np.mean(loop, axis=0) for loop in loops]
    assigned = {}
    for name, pt in far_points.items():
        pt = np.asarray(pt, dtype=float)
        for idx in sorted(range(len(loops)), key=lambda i: np.linalg.norm(loop_centroids[i] - pt)):
            if idx not in assigned:
                assigned[idx] = name
                break
    if len(assigned) != expected:
        raise ValueError(f"could not uniquely match every loop to a branch far point: {assigned}")

    tagged_wall = vtk.vtkPolyData()
    tagged_wall.DeepCopy(poly)
    _tag_cells(tagged_wall, FACE_ID_WALL)

    def build(cap_method_for_idx) -> "vtk.vtkPolyData":
        append = vtk.vtkAppendPolyData()
        append.AddInputData(tagged_wall)
        for idx, name in assigned.items():
            cap = cap_method_for_idx(idx)(loops[idx])
            _tag_cells(cap, FACE_ID_FOR[name])
            append.AddInputData(cap)
        append.Update()
        cleaner = vtk.vtkCleanPolyData()
        cleaner.SetInputData(append.GetOutput())
        cleaner.PointMergingOn()
        cleaner.Update()
        out = vtk.vtkPolyData()
        out.DeepCopy(cleaner.GetOutput())
        return out

    def _defect_count(p):
        # Both kinds of defect matter: non-manifold edges (shared by 3+
        # triangles) AND leftover boundary edges (shared by only 1) — the
        # latter is what an incompletely-triangulated cap leaves behind,
        # and it alone makes the surface non-watertight even though it
        # has zero non-manifold edges. See _count_boundary_edges().
        return _count_nonmanifold_edges(p), _count_boundary_edges(p)

    result = build(lambda idx: _fan_cap)
    n_bad = _defect_count(result)
    if any(n_bad):
        # _fan_cap (vtkContourTriangulator) left the merged surface
        # non-manifold or non-watertight — CONFIRMED this can happen on a
        # loop with real out-of-plane waviness even though the same
        # loop's own boundary-edge adjacency is a clean simple cycle (not
        # a bug in loop extraction). Retry only the specific loop(s)
        # whose own points are implicated, using the more robust (if
        # lower-quality) centroid fan instead — rather than
        # un-conditionally downgrading every cap, which would reintroduce
        # the thin-sliver failure mode _fan_cap was written to fix in the
        # first place.
        bad_pts = np.concatenate([_nonmanifold_points(result), _boundary_points(result)], axis=0)
        retry_idx = set()
        for idx in assigned:
            loop_arr = np.asarray(loops[idx])
            for bp in bad_pts:
                if np.min(np.linalg.norm(loop_arr - bp, axis=1)) < 1.0:
                    retry_idx.add(idx)
                    break
        result2 = build(lambda idx: _centroid_fan_cap if idx in retry_idx else _fan_cap)
        n_bad2 = _defect_count(result2)
        if n_bad2 <= n_bad:
            result, n_bad = result2, n_bad2

    if any(n_bad):
        n_nonmanifold, n_open = n_bad
        raise ValueError(
            f"could not produce a watertight capped surface after retry "
            f"({n_nonmanifold} non-manifold edge point(s), {n_open} open "
            f"boundary edge point(s) remaining) — TetGen would otherwise "
            f"fail opaquely on this input"
        )

    face_map = {FACE_ID_WALL: "wall"}
    for idx, name in assigned.items():
        face_map[FACE_ID_FOR[name]] = LABEL_FOR[name]
    return result, face_map



def _polydata_to_arrays(poly):
    pts = numpy_support.vtk_to_numpy(poly.GetPoints().GetData()).astype(np.float64)
    n_cells = poly.GetNumberOfCells()
    faces = np.zeros((n_cells, 3), dtype=np.int32)
    polys = poly.GetPolys()
    polys.InitTraversal()
    id_list = vtk.vtkIdList()
    i = 0
    while polys.GetNextCell(id_list):
        faces[i] = [id_list.GetId(0), id_list.GetId(1), id_list.GetId(2)]
        i += 1
    face_ids = numpy_support.vtk_to_numpy(poly.GetCellData().GetArray("ModelFaceID")).astype(np.int32)
    return pts, faces, face_ids


def _tetrahedralize(tagged_surface, global_edge_size_mm: float = GLOBAL_EDGE_SIZE_MM):
    """Runs plain TetGen (no SimVascular) on the already-tagged, already-
    watertight surface. minratio/mindihedral are TetGen's own standard
    quality-mesh knobs (not specific to vascular meshing); maxvolume
    caps element size at roughly a regular tetrahedron of edge
    global_edge_size_mm, the same role GLOBAL_EDGE_SIZE_MM played for
    SimVascular's own TetGenOptions.global_edge_size.
    Returns (nodes, elems, boundary_faces, triface_markers) —
    boundary_faces/triface_markers are parallel arrays (face's 3 node
    indices, that face's ModelFaceID) — see tag_surface_faces() for what
    the marker values mean. TetGen may have subdivided some of the input
    triangles to hit the target size, so boundary_faces isn't simply the
    input `faces` array — tg.trifaces is TetGen's own record of exactly
    which triangles ended up on the output boundary.

    mindihedral/minratio drive TetGen's own quality-improvement pass
    (Steiner point insertion to fix bad-angle tets/subfaces). CONFIRMED
    (on a real patient's surface with one near-degenerate sliver
    triangle — min face angle ~0.025 degrees, basically collapsed) that
    this pass can crash outright ("Internal TetGen error within
    `split_subface`") on such an input, even though the *unrefined*
    tetrahedralization of the exact same watertight surface succeeds
    immediately. A single sliver from upstream surface generation
    shouldn't block meshing entirely, so on that specific crash this
    retries once with the quality pass disabled — same surface, same
    target element size, just without the refinement step that choked
    — trading element quality near that one sliver for a mesh that
    actually exists."""
    pts, faces, face_ids = _polydata_to_arrays(tagged_surface)
    max_volume = (global_edge_size_mm ** 3) / (6 * np.sqrt(2))

    tg = tetgen.TetGen(pts, faces, face_ids)
    try:
        nodes, elems, _attr, triface_markers = tg.tetrahedralize(
            order=1, mindihedral=10, minratio=1.5, maxvolume=max_volume,
        )
    except RuntimeError as exc:
        log.warning(
            "TetGen quality pass failed (%s) — retrying without "
            "mindihedral/minratio quality refinement", exc,
        )
        tg = tetgen.TetGen(pts, faces, face_ids)
        nodes, elems, _attr, triface_markers = tg.tetrahedralize(
            order=1, maxvolume=max_volume,
        )

    boundary_faces = np.asarray(tg.trifaces)
    triface_markers = np.asarray(triface_markers)
    return nodes, elems, boundary_faces, triface_markers


def _write_volume_vtu_bytes(nodes, elems) -> bytes:
    points = vtk.vtkPoints()
    points.SetData(numpy_support.numpy_to_vtk(np.ascontiguousarray(nodes, dtype=np.float64)))

    grid = vtk.vtkUnstructuredGrid()
    grid.SetPoints(points)
    cell_types = np.full(len(elems), vtk.VTK_TETRA, dtype=np.uint8)
    # vtkCellArray's flat connectivity: [4, i0,i1,i2,i3, 4, i0,i1,i2,i3, ...]
    connectivity = np.empty((len(elems), 5), dtype=np.int64)
    connectivity[:, 0] = 4
    connectivity[:, 1:] = elems
    cell_array = vtk.vtkCellArray()
    cell_array.SetCells(len(elems), numpy_support.numpy_to_vtk(connectivity.ravel(), deep=True, array_type=vtk.VTK_ID_TYPE))
    grid.SetCells(numpy_support.numpy_to_vtk(cell_types, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR), cell_array)

    node_ids = numpy_support.numpy_to_vtk(np.arange(1, len(nodes) + 1, dtype=np.int32), deep=True, array_type=vtk.VTK_INT)
    node_ids.SetName("GlobalNodeID")
    grid.GetPointData().AddArray(node_ids)

    elem_ids = numpy_support.numpy_to_vtk(np.arange(1, len(elems) + 1, dtype=np.int32), deep=True, array_type=vtk.VTK_INT)
    elem_ids.SetName("GlobalElementID")
    grid.GetCellData().AddArray(elem_ids)

    region_ids = numpy_support.numpy_to_vtk(np.ones(len(elems), dtype=np.int32), deep=True, array_type=vtk.VTK_INT)
    region_ids.SetName("ModelRegionID")
    grid.GetCellData().AddArray(region_ids)

    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetWriteToOutputString(True)
    writer.SetInputData(grid)
    writer.Write()
    return writer.GetOutputString().encode("latin-1") if isinstance(writer.GetOutputString(), str) else bytes(writer.GetOutputString())


def _write_boundary_vtp_bytes(nodes, triface_markers_faces, triface_markers) -> bytes:
    """triface_markers_faces: (n_tri, 3) int array of node indices for every
    boundary triangle TetGen reports, in the same order as triface_markers
    (the ModelFaceID recovered for each one)."""
    points = vtk.vtkPoints()
    points.SetData(numpy_support.numpy_to_vtk(np.ascontiguousarray(nodes, dtype=np.float64)))
    poly = vtk.vtkPolyData()
    poly.SetPoints(points)

    cells = vtk.vtkCellArray()
    n = len(triface_markers_faces)
    connectivity = np.empty((n, 4), dtype=np.int64)
    connectivity[:, 0] = 3
    connectivity[:, 1:] = triface_markers_faces
    cells.SetCells(n, numpy_support.numpy_to_vtk(connectivity.ravel(), deep=True, array_type=vtk.VTK_ID_TYPE))
    poly.SetPolys(cells)

    face_id_arr = numpy_support.numpy_to_vtk(triface_markers.astype(np.int32), deep=True, array_type=vtk.VTK_INT)
    face_id_arr.SetName("ModelFaceID")
    poly.GetCellData().AddArray(face_id_arr)
    elem_ids = numpy_support.numpy_to_vtk(np.arange(1, n + 1, dtype=np.int32), deep=True, array_type=vtk.VTK_INT)
    elem_ids.SetName("GlobalElementID")
    poly.GetCellData().AddArray(elem_ids)
    # Same 1-indexed node numbering as the volume .vtu's own GlobalNodeID
    # (this poly's own points are the *full* shared `nodes` array, same
    # one the volume mesh uses, before vtkCleanPolyData below potentially
    # drops/reorders unused ones) — lets vtu_vtp_bytes_to_su2_mesh_text
    # map every boundary triangle's point back to its volume node index
    # without assuming point order survives cleaning unchanged.
    node_ids = numpy_support.numpy_to_vtk(np.arange(1, len(nodes) + 1, dtype=np.int32), deep=True, array_type=vtk.VTK_INT)
    node_ids.SetName("GlobalNodeID")
    poly.GetPointData().AddArray(node_ids)

    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(poly)
    cleaner.PointMergingOff()
    cleaner.Update()

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetWriteToOutputString(True)
    writer.SetInputData(cleaner.GetOutput())
    writer.Write()
    return writer.GetOutputString().encode("latin-1") if isinstance(writer.GetOutputString(), str) else bytes(writer.GetOutputString())


def build_su2_mesh(poly, series_by_branch: dict, global_edge_size_mm: float = GLOBAL_EDGE_SIZE_MM, on_stage=None):
    """Full pipeline: tag faces, tetrahedralize with plain TetGen, build
    the .vtu/.vtp byte payloads. Returns (vtu_bytes, vtp_bytes, face_map, diag).
    on_stage(stage: str, info: dict), if given, is called right after each
    stage finishes (so a caller polling a background job can report live
    progress — tetrahedralizing a real patient's surface only takes a
    couple of seconds, so this is the only point stage-by-stage progress
    is actually observable; there's no meaningful finer-grained hook
    inside TetGen itself to report from)."""
    def _report(stage, **info):
        if on_stage:
            on_stage(stage, info)

    diag = {}
    tagged, face_map = tag_surface_faces(poly, series_by_branch)
    diag["n_surface_points"] = tagged.GetNumberOfPoints()
    diag["n_surface_cells"] = tagged.GetNumberOfCells()
    diag["face_map"] = {str(k): v for k, v in face_map.items()}
    _report("tagged_faces", n_surface_points=diag["n_surface_points"], n_surface_cells=diag["n_surface_cells"])

    nodes, elems, boundary_faces, triface_markers = _tetrahedralize(tagged, global_edge_size_mm)
    diag["n_volume_nodes"] = len(nodes)
    diag["n_volume_elements"] = len(elems)
    diag["face_id_counts"] = {
        str(k): int((triface_markers == k).sum()) for k in np.unique(triface_markers)
    }
    _report("tetrahedralized", n_volume_nodes=diag["n_volume_nodes"], n_volume_elements=diag["n_volume_elements"])

    vtu_bytes = _write_volume_vtu_bytes(nodes, elems)
    vtp_bytes = _write_boundary_vtp_bytes(nodes, boundary_faces, triface_markers)
    _report("wrote_output", vtu_bytes=len(vtu_bytes), vtp_bytes=len(vtp_bytes))

    return vtu_bytes, vtp_bytes, face_map, diag


def vtp_bytes_to_stl_bytes(vtp_bytes: bytes) -> bytes:
    """Converts a mesh job's saved .vtp (its ModelFaceID-tagged boundary
    surface) to binary STL purely so the existing STL preview viewer
    (built for server/build_surface.py's own output) can also show a
    just-built mesh before it's saved, without a second
    parser/3D-viewer path on the Swift side for the mesh's own (XML,
    base64/zlib-encoded) .vtp format. Goes through temp files — the
    simplest reliable way to drive vtkXMLPolyDataReader/vtkSTLWriter,
    neither of which is meant to be handed an in-memory buffer here."""
    with tempfile.TemporaryDirectory(prefix="vtp2stl_") as tmp:
        tmp_dir = Path(tmp)
        vtp_path = tmp_dir / "mesh_surface.vtp"
        stl_path = tmp_dir / "mesh_surface.stl"
        vtp_path.write_bytes(vtp_bytes)

        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(str(vtp_path))
        reader.Update()

        writer = vtk.vtkSTLWriter()
        writer.SetFileName(str(stl_path))
        writer.SetInputData(reader.GetOutput())
        writer.SetFileTypeToBinary()
        writer.Write()

        return stl_path.read_bytes()


MM_TO_M = 1e-3


def vtu_vtp_bytes_to_su2_mesh_text(vtu_bytes: bytes, vtp_bytes: bytes, face_map: dict) -> str:
    """Converts a finished mesh job's own cached .vtu/.vtp pair into
    SU2's native ASCII .su2 mesh format — ported from the validated
    standalone pipeline's convert_generic.py (which converted a
    SimVascular-produced .vtu/.vtp pair the same way), adapted to read
    from bytes via tempfile (same pattern as vtp_bytes_to_stl_bytes
    above) rather than files on disk, and to use this module's own
    face_map directly as the SU2 marker tag names (its values are
    already "wall"/"cca_inlet"/"ica_outlet"/"eca_outlet" — no separate
    ModelFaceID -> tag translation table needed, unlike the SimVascular-
    sourced meshes convert_generic.py originally handled, which could
    have many more distinct ModelFaceIDs needing to be merged into one
    tag).

    face_map: {ModelFaceID(int): tag(str)}, as tag_surface_faces()
    returns it (string-keyed dict from a job's cache deserializes back
    to int keys by the caller).
    """
    with tempfile.TemporaryDirectory(prefix="vtu_vtp2su2_") as tmp:
        tmp_dir = Path(tmp)
        vtu_path = tmp_dir / "mesh.vtu"
        vtp_path = tmp_dir / "mesh_surface.vtp"
        vtu_path.write_bytes(vtu_bytes)
        vtp_path.write_bytes(vtp_bytes)

        vol_reader = vtk.vtkXMLUnstructuredGridReader()
        vol_reader.SetFileName(str(vtu_path))
        vol_reader.Update()
        vol = vol_reader.GetOutput()

        surf_reader = vtk.vtkXMLPolyDataReader()
        surf_reader.SetFileName(str(vtp_path))
        surf_reader.Update()
        surf = surf_reader.GetOutput()

    vol_gid = numpy_support.vtk_to_numpy(vol.GetPointData().GetArray("GlobalNodeID"))
    vol_pts = numpy_support.vtk_to_numpy(vol.GetPoints().GetData())
    n_pts = vol.GetNumberOfPoints()
    if vol_gid.min() != 1 or vol_gid.max() != n_pts:
        raise ValueError("volume GlobalNodeID not contiguous 1..N — cannot build .su2 mesh")
    gid_to_volidx = np.empty(n_pts + 1, dtype=np.int64)
    gid_to_volidx[vol_gid] = np.arange(n_pts)

    surf_gid = numpy_support.vtk_to_numpy(surf.GetPointData().GetArray("GlobalNodeID"))
    face_ids = numpy_support.vtk_to_numpy(surf.GetCellData().GetArray("ModelFaceID"))

    n_tets = vol.GetNumberOfCells()
    for i in range(n_tets):
        if vol.GetCellType(i) != vtk.VTK_TETRA:
            raise ValueError(f"non-tet cell at index {i} (type {vol.GetCellType(i)})")

    lines = ["NDIME= 3", f"NELEM= {n_tets}"]
    for i in range(n_tets):
        cell = vol.GetCell(i)
        ids = [cell.GetPointId(k) for k in range(4)]
        lines.append(f"10 {ids[0]} {ids[1]} {ids[2]} {ids[3]}")

    lines.append(f"NPOIN= {n_pts}")
    coords_m = vol_pts * MM_TO_M
    for i in range(n_pts):
        x, y, z = coords_m[i]
        lines.append(f"{x:.9f} {y:.9f} {z:.9f}")

    tags_in_order = list(dict.fromkeys(face_map.values()))  # first-seen order, de-duplicated
    tag_tris = {t: [] for t in tags_in_order}
    for ci in range(surf.GetNumberOfCells()):
        fid = int(face_ids[ci])
        if fid not in face_map:
            raise ValueError(f"ModelFaceID {fid} not in face_map — every boundary face needs a tag")
        tag = face_map[fid]
        cell = surf.GetCell(ci)
        if cell.GetNumberOfPoints() != 3:
            raise ValueError("expected a triangulated boundary surface")
        local_ids = [cell.GetPointId(k) for k in range(3)]
        gids = [int(surf_gid[lid]) for lid in local_ids]
        vol_ids = [int(gid_to_volidx[g]) for g in gids]
        tag_tris[tag].append(vol_ids)

    lines.append(f"NMARK= {len(tags_in_order)}")
    for tag in tags_in_order:
        tris = tag_tris[tag]
        lines.append(f"MARKER_TAG= {tag}")
        lines.append(f"MARKER_ELEMS= {len(tris)}")
        for a, b, c in tris:
            lines.append(f"5 {a} {b} {c}")

    return "\n".join(lines) + "\n"
