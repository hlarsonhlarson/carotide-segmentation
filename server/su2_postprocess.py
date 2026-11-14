"""Post-processes a finished SU2 CFD run's own output (the flow_*.vtu
snapshots server/su2_run.py's solver writes every OUTPUT_WRT_FREQ steps)
into the per-wall-point hemodynamic fields a results viewer actually shows
— instantaneous + time-averaged wall shear stress (WSS), oscillatory
shear index (OSI), relative residence time (RRT), and pressure — plus the
wall/cap geometry stats (areas, diameters, branch lengths) a "sizes and
volumes" panel needs.

Field extraction follows the same convention the validated standalone
pipeline's own extractor used (~/Desktop/SU2 Pipeline Code/shared/
extract_generic.py): WSS magnitude [Pa] = 0.5 * |Skin_Friction_Coefficient|
— SU2's own dimensional output under INC_NONDIM=DIMENSIONAL already scales
this by density/velocity internally, and the 0.5 factor is this project's
own established convention from real validated runs, not re-derived here.

OSI (oscillatory shear index) and RRT (relative residence time) are
standard derived hemodynamic metrics from the literature (CONFIRMED
formulas, not invented): given the per-step WSS *vector* w(t) at a wall
point,
    TAWSS = time-average of |w(t)|
    OSI   = 0.5 * (1 - |time-average of w(t)| / TAWSS)
    RRT   = 1 / ((1 - 2*OSI) * TAWSS)
OSI in [0, 0.5]: 0 means WSS never reverses direction over the cycle, 0.5
means it's purely oscillatory with zero net direction — the standard
atherosclerosis-risk marker low-TAWSS/high-OSI regions are built from.
RRT blows up as OSI -> 0.5, so it's clamped to a large finite value rather
than left as inf/nan (no finite "size" a UI threshold slider could use
otherwise).
"""
import glob
import logging
import re
from pathlib import Path

import numpy as np
import vtk
from vtk.util import numpy_support
from .cfd_temporal import validate_series, last_window, cycle_quality

log = logging.getLogger("carotid_server")

MM_TO_M = 1e-3
RRT_CLAMP_PA_INV = 1e6  # finite ceiling for RRT when OSI -> 0.5 (division blows up)


def _read_vtp_from_bytes(vtp_bytes: bytes, tmp_dir: Path):
    path = tmp_dir / "surface.vtp"
    path.write_bytes(vtp_bytes)
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    return reader.GetOutput()


def _wall_point_indices(surf, wall_face_id: int = 1):
    """Returns (wall_volume_node_indices, wall_triangles_local) — the
    *volume* mesh's own 0-indexed node ids for every point touched by a
    wall-tagged triangle, and that triangle connectivity re-indexed into
    the compact wall-only point list (0..len(wall)-1) for building a
    standalone wall-surface mesh to send to the client."""
    face_array = surf.GetCellData().GetArray("ModelFaceID")
    node_array = surf.GetPointData().GetArray("GlobalNodeID")
    if face_array is None or node_array is None:
        raise ValueError("CFD surface requires ModelFaceID and GlobalNodeID")
    face_ids = numpy_support.vtk_to_numpy(face_array)
    gid = numpy_support.vtk_to_numpy(node_array)
    if (gid.shape != (surf.GetNumberOfPoints(),) or not np.isfinite(gid).all()
            or np.any(gid < 1) or np.any(gid != np.floor(gid)) or len(np.unique(gid)) != len(gid)):
        raise ValueError("CFD surface has invalid or duplicate GlobalNodeID values")
    if face_ids.shape != (surf.GetNumberOfCells(),):
        raise ValueError("CFD surface has invalid ModelFaceID values")
    wall_cells = np.nonzero(face_ids == wall_face_id)[0]
    if len(wall_cells) == 0:
        raise ValueError("no wall-tagged triangles in this surface")

    used_local = []
    seen = set()
    tris_local = []
    local_of = {}
    for ci in wall_cells:
        cell = surf.GetCell(int(ci))
        if cell.GetNumberOfPoints() != 3:
            raise ValueError("CFD wall surface must contain triangles")
        ids = [cell.GetPointId(k) for k in range(3)]
        tri = []
        for lid in ids:
            if lid not in local_of:
                local_of[lid] = len(used_local)
                used_local.append(lid)
            tri.append(local_of[lid])
        tris_local.append(tri)

    wall_vol_idx = gid[used_local] - 1  # GlobalNodeID is 1-indexed; volume nodes array is 0-indexed
    return wall_vol_idx.astype(np.int64), np.asarray(tris_local, dtype=np.int32)


def _flow_files_sorted(run_dir: Path):
    files = sorted(
        glob.glob(str(run_dir / "flow_*.vtu")),
        key=lambda p: int(re.search(r"flow_(\d+)\.vtu", p).group(1)),
    )
    steps = [int(re.search(r"flow_(\d+)\.vtu", f).group(1)) for f in files]
    return list(zip(files, steps))


def extract_wall_time_series(run_dir: Path, vtp_bytes: bytes, time_step_s: float, on_frame=None):
    """Reads every flow_*.vtu snapshot SU2 wrote for this run and builds
    the wall-only surface (points in mm, triangles) plus per-frame WSS
    vectors [Pa] and pressure [Pa] at every wall point.

    Returns dict with:
      points_mm: (N,3), triangles: (M,3), times_s: (T,),
      wss_vec_pa: (T,N,3), pressure_pa: (T,N)
    on_frame(i, total), if given, is called after each frame is read (for
    progress reporting on what can be a few dozen multi-MB XML files).
    """
    import tempfile
    with tempfile.TemporaryDirectory(prefix="su2_post_") as tmp:
        surf = _read_vtp_from_bytes(vtp_bytes, Path(tmp))
    wall_vol_idx, triangles = _wall_point_indices(surf)
    surface_ids = numpy_support.vtk_to_numpy(surf.GetPointData().GetArray("GlobalNodeID"))
    order = np.argsort(surface_ids)
    surface_local = order[np.searchsorted(surface_ids[order], wall_vol_idx + 1)]
    expected_points_mm = numpy_support.vtk_to_numpy(surf.GetPoints().GetData())[surface_local]
    if not np.isfinite(time_step_s) or time_step_s <= 0:
        raise ValueError("CFD output time step must be positive")

    files = _flow_files_sorted(run_dir)
    if not files:
        raise ValueError(f"no flow_*.vtu snapshots found in {run_dir}")

    points_mm = None
    times_s = []
    wss_frames = []
    pressure_frames = []
    for i, (path, step) in enumerate(files):
        reader = vtk.vtkXMLUnstructuredGridReader()
        reader.SetFileName(path)
        reader.Update()
        ug = reader.GetOutput()
        pd = ug.GetPointData()
        if ug.GetPoints() is None or ug.GetNumberOfPoints() <= int(wall_vol_idx.max()):
            raise ValueError(f"CFD frame {step} does not contain the expected mesh nodes")
        all_pts = numpy_support.vtk_to_numpy(ug.GetPoints().GetData())
        frame_points_mm = all_pts[wall_vol_idx] * 1000.
        if not np.allclose(frame_points_mm, expected_points_mm, atol=1e-3, rtol=0):
            raise ValueError(f"CFD frame {step} node order or geometry does not match the exported wall")

        if points_mm is None:
            # The flow_*.vtu snapshot's own point coordinates are in
            # meters (SU2's internal convention — server/su2_run.py's
            # mesh.su2 writer converts mm -> m when building the input
            # mesh, and SU2 echoes that same unit back in its output) —
            # CONFIRMED (a real patient run's own points came back as
            # ~0.06, -0.04, -0.09) that leaving these as-is produces a
            # geometry a thousand times smaller than this field's own
            # name promises, invisible once rendered (not an error, just
            # a vanishingly small object relative to any reasonable
            # camera framing). Converting back to mm here keeps every
            # consumer of this field (the Swift viewer, any future one)
            # on the same mm convention every other surface in this app
            # already uses.
            points_mm = frame_points_mm.astype(np.float32)

        cf_array, pressure_array = pd.GetArray("Skin_Friction_Coefficient"), pd.GetArray("Pressure")
        if cf_array is None or pressure_array is None:
            raise ValueError(f"CFD frame {step} is missing skin friction or pressure output")
        cf = numpy_support.vtk_to_numpy(cf_array)
        pressure_values = numpy_support.vtk_to_numpy(pressure_array)
        if cf.shape != (ug.GetNumberOfPoints(), 3) or pressure_values.shape != (ug.GetNumberOfPoints(),):
            raise ValueError(f"CFD frame {step} has invalid WSS or pressure dimensions")
        wss_vec = (0.5 * cf[wall_vol_idx]).astype(np.float32)
        pressure = pressure_values[wall_vol_idx].astype(np.float32)
        if not np.isfinite(wss_vec).all() or not np.isfinite(pressure).all():
            raise ValueError(f"CFD frame {step} contains nonfinite wall fields")

        wss_frames.append(wss_vec)
        pressure_frames.append(pressure)
        times_s.append(step * time_step_s)
        if on_frame:
            on_frame(i + 1, len(files))

    return {
        "points_mm": points_mm,
        "triangles": triangles,
        "times_s": np.asarray(times_s, dtype=np.float64),
        "wss_vec_pa": np.stack(wss_frames, axis=0),
        "pressure_pa": np.stack(pressure_frames, axis=0),
    }


def compute_derived_fields(series: dict, cycle_s: float) -> dict:
    """TAWSS/OSI/RRT/mean pressure over the last recorded cycle, plus the
    raw per-frame WSS-magnitude/pressure time series (for an instantaneous
    frame-scrubbing view) — see module docstring for the formulas."""
    times_s = np.asarray(series["times_s"], dtype=float)
    wss_vec = np.asarray(series["wss_vec_pa"], dtype=float)  # (T, N, 3)
    pressure = np.asarray(series["pressure_pa"], dtype=float)  # (T, N)
    validate_series(times_s, wss_vec, pressure, cycle_s)

    t_settled, w = last_window(times_s, wss_vec, cycle_s)
    _, window_pressure = last_window(times_s, pressure, cycle_s)
    quality = cycle_quality(times_s, wss_vec, pressure, cycle_s)

    wss_mag = np.linalg.norm(wss_vec, axis=2)  # (T, N) — every frame, for scrubbing

    if len(t_settled) >= 2:
        tawss = np.trapezoid(np.linalg.norm(w, axis=2), t_settled, axis=0) / (t_settled[-1] - t_settled[0])
        mean_vec = np.trapezoid(w, t_settled, axis=0) / (t_settled[-1] - t_settled[0])
    else:
        tawss = np.linalg.norm(w, axis=2).mean(axis=0)
        mean_vec = w.mean(axis=0)
    mean_vec_mag = np.linalg.norm(mean_vec, axis=1)

    tawss_safe = np.maximum(tawss, 1e-9)
    osi = 0.5 * (1.0 - mean_vec_mag / tawss_safe)
    osi = np.clip(osi, 0.0, 0.5)
    undefined_osi = tawss <= 1e-9
    osi[undefined_osi] = 0.  # finite display placeholder, explicitly masked below
    quality["undefined_osi_nodes"] = int(undefined_osi.sum())

    denom = (1.0 - 2.0 * osi) * tawss_safe
    denom_safe = np.maximum(denom, 1.0 / RRT_CLAMP_PA_INV)  # avoid a real 1/0 before np.where selects it away
    rrt = np.where(denom > 1.0 / RRT_CLAMP_PA_INV, 1.0 / denom_safe, RRT_CLAMP_PA_INV)

    mean_pressure = (
        np.trapezoid(window_pressure, t_settled, axis=0) / (t_settled[-1] - t_settled[0])
        if len(t_settled) >= 2 else window_pressure.mean(axis=0)
    )

    return {
        "tawss_pa": tawss.astype(np.float32),
        "osi": osi.astype(np.float32),
        "rrt_pa_inv": rrt.astype(np.float32),
        "mean_pressure_pa": mean_pressure.astype(np.float32),
        "wss_mag_frames_pa": wss_mag.astype(np.float32),  # (T, N)
        "pressure_frames_pa": pressure.astype(np.float32),  # (T, N)
        "times_s": times_s,
        "temporal_quality": quality,
        "osi_valid_mask": ~undefined_osi,
        # Legacy field retained for old clients; quality reports periodicity.
        "settled_cycle_start_s": float(t_settled[0]) if len(t_settled) else 0.0,
    }


def _triangle_areas_mm2(points_mm, triangles):
    p0, p1, p2 = points_mm[triangles[:, 0]], points_mm[triangles[:, 1]], points_mm[triangles[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)


def _cap_outward_normal(points_mm, triangles) -> np.ndarray:
    """Average unit face-normal over a cap's own triangles — reliable as
    an "outward from the fluid domain" direction only because su2_export's
    capping (and SU2/VTK meshes generally) require consistently-wound,
    outward-facing triangles to be valid in the first place, not
    re-derived or checked here. Used as a cheap proxy for the vessel's
    local flow axis at that cap, to report a bifurcation angle alongside
    the cap diameters."""
    p0, p1, p2 = points_mm[triangles[:, 0]], points_mm[triangles[:, 1]], points_mm[triangles[:, 2]]
    face_normals = np.cross(p1 - p0, p2 - p0)
    lengths = np.linalg.norm(face_normals, axis=1, keepdims=True)
    valid = lengths[:, 0] > 1e-9
    if not valid.any():
        return np.array([0.0, 0.0, 1.0])
    unit_normals = face_normals[valid] / lengths[valid]
    avg = unit_normals.mean(axis=0)
    norm = np.linalg.norm(avg)
    return avg / norm if norm > 1e-9 else np.array([0.0, 0.0, 1.0])


def compute_geometry_stats(vtp_bytes: bytes, face_map: dict) -> dict:
    """Wall area + each cap's own area/equivalent-diameter/centroid/
    outward-normal, plus the angle(s) between outlet caps derived from
    those normals — the "sizes and angles" a geometry/annotation panel
    shows alongside the hemodynamic fields. face_map: {ModelFaceID(int):
    tag(str)}."""
    import tempfile
    with tempfile.TemporaryDirectory(prefix="su2_geom_") as tmp:
        surf = _read_vtp_from_bytes(vtp_bytes, Path(tmp))

    face_ids = numpy_support.vtk_to_numpy(surf.GetCellData().GetArray("ModelFaceID"))
    pts_mm = numpy_support.vtk_to_numpy(surf.GetPoints().GetData())

    def cell_points(ci):
        cell = surf.GetCell(int(ci))
        return [cell.GetPointId(k) for k in range(3)]

    stats = {"caps": {}}
    for fid, tag in face_map.items():
        cells = np.nonzero(face_ids == fid)[0]
        if len(cells) == 0:
            continue
        tris = np.asarray([cell_points(ci) for ci in cells], dtype=np.int64)
        area_mm2 = float(_triangle_areas_mm2(pts_mm, tris).sum())
        if tag == "wall":
            stats["wall_area_mm2"] = area_mm2
            continue
        pt_ids = np.unique(tris)
        centroid = pts_mm[pt_ids].mean(axis=0)
        d_eq_mm = 2.0 * np.sqrt(area_mm2 / np.pi)
        normal = _cap_outward_normal(pts_mm, tris)
        stats["caps"][tag] = {
            "area_mm2": area_mm2,
            "equivalent_diameter_mm": d_eq_mm,
            "centroid_mm": centroid.tolist(),
            "normal_mm": normal.tolist(),
        }

    # Angle between every pair of non-wall caps whose outward normals are
    # both known — with the usual two outlets + one inlet, that's just
    # the ICA/ECA bifurcation angle, reported at their shared midpoint so
    # a UI can anchor a label there.
    angles = []
    tags = sorted(stats["caps"].keys())
    for i in range(len(tags)):
        for j in range(i + 1, len(tags)):
            a, b = stats["caps"][tags[i]], stats["caps"][tags[j]]
            na, nb = np.asarray(a["normal_mm"]), np.asarray(b["normal_mm"])
            cos_angle = float(np.clip(np.dot(na, nb), -1.0, 1.0))
            degrees = float(np.degrees(np.arccos(cos_angle)))
            midpoint = ((np.asarray(a["centroid_mm"]) + np.asarray(b["centroid_mm"])) / 2.0).tolist()
            angles.append({
                "label": f"{tags[i]} / {tags[j]}",
                "degrees": degrees,
                "at_mm": midpoint,
            })
    stats["angles"] = angles
    return stats
