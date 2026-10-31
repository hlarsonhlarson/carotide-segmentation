"""In-memory case registry. One process, one user — no DB needed yet."""
import uuid
from dataclasses import dataclass, field

import numpy as np


@dataclass
class CaseState:
    case_id: str
    array: np.ndarray       # (z, y, x)
    spacing_mm: np.ndarray  # (z, y, x)
    origin_mm: np.ndarray   # (z, y, x)
    modality: str
    source_path: str
    # Which DICOM series (of possibly several in source_path) this case
    # is — None when the folder only had one and no choice was made.
    # Saved into the project manifest (see app.py's /projects/open) so
    # reopening a saved project reloads the exact same series rather
    # than re-prompting or guessing.
    series_uid: str | None = None
    # {(side, branch): [[z,y,x], ...]} — dense auto-stitched path, voxel
    # coords, rebuilt from control_points whenever one is added/moved/deleted.
    centerline_points: dict = field(default_factory=dict)
    # {(side, branch): [[z,y,x], ...]} — the sparse points the user actually
    # clicked/placed, in order. Separate from centerline_points so an
    # individual placed point can be moved or deleted (the dense path has no
    # notion of "which of these came from a click" once stitched).
    control_points: dict = field(default_factory=dict)
    # {(side, branch): [{point_mm, tangent_mm, contour_px, contour_mm,
    #                     radius_mm, method}, ...]} — the Series: one
    # resampled control point per entry, segmentation added as the user
    # works through it.
    segmentation_series: dict = field(default_factory=dict)
    # {side: stl_bytes} — the last built surface per side, kept in memory
    # so GET can serve it without re-building or touching the filesystem.
    built_surfaces: dict = field(default_factory=dict)
    # {side: {"state": "running"|"done"|"error", "vtu": bytes, "vtp": bytes,
    #         "face_map": dict, "diag": dict, "error": str}} — the SU2-
    # ready volume+surface mesh job, run in a background thread since a
    # real TetGen run still takes a few seconds.
    su2_mesh_jobs: dict = field(default_factory=dict)
    # {side: {"psv_cms", "edv_cms", "cardiac_cycle_s", "ica_fraction",
    #         "ica_map_mmhg", "eca_map_mmhg"}} — the per-side physiological
    # boundary-condition parameters a future SU2 run will be driven by
    # (CCA inlet waveform shape/magnitude, ICA/ECA outlet flow split, each
    # outlet's own pressure level — independently editable since the two
    # outlets' downstream beds don't necessarily share one operating
    # pressure).
    # Deliberately just the values that actually vary patient to patient
    # in the pysu2 pipeline this was modeled on (~/Desktop/SU2 Pipeline
    # Code/*/case_params.json) — fluid properties and solver numerics
    # were constant across every real case run there, so they stay fixed
    # defaults in the SU2 config itself rather than a per-case UI field.
    su2_params: dict = field(default_factory=dict)
    # {side: {"state": "running"|"done"|"error"|"cancelled", "step": int,
    #         "total_steps": int, "t_sim": float, ..., "process":
    #         subprocess.Popen, "output_dir": str}} — the actual SU2
    # pulsatile CFD solve (server/su2_run.py), run as a real subprocess
    # (mpirun + pysu2, not a thread — the solver owns MPI/C++ global
    # state that doesn't belong inside this process) since it can run for
    # a long time; a background thread owns reading its stdout and
    # updating this dict in place so a status poll always sees live
    # progress. "process" is the live Popen handle while running (used by
    # cancel_run), stripped once the run ends.
    su2_run_jobs: dict = field(default_factory=dict)
    # {side: {"state": "running"|"done"|"error", "frame": int,
    #         "total_frames": int, "result": {...}}} — the post-processed
    # hemodynamic results (TAWSS/OSI/RRT/pressure + wall geometry) built
    # from a finished su2_run_jobs run's own flow_*.vtu snapshots (see
    # server/su2_postprocess.py). Also runs in a background thread — not
    # because any single file is slow, but reading a few dozen multi-MB
    # XML snapshots adds up to several real seconds, long enough to want
    # progress rather than a blocking request.
    su2_postprocess_jobs: dict = field(default_factory=dict)
    # Folder the user picked to persist this patient's work into (see
    # app.py's /project endpoints) — None until they choose one. Every
    # workflow step's own "Save" button writes into this same folder, so
    # losing the in-memory case (a backend restart) doesn't mean losing
    # already-saved progress, only progress never explicitly saved.
    project_dir: str | None = None


_CASES: dict[str, CaseState] = {}


def create(array, spacing_mm, origin_mm, modality, source_path, series_uid=None) -> CaseState:
    case_id = uuid.uuid4().hex[:12]
    case = CaseState(
        case_id=case_id,
        array=array,
        spacing_mm=spacing_mm,
        origin_mm=origin_mm,
        modality=modality,
        source_path=source_path,
        series_uid=series_uid,
    )
    _CASES[case_id] = case
    return case


def get(case_id: str) -> CaseState | None:
    return _CASES.get(case_id)
