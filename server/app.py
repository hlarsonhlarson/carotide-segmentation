import io
import json
import logging
import tempfile
import threading

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from PIL import Image
from pydantic import BaseModel

from pathlib import Path

from . import (
    build_surface, cases, centerline_path, dicom_io, path_series, reslice, segment2d,
    su2_export, su2_postprocess, su2_run,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("carotid_server")

app = FastAPI(title="Carotid Segmentation Backend")

AXES = {"axial": 0, "coronal": 1, "sagittal": 2}
SIDES = {"left", "right"}
BRANCHES = {"cca_ica", "cca_eca"}


class LoadCaseRequest(BaseModel):
    path: str
    series_uid: str | None = None


class AddPointRequest(BaseModel):
    point: list[int]  # [z, y, x] voxel coords


class AutoTraceRequest(BaseModel):
    start: list[int]  # [z, y, x] voxel coords
    end: list[int]  # [z, y, x] voxel coords


class ReorderRequest(BaseModel):
    from_index: int
    to_index: int


class SegmentPointRequest(BaseModel):
    method: str  # "threshold" | "levelset" | "circle"
    radius_mm: float


class ManualContourRequest(BaseModel):
    contour_px: list[list[float]]  # [[row, col], ...], same pixel space as the oblique slice


class SU2ParamsRequest(BaseModel):
    psv_cms: float
    edv_cms: float
    cardiac_cycle_s: float
    ica_fraction: float
    # Separate outlet pressure levels — the pysu2 pipeline this is modeled
    # on (~/Desktop/SU2 Pipeline Code/*/pulsatile_run_generic.py) seeded
    # both outlets' Windkessel from the same single mean_arterial_pressure
    # value; CONFIRMED (per the user, working from real per-patient runs)
    # that the two outlets' own downstream beds don't necessarily sit at
    # the same operating pressure, so each needs its own editable value
    # rather than one shared MAP applied to both.
    ica_map_mmhg: float
    eca_map_mmhg: float


# Generic literature-ish starting point (not calibrated to any specific
# patient) — same role as the hardcoded values every case under
# ~/Desktop/SU2 Pipeline Code/*/case_params.json started from before being
# overwritten per-patient; the user is expected to adjust these, not trust
# them as-is.
SU2_PARAM_DEFAULTS = {
    "psv_cms": 85.0,
    "edv_cms": 30.0,
    "cardiac_cycle_s": 0.8,
    "ica_fraction": 0.65,
    "ica_map_mmhg": 93.0,
    "eca_map_mmhg": 93.0,
}


def _window_png(array2d: np.ndarray, wl: float, ww: float) -> bytes:
    lo, hi = wl - ww / 2.0, wl + ww / 2.0
    if hi <= lo:
        hi = lo + 1.0
    clipped = np.clip(array2d, lo, hi)
    normalized = ((clipped - lo) / (hi - lo) * 255.0).astype(np.uint8)
    img = Image.fromarray(normalized, mode="L")
    buf = io.BytesIO()
    # compress_level=1 (fastest, still lossless) instead of PIL's default 6
    # — this is a single grayscale slice refetched on every scroll step, so
    # encode speed matters more than the extra few KB a higher compression
    # level would save.
    img.save(buf, format="PNG", compress_level=1)
    return buf.getvalue()


def _default_window(array: np.ndarray, modality: str):
    """(center, width) in intensity units. CT: fixed, vessel-friendly (wide
    enough to show contrast-enhanced lumen without blowing out bone). MR has
    no fixed intensity scale, so stretch from this volume's own 1st/99th
    percentile instead — same percentile-based approach already used for
    TOF display elsewhere in this project (mra_pipeline's MIP diagnostics)."""
    if modality == "CT":
        return 300.0, 800.0
    lo, hi = np.percentile(array, [1.0, 99.0])
    if hi <= lo:
        hi = lo + 1.0
    return float((lo + hi) / 2.0), float(hi - lo)


@app.get("/series")
def get_series(path: str):
    try:
        return {"series": dicom_io.list_series(path)}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        log.exception("failed to list series at %s", path)
        raise HTTPException(status_code=400, detail=f"could not list DICOM series: {e}")


def _load_dicom_case(path: str, series_uid: str | None):
    """Shared by create_case and open_project — loads a DICOM series into
    a fresh in-memory CaseState and builds the CaseInfo response dict
    both endpoints return."""
    try:
        array, spacing_mm, origin_mm, modality = dicom_io.load_series(path, series_uid)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log.exception("failed to load series at %s", path)
        raise HTTPException(status_code=400, detail=f"could not load DICOM series: {e}")

    case = cases.create(array, spacing_mm, origin_mm, modality, path, series_uid=series_uid)
    wl, ww = _default_window(array, modality)
    info = {
        "case_id": case.case_id,
        "shape": list(array.shape),
        "spacing_mm": spacing_mm.tolist(),
        "origin_mm": origin_mm.tolist(),
        "modality": modality,
        "default_window": {"wl": wl, "ww": ww},
    }
    return case, info


@app.post("/cases")
def create_case(req: LoadCaseRequest):
    _case, info = _load_dicom_case(req.path, req.series_uid)
    return info


@app.get("/cases/{case_id}/meta")
def get_meta(case_id: str):
    case = cases.get(case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="unknown case_id")
    wl, ww = _default_window(case.array, case.modality)
    return {
        "case_id": case.case_id,
        "shape": list(case.array.shape),
        "spacing_mm": case.spacing_mm.tolist(),
        "origin_mm": case.origin_mm.tolist(),
        "modality": case.modality,
        "source_path": case.source_path,
        "default_window": {"wl": wl, "ww": ww},
    }


@app.get("/cases/{case_id}/slice")
def get_slice(case_id: str, axis: str, index: int, wl: float | None = None, ww: float | None = None):
    case = cases.get(case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="unknown case_id")
    if axis not in AXES:
        raise HTTPException(status_code=400, detail=f"axis must be one of {list(AXES)}")

    axis_i = AXES[axis]
    size = case.array.shape[axis_i]
    if not (0 <= index < size):
        raise HTTPException(status_code=400, detail=f"index out of range [0, {size})")

    if axis_i == 0:
        plane = case.array[index, :, :]
    elif axis_i == 1:
        # Coronal/sagittal planes have z (axis 0, increasing toward
        # superior — established convention throughout this codebase, see
        # mra_pipeline.py) as their row axis. Row 0 is the TOP of the
        # displayed image, so without flipping, the most inferior slice
        # (z=0) would render at the top — upside down relative to the
        # standard radiological convention of superior-up. Flip vertically
        # so z increasing (toward superior) goes up the image.
        plane = case.array[::-1, index, :]
    else:
        plane = case.array[::-1, :, index]

    if wl is None or ww is None:
        wl, ww = _default_window(case.array, case.modality)
    return Response(content=_window_png(plane, wl, ww), media_type="image/png")


@app.post("/cases/{case_id}/autoseed")
def autoseed(case_id: str):
    raise HTTPException(status_code=501, detail="not implemented yet")


def _check_side_branch(side: str, branch: str):
    if side not in SIDES:
        raise HTTPException(status_code=400, detail=f"side must be one of {sorted(SIDES)}")
    if branch not in BRANCHES:
        raise HTTPException(status_code=400, detail=f"branch must be one of {sorted(BRANCHES)}")


def _validate_point(point, shape):
    if len(point) != 3:
        raise HTTPException(status_code=400, detail="point must be [z, y, x]")
    p = [int(v) for v in point]
    for v, size in zip(p, shape):
        if not (0 <= v < size):
            raise HTTPException(status_code=400, detail=f"point {p} out of bounds {shape}")
    return p


def _restitch(case, side, branch):
    key = (side, branch)
    control = case.control_points.get(key, [])
    try:
        path = centerline_path.stitch_full_path(case.array, case.spacing_mm, case.modality, control)
    except Exception as e:
        log.exception("centerline stitch failed for %s/%s", side, branch)
        raise HTTPException(status_code=400, detail=f"could not stitch path: {e}")
    case.centerline_points[key] = path
    return path


def _invalidate_segmentation(case, side, branch):
    """Any edit to a branch's path — adding/moving/deleting/reordering a
    control point, or replacing the whole thing via auto-trace — changes
    where the resampled Series' own points and contours *should* sit.
    Leaving a previously-fit segmentation_series in place after that
    means it still answers GET .../segment/{side}/{branch} and still
    feeds build_surface, describing a path that no longer exists —
    CONFIRMED as a real, reproducible bug: re-tracing a branch (by hand
    or via auto-trace) after it was already segmented left the 3D
    preview showing the *new* centerline next to the *old* segmentation's
    own wall contours, nowhere near each other. clear_centerline already
    did this (see its own comment); every other path-mutating endpoint
    needs the same reset."""
    case.segmentation_series.pop((side, branch), None)


@app.post("/cases/{case_id}/centerline/{side}/{branch}/points")
def add_centerline_point(case_id: str, side: str, branch: str, req: AddPointRequest):
    case = cases.get(case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="unknown case_id")
    _check_side_branch(side, branch)
    new_point = _validate_point(req.point, case.array.shape)

    key = (side, branch)
    control = case.control_points.setdefault(key, [])
    control.append(new_point)
    path = _restitch(case, side, branch)
    _invalidate_segmentation(case, side, branch)
    return {"side": side, "branch": branch, "path": path, "control_points": control}


AUTO_TRACE_SEARCH_MARGIN_MM = 30.0


@app.post("/cases/{case_id}/centerline/{side}/{branch}/auto_trace")
def auto_trace_centerline(case_id: str, side: str, branch: str, req: AutoTraceRequest):
    """Full point-to-point auto-trace between exactly two clicked
    endpoints (start of the branch, tip of the branch) — replaces
    whatever path this branch already had. Reuses the exact same
    minimum-cost-path solver (centerline_path.stitch) that bridges
    consecutive manually-placed points, just run once over the whole
    branch with a much wider search margin: the default margin is tuned
    for the short hops between manually-guided points, but real carotid
    tortuosity can swing tens of mm off the straight line between a
    distant CCA origin and an ICA/ECA tip. A convenience for the easy,
    mostly-straight cases — not a replacement for manually guiding the
    path through tight bends or low-contrast segments, where the min-cost
    path can cut a corner across a loop instead of following it; the user
    reviews the result in 3D and adds intermediate points the usual way
    wherever that happens."""
    case = cases.get(case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="unknown case_id")
    _check_side_branch(side, branch)
    start = _validate_point(req.start, case.array.shape)
    end = _validate_point(req.end, case.array.shape)

    key = (side, branch)
    try:
        path = centerline_path.stitch_full_path(
            case.array, case.spacing_mm, case.modality, [start, end],
            search_margin_mm=AUTO_TRACE_SEARCH_MARGIN_MM,
        )
    except Exception as e:
        log.exception("auto-trace failed for %s/%s", side, branch)
        raise HTTPException(status_code=400, detail=f"auto-trace failed: {e}")

    case.control_points[key] = [start, end]
    case.centerline_points[key] = path
    _invalidate_segmentation(case, side, branch)
    return {"side": side, "branch": branch, "path": path, "control_points": [start, end]}


@app.put("/cases/{case_id}/centerline/{side}/{branch}/points/{index}")
def move_centerline_point(case_id: str, side: str, branch: str, index: int, req: AddPointRequest):
    case = cases.get(case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="unknown case_id")
    _check_side_branch(side, branch)
    new_point = _validate_point(req.point, case.array.shape)

    key = (side, branch)
    control = case.control_points.get(key, [])
    if not (0 <= index < len(control)):
        raise HTTPException(status_code=400, detail=f"index out of range [0, {len(control)})")
    control[index] = new_point
    path = _restitch(case, side, branch)
    _invalidate_segmentation(case, side, branch)
    return {"side": side, "branch": branch, "path": path, "control_points": control}


@app.delete("/cases/{case_id}/centerline/{side}/{branch}/points/{index}")
def delete_centerline_point(case_id: str, side: str, branch: str, index: int):
    case = cases.get(case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="unknown case_id")
    _check_side_branch(side, branch)

    key = (side, branch)
    control = case.control_points.get(key, [])
    if not (0 <= index < len(control)):
        raise HTTPException(status_code=400, detail=f"index out of range [0, {len(control)})")
    control.pop(index)
    path = _restitch(case, side, branch)
    _invalidate_segmentation(case, side, branch)
    return {"side": side, "branch": branch, "path": path, "control_points": control}


@app.put("/cases/{case_id}/centerline/{side}/{branch}/reorder")
def reorder_centerline_points(case_id: str, side: str, branch: str, req: ReorderRequest):
    case = cases.get(case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="unknown case_id")
    _check_side_branch(side, branch)

    key = (side, branch)
    control = case.control_points.get(key, [])
    if not (0 <= req.from_index < len(control)) or not (0 <= req.to_index < len(control)):
        raise HTTPException(status_code=400, detail=f"index out of range [0, {len(control)})")
    point = control.pop(req.from_index)
    control.insert(req.to_index, point)
    path = _restitch(case, side, branch)
    _invalidate_segmentation(case, side, branch)
    return {"side": side, "branch": branch, "path": path, "control_points": control}


@app.delete("/cases/{case_id}/centerline/{side}/{branch}")
def clear_centerline(case_id: str, side: str, branch: str):
    case = cases.get(case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="unknown case_id")
    _check_side_branch(side, branch)
    case.centerline_points.pop((side, branch), None)
    case.control_points.pop((side, branch), None)
    _invalidate_segmentation(case, side, branch)
    return {"ok": True}


@app.get("/cases/{case_id}/control_points")
def get_control_points(case_id: str):
    case = cases.get(case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="unknown case_id")
    return {f"{side}:{branch}": pts for (side, branch), pts in case.control_points.items()}


@app.get("/cases/{case_id}/centerline")
def get_centerlines(case_id: str):
    case = cases.get(case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="unknown case_id")
    return {f"{side}:{branch}": path for (side, branch), path in case.centerline_points.items()}


RESLICE_SIZE_PX = 160
RESLICE_PX_SPACING_MM = 0.25
MAX_CONTOUR_RADIUS_MM = 6.0
# A perfect circle scores 1.0 on _contour_compactness; a real vessel
# cross-section stays well above this even when imperfect, while a leak
# into adjacent tissue or a jagged partial detection drops much lower.
AUTO_SEED_COMPACTNESS_MIN = 0.55


def _get_series_point(case_id: str, side: str, branch: str, index: int):
    case = cases.get(case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="unknown case_id")
    _check_side_branch(side, branch)
    series = case.segmentation_series.get((side, branch))
    if series is None:
        raise HTTPException(status_code=400, detail="call /prepare for this branch first")
    if not (0 <= index < len(series)):
        raise HTTPException(status_code=400, detail=f"index out of range [0, {len(series)})")
    return case, series


def _polygon_radius_mm(contour_px, px_spacing_mm):
    """Area-equivalent radius (sqrt(area/pi)) via the shoelace formula —
    used only to seed the next point's initial radius slider, not stored as
    ground truth."""
    if contour_px is None or len(contour_px) < 3:
        return None
    pts = np.asarray(contour_px)
    x, y = pts[:, 1], pts[:, 0]
    area_px = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    if area_px <= 0:
        return None
    return float(np.sqrt(area_px / np.pi) * px_spacing_mm)


def _contour_compactness(contour_px):
    """4*pi*area / perimeter^2 via the shoelace formula — 1.0 for a
    perfect circle, dropping well below that for anything elongated or
    irregular (a leak into adjacent tissue, a jagged partial detection).
    Used to decide whether a Level Set result is trustworthy enough to
    keep as-is during auto-seeding, or rough enough that a plain circle
    is the more honest "I don't actually know this point's shape" answer."""
    if contour_px is None or len(contour_px) < 3:
        return 0.0
    pts = np.asarray(contour_px, dtype=np.float64)
    y, x = pts[:, 0], pts[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    perimeter = float(np.sum(np.hypot(np.diff(x, append=x[0]), np.diff(y, append=y[0]))))
    if perimeter <= 0:
        return 0.0
    return float(4 * np.pi * area / (perimeter ** 2))


@app.post("/cases/{case_id}/segment/{side}/{branch}/prepare")
def prepare_series(case_id: str, side: str, branch: str):
    case = cases.get(case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="unknown case_id")
    _check_side_branch(side, branch)
    path = case.centerline_points.get((side, branch))
    if not path:
        raise HTTPException(status_code=400, detail="no path marked for this branch yet")

    series = path_series.build_series(path, case.spacing_mm, case.origin_mm)
    case.segmentation_series[(side, branch)] = series
    return {
        "side": side, "branch": branch, "n_points": len(series),
        "points_mm": [p["point_mm"] for p in series],
        "tangents_mm": [p["tangent_mm"] for p in series],
    }


@app.get("/cases/{case_id}/segment/{side}/{branch}/slice/{index}")
def get_segment_slice(case_id: str, side: str, branch: str, index: int,
                       wl: float | None = None, ww: float | None = None):
    case, series = _get_series_point(case_id, side, branch, index)
    point = series[index]
    slice_2d, u_vec, v_vec, grid_origin = reslice.sample_oblique_slice(
        case.array, case.spacing_mm, case.origin_mm, point["point_mm"], point["tangent_mm"],
        size_px=RESLICE_SIZE_PX, px_spacing_mm=RESLICE_PX_SPACING_MM,
    )
    if wl is None or ww is None:
        wl, ww = _default_window(case.array, case.modality)
    png = _window_png(slice_2d, wl, ww)
    headers = {
        "X-Grid-Origin-Mm": ",".join(str(v) for v in grid_origin),
        "X-Grid-U": ",".join(str(v) for v in u_vec),
        "X-Grid-V": ",".join(str(v) for v in v_vec),
        "X-Px-Spacing-Mm": str(RESLICE_PX_SPACING_MM),
    }
    return Response(content=png, media_type="image/png", headers=headers)


@app.post("/cases/{case_id}/segment/{side}/{branch}/point/{index}")
def segment_point(case_id: str, side: str, branch: str, index: int, req: SegmentPointRequest):
    if req.method not in ("threshold", "levelset", "circle", "auto"):
        raise HTTPException(status_code=400, detail="method must be 'threshold', 'levelset', 'circle', or 'auto'")
    case, series = _get_series_point(case_id, side, branch, index)
    point = series[index]
    slice_2d, u_vec, v_vec, grid_origin = reslice.sample_oblique_slice(
        case.array, case.spacing_mm, case.origin_mm, point["point_mm"], point["tangent_mm"],
        size_px=RESLICE_SIZE_PX, px_spacing_mm=RESLICE_PX_SPACING_MM,
    )
    center_px = (RESLICE_SIZE_PX / 2.0, RESLICE_SIZE_PX / 2.0)
    max_radius_px = MAX_CONTOUR_RADIUS_MM / RESLICE_PX_SPACING_MM
    effective_method = req.method

    try:
        if req.method == "threshold":
            contour_px = segment2d.threshold_contour(slice_2d, center_px, case.modality, max_radius_px)
        elif req.method == "circle":
            # Not a detected wall at all — a perfect circle at the
            # requested radius, a plain fallback shape.
            radius_px = max(req.radius_mm, 0.5) / RESLICE_PX_SPACING_MM
            contour_px = segment2d.circle_contour(center_px, radius_px)
        elif req.method == "auto":
            # Used to seed every point right after a Series is prepared:
            # really try a real detection first, and only fall back to a
            # plain circle when the result doesn't look like a
            # trustworthy vessel cross-section — not "always draw
            # circles", which would throw away real detections that are
            # already correct. CT has a real HU cliff to grow against
            # (Threshold); MR doesn't, so Level Set's gradient-following
            # is the better default there.
            radius_px = max(req.radius_mm, 1.0) / RESLICE_PX_SPACING_MM
            if case.modality == "CT":
                contour_px = segment2d.threshold_contour(slice_2d, center_px, case.modality, max_radius_px)
            else:
                contour_px = segment2d.levelset_contour(slice_2d, center_px, radius_px, max_radius_px)
            if contour_px is None or _contour_compactness(contour_px) < AUTO_SEED_COMPACTNESS_MIN:
                contour_px = segment2d.circle_contour(center_px, radius_px)
                effective_method = "circle"
            else:
                effective_method = "levelset"
        else:
            radius_px = max(req.radius_mm, 1.0) / RESLICE_PX_SPACING_MM
            contour_px = segment2d.levelset_contour(slice_2d, center_px, radius_px, max_radius_px)
    except Exception as e:
        log.exception("segmentation failed for %s/%s/%s point %d", case_id, side, branch, index)
        raise HTTPException(status_code=400, detail=f"segmentation failed: {e}")

    if contour_px is None:
        raise HTTPException(status_code=422, detail="no contour found at this point — try a different method/radius")

    contour_px_list = [[float(r), float(c)] for r, c in contour_px]
    contour_mm = [
        (np.asarray(grid_origin) + c * RESLICE_PX_SPACING_MM * np.asarray(u_vec)
         + r * RESLICE_PX_SPACING_MM * np.asarray(v_vec)).tolist()
        for r, c in contour_px
    ]
    fitted_radius_mm = _polygon_radius_mm(contour_px, RESLICE_PX_SPACING_MM)

    series[index] = {
        **point,
        "contour_px": contour_px_list,
        "contour_mm": contour_mm,
        "radius_mm": fitted_radius_mm,
        "method": effective_method,
    }
    return {
        "side": side, "branch": branch, "index": index,
        "contour_px": contour_px_list, "contour_mm": contour_mm,
        "fitted_radius_mm": fitted_radius_mm,
    }


@app.post("/cases/{case_id}/segment/{side}/{branch}/point/{index}/manual")
def segment_point_manual(case_id: str, side: str, branch: str, index: int, req: ManualContourRequest):
    """Stores a ring the user drew by hand (clicked vertices on the
    oblique slice) instead of an algorithm's detection — for points where
    Threshold/Level Set pick up the wrong structure and tracing the wall
    directly is faster than fighting the method/radius controls. Recomputes
    this point's own oblique-slice grid basis (same as segment_point) just
    to convert the given pixel points to world-mm consistently; the slice
    image itself isn't needed here."""
    if len(req.contour_px) < 3:
        raise HTTPException(status_code=400, detail="a ring needs at least 3 points")
    case, series = _get_series_point(case_id, side, branch, index)
    point = series[index]
    _, u_vec, v_vec, grid_origin = reslice.sample_oblique_slice(
        case.array, case.spacing_mm, case.origin_mm, point["point_mm"], point["tangent_mm"],
        size_px=RESLICE_SIZE_PX, px_spacing_mm=RESLICE_PX_SPACING_MM,
    )
    contour_px_list = [[float(r), float(c)] for r, c in req.contour_px]
    contour_mm = [
        (np.asarray(grid_origin) + c * RESLICE_PX_SPACING_MM * np.asarray(u_vec)
         + r * RESLICE_PX_SPACING_MM * np.asarray(v_vec)).tolist()
        for r, c in req.contour_px
    ]
    fitted_radius_mm = _polygon_radius_mm(req.contour_px, RESLICE_PX_SPACING_MM)

    series[index] = {
        **point,
        "contour_px": contour_px_list,
        "contour_mm": contour_mm,
        "radius_mm": fitted_radius_mm,
        "method": "manual",
    }
    return {
        "side": side, "branch": branch, "index": index,
        "contour_px": contour_px_list, "contour_mm": contour_mm,
        "fitted_radius_mm": fitted_radius_mm,
    }


@app.delete("/cases/{case_id}/segment/{side}/{branch}/point/{index}")
def delete_segment_point(case_id: str, side: str, branch: str, index: int):
    """Clears a point back to "not segmented" — for a result that's
    plain wrong (leaked into adjacent tissue, auto-seeded badly) where
    leaving a wrong ring in the wall preview is worse than a gap."""
    case, series = _get_series_point(case_id, side, branch, index)
    point = series[index]
    series[index] = {"point_mm": point["point_mm"], "tangent_mm": point["tangent_mm"]}
    return {"side": side, "branch": branch, "index": index}


@app.get("/cases/{case_id}/segment/{side}/su2_params")
def get_su2_params(case_id: str, side: str):
    """Returns this side's stored boundary-condition parameters, or the
    generic defaults if none have been set yet — so the Boundary
    Conditions form always has something sensible to prefill with.
    Registered *before* the generic GET /segment/{side}/{branch} route
    below — CONFIRMED (via a live round-trip test that came back with
    that route's own "branch must be one of [...]" error) that Starlette
    matches path routes in registration order, not by specificity, so a
    same-method route registered after a more generic one that also
    structurally matches ("su2_params" binds to that route's own
    {branch} just fine) never gets reached at all."""
    case = cases.get(case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="unknown case_id")
    if side not in SIDES:
        raise HTTPException(status_code=400, detail=f"side must be one of {sorted(SIDES)}")
    return case.su2_params.get(side, SU2_PARAM_DEFAULTS)


@app.post("/cases/{case_id}/segment/{side}/su2_params")
def set_su2_params(case_id: str, side: str, req: SU2ParamsRequest):
    case = cases.get(case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="unknown case_id")
    if side not in SIDES:
        raise HTTPException(status_code=400, detail=f"side must be one of {sorted(SIDES)}")
    case.su2_params[side] = req.model_dump()
    return case.su2_params[side]


@app.get("/cases/{case_id}/segment/{side}/{branch}")
def get_segmentation_series(case_id: str, side: str, branch: str):
    case = cases.get(case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="unknown case_id")
    _check_side_branch(side, branch)
    series = case.segmentation_series.get((side, branch), [])
    return {"side": side, "branch": branch, "points": series}


@app.post("/cases/{case_id}/segment/{side}/build_surface")
def build_surface_endpoint(case_id: str, side: str):
    case = cases.get(case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="unknown case_id")
    if side not in SIDES:
        raise HTTPException(status_code=400, detail=f"side must be one of {sorted(SIDES)}")

    series_by_branch = {
        branch: case.segmentation_series.get((side, branch), [])
        for branch in BRANCHES
    }
    if not series_by_branch.get("cca_ica"):
        raise HTTPException(status_code=400, detail="CCA→ICA path must be segmented before building a surface")

    try:
        poly, diag = build_surface.build_surface(case.array.shape, case.spacing_mm, case.origin_mm, series_by_branch)
    except Exception as e:
        log.exception("build_surface failed for %s/%s", case_id, side)
        raise HTTPException(status_code=400, detail=f"surface build failed: {e}")

    if poly is None:
        raise HTTPException(status_code=400, detail=diag.get("error", "surface build failed"))

    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    build_surface.write_stl(poly, tmp_path)
    case.built_surfaces[side] = tmp_path.read_bytes()
    tmp_path.unlink(missing_ok=True)

    return diag


@app.get("/cases/{case_id}/mesh/{side}")
def get_mesh(case_id: str, side: str):
    case = cases.get(case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="unknown case_id")
    stl_bytes = case.built_surfaces.get(side)
    if stl_bytes is None:
        raise HTTPException(status_code=404, detail="no surface built yet for this side — call build_surface first")
    return Response(content=stl_bytes, media_type="model/stl")


@app.post("/cases/{case_id}/segment/{side}/su2_mesh")
def start_su2_mesh(case_id: str, side: str):
    """Kicks off a CFD-ready volume mesh (.vtu) + boundary surface (.vtp,
    ModelFaceID-tagged) build in a background thread — a real TetGen run
    still takes a few seconds, so this returns immediately; poll
    .../su2_mesh/status."""
    case = cases.get(case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="unknown case_id")
    if side not in SIDES:
        raise HTTPException(status_code=400, detail=f"side must be one of {sorted(SIDES)}")

    existing = case.su2_mesh_jobs.get(side)
    if existing and existing.get("state") == "running":
        raise HTTPException(status_code=409, detail="a mesh job is already running for this side")

    series_by_branch = {
        branch: case.segmentation_series.get((side, branch), [])
        for branch in BRANCHES
    }
    if not series_by_branch.get("cca_ica"):
        raise HTTPException(status_code=400, detail="CCA→ICA path must be segmented before building a mesh")

    try:
        poly, diag = build_surface.build_surface(case.array.shape, case.spacing_mm, case.origin_mm, series_by_branch)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"surface build failed: {e}")
    if poly is None:
        raise HTTPException(status_code=400, detail=diag.get("error", "surface build failed"))

    case.su2_mesh_jobs[side] = {"state": "running", "stage": "tagging_faces", "stage_info": {}}
    # A rebuilt mesh invalidates any previous run/results for this side —
    # without this, an old finished (or restored-from-disk) su2_run/
    # su2_postprocess stays reachable via its own status/result endpoints
    # even though it describes a now-replaced geometry, and the client's
    # "View Results" happily shows it as if it were current (CONFIRMED: a
    # real session where rebuilding the mesh from a new path still showed
    # the *previous* mesh's own CFD results — wrong vessel, no error
    # anywhere to say so).
    case.su2_run_jobs.pop(side, None)
    case.su2_postprocess_jobs.pop(side, None)

    def on_stage(stage, info):
        # Mutating the existing dict's stage/stage_info in place (not
        # replacing case.su2_mesh_jobs[side]) so a status poll racing this
        # background thread can never see a "running" entry that's
        # momentarily missing "stage" — the dict is always complete.
        job = case.su2_mesh_jobs.get(side)
        if job is not None and job.get("state") == "running":
            job["stage"] = stage
            job["stage_info"] = info

    def run_job():
        try:
            vtu_bytes, vtp_bytes, face_map, mesh_diag = su2_export.build_su2_mesh(
                poly, series_by_branch, on_stage=on_stage
            )
            case.su2_mesh_jobs[side] = {
                "state": "done", "vtu": vtu_bytes, "vtp": vtp_bytes,
                "face_map": {str(k): v for k, v in face_map.items()}, "diag": mesh_diag,
            }
        except Exception as e:
            log.exception("su2 mesh job failed for %s/%s", case_id, side)
            case.su2_mesh_jobs[side] = {"state": "error", "error": str(e)}

    threading.Thread(target=run_job, daemon=True).start()
    return {"side": side, "state": "running"}


@app.get("/cases/{case_id}/segment/{side}/su2_mesh/status")
def su2_mesh_status(case_id: str, side: str):
    case = cases.get(case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="unknown case_id")
    job = case.su2_mesh_jobs.get(side)
    if job is None:
        return {"side": side, "state": "idle"}
    result = {"side": side, "state": job["state"]}
    if job["state"] == "running":
        result["stage"] = job.get("stage")
        result["stage_info"] = job.get("stage_info")
    elif job["state"] == "done":
        result["face_map"] = job.get("face_map")
        result["diag"] = job.get("diag")
    elif job["state"] == "error":
        result["error"] = job.get("error")
    return result


@app.get("/cases/{case_id}/segment/{side}/su2_mesh/vtu")
def get_su2_mesh_vtu(case_id: str, side: str):
    case = cases.get(case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="unknown case_id")
    job = case.su2_mesh_jobs.get(side)
    if not job or job.get("state") != "done":
        raise HTTPException(status_code=404, detail="no finished mesh job for this side")
    return Response(content=job["vtu"], media_type="application/octet-stream")


@app.get("/cases/{case_id}/segment/{side}/su2_mesh/vtp")
def get_su2_mesh_vtp(case_id: str, side: str):
    case = cases.get(case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="unknown case_id")
    job = case.su2_mesh_jobs.get(side)
    if not job or job.get("state") != "done":
        raise HTTPException(status_code=404, detail="no finished mesh job for this side")
    return Response(content=job["vtp"], media_type="application/octet-stream")


@app.get("/cases/{case_id}/segment/{side}/su2_mesh/preview_stl")
def get_su2_mesh_preview_stl(case_id: str, side: str):
    """Binary-STL conversion of the finished mesh's own boundary surface
    — lets the Swift client's existing STL preview viewer show a just-
    built mesh before it's saved, the same way it already does for
    Build Surface's output, without a second (XML .vtp) parser on that
    side. See su2_export.vtp_bytes_to_stl_bytes."""
    case = cases.get(case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="unknown case_id")
    job = case.su2_mesh_jobs.get(side)
    if not job or job.get("state") != "done":
        raise HTTPException(status_code=404, detail="no finished mesh job for this side")
    stl_bytes = su2_export.vtp_bytes_to_stl_bytes(job["vtp"])
    return Response(content=stl_bytes, media_type="model/stl")


@app.get("/cases/{case_id}/segment/{side}/su2_mesh/geometry")
def su2_mesh_geometry(case_id: str, side: str):
    """Cap diameters/centroids/outward-normals + the angle(s) between
    outlet caps, straight from the finished mesh's own boundary surface —
    no CFD run needed (unlike su2_postprocess's own geometry stats, which
    piggyback on a finished solve). Lets the mesh preview annotate itself
    with diameter/angle callouts right after Build SU2 Mesh, the step
    where a bad or unexpectedly-sized cap is actually useful to catch."""
    case = cases.get(case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="unknown case_id")
    job = case.su2_mesh_jobs.get(side)
    if not job or job.get("state") != "done":
        raise HTTPException(status_code=404, detail="no finished mesh job for this side")
    face_map = {int(k): v for k, v in job["face_map"].items()}
    return su2_postprocess.compute_geometry_stats(job["vtp"], face_map)


def _inlet_direction_xyz(case, side):
    """Unit flow-direction vector at the CCA inlet, in the surface's own
    (x,y,z) convention — the inlet path point's own tangent already
    points from the trunk toward the bifurcation (the direction blood
    actually enters the domain), so no separate geometric computation is
    needed beyond the same _xyz() axis flip build_surface.py/su2_export.py
    already use for every other world point."""
    series = case.segmentation_series.get((side, "cca_ica")) or case.segmentation_series.get((side, "cca_eca"))
    if not series:
        raise HTTPException(status_code=400, detail="CCA path must be segmented before running CFD")
    tangent = series[0].get("tangent_mm")
    if not tangent:
        raise HTTPException(status_code=400, detail="inlet point is missing a tangent — re-segment the path")
    return su2_export._xyz(tangent)


@app.post("/cases/{case_id}/segment/{side}/su2_run")
def start_su2_run(case_id: str, side: str):
    """Kicks off the actual SU2 pulsatile CFD solve (mpirun + pysu2) as a
    background subprocess — see server/su2_run.py. Returns immediately;
    poll .../su2_run/status for live progress/ETA."""
    case = cases.get(case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="unknown case_id")
    if side not in SIDES:
        raise HTTPException(status_code=400, detail=f"side must be one of {sorted(SIDES)}")

    existing = case.su2_run_jobs.get(side)
    if existing and existing.get("state") in ("running", "cancelling"):
        raise HTTPException(status_code=409, detail="a CFD run is already in progress for this side")

    mesh_job = case.su2_mesh_jobs.get(side)
    if not mesh_job or mesh_job.get("state") != "done":
        raise HTTPException(status_code=400, detail="build the SU2 mesh for this side before running CFD")

    su2_params = case.su2_params.get(side, SU2_PARAM_DEFAULTS)
    inlet_direction = _inlet_direction_xyz(case, side)
    face_map = {int(k): v for k, v in mesh_job["face_map"].items()}

    if case.project_dir:
        run_dir = Path(case.project_dir) / f"su2_run_{side}"
    else:
        run_dir = Path(tempfile.mkdtemp(prefix=f"carotid_su2_run_{case_id}_{side}_"))

    try:
        _mesh_path, cfg_path, params_path, time_iter = su2_run.prepare_run_files(
            run_dir, mesh_job["vtu"], mesh_job["vtp"], face_map, su2_params, inlet_direction,
            case_name=f"{case_id}_{side}",
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"failed to prepare CFD run: {e}")

    job = {"state": "running", "step": 0, "total_steps": time_iter, "output_dir": str(run_dir)}
    case.su2_run_jobs[side] = job
    # A fresh run invalidates whatever postprocessed results the *previous*
    # run left behind (possibly restored from a saved project, describing
    # an even older run) — otherwise they stay reachable and "View
    # Results" shows stale hemodynamics while this run is still going, or
    # if it's never post-processed again afterward.
    case.su2_postprocess_jobs.pop(side, None)
    su2_run.start_run(job, run_dir, cfg_path, params_path)
    return {"side": side, "state": "running", "total_steps": time_iter, "output_dir": str(run_dir),
            "mpi_processes": su2_run.mpi_processes()}


@app.get("/cases/{case_id}/segment/{side}/su2_run/status")
def su2_run_status(case_id: str, side: str):
    case = cases.get(case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="unknown case_id")
    job = case.su2_run_jobs.get(side)
    if job is None:
        return {"side": side, "state": "idle"}
    result = {k: v for k, v in job.items() if k != "process"}
    result["side"] = side
    return result


@app.post("/cases/{case_id}/segment/{side}/su2_run/cancel")
def cancel_su2_run(case_id: str, side: str):
    case = cases.get(case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="unknown case_id")
    job = case.su2_run_jobs.get(side)
    if job is None or job.get("state") != "running":
        raise HTTPException(status_code=400, detail="no running CFD job for this side")
    ok = su2_run.cancel_run(job)
    return {"side": side, "cancelling": ok}


@app.post("/cases/{case_id}/segment/{side}/su2_postprocess")
def start_su2_postprocess(case_id: str, side: str):
    """Builds the hemodynamic results (TAWSS/OSI/RRT/pressure on the
    wall + geometry stats) from a finished CFD run's own flow_*.vtu
    snapshots — see server/su2_postprocess.py. Works on a run that's
    still "running" or was "cancelled" too (whatever snapshots exist so
    far), not just "done" — a provisional look at a long run in progress
    is more useful than forcing a wait for the full 2 cycles."""
    case = cases.get(case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="unknown case_id")
    if side not in SIDES:
        raise HTTPException(status_code=400, detail=f"side must be one of {sorted(SIDES)}")

    existing = case.su2_postprocess_jobs.get(side)
    if existing and existing.get("state") == "running":
        raise HTTPException(status_code=409, detail="a postprocess job is already running for this side")

    run_job = case.su2_run_jobs.get(side)
    if not run_job or not run_job.get("output_dir"):
        raise HTTPException(status_code=400, detail="run the CFD solve for this side first")
    mesh_job = case.su2_mesh_jobs.get(side)
    if not mesh_job or mesh_job.get("state") != "done":
        raise HTTPException(status_code=400, detail="no finished mesh job for this side")

    run_dir = Path(run_job["output_dir"])
    vtp_bytes = mesh_job["vtp"]
    face_map = {int(k): v for k, v in mesh_job["face_map"].items()}
    su2_params = case.su2_params.get(side, SU2_PARAM_DEFAULTS)
    cycle_s = su2_params["cardiac_cycle_s"]

    job = {"state": "running", "frame": 0, "total_frames": 0}
    case.su2_postprocess_jobs[side] = job

    def on_frame(i, total):
        j = case.su2_postprocess_jobs.get(side)
        if j is not None and j.get("state") == "running":
            j["frame"] = i
            j["total_frames"] = total

    def run_job_thread():
        try:
            series = su2_postprocess.extract_wall_time_series(
                run_dir, vtp_bytes, su2_run.TIME_STEP_S, on_frame=on_frame
            )
            derived = su2_postprocess.compute_derived_fields(series, cycle_s)
            geometry = su2_postprocess.compute_geometry_stats(vtp_bytes, face_map)
            # wss_mag_frames_pa/pressure_frames_pa are (T, N) — at a real
            # patient mesh's point count and a real run's frame count,
            # serializing both into one JSON response is tens of MB
            # (CONFIRMED: ~104KB per single N-length array here, so a
            # 140-frame run is ~29MB for the two fields combined) — far
            # too much for one response a UI should wait on. Kept here as
            # numpy, served one frame at a time by the /frame endpoint
            # below instead, so a time-scrubbing UI only ever pays for
            # the one frame it's currently showing.
            case.su2_postprocess_jobs[side] = {
                "state": "done",
                "result": {
                    "points_mm": series["points_mm"].tolist(),
                    "triangles": series["triangles"].tolist(),
                    "times_s": derived["times_s"].tolist(),
                    "settled_cycle_start_s": derived["settled_cycle_start_s"],
                    "tawss_pa": derived["tawss_pa"].tolist(),
                    "osi": derived["osi"].tolist(),
                    "rrt_pa_inv": derived["rrt_pa_inv"].tolist(),
                    "mean_pressure_pa": derived["mean_pressure_pa"].tolist(),
                    "geometry": geometry,
                },
                "frames": {
                    "wss_mag_pa": derived["wss_mag_frames_pa"],
                    "pressure_pa": derived["pressure_frames_pa"],
                },
            }
        except Exception as e:
            log.exception("su2 postprocess job failed for %s/%s", case_id, side)
            case.su2_postprocess_jobs[side] = {"state": "error", "error": str(e)}

    threading.Thread(target=run_job_thread, daemon=True).start()
    return {"side": side, "state": "running"}


@app.get("/cases/{case_id}/segment/{side}/su2_postprocess/status")
def su2_postprocess_status(case_id: str, side: str):
    case = cases.get(case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="unknown case_id")
    job = case.su2_postprocess_jobs.get(side)
    if job is None:
        return {"side": side, "state": "idle"}
    result = {"side": side, "state": job["state"]}
    if job["state"] == "running":
        result["frame"] = job.get("frame")
        result["total_frames"] = job.get("total_frames")
    elif job["state"] == "error":
        result["error"] = job.get("error")
    return result


@app.get("/cases/{case_id}/segment/{side}/su2_postprocess/result")
def su2_postprocess_result(case_id: str, side: str):
    case = cases.get(case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="unknown case_id")
    job = case.su2_postprocess_jobs.get(side)
    if not job or job.get("state") != "done":
        raise HTTPException(status_code=404, detail="no finished postprocess result for this side")
    return job["result"]


@app.get("/cases/{case_id}/segment/{side}/su2_postprocess/frame/{index}")
def su2_postprocess_frame(case_id: str, side: str, index: int, field: str = "wss"):
    """One instantaneous frame's per-wall-point values — see
    start_su2_postprocess's own comment for why this is split out of
    /result instead of shipping every frame at once. field: "wss"
    (magnitude, Pa) or "pressure" (Pa)."""
    case = cases.get(case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="unknown case_id")
    job = case.su2_postprocess_jobs.get(side)
    if not job or job.get("state") != "done":
        raise HTTPException(status_code=404, detail="no finished postprocess result for this side")
    key = {"wss": "wss_mag_pa", "pressure": "pressure_pa"}.get(field)
    if key is None:
        raise HTTPException(status_code=400, detail="field must be 'wss' or 'pressure'")
    frames = job["frames"][key]
    if not (0 <= index < len(frames)):
        raise HTTPException(status_code=400, detail=f"index out of range 0..{len(frames) - 1}")
    times_s = job["result"]["times_s"]
    return {"index": index, "t": times_s[index], "values": frames[index].tolist()}


class SetProjectRequest(BaseModel):
    path: str


@app.post("/cases/{case_id}/project")
def set_project_dir(case_id: str, req: SetProjectRequest):
    """Every workflow step's own "Save" button (see the /save/* endpoints
    below) writes into this same folder, chosen once per case right after
    loading it — so losing the in-memory case (a backend restart) only
    loses progress that was never explicitly saved, not everything."""
    case = cases.get(case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="unknown case_id")
    folder = Path(req.path)
    folder.mkdir(parents=True, exist_ok=True)
    case.project_dir = str(folder)
    # Written once here (not inside each individual /save/* endpoint)
    # since it never changes across those saves — source_path/series_uid
    # are fixed for the lifetime of a case. This is what /projects/open
    # reads to know which DICOM series to reload.
    manifest = {"source_path": case.source_path, "series_uid": case.series_uid}
    (folder / "project.json").write_text(json.dumps(manifest, indent=2))
    return {"project_dir": case.project_dir}


@app.get("/cases/{case_id}/project")
def get_project_dir(case_id: str):
    case = cases.get(case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="unknown case_id")
    return {"project_dir": case.project_dir}


def _require_project_dir(case) -> Path:
    if not case.project_dir:
        raise HTTPException(status_code=400, detail="no project folder chosen yet for this case")
    folder = Path(case.project_dir)
    folder.mkdir(parents=True, exist_ok=True)
    return folder


@app.post("/cases/{case_id}/save/centerline/{side}")
def save_centerline_to_project(case_id: str, side: str):
    """Pathway step's own Save — both branches' click points (the sparse
    ones the user actually placed, so re-opening this later can be
    edited the same way) and the dense auto-stitched path."""
    case = cases.get(case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="unknown case_id")
    if side not in SIDES:
        raise HTTPException(status_code=400, detail=f"side must be one of {sorted(SIDES)}")
    folder = _require_project_dir(case)
    payload = {
        branch: {
            "control_points": case.control_points.get((side, branch), []),
            "centerline_points": case.centerline_points.get((side, branch), []),
        }
        for branch in BRANCHES
    }
    path = folder / f"{side}_centerline.json"
    path.write_text(json.dumps(payload, indent=2))
    return {"saved_to": str(path)}


@app.post("/cases/{case_id}/save/segmentation/{side}")
def save_segmentation_to_project(case_id: str, side: str):
    """Segmentation step's own Save — the full per-point Series (contours,
    radii, method) for both branches, the single most time-consuming
    thing to redo from scratch if it's ever lost."""
    case = cases.get(case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="unknown case_id")
    if side not in SIDES:
        raise HTTPException(status_code=400, detail=f"side must be one of {sorted(SIDES)}")
    folder = _require_project_dir(case)
    payload = {branch: case.segmentation_series.get((side, branch), []) for branch in BRANCHES}
    path = folder / f"{side}_segmentation.json"
    path.write_text(json.dumps(payload, indent=2))
    return {"saved_to": str(path)}


@app.post("/cases/{case_id}/save/surface/{side}")
def save_surface_to_project(case_id: str, side: str):
    """Surface step's own Save — the last Build Surface STL for this
    side, already cached in case.built_surfaces (no rebuild needed)."""
    case = cases.get(case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="unknown case_id")
    stl_bytes = case.built_surfaces.get(side)
    if stl_bytes is None:
        raise HTTPException(status_code=404, detail="no surface built yet for this side — call build_surface first")
    folder = _require_project_dir(case)
    path = folder / f"{side}_surface.stl"
    path.write_bytes(stl_bytes)
    return {"saved_to": str(path)}


@app.post("/cases/{case_id}/save/mesh/{side}")
def save_mesh_to_project(case_id: str, side: str):
    """Meshing step's own Save — the finished SU2 mesh's .vtu + .vtp."""
    case = cases.get(case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="unknown case_id")
    job = case.su2_mesh_jobs.get(side)
    if not job or job.get("state") != "done":
        raise HTTPException(status_code=404, detail="no finished mesh job for this side")
    folder = _require_project_dir(case)
    vtu_path = folder / f"{side}_mesh.vtu"
    vtp_path = folder / f"{side}_mesh_surface.vtp"
    vtu_path.write_bytes(job["vtu"])
    vtp_path.write_bytes(job["vtp"])
    return {"saved_to": [str(vtu_path), str(vtp_path)]}


@app.post("/cases/{case_id}/save/su2_params/{side}")
def save_su2_params_to_project(case_id: str, side: str):
    """Boundary Conditions step's own Save — whatever's currently stored
    for this side (the defaults, if the user never changed anything)."""
    case = cases.get(case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="unknown case_id")
    if side not in SIDES:
        raise HTTPException(status_code=400, detail=f"side must be one of {sorted(SIDES)}")
    folder = _require_project_dir(case)
    payload = case.su2_params.get(side, SU2_PARAM_DEFAULTS)
    path = folder / f"{side}_su2_params.json"
    path.write_text(json.dumps(payload, indent=2))
    return {"saved_to": str(path)}


@app.post("/cases/{case_id}/save/cfd_results/{side}")
def save_cfd_results_to_project(case_id: str, side: str):
    """CFD Computation step's own Save — the post-processed hemodynamic
    result (TAWSS/OSI/RRT/pressure + geometry, from
    server/su2_postprocess.py) plus the full per-frame WSS/pressure time
    series, so reopening this project later can drive CFDResultsView's
    instantaneous-frame scrubbing (see /projects/open) without re-running
    the actual CFD solve — unlike surface/mesh, which rebuild in seconds
    and are deliberately *not* persisted this way, a real pulsatile solve
    is minutes of mpirun+pysu2 compute, worth not repeating."""
    case = cases.get(case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="unknown case_id")
    if side not in SIDES:
        raise HTTPException(status_code=400, detail=f"side must be one of {sorted(SIDES)}")
    job = case.su2_postprocess_jobs.get(side)
    if not job or job.get("state") != "done":
        raise HTTPException(
            status_code=404,
            detail="no finished CFD results for this side — run CFD and View Results first",
        )
    folder = _require_project_dir(case)
    result_path = folder / f"{side}_cfd_results.json"
    result_path.write_text(json.dumps(job["result"], indent=2))
    frames_path = folder / f"{side}_cfd_frames.npz"
    np.savez_compressed(
        frames_path,
        wss_mag_pa=job["frames"]["wss_mag_pa"],
        pressure_pa=job["frames"]["pressure_pa"],
    )
    return {"saved_to": [str(result_path), str(frames_path)]}


class OpenProjectRequest(BaseModel):
    path: str


@app.post("/projects/open")
def open_project(req: OpenProjectRequest):
    """Reopens a previously-saved project folder: reloads the same DICOM
    series fresh (the array itself is never saved to the project folder
    — far too large, and trivially reproducible from the same source
    files) into a brand-new in-memory CaseState, then restores whatever
    per-side centerline/segmentation files that folder has (written by
    the /save/* endpoints). Deliberately does *not* restore a saved
    surface.stl or mesh .vtu/.vtp — both rebuild from the restored
    segmentation in a couple of seconds with today's pipeline, and doing
    that instead of deserializing a possibly-stale cached surface means
    "Build Surface"/"Build SU2 Mesh" after reopening always reflects
    whatever the current code actually produces, not whatever it
    produced when the project was last saved."""
    folder = Path(req.path)
    manifest_path = folder / "project.json"
    if not manifest_path.exists():
        raise HTTPException(
            status_code=400,
            detail="no project.json here — this doesn't look like a project folder saved by this app",
        )
    try:
        manifest = json.loads(manifest_path.read_text())
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"could not read project.json: {e}")

    source_path = manifest.get("source_path")
    if not source_path:
        raise HTTPException(status_code=400, detail="project.json is missing source_path")

    case, info = _load_dicom_case(source_path, manifest.get("series_uid"))
    case.project_dir = str(folder)

    restored = {}
    for side in SIDES:
        side_restored = {"centerline": [], "segmentation": []}

        centerline_path = folder / f"{side}_centerline.json"
        if centerline_path.exists():
            try:
                data = json.loads(centerline_path.read_text())
                for branch in BRANCHES:
                    branch_data = data.get(branch) or {}
                    if branch_data.get("control_points"):
                        case.control_points[(side, branch)] = branch_data["control_points"]
                    if branch_data.get("centerline_points"):
                        case.centerline_points[(side, branch)] = branch_data["centerline_points"]
                        side_restored["centerline"].append(branch)
            except Exception as e:
                log.warning("failed to restore %s centerline from %s: %s", side, centerline_path, e)

        segmentation_path = folder / f"{side}_segmentation.json"
        if segmentation_path.exists():
            try:
                data = json.loads(segmentation_path.read_text())
                for branch in BRANCHES:
                    series = data.get(branch) or []
                    if not series:
                        continue
                    # contour_mm/point_mm were saved as *absolute* world-mm
                    # at segmentation time (origin_mm + voxel*spacing with
                    # whatever origin that session's DICOM load resolved
                    # to). A later reload that resolves to a different
                    # origin_mm — e.g. an older save from before a
                    # since-fixed slice-ordering bug, or any other reason
                    # the same series loads with a shifted origin — makes
                    # that saved geometry silently land tens to hundreds
                    # of mm away from the freshly-restored path, even
                    # though the raw voxel-index path itself still looks
                    # fine (CONFIRMED on a real project: the 3D preview's
                    # wall contours rendered ~470mm from the path with no
                    # error anywhere). Cross-check the first contour's own
                    # centroid against path_series's fresh point_mm for
                    # that same index, computed from the centerline just
                    # restored above (same origin_mm this reload actually
                    # has) — if they don't land within a generous margin
                    # of each other, this branch's saved segmentation is
                    # stale and must be re-run, not silently trusted.
                    centerline = case.centerline_points.get((side, branch))
                    is_stale = False
                    if centerline and len(centerline) > 1:
                        fresh_series = path_series.build_series(centerline, case.spacing_mm, case.origin_mm)
                        first_idx, first_point = next(
                            ((i, p) for i, p in enumerate(series) if p.get("contour_mm")), (None, None)
                        )
                        if first_point is not None and first_idx is not None and first_idx < len(fresh_series):
                            fresh_pt = np.asarray(fresh_series[first_idx]["point_mm"])
                            saved_centroid = np.asarray(first_point["contour_mm"]).mean(axis=0)
                            if np.linalg.norm(fresh_pt - saved_centroid) > 20.0:
                                is_stale = True
                    if is_stale:
                        log.warning(
                            "discarding stale %s/%s segmentation on restore — saved contour_mm "
                            "doesn't match the freshly-reloaded DICOM's own origin_mm; re-run "
                            "Segment Along Path for this branch", side, branch,
                        )
                        continue
                    case.segmentation_series[(side, branch)] = series
                    side_restored["segmentation"].append(branch)
            except Exception as e:
                log.warning("failed to restore %s segmentation from %s: %s", side, segmentation_path, e)

        su2_params_path = folder / f"{side}_su2_params.json"
        if su2_params_path.exists():
            try:
                case.su2_params[side] = json.loads(su2_params_path.read_text())
                side_restored["su2_params"] = True
            except Exception as e:
                log.warning("failed to restore %s su2_params from %s: %s", side, su2_params_path, e)

        # Unlike surface/mesh (cheap to rebuild, so deliberately not
        # persisted), a saved CFD result *is* restored directly into
        # su2_postprocess_jobs — re-deriving it means re-running the
        # actual pulsatile solve, minutes of real compute, not something
        # reopening a project should force.
        cfd_result_path = folder / f"{side}_cfd_results.json"
        cfd_frames_path = folder / f"{side}_cfd_frames.npz"
        if cfd_result_path.exists() and cfd_frames_path.exists():
            try:
                result = json.loads(cfd_result_path.read_text())
                frames_npz = np.load(cfd_frames_path)
                case.su2_postprocess_jobs[side] = {
                    "state": "done",
                    "result": result,
                    "frames": {
                        "wss_mag_pa": frames_npz["wss_mag_pa"],
                        "pressure_pa": frames_npz["pressure_pa"],
                    },
                }
                side_restored["cfd_results"] = True
            except Exception as e:
                log.warning("failed to restore %s CFD results from %s: %s", side, cfd_result_path, e)

        side_restored["surface_file_exists"] = (folder / f"{side}_surface.stl").exists()
        side_restored["mesh_files_exist"] = (folder / f"{side}_mesh.vtu").exists()
        restored[side] = side_restored

    info["project_dir"] = case.project_dir
    info["restored"] = restored
    return info
