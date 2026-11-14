"""Background automatic carotid seed generation.

This module deliberately keeps model execution out of request handlers.  The
result is a draft: it creates editable CCA→ICA / CCA→ECA paths and records the
pipeline QA, but never marks a study clinically validated.
"""
from __future__ import annotations

import os
import tempfile
import threading
import traceback
import uuid
import io
import json
import urllib.request
import urllib.error
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize

from carotid_seg.treeanalysis import _build_graph
from . import path_series, reslice, segment2d
from .lumen_profiles import fwhm_contour


LABELS = {
    "right": {"cca": 1, "ica": 3, "eca": 5},
    "left": {"cca": 2, "ica": 4, "eca": 6},
}
RESLICE_SIZE_PX = 160
RESLICE_PX_SPACING_MM = 0.25
MAX_CONTOUR_RADIUS_MM = 6.0
AUTO_CONTOUR_COMPACTNESS_MIN = 0.55


def _deepest_point(points: np.ndarray, other_points: np.ndarray, spacing_mm) -> tuple[int, int, int]:
    """Return the point furthest from the other branch, in physical space."""
    distances = cKDTree(other_points * spacing_mm).query(points * spacing_mm)[0]
    return tuple(int(v) for v in points[int(np.argmax(distances))])


def _path_between_labels(labels: np.ndarray, cca_code: int, target_code: int, spacing_mm) -> list[list[int]]:
    """Extract one editable centerline from the CCA to ICA/ECA labels.

    This is intentionally only a path proposal.  The user can edit its two
    endpoint controls or add intermediate controls in the existing editor.
    """
    mask = (labels == cca_code) | (labels == target_code)
    cca = np.argwhere(labels == cca_code)
    target = np.argwhere(labels == target_code)
    if not cca.size or not target.size:
        return []
    graph = _build_graph(skeletonize(mask), spacing_mm)
    if not graph.number_of_nodes():
        return []
    nodes = np.asarray(list(graph.nodes), dtype=float)
    tree = cKDTree(nodes * spacing_mm)
    start = tuple(nodes[tree.query(np.asarray(_deepest_point(cca, target, spacing_mm)) * spacing_mm)[1]].astype(int))
    end = tuple(nodes[tree.query(np.asarray(_deepest_point(target, cca, spacing_mm)) * spacing_mm)[1]].astype(int))
    try:
        import networkx as nx

        path = nx.shortest_path(graph, start, end, weight="weight")
    except Exception:
        return []
    return [[int(v) for v in point] for point in path]


def paths_from_labels(labels: np.ndarray, spacing_mm) -> dict[tuple[str, str], list[list[int]]]:
    paths = {}
    for side, codes in LABELS.items():
        for branch, target in (("cca_ica", "ica"), ("cca_eca", "eca")):
            path = _path_between_labels(labels, codes["cca"], codes[target], spacing_mm)
            if path:
                paths[side, branch] = path
    return paths


def _contour_metrics(contour_px) -> tuple[float, float]:
    """Return compactness and area-equivalent radius, both in physical units."""
    if contour_px is None or len(contour_px) < 3:
        return 0.0, 0.0
    points = np.asarray(contour_px, dtype=float)
    row, col = points[:, 0], points[:, 1]
    area_px = 0.5 * abs(np.dot(col, np.roll(row, 1)) - np.dot(row, np.roll(col, 1)))
    perimeter_px = float(np.sum(np.hypot(np.diff(col, append=col[0]), np.diff(row, append=row[0]))))
    compactness = 4 * np.pi * area_px / (perimeter_px ** 2) if perimeter_px > 0 else 0.0
    radius_mm = float(np.sqrt(area_px / np.pi) * RESLICE_PX_SPACING_MM) if area_px > 0 else 0.0
    return compactness, radius_mm


def draft_series(case, paths: dict[tuple[str, str], list[list[int]]], job) -> tuple[dict, dict]:
    """Create only defensible CT contours along extracted centerlines.

    A failed or implausible section is retained in the Series *without* a
    contour.  That makes the existing editor show an explicit gap and keeps
    surface/CFD generation blocked until an operator repairs it.
    """
    series_by_key, qa = {}, {}
    total = sum(bool(path) for path in paths.values())
    completed = 0
    for key, path in paths.items():
        series = path_series.build_series(path, case.spacing_mm, case.origin_mm)
        accepted, rejected = 0, []
        for index, point in enumerate(series):
            slice_2d, u_vec, v_vec, grid_origin = reslice.sample_oblique_slice(
                case.array, case.spacing_mm, case.origin_mm, point["point_mm"], point["tangent_mm"],
                size_px=RESLICE_SIZE_PX, px_spacing_mm=RESLICE_PX_SPACING_MM,
                background_value=case.array_min,
                frame=(point["u_vec"], point["v_vec"]) if "u_vec" in point else None,
            )
            center = ((RESLICE_SIZE_PX - 1) / 2.0,) * 2
            if case.modality == "CT":
                contour, profile_qa = fwhm_contour(
                    slice_2d, center, RESLICE_PX_SPACING_MM, MAX_CONTOUR_RADIUS_MM)
            else:
                contour = segment2d.threshold_contour(
                    slice_2d, center, case.modality, MAX_CONTOUR_RADIUS_MM / RESLICE_PX_SPACING_MM)
                profile_qa = {}
            compactness, radius_mm = _contour_metrics(contour)
            if contour is None or compactness < AUTO_CONTOUR_COMPACTNESS_MIN:
                rejected.append({"index": index, "reason": profile_qa.get("reason", "no_contour") if contour is None else "low_compactness",
                                 "compactness": round(compactness, 3)})
                continue
            contour_px = [[float(row), float(col)] for row, col in contour]
            contour_mm = [
                (np.asarray(grid_origin) + col * RESLICE_PX_SPACING_MM * np.asarray(u_vec)
                 + row * RESLICE_PX_SPACING_MM * np.asarray(v_vec)).tolist()
                for row, col in contour
            ]
            series[index] = {**point, "contour_px": contour_px, "contour_mm": contour_mm,
                             "radius_mm": radius_mm,
                             "method": "autoseed_fwhm" if case.modality == "CT" else "autoseed_threshold",
                             "profile_qa": profile_qa}
            accepted += 1
        series_by_key[key] = series
        qa[f"{key[0]}:{key[1]}"] = {
            "total_sections": len(series), "accepted_sections": accepted,
            "rejected_sections": rejected,
            "review_required": bool(rejected),
        }
        completed += 1
        job.update(stage="segmenting_contours", progress=85 + int(10 * completed / max(total, 1)))
    return series_by_key, qa


def _run(case, job):
    """Run TotalSegmentator + rule-based refinement, updating ``job`` in place."""
    try:
        job.update(state="running", stage="preparing", progress=5)
        with tempfile.TemporaryDirectory(prefix="carotid-autoseed-") as temp:
            temp_path = Path(temp)
            source = temp_path / "input.nii.gz"
            # The local worker owns TotalSegmentator/PyTorch.  Keep the Docker
            # backend small and send it a temporary NIfTI over localhost.
            import SimpleITK as sitk
            image = sitk.GetImageFromArray(case.array.astype(np.float32, copy=False))
            image.SetSpacing(tuple(float(v) for v in case.spacing_mm[::-1]))
            image.SetOrigin(tuple(float(v) for v in case.origin_mm[::-1]))
            image.SetDirection(tuple(float(v) for v in case.direction_lps.ravel()))
            sitk.WriteImage(image, str(source))
            job.update(stage="segmenting", progress=15)
            worker_url = os.environ.get("CAROTID_AUTOSEED_WORKER_URL", "http://host.docker.internal:8011/segment")
            request = urllib.request.Request(
                worker_url,
                data=source.read_bytes(),
                headers={"Content-Type": "application/gzip"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=60 * 60) as response:
                    result = response.read()
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"Segmentation worker returned HTTP {exc.code}: {detail}") from exc
            except Exception as exc:
                raise RuntimeError(
                    "Local automatic-segmentation worker is unavailable. Start it with "
                    "`.venv/bin/python scripts/local_autoseed_worker.py`."
                ) from exc
            with np.load(io.BytesIO(result), allow_pickle=False) as result_data:
                labels = result_data["labels"].astype(np.uint8, copy=False)
                qa = json.loads(str(result_data["qa"].item()))
            job.update(stage="extracting_paths", progress=85)
            if labels.shape != case.array.shape:
                raise RuntimeError(f"automatic label geometry {labels.shape} does not match case {case.array.shape}")
            paths = paths_from_labels(labels, case.spacing_mm)
            job.update(stage="segmenting_contours", progress=85)
            draft, contour_qa = draft_series(case, paths, job)

        # Keep state mutations serialized with normal edits.  Do not overwrite
        # work performed while the model was running: the revision captured at
        # launch is the optimistic-concurrency guard.
        with case.lock:
            if job.get("revision") != case.revisions:
                job.update(state="cancelled", stage="superseded", progress=100,
                           error="Study changed while automatic segmentation was running; draft was discarded.")
                return
            for side in ("left", "right"):
                for branch in ("cca_ica", "cca_eca"):
                    key = (side, branch)
                    path = paths.get(key, [])
                    if path:
                        case.centerline_points[key] = path
                        case.control_points[key] = [path[0], path[-1]]
                        case.segmentation_series[key] = draft.get(key, [])
                    else:
                        case.centerline_points.pop(key, None)
                        case.control_points.pop(key, None)
                        case.segmentation_series.pop(key, None)
            # The resulting paths invalidate all geometry derived from old
            # paths; increment once per side, rather than once per branch.
            from . import workflow
            for side in ("left", "right"):
                workflow.invalidate(case, side)
            job.update(
                state="done",
                stage="complete",
                progress=100,
                qa=qa,
                contour_qa=contour_qa,
                paths={f"{side}:{branch}": len(path) for (side, branch), path in paths.items()},
                review_required=True,
                review_accepted=False,
                message="Automatic draft created. Review paths and contours before surface or CFD generation.",
            )
            case.autoseed_review = {"job_id": job["id"], "accepted": False}
    except Exception as exc:
        job.update(state="error", stage="error", progress=100, error=str(exc), traceback=traceback.format_exc())


def start(case) -> dict:
    """Start a single automatic-segmentation job for a case."""
    existing = case.autoseed_job
    if existing and existing.get("state") in {"queued", "running"}:
        return existing
    job = {
        "id": uuid.uuid4().hex,
        "state": "queued",
        "stage": "queued",
        "progress": 0,
        "revision": {side: dict(values) for side, values in case.revisions.items()},
        "review_required": True,
        "review_accepted": False,
    }
    case.autoseed_job = job
    thread = threading.Thread(target=_run, args=(case, job), daemon=True, name=f"autoseed-{case.case_id}")
    thread.start()
    return job
