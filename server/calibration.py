"""Compare operator contours with the automatic methods on the same sections.

A project segmented by hand is the reference. For every section whose lumen
was drawn manually, each automatic lumen method is run on the identical MPR
section and scored against it; for every manually drawn outer wall, the
automatic wall detector is scored the same way. The report says how far each
method is from the operator and how often it fails, which is what parameter
calibration needs. It does not change the project.
"""
from __future__ import annotations

import numpy as np
from skimage.draw import polygon

from . import segment2d
from .lumen_profiles import fwhm_contour
from .wall_analysis import analyze_wall

SIZE_PX = 160
PX_MM = 0.25
MAX_RADIUS_MM = 6.0
WALL_SOFT_TISSUE_MAX_HU = 100.0


def mask(contour_px, shape=(SIZE_PX, SIZE_PX)):
    contour = np.asarray(contour_px, dtype=float)
    out = np.zeros(shape, dtype=bool)
    if len(contour) >= 3:
        out[polygon(contour[:, 0], contour[:, 1], shape)] = True
    return out


def _boundary_points(contour_px, step_px=0.5):
    """Densely resampled closed contour, so distances do not depend on vertex count."""
    c = np.asarray(contour_px, dtype=float)
    closed = np.vstack([c, c[:1]])
    pts = []
    for a, b in zip(closed[:-1], closed[1:]):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / step_px)))
        pts.append(a + (b - a) * np.linspace(0, 1, n, endpoint=False)[:, None])
    return np.vstack(pts)


def compare(reference_px, candidate_px, px_mm=PX_MM):
    """Overlap and boundary agreement of two contours in one section (mm)."""
    ref, cand = mask(reference_px), mask(candidate_px)
    inter, total = np.logical_and(ref, cand).sum(), ref.sum() + cand.sum()
    a, b = _boundary_points(reference_px), _boundary_points(candidate_px)
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    a_to_b, b_to_a = d.min(axis=1), d.min(axis=0)
    ref_area = ref.sum() * px_mm ** 2
    return {
        "dice": float(2 * inter / total) if total else 0.0,
        "area_reference_mm2": float(ref_area),
        "area_candidate_mm2": float(cand.sum() * px_mm ** 2),
        "area_error_pct": float(100 * (cand.sum() - ref.sum()) / ref.sum()) if ref.sum() else None,
        "mean_boundary_mm": float((a_to_b.mean() + b_to_a.mean()) / 2 * px_mm),
        "hausdorff_mm": float(max(a_to_b.max(), b_to_a.max()) * px_mm),
    }


def lumen_methods(modality):
    """Automatic lumen methods as name -> f(image) -> contour or None."""
    center = ((SIZE_PX - 1) / 2,) * 2
    max_px = MAX_RADIUS_MM / PX_MM
    methods = {
        "threshold": lambda image: segment2d.threshold_contour(image, center, modality, max_px),
        "levelset": lambda image: segment2d.levelset_contour(image, center, 3.0 / PX_MM, max_px),
    }
    if modality == "CT":
        methods["fwhm"] = lambda image: fwhm_contour(image, center, PX_MM, MAX_RADIUS_MM)[0]
    return methods


def _run(method, image):
    try:
        contour = method(image)
    except Exception as exc:  # a failed method is a result, not a crash
        return None, str(exc)
    if contour is None or len(contour) < 3:
        return None, "no contour"
    return contour, None


def calibrate_section(image, point, modality, calcium_hu=600.0):
    """Scores for one section; only manual annotations are used as reference."""
    rows = []
    if point.get("method") == "manual" and point.get("contour_px"):
        for name, method in lumen_methods(modality).items():
            contour, failure = _run(method, image)
            row = {"target": "lumen", "method": name, "failure": failure}
            if contour is not None:
                row.update(compare(point["contour_px"], contour))
            rows.append(row)
    wall = point.get("wall_analysis") or {}
    if wall.get("status") == "manual" and point.get("contour_px") and modality == "CT":
        row = {"target": "outer_wall", "method": "wall_detector", "failure": None}
        try:
            # Same defaults as the editor's wall request (server.app.WallAnalysisRequest).
            auto = analyze_wall(image, point["contour_px"], PX_MM, calcium_hu,
                                soft_tissue_max_hu=WALL_SOFT_TISSUE_MAX_HU)
            row.update(compare(wall["wall_contour_px"], auto["wall_contour_px"]))
        except ValueError as exc:
            row["failure"] = str(exc)
        rows.append(row)
    return rows


def summarize(rows):
    """Per target and method: n, failure rate and robust error statistics."""
    out = {}
    for row in rows:
        key = f"{row['target']}:{row['method']}"
        out.setdefault(key, []).append(row)
    summary = {}
    for key, group in out.items():
        ok = [r for r in group if r.get("failure") is None]
        def med(field):
            values = [r[field] for r in ok if r.get(field) is not None]
            return float(np.median(values)) if values else None
        def p90(field):
            values = [abs(r[field]) for r in ok if r.get(field) is not None]
            return float(np.percentile(values, 90)) if values else None
        summary[key] = {
            "sections": len(group),
            "failures": len(group) - len(ok),
            "median_dice": med("dice"),
            "median_area_error_pct": med("area_error_pct"),
            "p90_abs_area_error_pct": p90("area_error_pct"),
            "median_mean_boundary_mm": med("mean_boundary_mm"),
            "p90_hausdorff_mm": p90("hausdorff_mm"),
        }
    return summary
