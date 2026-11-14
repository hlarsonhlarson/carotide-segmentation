"""Sampled section profiles and explicitly coverage-limited volume estimates."""
import numpy as np

from .lumen_measurements import contour_metrics


def analyze_series(series, start=0, end=None, max_step_mm=4.):
    end = len(series) - 1 if end is None else end
    if not 0 <= start <= end < len(series):
        raise ValueError("Select a nonempty range within the section series")
    if not np.isfinite(max_step_mm) or max_step_mm <= 0:
        raise ValueError("Maximum integration step must be positive")
    centers = np.asarray([p["point_mm"] for p in series], dtype=float)
    if centers.shape != (len(series), 3) or not np.isfinite(centers).all():
        raise ValueError("Invalid section positions")
    steps = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    arc = np.r_[0., np.cumsum(steps)]
    sections = []
    for index in range(start, end + 1):
        point = series[index]
        section = {"index": index, "distance_mm": float(arc[index]), "lumen": None, "wall": None}
        try:
            if point.get("method") == "circle":
                raise ValueError("Estimated circle")
            section["lumen"] = contour_metrics(point.get("contour_mm", []))
        except ValueError as exc:
            section["reason"] = str(exc)
        wall = point.get("wall_analysis")
        if section["lumen"] is not None and wall:
            values = [wall.get("annulus_area_mm2"), wall.get("high_density_area_mm2")]
            try:
                valid = np.isfinite(values).all() and 0 <= values[1] <= values[0]
            except TypeError:
                valid = False
            if valid:
                section["wall"] = {
                    "annulus_area_mm2": float(values[0]), "high_density_area_mm2": float(values[1]),
                    "status": wall.get("status", "proposal"),
                    "parameters": wall.get("parameters", {}),
                }
        sections.append(section)

    fields = {
        "lumen": ("lumen", "area_mm2"),
        "annulus": ("wall", "annulus_area_mm2"),
        "high_density": ("wall", "high_density_area_mm2"),
    }
    totals = {key: {"volume_mm3": None, "covered_length_mm": 0., "intervals": 0,
                    "excluded_intervals": []} for key in fields}
    density_parameters = {
        (s["wall"]["parameters"].get("calcium_hu"),
         s["wall"]["parameters"].get("min_component_mm2"))
        for s in sections if s["wall"] is not None
    }
    consistent_density = len(density_parameters) == 1 and all(
        value is not None and np.isfinite(value)
        for pair in density_parameters for value in pair)
    for left, right in zip(sections, sections[1:]):
        i = left["index"]
        step = steps[i]
        geometry_reason = None
        if step <= 1e-6:
            geometry_reason = "duplicate_position"
        elif step > max_step_mm:
            geometry_reason = "sampling_gap"
        else:
            tangents = [np.asarray(series[j].get("tangent_mm", []), dtype=float) for j in (i, i + 1)]
            if any(t.shape != (3,) or not np.isfinite(t).all() for t in tangents):
                geometry_reason = "invalid_frame"
            else:
                tangents = np.asarray(tangents)
                lengths = np.linalg.norm(tangents, axis=1)
                if np.any(lengths < 1e-8):
                    geometry_reason = "invalid_frame"
                elif np.dot(*tangents) / np.prod(lengths) < np.cos(np.deg2rad(30)):
                    geometry_reason = "sharp_turn"
        for name, (group, field) in fields.items():
            reason = geometry_reason
            if reason is None and (left[group] is None or right[group] is None):
                reason = "missing_measurement"
            if reason is None and name == "high_density":
                if not consistent_density:
                    reason = "different_density_parameters"
            total = totals[name]
            if reason:
                total["excluded_intervals"].append({"from_index": i, "to_index": i + 1, "reason": reason})
                continue
            volume = .5 * (left[group][field] + right[group][field]) * step
            total["volume_mm3"] = float((total["volume_mm3"] or 0.) + volume)
            total["covered_length_mm"] += float(step)
            total["intervals"] += 1
    valid = [s for s in sections if s["lumen"] is not None]
    return {
        "sections": sections, "start_index": start, "end_index": end,
        "selected_length_mm": float(arc[end] - arc[start]),
        "minimum_area_index": min(valid, key=lambda s: s["lumen"]["area_mm2"])["index"] if valid else None,
        "minimum_diameter_index": min(valid, key=lambda s: s["lumen"]["min_feret_mm"])["index"] if valid else None,
        "integrals": totals, "max_step_mm": max_step_mm,
        "density_parameters": dict(zip(("calcium_hu", "min_component_mm2"), next(iter(density_parameters))))
        if consistent_density else None,
        "method": "trapezoidal_section_integral", "validated": False,
        "limitations": "Sampled estimates, not voxel volumes or plaque classification. "
                       "No interpolation across missing sections, gaps or sharp turns. "
                       "Overlapping curved sections and bifurcations require review; do not sum overlapping branches.",
    }
