"""Signed boundary-flow diagnostics for a rigid incompressible domain."""
import csv
from pathlib import Path

import numpy as np


def flow_metrics(inlet_outward, ica_outward, eca_outward):
    flows = np.asarray([inlet_outward, ica_outward, eca_outward], dtype=float)
    if not np.isfinite(flows).all():
        raise ValueError("Nonfinite boundary flux")
    entering = -float(inlet_outward)
    net = float(flows.sum())
    return {
        "q_inlet_mlps": entering * 1e6,
        "q_ica_mlps": float(ica_outward) * 1e6,
        "q_eca_mlps": float(eca_outward) * 1e6,
        "mass_imbalance_mlps": net * 1e6,
        "mass_imbalance_relative": abs(net) / abs(entering) if abs(entering) > 1e-12 else None,
        "inlet_reversed": entering < -1e-12,
        "ica_reversed": bool(ica_outward < -1e-12),
        "eca_reversed": bool(eca_outward < -1e-12),
    }


def summarize_balance(run_dir, cycle_s, end_time_s=None):
    path = Path(run_dir) / "boundary_flows.csv"
    if not path.is_file():
        return {"status": "unavailable", "reason": "No recorded boundary fluxes"}
    with path.open() as handle:
        lines = handle.readlines()
    # A live solver may currently be writing the final line.
    if lines and not lines[-1].endswith("\n"):
        lines.pop()
    rows = list(csv.DictReader(lines))
    if end_time_s is not None:
        rows = [row for row in rows if float(row["time_s"]) <= end_time_s + 1e-10]
    if len(rows) < 2:
        return {"status": "insufficient_samples"}
    times = np.array([float(row["time_s"]) for row in rows])
    flows = np.array([[float(row[key]) for key in ("q_inlet_mlps", "q_ica_mlps", "q_eca_mlps")]
                      for row in rows])
    if (not np.isfinite(cycle_s) or cycle_s <= 0 or not np.isfinite(times).all()
            or np.any(np.diff(times) <= 0) or not np.isfinite(flows).all()):
        raise ValueError("Invalid boundary flux time series")
    start = max(times[0], times[-1] - cycle_s)
    selected = times > start
    t = np.r_[start, times[selected]]
    q = np.vstack(([np.interp(start, times, flows[:, i]) for i in range(3)], flows[selected]))
    duration = t[-1] - t[0]
    net = q[:, 1] + q[:, 2] - q[:, 0]
    mean_inlet = float(np.trapezoid(abs(q[:, 0]), t) / duration)
    mean_absolute_error = float(np.trapezoid(abs(net), t) / duration)
    relative = mean_absolute_error / mean_inlet if mean_inlet > 1e-6 else None
    clamped = [row for row in rows if float(row["time_s"]) >= start
               and (int(row["ica_pressure_clamped"]) or int(row["eca_pressure_clamped"]))]
    return {
        "status": "measured", "validated": False,
        "method": "nodal_boundary_velocity_quadrature",
        "window_start_s": float(start), "window_end_s": float(t[-1]),
        "full_cycle": bool(times[-1] - times[0] >= cycle_s - 1e-10),
        "mean_absolute_imbalance_mlps": mean_absolute_error,
        "peak_absolute_imbalance_mlps": float(abs(net).max()),
        "relative_mean_absolute_imbalance": relative,
        "inlet_volume_ml": float(np.trapezoid(q[:, 0], t)),
        "outlet_volume_ml": float(np.trapezoid(q[:, 1] + q[:, 2], t)),
        "reverse_samples": {name: int(np.sum(q[:, i] < -1e-6))
                            for i, name in enumerate(("inlet", "ica", "eca"))},
        "pressure_clamped_steps": len(clamped),
        "sign_convention": "inlet positive into domain; outlets positive out of domain",
        "limitations": "Boundary nodal velocity integral, not the solver's conservative numerical flux residual.",
    }
