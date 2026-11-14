"""Time-window integration support and sampled cycle-to-cycle diagnostics."""
import numpy as np


def validate_series(times, wss, pressure, cycle_s):
    if not np.isfinite(cycle_s) or cycle_s <= 0:
        raise ValueError("Cardiac cycle must be positive")
    if times.ndim != 1 or not len(times) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("CFD times must be finite and strictly increasing")
    if (wss.ndim != 3 or wss.shape[0] != len(times) or wss.shape[2] != 3
            or wss.shape[1] == 0 or pressure.shape != wss.shape[:2]
            or not np.isfinite(wss).all() or not np.isfinite(pressure).all()):
        raise ValueError("Invalid CFD WSS or pressure samples")


def interpolate(times, values, time):
    """Linear interpolation within recorded time support; never extrapolate."""
    if time < times[0] - 1e-10 or time > times[-1] + 1e-10:
        raise ValueError("Requested time is outside recorded CFD frames")
    index = int(np.searchsorted(times, time))
    if index == 0:
        return values[0]
    if index == len(times):
        return values[-1]
    fraction = (time - times[index - 1]) / (times[index] - times[index - 1])
    return values[index - 1] * (1 - fraction) + values[index] * fraction


def last_window(times, values, cycle_s):
    start = max(float(times[0]), float(times[-1] - cycle_s))
    selected = times > start
    window_times = np.r_[start, times[selected]]
    window_values = np.concatenate((interpolate(times, values, start)[None], values[selected]), axis=0)
    return window_times, window_values


def _max_gap_covering(times, start):
    # Include the bracketing interval used to interpolate the window start.
    # Clipping that interval at start would conceal sparse source sampling.
    left = max(0, int(np.searchsorted(times, start, side="left")) - 1)
    if left + 1 < len(times) and times[left + 1] == start:
        left += 1
    gaps = np.diff(times[left:])
    return float(gaps.max()) if len(gaps) else None


def cycle_quality(times, wss, pressure, cycle_s, relative_tolerance=.05):
    """Global RMS agreement of the two last cycles, not solver validation.

    Phase samples compare full WSS vectors, so directional reversals matter.
    Absolute pressure differences are reported alongside relative errors.
    """
    duration = float(times[-1] - times[0])
    start = max(float(times[0]), float(times[-1] - cycle_s))
    max_gap = _max_gap_covering(times, start)
    result = {
        "window_start_s": start, "window_end_s": float(times[-1]),
        "window_duration_s": float(times[-1] - start),
        "full_cycle": bool(duration >= cycle_s - 1e-10),
        "max_snapshot_gap_s": max_gap,
        "relative_tolerance": relative_tolerance, "status": "insufficient_cycles",
        "comparison": None, "validated": False,
        "method": "global_phase_aligned_rms",
    }
    if duration < 2 * cycle_s - 1e-10:
        return result
    comparison_start = float(times[-1] - 2 * cycle_s)
    result["comparison_max_snapshot_gap_s"] = _max_gap_covering(times, comparison_start)
    metrics = {}
    for name, values in (("wss", wss), ("pressure", pressure)):
        difference_energy, reference_energy = 0., 0.
        max_difference = 0.
        for i, phase in enumerate(np.linspace(0, cycle_s, 101)):
            previous = interpolate(times, values, comparison_start + phase)
            current = interpolate(times, values, comparison_start + cycle_s + phase)
            weight = .5 if i in (0, 100) else 1.
            difference_energy += weight * float(np.mean((current - previous)**2))
            reference_energy += weight * float(np.mean(previous**2))
            max_difference = max(max_difference, float(np.max(abs(current - previous))))
        rms = float(np.sqrt(difference_energy / 100))
        reference_rms = float(np.sqrt(reference_energy / 100))
        metrics[name] = {
            "rms_difference_pa": rms, "reference_rms_pa": reference_rms,
            "relative_rms": rms / reference_rms if reference_rms > 1e-9 else None,
            "max_component_difference_pa": max_difference,
        }
    result["comparison"] = metrics
    if result["comparison_max_snapshot_gap_s"] > cycle_s / 10 + 1e-10:
        result["status"] = "sparse_sampling"
    elif any(m["relative_rms"] is None for m in metrics.values()):
        result["status"] = "insufficient_signal"
    elif all(m["relative_rms"] <= relative_tolerance for m in metrics.values()):
        result["status"] = "within_tolerance"
    else:
        result["status"] = "not_periodic"
    return result
