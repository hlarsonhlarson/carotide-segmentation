"""CCA inlet velocity waveform, calibrated to a target peak-systolic (PSV)
and end-diastolic (EDV) velocity.

Ported verbatim from the validated standalone pipeline
(~/Desktop/SU2 Pipeline Code/shared/cca_waveform_psv_edv.py). The shape
itself (the landmark points below) is a reconstructed approximation of a
real carotid Doppler waveform, calibrated to Holdsworth et al. 1999's
published summary statistics (peak/notch timing and magnitude) — not
verbatim digitized patient data, since the canonical dataset's own URL is
dead. Rescaled so its per-cycle max/min land exactly on the given
PSV/EDV. The same shape (as a dense precomputed table instead of a live
scipy spline) drives the live preview chart in BoundaryConditionsView.swift
— CCAWaveformShape.swift's table was generated from this exact spline, so
the Swift preview and the actual SU2 run are driven by the same curve.
"""
import numpy as np
from scipy.interpolate import CubicSpline

_LANDMARKS = [
    (0.000, 0.75), (0.050, 1.00), (0.100, 2.05), (0.152, 2.79), (0.200, 1.85),
    (0.280, 0.95), (0.340, 0.60), (0.398, 0.50), (0.450, 0.68), (0.600, 0.78),
    (0.800, 0.80), (1.000, 0.75),
]


def _build_shape_spline():
    t = np.array([p[0] for p in _LANDMARKS])
    q = np.array([p[1] for p in _LANDMARKS])
    spline = CubicSpline(t, q, bc_type="periodic")
    tt = np.linspace(0, 1, 2001)
    mean_val = np.trapezoid(spline(tt), tt)
    return CubicSpline(t, q / mean_val, bc_type="periodic")


_SPLINE = _build_shape_spline()
_TT = np.linspace(0, 1, 2001)
_QVALS = _SPLINE(_TT)
_QMIN, _QMAX = float(_QVALS.min()), float(_QVALS.max())


def shape(t_frac):
    return float(_SPLINE(np.mod(t_frac, 1.0)))


def inlet_velocity_ms(t_seconds, period_s, psv_cms, edv_cms):
    """Instantaneous CCA inlet velocity [m/s], shaped like the
    reconstructed physiological waveform but rescaled so the cycle max =
    psv_cms and cycle min = edv_cms (both in cm/s)."""
    q = shape(t_seconds / period_s)
    frac = (q - _QMIN) / (_QMAX - _QMIN)
    v_cms = edv_cms + (psv_cms - edv_cms) * frac
    return v_cms / 100.0


def mean_velocity_ms(period_s, psv_cms, edv_cms, n=400):
    tt = np.linspace(0, period_s, n, endpoint=False)
    v = np.array([inlet_velocity_ms(t, period_s, psv_cms, edv_cms) for t in tt])
    return float(v.mean())
