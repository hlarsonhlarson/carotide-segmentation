import numpy as np
import pytest

from server.su2_postprocess import compute_derived_fields


def samples(times, x=None, pressure=None):
    times = np.asarray(times, float)
    w = np.zeros((len(times), 2, 3))
    w[:, :, 0] = (np.full(len(times), 2.) if x is None else np.asarray(x))[:, None]
    p = np.full((len(times), 2), 10000.) if pressure is None else np.repeat(np.asarray(pressure)[:, None], 2, axis=1)
    return {"times_s": times, "wss_vec_pa": w, "pressure_pa": p}


def test_constant_flow_is_periodic_and_has_zero_osi():
    result = compute_derived_fields(samples(np.linspace(0, 3, 61)), 1.)
    np.testing.assert_allclose(result["tawss_pa"], 2)
    np.testing.assert_allclose(result["osi"], 0)
    np.testing.assert_allclose(result["rrt_pa_inv"], .5)
    assert result["temporal_quality"]["status"] == "within_tolerance"
    assert result["temporal_quality"]["window_duration_s"] == 1.


def test_window_includes_interpolated_start_and_exact_duration():
    times = np.array([0., .3, .6, .9, 1.2, 1.5])
    result = compute_derived_fields(samples(times, x=times + 1, pressure=times * 10), 1.)
    assert result["settled_cycle_start_s"] == pytest.approx(.5)
    np.testing.assert_allclose(result["tawss_pa"], 2)
    np.testing.assert_allclose(result["mean_pressure_pa"], 10)


def test_periodic_reversing_flow_has_high_osi():
    times = np.linspace(0, 3, 121)
    result = compute_derived_fields(samples(times, x=np.sin(2*np.pi*times)), 1.)
    np.testing.assert_allclose(result["osi"], .5, atol=1e-6)
    assert result["temporal_quality"]["status"] == "within_tolerance"
    assert result["tawss_pa"][0] == pytest.approx(2/np.pi, rel=.003)


def test_drift_is_not_reported_as_periodic():
    times = np.linspace(0, 3, 61)
    result = compute_derived_fields(samples(times, x=times + 1), 1.)
    assert result["temporal_quality"]["status"] == "not_periodic"


def test_vector_direction_changes_are_detected():
    times = np.linspace(0, 3, 61)
    series = samples(times)
    series["wss_vec_pa"][:, :, 0] = np.cos(np.pi * times)[:, None]
    series["wss_vec_pa"][:, :, 1] = np.sin(np.pi * times)[:, None]
    result = compute_derived_fields(series, 1.)
    assert result["temporal_quality"]["status"] == "not_periodic"


def test_partial_and_sparse_runs_are_not_called_converged():
    partial = compute_derived_fields(samples([0., .1, .2]), 1.)
    assert not partial["temporal_quality"]["full_cycle"]
    assert partial["temporal_quality"]["status"] == "insufficient_cycles"
    sparse = compute_derived_fields(samples([0., 1., 2., 3.]), 1.)
    assert sparse["temporal_quality"]["status"] == "sparse_sampling"


def test_zero_wss_has_explicit_undefined_osi_mask():
    times = np.linspace(0, 3, 61)
    result = compute_derived_fields(samples(times, x=np.zeros_like(times)), 1.)
    assert result["temporal_quality"]["undefined_osi_nodes"] == 2
    assert not result["osi_valid_mask"].any()
    assert result["temporal_quality"]["status"] == "insufficient_signal"
    assert np.isfinite(result["rrt_pa_inv"]).all()


def test_sparse_interval_bracketing_cycle_start_is_not_hidden():
    times = np.r_[0., .5, np.arange(1.01, 3, .05), 3.]
    result = compute_derived_fields(samples(times), 1.)
    assert result["temporal_quality"]["status"] == "sparse_sampling"
    assert result["temporal_quality"]["comparison_max_snapshot_gap_s"] == pytest.approx(.51)


@pytest.mark.parametrize("times, cycle", [([], 1), ([0, 0], 1), ([1, 0], 1),
                                          ([0, np.nan], 1), ([0, 1], 0)])
def test_invalid_time_series_rejected(times, cycle):
    with pytest.raises(ValueError):
        compute_derived_fields(samples(times), cycle)
