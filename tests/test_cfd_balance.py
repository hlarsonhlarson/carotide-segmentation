import csv

import numpy as np
import pytest

from server.cfd_balance import flow_metrics, summarize_balance


def record(tmp_path, flows, clamp=None):
    with (tmp_path / "boundary_flows.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "time_s", "q_inlet_mlps", "q_ica_mlps", "q_eca_mlps",
            "ica_pressure_clamped", "eca_pressure_clamped",
        ])
        writer.writeheader()
        for index, (qin, qica, qeca) in enumerate(flows):
            writer.writerow(dict(time_s=index*.1, q_inlet_mlps=qin, q_ica_mlps=qica,
                                 q_eca_mlps=qeca, ica_pressure_clamped=int(index == clamp),
                                 eca_pressure_clamped=0))


def test_flux_sign_and_units():
    import json
    result = flow_metrics(-10e-6, 6e-6, 4e-6)
    assert result["q_inlet_mlps"] == pytest.approx(10)
    assert result["mass_imbalance_relative"] == pytest.approx(0, abs=1e-15)
    assert not result["ica_reversed"]
    assert flow_metrics(-10e-6, -2e-6, 12e-6)["ica_reversed"]
    assert flow_metrics(0, 1e-6, 0)["mass_imbalance_relative"] is None
    json.dumps(flow_metrics(np.float64(-1), np.float64(.5), np.float64(.5)))


def test_mass_error_cannot_cancel_in_time_average(tmp_path):
    record(tmp_path, [(10, 6, 2), (10, 6, 6), (10, 6, 2)])
    result = summarize_balance(tmp_path, 1.)
    assert result["inlet_volume_ml"] == pytest.approx(result["outlet_volume_ml"])
    assert result["relative_mean_absolute_imbalance"] == pytest.approx(.2)
    assert not result["full_cycle"]


def test_cycle_window_and_clamp_recording(tmp_path):
    record(tmp_path, [(10, 6, 4)] * 31, clamp=25)
    result = summarize_balance(tmp_path, 1.)
    assert result["full_cycle"]
    assert result["inlet_volume_ml"] == pytest.approx(10)
    assert result["pressure_clamped_steps"] == 1
    assert result["relative_mean_absolute_imbalance"] == 0


def test_live_summary_stops_at_last_saved_frame(tmp_path):
    record(tmp_path, [(10, 6, 4)] * 11 + [(10, -5, 4)] * 10)
    result = summarize_balance(tmp_path, 1., end_time_s=1.)
    assert result["reverse_samples"]["ica"] == 0
    assert result["window_end_s"] == pytest.approx(1.)


def test_missing_and_partial_files(tmp_path):
    assert summarize_balance(tmp_path, 1.)["status"] == "unavailable"
    record(tmp_path, [(10, 6, 4)])
    with (tmp_path / "boundary_flows.csv").open("a") as handle:
        handle.write("0.1,10,6")
    assert summarize_balance(tmp_path, 1.)["status"] == "insufficient_samples"


def test_nonfinite_flux_rejected():
    with pytest.raises(ValueError):
        flow_metrics(np.nan, 0, 0)


def test_su2_inward_normals_are_converted_to_outward_flux(monkeypatch):
    import sys
    from types import SimpleNamespace
    from server.su2_pulsatile_runner import marker_area_and_flow
    monkeypatch.setitem(sys.modules, "mpi4py", SimpleNamespace(MPI=SimpleNamespace(SUM=object())))
    class Driver:
        def GetNumberMarkerNodes(self, marker):
            return 2
        def MarkerPrimitives(self, marker):
            return lambda node, component: .2 if component == 0 else 0.
        def GetMarkerNode(self, marker, node):
            return node
        def GetNodeDomain(self, node):
            return node == 0  # exclude a halo copy
        def GetMarkerVertexNormals(self, marker, node, normalized):
            return [1e-5 if marker == 0 else -1e-5, 0., 0.]
    comm = SimpleNamespace(allreduce=lambda value, op: value)
    area, inlet = marker_area_and_flow(Driver(), 0, [0, 1, 2], comm)
    _, outlet = marker_area_and_flow(Driver(), 1, [0, 1, 2], comm)
    assert area == pytest.approx(1e-5)
    assert inlet == pytest.approx(-2e-6)
    assert outlet == pytest.approx(2e-6)
