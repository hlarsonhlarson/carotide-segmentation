import numpy as np

from server import cases


def _new():
    return cases.create(np.zeros((2, 2, 2)), np.ones(3), np.zeros(3), "CT", "synthetic")


def test_registry_evicts_least_recent_case_without_losing_work(monkeypatch):
    monkeypatch.setattr(cases, "MAX_CASES", 2)
    cases._CASES.clear()
    try:
        unsaved = _new()
        unsaved.control_points[("left", "cca_ica")] = [[0, 0, 0]]
        idle = _new()
        running = _new()
        running.su2_mesh_jobs["left"] = {"state": "running"}
        # idle is the only case with nothing to lose; unsaved edits and running jobs stay.
        assert set(cases._CASES) == {unsaved.case_id, running.case_id}
        saved = _new()
        assert saved.case_id in cases._CASES

        unsaved.project_dir = "/projects/a"
        cases.get(running.case_id)
        newest = _new()
        assert unsaved.case_id not in cases._CASES
        assert {running.case_id, newest.case_id} <= set(cases._CASES)
    finally:
        cases._CASES.clear()
