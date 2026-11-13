import numpy as np
import pytest
from fastapi.testclient import TestClient

from server import cases
from server.app import app
from server.lumen_measurements import contour_metrics, nascet_estimate


def circle(radius, n=128):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack((np.zeros(n), radius * np.cos(t), radius * np.sin(t)))


def test_circle_measurements_are_rotation_and_translation_invariant():
    ring = circle(3)
    angle = .7
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                         [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    metrics = contour_metrics(ring @ rotation.T + [40, -30, 18])
    assert metrics["area_mm2"] == pytest.approx(9 * np.pi, rel=.001)
    assert metrics["min_feret_mm"] == pytest.approx(6, rel=.001)
    assert metrics["max_feret_mm"] == pytest.approx(6)


def test_nascet_uses_diameter_not_area():
    result = nascet_estimate(contour_metrics(circle(1)), contour_metrics(circle(2)))
    assert result["percent"] == pytest.approx(50)
    assert not result["validated"]


@pytest.mark.parametrize("ring", [
    [[0, 0, 0], [0, 1, 1], [0, 0, 1], [0, 1, 0]],
    [[0, 0, 0], [0, 0, 0], [0, 1, 0]],
    [[0, 0, 0], [0, 1, 0], [0, 2, 0]],
    [[0, 0, 0], [0, 1, 0], [np.nan, 0, 1]],
])
def test_invalid_contours_do_not_produce_measurements(ring):
    with pytest.raises(ValueError):
        contour_metrics(ring)


def test_measurement_endpoint_requires_reference_confirmation_and_real_contours():
    case = cases.create(np.zeros((5, 5, 5)), np.ones(3), np.zeros(3), "CT", "synthetic")
    case.segmentation_series["right", "cca_ica"] = [
        {"contour_mm": circle(1).tolist(), "method": "manual"},
        {"contour_mm": circle(2).tolist(), "method": "manual"},
    ]
    route = f"/cases/{case.case_id}/segment/right/cca_ica/measurements"
    try:
        with TestClient(app) as client:
            assert client.post(route, json={"index": 0}).status_code == 200
            assert client.post(route, json={"index": 0, "reference_index": 1}).status_code == 422
            payload = {"index": 0, "reference_index": 1, "confirmed_ica_reference": True}
            response = client.post(route, json=payload)
            assert response.status_code == 200, response.text
            assert response.json()["nascet"]["percent"] == pytest.approx(50)
            case.segmentation_series["right", "cca_ica"][1]["method"] = "circle"
            assert client.post(route, json=payload).status_code == 422
    finally:
        cases._CASES.pop(case.case_id, None)


def test_fwhm_is_accepted_by_request_schema():
    from server.app import SegmentPointRequest
    assert SegmentPointRequest(method="fwhm", radius_mm=3).method == "fwhm"
