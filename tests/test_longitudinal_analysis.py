import numpy as np
import pytest

from server.longitudinal_analysis import analyze_series


def tube(radii=(2., 2., 2., 2.), positions=(0., 2., 4., 6.)):
    angles = np.linspace(0, 2*np.pi, 64, endpoint=False)
    return [
        {"point_mm": [z, 0., 0.], "tangent_mm": [1., 0., 0.], "method": "manual",
         "contour_mm": np.column_stack((np.full(64, z), r*np.sin(angles), r*np.cos(angles))).tolist(),
         "wall_analysis": {"annulus_area_mm2": 10., "high_density_area_mm2": 2.,
                           "parameters": {"calcium_hu": 600, "min_component_mm2": .5}}}
        for r, z in zip(radii, positions)
    ]


def test_cylinder_integral_and_physical_length():
    result = analyze_series(tube())
    assert result["selected_length_mm"] == 6
    assert result["integrals"]["lumen"]["volume_mm3"] == pytest.approx(np.pi * 4 * 6, rel=.002)
    assert result["integrals"]["annulus"]["volume_mm3"] == 60
    assert result["integrals"]["high_density"]["volume_mm3"] == 12


def test_missing_section_does_not_bridge_gap_or_count_wall():
    series = tube()
    series[1].pop("contour_mm")
    result = analyze_series(series)
    for value in result["integrals"].values():
        assert value["covered_length_mm"] == 2
        assert value["intervals"] == 1
        assert len(value["excluded_intervals"]) == 2
    assert result["sections"][1]["wall"] is None


def test_minimum_and_selected_range():
    result = analyze_series(tube(radii=(1., 3., 2., 3.)), 1, 3)
    assert result["minimum_area_index"] == 2
    assert result["minimum_diameter_index"] == 2
    assert result["sections"][0]["distance_mm"] == 2
    assert result["selected_length_mm"] == 4


def test_single_section_has_no_volume():
    result = analyze_series(tube(), 1, 1)
    assert all(v["volume_mm3"] is None for v in result["integrals"].values())


def test_missing_frame_excludes_intervals_instead_of_failing_profile():
    series = tube()
    series[2].pop("tangent_mm")
    result = analyze_series(series)
    assert result["integrals"]["lumen"]["covered_length_mm"] == 2
    assert result["integrals"]["lumen"]["excluded_intervals"][0]["reason"] == "invalid_frame"


def test_zero_density_volume_is_distinct_from_missing_data():
    series = tube()
    for point in series:
        point["wall_analysis"]["high_density_area_mm2"] = 0.
    result = analyze_series(series)
    assert result["integrals"]["high_density"]["volume_mm3"] == 0.
    assert result["density_parameters"] == {"calcium_hu": 600, "min_component_mm2": .5}


def test_density_thresholds_are_not_combined_across_regions():
    series = tube()
    for point in series[2:]:
        point["wall_analysis"]["parameters"]["calcium_hu"] = 800
    result = analyze_series(series)
    assert result["integrals"]["high_density"]["volume_mm3"] is None
    assert result["integrals"]["annulus"]["volume_mm3"] == 60


@pytest.mark.parametrize("change, reason", [
    ("gap", "sampling_gap"), ("turn", "sharp_turn"), ("circle", "missing_measurement"),
])
def test_bad_intervals_excluded(change, reason):
    series = tube()
    if change == "gap":
        series[2]["point_mm"][0] = 20
    elif change == "turn":
        series[2]["tangent_mm"] = [0, 1, 0]
    else:
        series[2]["method"] = "circle"
    result = analyze_series(series)
    assert result["integrals"]["lumen"]["covered_length_mm"] == 2
    assert all(i["reason"] == reason for i in result["integrals"]["lumen"]["excluded_intervals"])


def test_profile_api_validates_range():
    from fastapi.testclient import TestClient
    from server import cases
    from server.app import app
    case = cases.create(np.zeros((5, 5, 5)), np.ones(3), np.zeros(3), "CT", "synthetic")
    case.segmentation_series["right", "cca_ica"] = tube()
    try:
        with TestClient(app) as client:
            route = f"/cases/{case.case_id}/segment/right/cca_ica/profile"
            assert client.post(route, json={}).status_code == 200
            assert client.post(route, json={"end_index": 100}).status_code == 422
    finally:
        cases._CASES.pop(case.case_id, None)
