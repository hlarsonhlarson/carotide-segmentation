import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from server.wall_analysis import analyze_wall, measure_wall_section, _closed_boundary


def phantom(wall=True, calcium=False):
    r, c = np.indices((161, 161))
    radius = np.hypot(r - 80, c - 80) * .15
    image = np.where(radius < 2, 300., np.where((radius < 3.5) & wall, 65., -60.))
    if calcium:
        image[(r - 80)**2 + (c - 98)**2 < 9] = 1000
    theta = np.linspace(0, 2*np.pi, 96, endpoint=False)
    lumen = np.column_stack((80 + 2/.15*np.sin(theta), 80 + 2/.15*np.cos(theta)))
    return gaussian_filter(image, .7), lumen


def test_visible_wall_and_sectional_high_density():
    image, lumen = phantom(calcium=True)
    result = analyze_wall(image, lumen, .15, 600)
    assert result["status"] == "proposal"
    assert result["max_radial_thickness_mm"] == pytest.approx(1.5, abs=.3)
    assert result["annulus_area_mm2"] == pytest.approx(np.pi*(3.5**2 - 2**2), abs=3)
    assert result["high_density_area_mm2"] > .5


def test_absent_wall_is_not_assumed():
    image, lumen = phantom(wall=False)
    with pytest.raises(ValueError):
        analyze_wall(image, lumen, .15, 600)


def test_cyclic_path_enforces_seam():
    cost = np.zeros((5, 3))
    cost[0, 0] = -10
    cost[-1, 2] = -10
    radii = np.tile([1., 1.5, 2.], (5, 1))
    path = _closed_boundary(cost, radii, max_jump_mm=.6)
    assert abs(radii[0, path[0]] - radii[-1, path[-1]]) <= .6


@pytest.mark.parametrize("kwargs", [
    {"pixel_mm": float("nan")}, {"min_component_mm2": float("nan")},
    {"min_thickness_mm": 5, "max_thickness_mm": 2},
])
def test_invalid_parameters(kwargs):
    image, lumen = phantom()
    params = dict(pixel_mm=.15, calcium_hu=600)
    params.update(kwargs)
    with pytest.raises(ValueError):
        analyze_wall(image, lumen, **params)


def test_wall_endpoint_persists_and_lumen_edits_invalidate(monkeypatch):
    from fastapi.testclient import TestClient
    from server import app as backend, cases, projects
    image, lumen = phantom()
    case = cases.create(np.zeros((5, 5, 5)), np.ones(3), np.zeros(3), "CT", "synthetic")
    frame = {"point_mm": [2., 2., 2.], "tangent_mm": [1., 0., 0.],
             "u_vec": [0., 1., 0.], "v_vec": [0., 0., 1.]}
    case.segmentation_series["right", "cca_ica"] = [{
        **frame, "contour_px": lumen.tolist(), "method": "manual",
    }]
    monkeypatch.setattr(backend, "RESLICE_PX_SPACING_MM", .15)
    monkeypatch.setattr(backend.reslice, "sample_oblique_slice",
                        lambda *args, **kwargs: (image, np.array(frame["u_vec"]),
                                                np.array(frame["v_vec"]), np.zeros(3)))
    route = f"/cases/{case.case_id}/segment/right/cca_ica/point/0"
    try:
        with TestClient(backend.app) as client:
            response = client.post(route + "/wall", json={"calcium_hu": 600})
            assert response.status_code == 200, response.text
            saved = projects.snapshot(case)["sides"]["right"]["branches"]["cca_ica"]["segmentation"][0]
            assert saved["wall_analysis"]["parameters"]["calcium_hu"] == 600
            manual_wall = response.json()["wall_contour_px"]
            response = client.post(route + "/wall", json={"calcium_hu": 600, "wall_contour_px": manual_wall})
            assert response.status_code == 200, response.text
            assert response.json()["status"] == "manual"
            assert response.json()["edge_support_fraction"] is None
            assert client.delete(route + "/wall").status_code == 200
            assert "wall_analysis" not in case.segmentation_series["right", "cca_ica"][0]
            assert case.segmentation_series["right", "cca_ica"][0]["contour_px"] == lumen.tolist()
            assert client.post(route + "/wall", json={"calcium_hu": 600}).status_code == 200
            assert client.post(route + "/manual", json={"contour_px": lumen.tolist()}).status_code == 200
            assert "wall_analysis" not in case.segmentation_series["right", "cca_ica"][0]
            assert client.delete(route).status_code == 200
            assert case.segmentation_series["right", "cca_ica"][0] == frame
    finally:
        cases._CASES.pop(case.case_id, None)


def test_manual_boundary_measures_without_visible_edge():
    image, lumen = phantom(wall=False)
    outer = (lumen - 80) * 1.75 + 80
    result = measure_wall_section(image, lumen, outer, .15, 600)
    assert result["annulus_area_mm2"] == pytest.approx(np.pi * (3.5**2 - 2**2), abs=1)
    assert result["high_density_area_mm2"] == 0


def test_manual_wall_must_contain_lumen_without_crossing_concavity():
    image = np.zeros((20, 20))
    lumen = [[5, 1], [5, 9], [8, 9], [8, 1]]
    wall = [[0, 0], [0, 10], [10, 10], [10, 7], [3, 7], [3, 3], [10, 3], [10, 0]]
    with pytest.raises(ValueError, match="intersect"):
        measure_wall_section(image, lumen, wall, .2, 600)


def test_manual_wall_outside_image_or_inside_lumen_rejected():
    image, lumen = phantom()
    for wall in ((lumen - 80) * .5 + 80, lumen + 200):
        with pytest.raises(ValueError):
            measure_wall_section(image, lumen, wall, .15, 600)
