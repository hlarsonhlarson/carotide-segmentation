import numpy as np
import pytest

from server.longitudinal_analysis import analyze_series
from server.plaque_composition import classify_plaque
from server.wall_analysis import measure_wall_section

PX = .1  # mm


def _rings():
    theta = np.linspace(0, 2 * np.pi, 120, endpoint=False)
    ring = lambda r_mm: np.column_stack((80 + r_mm / PX * np.sin(theta), 80 + r_mm / PX * np.cos(theta)))
    return ring(2.0), ring(5.0)


def _phantom():
    r, c = np.indices((161, 161))
    radius = np.hypot(r - 80, c - 80) * PX
    image = np.where(radius < 2, 350., np.where(radius < 5, 80., -100.))
    # Lipid core well inside the wall (3.5 mm from centre, 1 mm across).
    lipid = np.hypot(r - 80, c - 115) * PX < .5
    image[lipid] = 20.
    # Calcification straddling the lumen contour, as on a threshold CTA lumen.
    calcium = np.hypot(r - 80, c - 58) * PX < .6
    image[calcium] = 1100.
    return image, lipid, calcium


def test_components_are_separated_and_sum_to_plaque():
    image, lipid, calcium = _phantom()
    lumen, wall = _rings()
    result = measure_wall_section(image, lumen, wall, PX, 600)
    comp = result["composition"]
    assert comp["lipid_area_mm2"] == pytest.approx(lipid.sum() * PX**2, rel=.05)
    assert comp["calcified_area_mm2"] == pytest.approx(calcium.sum() * PX**2, rel=.05)
    # Part of the calcium lies inside the lumen contour and is still plaque.
    assert comp["calcium_in_lumen_area_mm2"] > .1
    parts = comp["calcified_area_mm2"] + comp["lipid_area_mm2"] + comp["fibrous_area_mm2"]
    assert parts == pytest.approx(comp["plaque_area_mm2"])
    assert comp["plaque_area_mm2"] == pytest.approx(
        result["annulus_area_mm2"] + comp["calcium_in_lumen_area_mm2"], abs=.05)
    assert comp["contrast_partial_volume_area_mm2"] == 0
    # Only polygon rasterization slivers beyond the 5 mm wall reach the -100 HU fat.
    assert comp["excluded_low_hu_area_mm2"] < .1


def test_partial_volume_with_fat_at_outer_edge_is_not_lipid():
    image, _, _ = _phantom()
    image[:, :] = np.where(image == 20., 80., image)  # remove the core
    lumen, wall = _rings()
    r, c = np.indices(image.shape)
    rim = (np.hypot(r - 80, c - 80) * PX > 4.7) & (np.hypot(r - 80, c - 80) * PX < 5)
    image[rim] = 10.  # fat blurred into the outermost 0.3 mm of the wall
    comp = measure_wall_section(image, lumen, wall, PX, 600)["composition"]
    assert comp["lipid_area_mm2"] == 0


def test_threshold_order_is_enforced():
    image, _, _ = _phantom()
    mask = np.zeros(image.shape, bool)
    with pytest.raises(ValueError):
        classify_plaque(image, mask, mask, PX, 600, .5, lipid_hu=700)


def test_longitudinal_volumes_include_composition():
    image, _, _ = _phantom()
    lumen, wall = _rings()
    section = measure_wall_section(image, lumen, wall, PX, 600)
    contour_mm = [[0, x * PX, y * PX] for x, y in lumen]
    series = [{"point_mm": [z, 0, 0], "tangent_mm": [1, 0, 0], "method": "threshold",
               "contour_mm": [[z, y, x] for _, y, x in contour_mm], "wall_analysis": section}
              for z in (0., 1., 2.)]
    profile = analyze_series(series)
    comp = section["composition"]
    assert profile["integrals"]["lipid"]["volume_mm3"] == pytest.approx(2 * comp["lipid_area_mm2"])
    assert profile["integrals"]["calcified"]["volume_mm3"] == pytest.approx(2 * comp["calcified_area_mm2"])
    assert profile["composition_parameters"]["lipid_hu"] == 50


def test_old_results_without_composition_are_not_counted():
    image, _, _ = _phantom()
    lumen, wall = _rings()
    section = measure_wall_section(image, lumen, wall, PX, 600)
    section.pop("composition")
    series = [{"point_mm": [z, 0, 0], "tangent_mm": [1, 0, 0], "method": "threshold",
               "contour_mm": [[z, x * PX, y * PX] for x, y in lumen], "wall_analysis": section}
              for z in (0., 1.)]
    profile = analyze_series(series)
    assert profile["integrals"]["plaque"]["volume_mm3"] is None
    assert profile["integrals"]["annulus"]["volume_mm3"] is not None


def test_blurred_contrast_rim_is_not_plaque():
    image, _, _ = _phantom()
    lumen, wall = _rings()
    r, c = np.indices(image.shape)
    rim = (np.hypot(r - 80, c - 80) * PX >= 2) & (np.hypot(r - 80, c - 80) * PX < 2.4) & (image == 80.)
    image[rim] = 250.  # lumen contrast smeared just outside the lumen contour
    comp = measure_wall_section(image, lumen, wall, PX, 600)["composition"]
    assert comp["contrast_partial_volume_area_mm2"] == pytest.approx(rim.sum() * PX**2, rel=.1)
    parts = comp["calcified_area_mm2"] + comp["lipid_area_mm2"] + comp["fibrous_area_mm2"]
    assert parts == pytest.approx(comp["plaque_area_mm2"])


def test_calcium_threshold_rises_above_bright_lumen():
    image, _, calcium = _phantom()
    image[image == 350.] = 700.  # lumen brighter than the 600 HU calcium setting
    lumen, wall = _rings()
    comp = measure_wall_section(image, lumen, wall, PX, 600)["composition"]
    assert comp["calcium_hu_effective"] == pytest.approx(910)
    assert comp["calcified_area_mm2"] == pytest.approx(calcium.sum() * PX**2, rel=.05)


def test_low_attenuation_continuing_outside_the_wall_is_not_atheroma():
    image, _, _ = _phantom()
    lumen, wall = _rings()
    r, c = np.indices(image.shape)
    # A fat tongue (20 HU) crossing the wall contour from outside, as when the
    # outer boundary overreaches into perivascular fat.
    tongue = (np.abs(r - 80) < 6) & (c > 80 + int(3.3 / PX))
    image[tongue] = 20.
    comp = measure_wall_section(image, lumen, wall, PX, 600)["composition"]
    assert comp["perivascular_low_hu_area_mm2"] > 1
    # The tongue (which here also covers the phantom core) is not atheroma.
    assert comp["lipid_area_mm2"] < 0.2
    parts = comp["calcified_area_mm2"] + comp["lipid_area_mm2"] + comp["fibrous_area_mm2"]
    assert parts == pytest.approx(comp["plaque_area_mm2"])
