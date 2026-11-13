import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from server.lumen_profiles import fwhm_contour


@pytest.mark.parametrize("enhancement", [180., 300., 500.])
def test_half_height_tracks_blurred_lumen_with_different_contrast(enhancement):
    r, c = np.indices((151, 151))
    image = np.where(np.hypot(r - 75, c - 75) * .2 <= 2.5, enhancement, 30.)
    contour, qa = fwhm_contour(gaussian_filter(image, 1), (75, 75), .2)
    assert contour is not None
    assert abs(np.median(qa["radii_mm"]) - 2.5) < .2


def test_air_and_uniform_bright_region_are_not_vessels():
    for value in [-1000., 40., 300.]:
        contour, _ = fwhm_contour(np.full((151, 151), value), (75, 75), .2)
        assert contour is None


def test_bright_outer_peak_is_flagged_not_filled():
    r, c = np.indices((151, 151))
    radius = np.hypot(r - 75, c - 75) * .2
    image = np.where(radius <= 2.5, 300., 30.)
    image[(radius > 1.7) & (radius < 2.5) & (c > 75)] = 1000
    contour, qa = fwhm_contour(image, (75, 75), .2)
    assert contour is None
    assert qa["reason"] == "bright_peak_requires_review"
