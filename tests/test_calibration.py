import numpy as np
import pytest

from server import calibration


def _ring(radius_px, center=(79.5, 79.5), n=64, dx=0.0):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack((center[0] + radius_px * np.sin(t), center[1] + dx + radius_px * np.cos(t)))


def _image(radius_px=12):
    yy, xx = np.mgrid[0:160, 0:160]
    image = np.full((160, 160), 40.0)
    image[np.hypot(yy - 79.5, xx - 79.5) <= radius_px] = 400.0
    return image


def test_compare_identical_and_shifted_contours():
    same = calibration.compare(_ring(12), _ring(12))
    assert same["dice"] == pytest.approx(1.0) and same["mean_boundary_mm"] == pytest.approx(0, abs=0.05)
    bigger = calibration.compare(_ring(12), _ring(14))
    assert bigger["area_error_pct"] == pytest.approx(100 * (14 ** 2 / 12 ** 2 - 1), rel=0.08)
    assert bigger["mean_boundary_mm"] == pytest.approx(2 * calibration.PX_MM, abs=0.1)
    shifted = calibration.compare(_ring(12), _ring(12, dx=4))
    assert shifted["hausdorff_mm"] == pytest.approx(4 * calibration.PX_MM, abs=0.15)


def test_only_manual_annotations_are_scored():
    image = _image()
    manual = {"method": "manual", "contour_px": _ring(12).tolist()}
    rows = calibration.calibrate_section(image, manual, "CT")
    methods = {r["method"] for r in rows}
    assert {"threshold", "fwhm", "levelset"} <= methods
    threshold = next(r for r in rows if r["method"] == "threshold")
    assert threshold["failure"] is None and threshold["dice"] > 0.9
    automatic = {"method": "threshold", "contour_px": _ring(12).tolist()}
    assert calibration.calibrate_section(image, automatic, "CT") == []


def test_summary_counts_failures_and_errors():
    rows = [{"target": "lumen", "method": "a", "failure": None, "dice": 0.9, "area_error_pct": 10.0,
             "mean_boundary_mm": 0.2, "hausdorff_mm": 0.5},
            {"target": "lumen", "method": "a", "failure": "no contour"}]
    s = calibration.summarize(rows)["lumen:a"]
    assert s["sections"] == 2 and s["failures"] == 1 and s["median_dice"] == 0.9
