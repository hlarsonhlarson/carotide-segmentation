"""Regression checks for cached image data and surface reconstruction.

Run with: .venv/bin/python -m unittest discover -s server/tests -v
"""
import copy
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import numpy as np
from vtk.util.numpy_support import vtk_to_numpy

from server import app, build_surface, cases, reslice, surface_cache, workflow


def make_case(modality="CT"):
    image = np.random.default_rng(7).normal(200, 50, (64, 48, 48)).astype(np.float32)
    return cases.create(image, np.ones(3), np.zeros(3), modality, "synthetic")


def vessel_series():
    angles = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    return {"cca_ica": [
        {
            "point_mm": [float(z), 24.0, 24.0],
            "tangent_mm": [1.0, 0.0, 0.0],
            "radius_mm": 3.0,
            "contour_mm": [[float(z), float(24 + 3 * np.sin(a)), float(24 + 3 * np.cos(a))] for a in angles],
        }
        for z in range(8, 57, 2)
    ], "cca_eca": []}


class ImageCacheTests(unittest.TestCase):
    def tearDown(self):
        cases._CASES.clear()

    def test_oblique_pixels_and_basis_are_identical_including_outside_volume(self):
        case = make_case()
        for point in ([32, 24, 24], [0, 0, 0], [63, 47, 47]):
            args = (case.array, case.spacing_mm, case.origin_mm, point, [1, 2, 3])
            reference = reslice.sample_oblique_slice(*args)
            cached = reslice.sample_oblique_slice(*args, background_value=case.array_min)
            for expected, actual in zip(reference, cached):
                np.testing.assert_array_equal(actual, expected)

    def test_mr_percentiles_are_computed_only_at_load(self):
        with patch.object(cases.np, "percentile", wraps=np.percentile) as percentile:
            case = make_case("MR")
            case.segmentation_series[("right", "cca_ica")] = vessel_series()["cca_ica"]
            for _ in range(3):
                meta = app.get_meta(case.case_id)
                app.get_slice(case.case_id, "axial", 10)
                app.get_segment_slice(case.case_id, "right", "cca_ica", 0)
            self.assertEqual(percentile.call_count, 1)
        lo, hi = np.percentile(case.array, [1, 99])
        self.assertEqual(meta["default_window"], {"wl": float((lo + hi) / 2), "ww": float(hi - lo)})
        self.assertEqual(make_case().default_window, (300.0, 800.0))

    def test_constant_mr_window_keeps_positive_width(self):
        case = cases.create(np.full((4, 4, 4), 7, dtype=np.float32), np.ones(3), np.zeros(3), "MR", "synthetic")
        self.assertEqual(case.default_window, (7.5, 1.0))


class SurfaceCacheTests(unittest.TestCase):
    def setUp(self):
        self.case = make_case()
        self.series = vessel_series()

    def tearDown(self):
        cases._CASES.clear()

    def test_real_surface_matches_uncached_and_consumers_are_isolated(self):
        expected, expected_diag = build_surface.build_surface(
            self.case.array.shape, self.case.spacing_mm, self.case.origin_mm, self.series,
        )
        self.assertIsNotNone(expected)
        with patch.object(build_surface, "build_surface", wraps=build_surface.build_surface) as builder:
            first, diag = surface_cache.get_surface(self.case, "right", self.series)
            np.testing.assert_array_equal(vtk_to_numpy(first.GetPoints().GetData()), vtk_to_numpy(expected.GetPoints().GetData()))
            np.testing.assert_array_equal(vtk_to_numpy(first.GetPolys().GetData()), vtk_to_numpy(expected.GetPolys().GetData()))
            self.assertEqual(diag, expected_diag)
            first.GetPoints().SetPoint(0, 999, 999, 999)
            diag["branches_used"].append("corrupted")
            second, second_diag = surface_cache.get_surface(self.case, "right", copy.deepcopy(self.series))
            self.assertEqual(builder.call_count, 1)
            np.testing.assert_array_equal(vtk_to_numpy(second.GetPoints().GetData()), vtk_to_numpy(expected.GetPoints().GetData()))
            self.assertEqual(second_diag, expected_diag)

    def test_contour_edits_side_image_geometry_and_settings_cause_reconstruction(self):
        with patch.object(build_surface, "build_surface", wraps=build_surface.build_surface) as builder:
            original, _ = surface_cache.get_surface(self.case, "right", self.series)
            for point in self.series["cca_ica"]:
                point["point_mm"][1] += 1
                for vertex in point["contour_mm"]:
                    vertex[1] += 1
            changed, _ = surface_cache.get_surface(self.case, "right", self.series)
            self.assertNotEqual(original.GetBounds(), changed.GetBounds())
            surface_cache.get_surface(self.case, "left", self.series)
            self.case.origin_mm = np.array([0, 0, 0.25])
            surface_cache.get_surface(self.case, "right", self.series)
            with patch.object(build_surface, "SMOOTHING_ITERATIONS", build_surface.SMOOTHING_ITERATIONS + 1):
                surface_cache.get_surface(self.case, "right", self.series)
            self.assertEqual(builder.call_count, 5)

    def test_concurrent_requests_build_once(self):
        with patch.object(build_surface, "build_surface", wraps=build_surface.build_surface) as builder:
            with ThreadPoolExecutor(max_workers=4) as pool:
                outputs = list(pool.map(lambda _: surface_cache.get_surface(self.case, "right", self.series), range(4)))
            self.assertEqual(builder.call_count, 1)
            self.assertEqual(len({id(poly) for poly, _ in outputs}), 4)

    def test_failed_build_is_not_cached(self):
        with patch.object(build_surface, "build_surface", return_value=(None, {"error": "incomplete"})) as builder:
            for _ in range(2):
                self.assertIsNone(surface_cache.get_surface(self.case, "right", self.series)[0])
            self.assertEqual(builder.call_count, 2)

    def test_workflow_invalidates_only_the_edited_sides_geometry_cache(self):
        surface_cache.get_surface(self.case, "right", self.series)
        surface_cache.get_surface(self.case, "left", self.series)
        workflow.invalidate(self.case, "right", "cfd")
        self.assertIn("right", self.case.surface_cache)
        workflow.invalidate(self.case, "right", "surface")
        self.assertNotIn("right", self.case.surface_cache)
        self.assertIn("left", self.case.surface_cache)

    def test_preview_and_meshing_reuse_the_same_reconstruction(self):
        self.case.segmentation_series[("right", "cca_ica")] = self.series["cca_ica"]
        with patch.object(build_surface, "build_surface", wraps=build_surface.build_surface) as builder:
            app.build_surface_endpoint(self.case.case_id, "right")
            # Don't launch TetGen here; verify the mesh endpoint's surface input.
            with patch.object(app.threading, "Thread"):
                app.start_su2_mesh(self.case.case_id, "right")
            self.assertEqual(builder.call_count, 1)


if __name__ == "__main__":
    unittest.main()
