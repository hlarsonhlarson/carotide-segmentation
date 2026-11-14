import numpy as np
import pytest

from server.coordinates import voxel_to_lps, lps_to_voxel
from server.path_series import resample_by_arclength, parallel_transport_frames
from server.reslice import sample_oblique_slice


def test_oblique_lps_roundtrip():
    angle = 0.43
    direction = np.array([[np.cos(angle), -np.sin(angle), 0],
                          [np.sin(angle), np.cos(angle), 0], [0, 0, -1]])
    points = np.array([[3, 4, 5], [17, 21, 2.5]])
    spacing, origin = [1.3, 0.7, 0.5], [-40, 20, 35]
    lps = voxel_to_lps(points, spacing, origin, direction)
    np.testing.assert_allclose(lps_to_voxel(lps, spacing, origin, direction), points)
    np.testing.assert_allclose(voxel_to_lps([0, 0, 0], spacing, origin, direction), origin[::-1])


def test_arclength_interpolates_long_segments_and_keeps_endpoint():
    points = [[0, 0, 0], [0, 0, 0], [0, 0, 2.3]]
    samples = resample_by_arclength(points, 0.5)
    np.testing.assert_allclose(np.array(samples)[:, 2], [0, .5, 1, 1.5, 2, 2.3])
    with pytest.raises(ValueError):
        resample_by_arclength(points, 0)


def test_transported_frames_do_not_jump_at_old_up_threshold():
    theta = np.linspace(.1, 1.0, 100)
    tangents = np.column_stack((np.sin(theta), np.zeros(100), np.cos(theta)))
    u, v = parallel_transport_frames(tangents)
    np.testing.assert_allclose(np.sum(u * tangents, axis=1), 0, atol=1e-12)
    np.testing.assert_allclose(np.cross(u, v), tangents, atol=1e-12)
    assert np.min(np.sum(u[1:] * u[:-1], axis=1)) > .99


def test_mpr_pixel_centers_match_reported_world_coordinates():
    z, y, x = np.indices((25, 25, 25), dtype=float)
    volume = 3 * z + 2 * y + x
    frame = ([0, 1, 0], [0, 0, 1])
    image, u, v, origin = sample_oblique_slice(
        volume, [1, 1, 1], [0, 0, 0], [12, 12, 12], [1, 0, 0],
        size_px=10, px_spacing_mm=.2, frame=frame)
    rows, cols = np.indices(image.shape)
    world = origin + cols[..., None] * .2 * u + rows[..., None] * .2 * v
    np.testing.assert_allclose(image, world @ [3, 2, 1], rtol=1e-6)
