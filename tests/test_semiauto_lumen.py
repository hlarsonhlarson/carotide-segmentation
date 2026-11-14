"""Synthetic checks for the semi-automatic CTA lumen rules.

These do not estimate clinical accuracy. They check that the implementation
keeps stenosis, bulb and tortuosity, refuses to bridge a gap, does not cut
on the operator points, and reports surface distance plus minimum area.
"""
import numpy as np
import pytest
import SimpleITK as sitk

from carotid_seg.semiauto import LumenParams, segment_lumen
from carotid_seg.semiauto.geometry import physical_to_voxel


SOFT = 35.0
LUMEN = 320.0
VEIN = 260.0
CALCIUM = 980.0
BONE = 1200.0


def _image(shape_zyx, spacing_xyz=(0.8, 0.8, 1.0), origin=(0.0, 0.0, 0.0), direction=None):
    array = np.full(shape_zyx, SOFT, dtype=np.float32)
    image = sitk.GetImageFromArray(array)
    image.SetSpacing(tuple(spacing_xyz))
    image.SetOrigin(tuple(origin))
    if direction is not None:
        image.SetDirection(tuple(direction))
    return image, array


def _write(array, image):
    out = sitk.GetImageFromArray(array)
    out.CopyInformation(image)
    return out


def _segment_distance(points, a, b):
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom == 0:
        return np.linalg.norm(points - a, axis=1)
    t = np.clip(((points - a) @ ab) / denom, 0.0, 1.0)
    return np.linalg.norm(points - (a + t[:, None] * ab), axis=1)


def _fill_capsule(array, image, a, b, radius, value):
    a, b = np.asarray(a, float), np.asarray(b, float)
    spacing = np.array(image.GetSpacing())
    origin = np.array(image.GetOrigin())
    size = np.array(image.GetSize())
    indexes = []
    for point in (a, b):
        indexes.append(np.array(image.TransformPhysicalPointToContinuousIndex(tuple(float(v) for v in point))))
    pad = radius / spacing + 2
    lo = np.maximum(0, np.floor(np.min(indexes, axis=0) - pad).astype(int))
    hi = np.minimum(size - 1, np.ceil(np.max(indexes, axis=0) + pad).astype(int))
    xs = np.arange(lo[0], hi[0] + 1)
    ys = np.arange(lo[1], hi[1] + 1)
    zs = np.arange(lo[2], hi[2] + 1)
    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
    physical = np.stack([
        origin[0] + xx * spacing[0],
        origin[1] + yy * spacing[1],
        origin[2] + zz * spacing[2],
    ], axis=-1)
    direction = np.array(image.GetDirection()).reshape(3, 3)
    if not np.allclose(direction, np.eye(3)):
        centered = np.stack([xx * spacing[0], yy * spacing[1], zz * spacing[2]], axis=-1)
        rotated = centered @ direction.T
        physical = origin + rotated
    flat = physical.reshape(-1, 3)
    inside = _segment_distance(flat, a, b) <= radius
    coords = np.stack([zz, yy, xx], axis=-1).reshape(-1, 3)[inside]
    for z, y, x in coords:
        array[int(z), int(y), int(x)] = value


def _fill_ellipse(array, image, z0, z1, center, rx, ry, value):
    spacing = np.array(image.GetSpacing())
    origin = np.array(image.GetOrigin())
    size = np.array(image.GetSize())
    p0 = np.array([center[0], center[1], z0])
    p1 = np.array([center[0], center[1], z1])
    indexes = [np.array(image.TransformPhysicalPointToContinuousIndex(tuple(float(v) for v in p))) for p in (p0, p1)]
    pad = np.array([rx, ry, 1.0]) / spacing + 2
    lo = np.maximum(0, np.floor(np.min(indexes, axis=0) - pad).astype(int))
    hi = np.minimum(size - 1, np.ceil(np.max(indexes, axis=0) + pad).astype(int))
    for x in range(lo[0], hi[0] + 1):
        for y in range(lo[1], hi[1] + 1):
            px = origin[0] + x * spacing[0]
            py = origin[1] + y * spacing[1]
            if ((px - center[0]) / rx) ** 2 + ((py - center[1]) / ry) ** 2 > 1:
                continue
            for z in range(lo[2], hi[2] + 1):
                pz = origin[2] + z * spacing[2]
                if z0 <= pz <= z1:
                    array[z, y, x] = value


def _ica_x(z):
    return 40.0 + 4.0 * np.sin((z - 80.0) / 10.0)


def _ica_points(z0, z1, step=2.0):
    zs = list(np.arange(z0, z1, step)) + [z1]
    return [(float(_ica_x(z)), 40.0, float(z)) for z in zs]


def _fill_path(array, image, points, radius, value):
    for start, end in zip(points, points[1:]):
        _fill_capsule(array, image, start, end, radius, value)


def build_carotid(gap_z=None):
    image, array = _image((150, 110, 110))
    _fill_capsule(array, image, (40, 40, 22), (40, 40, 78), 4.0, LUMEN)
    _fill_ellipse(array, image, 66, 80, (40, 40), 7.0, 4.2, LUMEN)
    _fill_path(array, image, _ica_points(80, 102), 2.5, LUMEN)
    _fill_path(array, image, _ica_points(102, 110), 1.35, LUMEN)
    _fill_path(array, image, _ica_points(110, 140), 2.5, LUMEN)
    # Parent vessel below the CCA. The seed is not this end.
    for z, radius in ((20, 5.5), (16, 8.0), (13, 11.0)):
        _fill_capsule(array, image, (40, 40, z + 3), (40, 40, z), radius, LUMEN)
    _fill_capsule(array, image, (40, 40, 80), (40, 54, 84), 2.0, LUMEN)
    _fill_capsule(array, image, (40, 54, 84), (40, 72, 84), 1.8, LUMEN)
    _fill_capsule(array, image, (40, 54, 84), (58, 54, 90), 1.8, LUMEN)
    _fill_capsule(array, image, (62, 40, 30), (62, 40, 110), 2.8, VEIN)
    _fill_capsule(array, image, (46.2, 40, 48), (46.2, 40, 52), 1.4, CALCIUM)
    _fill_canal(array, image, 128, 145, 3.4, 8.0, BONE)
    if gap_z is not None:
        _fill_capsule(array, image, (_ica_x(gap_z), 40, gap_z - 2), (_ica_x(gap_z), 40, gap_z + 2), 4.0, SOFT)
    return _write(array, image)


def _fill_canal(array, image, z0, z1, inner, outer, value):
    spacing = np.array(image.GetSpacing())
    origin = np.array(image.GetOrigin())
    size = np.array(image.GetSize())
    for z in range(size[2]):
        pz = origin[2] + z * spacing[2]
        if pz < z0 or pz > z1:
            continue
        cx, cy = _ica_x(pz), 40.0
        for y in range(size[1]):
            py = origin[1] + y * spacing[1]
            for x in range(size[0]):
                px = origin[0] + x * spacing[0]
                radial = np.hypot(px - cx, py - cy)
                if inner <= radial <= outer and array[z, y, x] < 200:
                    array[z, y, x] = value


@pytest.fixture(scope="module")
def carotid():
    image = build_carotid()
    result = segment_lumen(image, (40, 40, 42), (_ica_x(118), 40, 118), params=LumenParams())
    return image, result


def _labels(image, result):
    return sitk.GetArrayFromImage(result.labels), sitk.GetArrayFromImage(result.component)


def test_seed_model_refuses_a_point_without_local_contrast():
    image, _ = _image((30, 30, 30), spacing_xyz=(1, 1, 1))
    array = sitk.GetArrayFromImage(image)
    array[:] = 40
    image = _write(array, image)
    result = segment_lumen(image, (15, 15, 15), (15, 15, 20))
    assert result.status == "seed_failed"
    assert result.requests


def test_segments_do_not_end_at_the_seeds_and_keep_shape(carotid):
    image, result = carotid
    labels, component = _labels(image, result)
    cca = physical_to_voxel(image, (40, 40, 42))
    ica = physical_to_voxel(image, (_ica_x(118), 40, 118))
    assert labels[cca] == 1, result.requests
    assert labels[ica] == 2, result.requests
    cca_z = np.where(labels == 1)[0]
    ica_z = np.where(labels == 2)[0]
    # Spacing on z is 1 mm, so index approximates the physical z at origin 0.
    assert cca_z.min() < cca[0] - 8
    assert cca_z.min() > 6
    assert ica_z.max() > ica[0] + 6
    assert ica_z.max() < 132
    bulb = np.argwhere(labels[72] == 1)
    coords = bulb[:, ::-1].astype(float) * np.array(image.GetSpacing()[:2])
    cov = np.cov(coords.T)
    axes = np.sort(np.linalg.eigvalsh(cov))
    assert axes[-1] / axes[0] > 1.3
    areas = [sample["area_mm2"] for sample in result.samples if sample["label"] == 2 and sample["from_end_mm"] >= 3]
    assert min(areas) < 0.55 * np.median(areas)
    calcium = physical_to_voxel(image, (46.2, 40, 50))
    vein = physical_to_voxel(image, (62, 40, 70))
    canal = physical_to_voxel(image, (_ica_x(136), 40, 136))
    assert not component[calcium]
    assert not component[vein]
    assert not labels[canal]
    assert (labels == 3).any()
    child = physical_to_voxel(image, (40, 68, 84))
    assert labels[child] != 3


def test_moving_the_seed_does_not_move_the_ends(carotid):
    image, first = carotid
    second = segment_lumen(image, (40, 40, 52), (_ica_x(112), 40, 112), params=LumenParams())
    first_labels = sitk.GetArrayFromImage(first.labels)
    second_labels = sitk.GetArrayFromImage(second.labels)
    assert abs(int(np.where(first_labels == 1)[0].min()) - int(np.where(second_labels == 1)[0].min())) <= 3
    assert abs(int(np.where(first_labels == 2)[0].max()) - int(np.where(second_labels == 2)[0].max())) <= 3


def test_gap_is_not_bridged():
    image = build_carotid(gap_z=96)
    result = segment_lumen(image, (40, 40, 42), (_ica_x(118), 40, 118), params=LumenParams())
    assert result.status == "gap"
    gap = physical_to_voxel(image, (_ica_x(96), 40, 96))
    assert not sitk.GetArrayFromImage(result.component)[gap]
    assert "не связен" in result.requests[0]


def _expected_lumen():
    image, array = _image((150, 110, 110))
    array[:] = 0
    _fill_capsule(array, image, (40, 40, 18), (40, 40, 66), 4.0, 1)
    _fill_ellipse(array, image, 66, 78, (40, 40), 7.0, 4.2, 1)
    _fill_path(array, image, _ica_points(78, 102), 2.5, 1)
    _fill_path(array, image, _ica_points(102, 110), 1.35, 1)
    _fill_path(array, image, _ica_points(110, 127), 2.5, 1)
    _fill_capsule(array, image, (40, 40, 80), (40, 52, 83), 2.0, 1)
    return array.astype(bool)


def test_surface_distance_and_min_area_are_reported(carotid):
    image, result = carotid
    lumen = sitk.GetArrayFromImage(result.lumen).astype(bool)
    report = result.qa
    assert report["min_area_mm2"]["ica"]["area_mm2"] < report["min_area_mm2"]["cca"]["area_mm2"]
    assert report["dice_is_not_a_pass_criterion"] is True
    from carotid_seg.semiauto.qa import surface_errors
    errors = surface_errors(lumen, _expected_lumen(), image)
    # Mean distance tracks the wall. The larger HD95 on this phantom is the
    # mismatch of the end caps, which the reference only approximates.
    assert errors["mean_surface_mm"] < 1.2
    assert errors["hd95_mm"] < 6.0
    shifted = np.zeros_like(lumen)
    shifted[:-2] = lumen[2:]
    assert surface_errors(shifted, lumen, image)["mean_surface_mm"] > 0


def test_orientation_uses_physical_points():
    image, array = _image((40, 40, 40), spacing_xyz=(1, 1, 1), origin=(10, -5, 3),
                           direction=(0, -1, 0, 1, 0, 0, 0, 0, 1))
    # Tube along the image k axis in index space, expressed through physical points.
    start = image.TransformContinuousIndexToPhysicalPoint((20, 20, 5))
    end = image.TransformContinuousIndexToPhysicalPoint((20, 20, 34))
    _fill_capsule(array, image, start, end, 3.0, LUMEN)
    oriented = _write(array, image)
    cca = image.TransformContinuousIndexToPhysicalPoint((20, 20, 12))
    ica = image.TransformContinuousIndexToPhysicalPoint((20, 20, 28))
    result = segment_lumen(oriented, cca, ica, params=LumenParams(max_radius_factor=3.0, eca_max_length_mm=4))
    labels = sitk.GetArrayFromImage(result.labels)
    assert labels[12, 20, 20] == 1, result.requests
    assert labels[28, 20, 20] == 2, result.requests
    assert labels[8, 20, 20] == 1
    assert labels[32, 20, 20] == 2


def test_ambiguous_eca_waits_for_a_third_point():
    image, array = _image((80, 50, 70), spacing_xyz=(1, 1, 1))
    _fill_capsule(array, image, (25, 25, 10), (25, 25, 40), 3.0, LUMEN)
    _fill_capsule(array, image, (25, 25, 40), (25, 25, 70), 2.2, LUMEN)
    _fill_capsule(array, image, (25, 25, 40), (25, 45, 48), 2.0, LUMEN)
    _fill_capsule(array, image, (25, 25, 40), (45, 25, 48), 1.9, LUMEN)
    built = _write(array, image)
    params = LumenParams(eca_max_length_mm=12, max_radius_factor=3)
    waiting = segment_lumen(built, (25, 25, 20), (25, 25, 60), params=params)
    assert waiting.qa["status_notes"]["ambiguous_eca"] is True
    assert not (sitk.GetArrayFromImage(waiting.labels) == 3).any()
    chosen = segment_lumen(built, (25, 25, 20), (25, 25, 60), eca_point=(25, 36, 44), params=params)
    labels = sitk.GetArrayFromImage(chosen.labels)
    assert labels[physical_to_voxel(built, (25, 36, 44))] == 3
    assert labels[physical_to_voxel(built, (40, 25, 46))] != 3
