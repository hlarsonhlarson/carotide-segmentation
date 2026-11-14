"""Display corrections preserve source voxel identity and saved annotations."""
import numpy as np
import pytest
import SimpleITK as sitk
from server import dicom_io


@pytest.mark.parametrize('diagonal, expected', [
    ([1, 1, 1], {'axial': (False, False), 'coronal': (False, False), 'sagittal': (False, False)}),
    ([1, 1, -1], {'axial': (False, False), 'coronal': (False, True), 'sagittal': (False, True)}),
    ([-1, -1, 1], {'axial': (True, True), 'coronal': (True, False), 'sagittal': (True, False)}),
])
def test_acquisition_direction_controls_display_flips(diagonal, expected):
    actual = dicom_io.display_flips(np.diag(diagonal).ravel())
    assert {axis: (flips['x'], flips['y']) for axis, flips in actual.items()} == expected


def test_descending_stack_preserves_pixels_and_voxel_geometry(tmp_path, monkeypatch):
    array = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6)
    image = sitk.GetImageFromArray(array)
    image.SetDirection((1., 0., 0., 0., 1., 0., 0., 0., -1.))
    image.SetSpacing((0.5, 0.75, 2.))
    image.SetOrigin((10., 20., 30.))
    monkeypatch.setattr(dicom_io, '_group_series', lambda path: {'test': {'files': ['test'], 'modality': 'CT'}})
    monkeypatch.setattr(dicom_io, '_read_image', lambda files: image)
    loaded, spacing, origin, modality, flips = dicom_io.load_series(str(tmp_path), 'test', include_display_flips=True)
    np.testing.assert_array_equal(loaded, array)
    np.testing.assert_array_equal(spacing, [2., 0.75, 0.5])
    np.testing.assert_array_equal(origin, [30., 20., 10.])
    assert modality == 'CT'
    # get_slice reverses Z rows for coronal PNGs; the viewer reverses their display
    # for descending acquisitions, so superior (voxel Z=0 here) is at the top.
    png_rows = loaded[::-1, 2, :]
    displayed = png_rows[::-1] if flips['coronal']['y'] else png_rows
    np.testing.assert_array_equal(displayed[0], array[0, 2, :])
    assert len(dicom_io.load_series(str(tmp_path), 'test')) == 4
