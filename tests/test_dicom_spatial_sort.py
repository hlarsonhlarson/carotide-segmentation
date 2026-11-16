from pydicom.dataset import Dataset

from server.dicom_io import _slice_sort_key


def slice_at(position, instance, orientation):
    ds = Dataset()
    ds.ImagePositionPatient = position
    ds.ImageOrientationPatient = orientation
    ds.InstanceNumber = instance
    return ds


def test_spatial_sort_ignores_reversed_instance_numbers():
    orientation = [1, 0, 0, 0, 1, 0]
    slices = [slice_at([0, 0, z], 10 - z, orientation) for z in (2, 0, 1)]
    assert [float(ds.ImagePositionPatient[2]) for ds in sorted(slices, key=_slice_sort_key)] == [0, 1, 2]


def test_sagittal_stack_with_constant_z_uses_normal_projection():
    orientation = [0, 1, 0, 0, 0, 1]
    slices = [slice_at([x, 0, 50], 10 - x, orientation) for x in (2, 0, 1)]
    assert [float(ds.ImagePositionPatient[0]) for ds in sorted(slices, key=_slice_sort_key)] == [0, 1, 2]


def test_missing_geometry_falls_back_to_instance():
    ds = Dataset()
    ds.InstanceNumber = 7
    assert _slice_sort_key(ds) == (1, 7)
