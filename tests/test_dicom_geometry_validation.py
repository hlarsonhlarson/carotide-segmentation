import numpy as np
import pytest
from pydicom.dataset import Dataset

from server.dicom_io import _validate_stack_geometry, _validate_multiframe_geometry


def stack():
    angle = .42
    u = np.array([np.cos(angle), 0, np.sin(angle)])
    v = np.array([0., 1., 0.])
    n = np.cross(u, v)
    result = []
    for i in range(5):
        ds = Dataset()
        ds.ImagePositionPatient = (np.array([30., -10., 50.]) + i * .625 * n).tolist()
        ds.ImageOrientationPatient = [*u, *v]
        ds.PixelSpacing = [.4375, .4375]
        ds.Rows, ds.Columns = 512, 512
        result.append(ds)
    return result


def test_valid_oblique_stack_and_rounding():
    headers = stack()[::-1]
    headers[2].ImagePositionPatient = [round(float(x), 4) for x in headers[2].ImagePositionPatient]
    _validate_stack_geometry(headers)


@pytest.mark.parametrize("problem, message", [
    ("duplicate", "duplicate"), ("missing_slice", "nonuniform"),
    ("orientation", "orientations"), ("pixel_spacing", "pixel spacings"),
    ("dimensions", "dimensions"), ("missing_metadata", "missing spatial"),
    ("shear", "sheared"), ("invalid_basis", "orthonormal"),
])
def test_reject_unrepresentable_geometry(problem, message):
    headers = stack()
    if problem == "duplicate":
        headers[2].ImagePositionPatient = headers[1].ImagePositionPatient
    elif problem == "missing_slice":
        del headers[2]
    elif problem == "orientation":
        headers[2].ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    elif problem == "pixel_spacing":
        headers[2].PixelSpacing = [.5, .5]
    elif problem == "dimensions":
        headers[2].Rows = 256
    elif problem == "missing_metadata":
        del headers[2].ImagePositionPatient
    elif problem == "shear":
        for i, ds in enumerate(headers):
            ds.ImagePositionPatient[1] += i * .1
    elif problem == "invalid_basis":
        headers[2].ImageOrientationPatient = [1, 0, 0, 1, 0, 0]
    with pytest.raises(ValueError, match=message):
        _validate_stack_geometry(headers)


def test_enhanced_shared_geometry_and_per_frame_positions():
    headers = stack()
    ds = Dataset()
    ds.NumberOfFrames = len(headers)
    ds.Rows, ds.Columns = 512, 512
    shared = Dataset()
    shared.PlaneOrientationSequence = [headers[0]]
    shared.PixelMeasuresSequence = [headers[0]]
    ds.SharedFunctionalGroupsSequence = [shared]
    frames = []
    for header in headers:
        frame = Dataset()
        frame.PlanePositionSequence = [header]
        frames.append(frame)
    ds.PerFrameFunctionalGroupsSequence = frames
    _validate_multiframe_geometry(ds)
    # A multi-temporal enhanced file cannot be flattened into a spatial grid.
    frames[2].PlanePositionSequence = [headers[1]]
    with pytest.raises(ValueError, match="duplicate"):
        _validate_multiframe_geometry(ds)


def test_scrambled_frame_order_rejected():
    headers = stack()
    headers[1], headers[2] = headers[2], headers[1]
    with pytest.raises(ValueError, match="spatial order"):
        _validate_stack_geometry(headers)
