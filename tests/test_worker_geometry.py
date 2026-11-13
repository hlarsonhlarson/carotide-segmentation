import numpy as np
import pytest
import SimpleITK as sitk


@pytest.mark.parametrize("direction", [
    np.eye(3),
    np.diag([-1., -1., 1.]),
    np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., -1.]]),
])
def test_canonical_mask_restores_asymmetric_landmarks(tmp_path, direction):
    nib = pytest.importorskip("nibabel")
    from scripts.local_autoseed_worker import align_labels

    labels = np.zeros((11, 13, 17), dtype=np.uint8)
    labels[2:4, 3:6, 8:12] = 1
    labels[7:9, 9:11, 2:4] = 4
    image = sitk.GetImageFromArray(labels)
    image.SetSpacing((.45, .65, 1.2))
    image.SetOrigin((32., -57., 90.))
    image.SetDirection(tuple(direction.ravel()))
    source = tmp_path / "source.nii.gz"
    canonical = tmp_path / "canonical.nii.gz"
    sitk.WriteImage(image, str(source))
    nib.save(nib.as_closest_canonical(nib.load(source)), canonical)
    restored = align_labels(canonical, source)
    np.testing.assert_array_equal(restored, labels)
