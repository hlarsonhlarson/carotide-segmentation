import numpy as np
from scipy import ndimage as ndi
from skimage.measure import label as cc_label


def propagate_labels(label_skel: np.ndarray, tree_mask: np.ndarray, spacing_mm) -> np.ndarray:
    """Assign every voxel in `tree_mask` the label of the nearest labeled
    skeleton voxel (anisotropic nearest-neighbor via a Euclidean distance
    transform's index map) — a cheap, dependency-free stand-in for a
    marker-based watershed, appropriate since segments are non-overlapping
    tubes around their own centerline.
    """
    labeled_points = label_skel > 0
    _, indices = ndi.distance_transform_edt(
        ~labeled_points, sampling=spacing_mm, return_indices=True
    )
    nearest_label = label_skel[tuple(indices)]
    out = np.zeros_like(label_skel)
    out[tree_mask] = nearest_label[tree_mask]
    return out


def qa_report(label_vol: np.ndarray, spacing_mm, side: str) -> dict:
    voxel_vol_mm3 = float(np.prod(spacing_mm))
    report = {"side": side}
    for name, val in (("cca", 1), ("ica", 2), ("eca", 3)):
        mask = label_vol == val
        n_components = cc_label(mask, connectivity=3).max() if mask.any() else 0
        report[f"{name}_volume_mm3"] = float(mask.sum()) * voxel_vol_mm3
        report[f"{name}_components"] = int(n_components)
        report[f"{name}_multi_component"] = n_components > 1
    return report
