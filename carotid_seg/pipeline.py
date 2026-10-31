import logging
from pathlib import Path

import nibabel as nib
import numpy as np
from skimage.morphology import skeletonize

from . import totalseg_steps as ts
from .vesselness import vessel_lumen_mask, largest_component_touching, bridge_gap
from .treeanalysis import find_bifurcation_and_label
from .volumize import propagate_labels, qa_report

log = logging.getLogger("carotid_seg")

# final multi-label output codes
LABELS = {
    ("cca", "right"): 1, ("cca", "left"): 2,
    ("ica", "right"): 3, ("ica", "left"): 4,
    ("eca", "right"): 5, ("eca", "left"): 6,
}
ROI_MARGIN_MM = 20.0


def _load_canonical(path: Path):
    img = nib.as_closest_canonical(nib.load(str(path)))
    return np.asarray(img.dataobj), img.affine, img.header.get_zooms()[:3]


def _bbox_with_margin(mask: np.ndarray, spacing_mm, margin_mm: float, shape):
    idx = np.argwhere(mask)
    lo = idx.min(axis=0)
    hi = idx.max(axis=0) + 1
    margin_vox = np.ceil(margin_mm / np.asarray(spacing_mm)).astype(int)
    lo = np.maximum(lo - margin_vox, 0)
    hi = np.minimum(hi + margin_vox, np.asarray(shape))
    return tuple(slice(a, b) for a, b in zip(lo, hi))


def _process_side(ct, spacing_mm, cca_mask, ica_mask, side: str):
    """Returns (label_vol_full_shape, qa_dict) with values in {0,1,2,3} for
    none/cca/ica/eca, already placed back at full-volume shape/position.
    """
    full_label = np.zeros(ct.shape, dtype=np.uint8)
    if not cca_mask.any() or not ica_mask.any():
        log.warning("%s: missing CCA or ICA seed mask, skipping", side)
        return full_label, {"side": side, "bifurcation_found": False, "skipped": True}

    roi = _bbox_with_margin(cca_mask | ica_mask, spacing_mm, ROI_MARGIN_MM, ct.shape)
    ct_roi, cca_roi, ica_roi = ct[roi], cca_mask[roi], ica_mask[roi]

    lumen = vessel_lumen_mask(ct_roi, spacing_mm, cca_roi | ica_roi)
    tree_mask = largest_component_touching(lumen, cca_roi, ica_roi)
    if not tree_mask.any():
        lumen = bridge_gap(lumen, ct_roi, spacing_mm, cca_roi, ica_roi)
        tree_mask = largest_component_touching(lumen, cca_roi, ica_roi)
    if not tree_mask.any():
        log.warning("%s: CCA and ICA seeds did not join into one component", side)
        out = np.zeros(ct_roi.shape, dtype=np.uint8)
        out[cca_roi] = 1
        out[ica_roi] = 2
        full_label[roi] = out
        return full_label, {"side": side, "bifurcation_found": False, "skipped": False}

    skeleton = skeletonize(tree_mask)
    label_skel, found = find_bifurcation_and_label(skeleton, spacing_mm, cca_roi, ica_roi)

    if found:
        label_roi = propagate_labels(label_skel, tree_mask, spacing_mm)
    else:
        log.warning("%s: bifurcation not resolved, falling back to raw CCA/ICA seeds only", side)
        label_roi = np.zeros(tree_mask.shape, dtype=np.uint8)
        label_roi[cca_roi] = 1
        label_roi[ica_roi] = 2

    full_label[roi] = label_roi
    qa = qa_report(label_roi, spacing_mm, side)
    qa["bifurcation_found"] = found
    qa["skipped"] = False
    return full_label, qa


def cap_ica_at_landmark(final_label: np.ndarray, styloid_mask: np.ndarray, code: int,
                         margin_mm: float, spacing_mm):
    """Zero out `code` voxels above (more superior than, canonical axis 2)
    the styloid process tip + margin — an approximate skull-base-entry cap.
    Only applied if explicitly requested; the model's native extent is used
    otherwise.
    """
    if not styloid_mask.any():
        return final_label
    tip = styloid_mask.nonzero()[2].max()
    cutoff = tip + int(round(margin_mm / spacing_mm[2]))
    out = final_label.copy()
    out[:, :, cutoff + 1:][out[:, :, cutoff + 1:] == code] = 0
    return out


def process_case(ct_path: Path, output_dir: Path, work_dir: Path,
                  device: str = "gpu", cap_ica_at_skull_base: bool = False,
                  skull_base_margin_mm: float = 10.0) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    class_paths = ts.run_bootstrap_segmentations(ct_path, work_dir, device=device)

    ct, affine, spacing = _load_canonical(ct_path)

    def mask_for(name):
        if name not in class_paths:
            return np.zeros(ct.shape, dtype=bool)
        data, _, _ = _load_canonical(class_paths[name])
        return data.astype(bool)

    final_label = np.zeros(ct.shape, dtype=np.uint8)
    qa_all = {"vertebrae_found": [v for v in ts.NECK_VERTEBRAE if v in class_paths]}

    for side in ("right", "left"):
        cca = mask_for(f"common_carotid_artery_{side}")
        ica = mask_for(f"internal_carotid_artery_{side}")
        side_label, qa = _process_side(ct, spacing, cca, ica, side)
        qa_all[side] = qa

        for name, code in (("cca", LABELS[("cca", side)]),
                            ("ica", LABELS[("ica", side)]),
                            ("eca", LABELS[("eca", side)])):
            value = {"cca": 1, "ica": 2, "eca": 3}[name]
            final_label[side_label == value] = code

        if cap_ica_at_skull_base:
            styloid = mask_for(f"styloid_process_{side}")
            final_label = cap_ica_at_landmark(
                final_label, styloid, LABELS[("ica", side)], skull_base_margin_mm, spacing
            )

    out_path = output_dir / "carotid_segments.nii.gz"
    nib.save(nib.Nifti1Image(final_label, affine), str(out_path))
    qa_all["output"] = str(out_path)
    qa_all["label_legend"] = {f"{k[0]}_{k[1]}": v for k, v in LABELS.items()}
    return qa_all
