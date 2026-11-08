"""Checks that do not treat Dice as a pass.

Connectivity, unlabeled extensions of the grown component, surface distance
when a reference exists, and the minimum cross-sectional area of the lumen.
"""
import numpy as np
from scipy import ndimage as ndi

from .geometry import spacing_zyx


def surface_voxels(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return mask
    eroded = ndi.binary_erosion(mask, ndi.generate_binary_structure(3, 1))
    return mask & ~eroded


def surface_errors(prediction: np.ndarray, reference: np.ndarray, image) -> dict:
    """Symmetric distance between lumen surfaces, in millimetres, plus Dice.

    Dice is reported only as a secondary number. A smooth wrong surface can
    score well on Dice and still move the boundary.
    """
    pred = prediction.astype(bool)
    ref = reference.astype(bool)
    inter = np.logical_and(pred, ref).sum()
    denom = pred.sum() + ref.sum()
    dice = float(2 * inter / denom) if denom else 1.0
    pred_s = surface_voxels(pred)
    ref_s = surface_voxels(ref)
    if not pred_s.any() or not ref_s.any():
        return {"dice": dice, "mean_surface_mm": None, "hd95_mm": None}
    sampling = spacing_zyx(image)
    to_ref = ndi.distance_transform_edt(~ref_s, sampling=sampling)
    to_pred = ndi.distance_transform_edt(~pred_s, sampling=sampling)
    distances = np.concatenate([to_ref[pred_s], to_pred[ref_s]])
    return {
        "dice": dice,
        "mean_surface_mm": float(np.mean(distances)),
        "hd95_mm": float(np.percentile(distances, 95)),
    }


def connectivity(labels: np.ndarray) -> dict:
    structure = ndi.generate_binary_structure(3, 1)
    report = {}
    for value, name in ((1, "cca"), (2, "ica"), (3, "eca")):
        count = int(ndi.label(labels == value, structure=structure)[1])
        report[name] = {"components": count, "voxels": int((labels == value).sum())}
    return report


def extension_volume_mm3(component: np.ndarray, labels: np.ndarray, image) -> float:
    voxel = float(np.prod(spacing_zyx(image)))
    return float((component & (labels == 0)).sum() * voxel)


def min_area(samples: list, label: int | None = None) -> dict | None:
    chosen = samples
    if label is not None:
        chosen = [sample for sample in samples if sample["label"] == label]
    chosen = [sample for sample in chosen if sample["from_end_mm"] >= 3.0]
    if not chosen:
        return None
    sample = min(chosen, key=lambda item: item["area_mm2"])
    return {
        "area_mm2": sample["area_mm2"],
        "point_xyz": sample["point_xyz"],
        "label": sample["label"],
    }


def overlay_slice(ct: np.ndarray, labels: np.ndarray, uncertainty: np.ndarray, models, axis: int, index: int) -> np.ndarray:
    """RGB review slice. Window follows the seed contrast, not a fixed HU window."""
    sl = (slice(None),) * axis + (index,)
    hu = ct[sl].astype(float)
    lab = labels[sl]
    unsure = uncertainty[sl]
    if models:
        lumen = float(np.mean([model.lumen for model in models]))
        soft = float(np.mean([model.soft for model in models]))
    else:
        soft, lumen = np.percentile(hu, [10, 90])
    width = max(lumen - soft, 1.0)
    gray = np.clip((hu - (soft - 0.2 * width)) / (1.4 * width), 0, 1)
    rgb = np.stack([gray, gray, gray], axis=-1)
    colors = {
        1: np.array([0.86, 0.22, 0.18]),
        2: np.array([0.20, 0.45, 0.86]),
        3: np.array([0.15, 0.70, 0.32]),
    }
    for value, color in colors.items():
        rgb[lab == value] = 0.45 * rgb[lab == value] + 0.55 * color
    mark = (unsure > 0) & (lab == 0)
    rgb[mark] = 0.35 * rgb[mark] + 0.65 * np.array([0.90, 0.78, 0.15])
    return (rgb * 255).astype(np.uint8)


def summarize(component, labels, uncertainty, samples, image, connected: bool, reference=None) -> dict:
    areas = {
        "cca": min_area(samples, 1),
        "ica": min_area(samples, 2),
        "eca": min_area(samples, 3),
    }
    report = {
        "connected_seeds": connected,
        "labels": connectivity(labels),
        "unlabeled_extension_mm3": extension_volume_mm3(component, labels, image),
        "uncertain_mm3": float(((uncertainty > 0).sum()) * float(np.prod(spacing_zyx(image)))),
        "min_area_mm2": areas,
        "dice_is_not_a_pass_criterion": True,
    }
    if reference is not None:
        report["surface"] = surface_errors(labels > 0, reference, image)
    return report
