"""Semi-automatic lumen segmentation of the cervical carotid on CTA.

Operator input is two points inside the lumen (CCA and distal cervical ICA)
and, if the ECA is ambiguous, a third point inside the proximal ECA. The
points identify the vessel. They are not the ends of the segments.

The lumen is the face-connected component whose intensity matches the local
contrast around those points. Plaque, calcium, vein and bone stay out because
they fail that local comparison or are not connected, not because of a fixed
HU gate. Vesselness is not used to grow the mask. Centerlines and CCA/ICA/ECA
labels are derived only after the mask exists.

This does not claim clinical accuracy. Synthetic checks live in
``tests/test_semiauto_lumen.py``. A real series still needs visual review of
the overlays, connectivity, unlabeled extensions and minimum lumen area.
"""
from dataclasses import dataclass, field

import numpy as np
import SimpleITK as sitk

from .appearance import fit_seed_model
from .branches import label_branches
from .geometry import image_from_array, physical_to_voxel
from .grow import component_contains_both, grow_lumen, paint
from .params import LumenParams
from .qa import overlay_slice, summarize

# Uncertainty bits. A voxel may carry more than one.
BOUNDARY = 1
BLOOM = 2
GAP = 4
CANAL = 8
ECA_AMBIGUOUS = 16
UNLABELED = 32
SUBOCCLUSION = 64
FOV = 128
VOLUME_CAP = 256


@dataclass
class SegmentationResult:
    status: str
    requests: list
    lumen: sitk.Image
    labels: sitk.Image
    uncertainty: sitk.Image
    component: sitk.Image
    qa: dict
    models: list = field(default_factory=list)
    samples: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def _blank(reference, dtype):
    array = np.zeros(sitk.GetArrayViewFromImage(reference).shape, dtype=dtype)
    return image_from_array(array, reference)


def _mark_boundary(array, accepted, scores, params, uncertainty):
    for zyx, score in scores.items():
        if score >= params.membership_min + params.boundary_band:
            continue
        z, y, x = zyx
        for dz, dy, dx in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
            nxt = (z + dz, y + dy, x + dx)
            if nxt not in accepted:
                uncertainty[zyx] |= BOUNDARY
                break


def _mark_bloom(array, accepted, rejected, models, params, uncertainty):
    """Intermediate voxels between lumen and a brighter core, not the core itself."""
    for zyx, reason in rejected.items():
        if reason != "calcium":
            continue
        z, y, x = zyx
        touches_lumen = False
        for dz, dy, dx in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
            if (z + dz, y + dy, x + dx) in accepted:
                touches_lumen = True
                break
        if touches_lumen:
            uncertainty[zyx] |= BLOOM


def segment_lumen(image: sitk.Image, cca_point, ica_point, eca_point=None, params: LumenParams | None = None,
                  reference: sitk.Image | None = None) -> SegmentationResult:
    """Segment one cervical carotid.

    ``cca_point``, ``ica_point`` and optional ``eca_point`` are physical (x, y, z)
    millimetres in the same frame as ``image``. Outputs copy that geometry.
    """
    params = params or LumenParams()
    array = np.asarray(sitk.GetArrayFromImage(image), dtype=np.float32)
    cca = fit_seed_model(image, array, cca_point, "cca", params)
    ica = fit_seed_model(image, array, ica_point, "ica", params)
    models = [cca, ica]
    failed = [model for model in models if not model.reliable]
    if failed:
        message = " ".join(model.message for model in failed)
        return _failed(image, "seed_failed", [message], models)

    grown = grow_lumen(image, array, models, params)
    cca_voxel = physical_to_voxel(image, cca_point)
    ica_voxel = physical_to_voxel(image, ica_point)
    connected = component_contains_both(grown.accepted, cca_voxel, ica_voxel)
    component = paint(array.shape, grown.accepted)
    uncertainty = np.zeros(array.shape, dtype=np.uint16)
    _mark_boundary(array, grown.accepted, grown.scores, params, uncertainty)
    _mark_bloom(array, grown.accepted, grown.rejected, models, params, uncertainty)
    requests = list(grown.notes)
    if np.any(uncertainty & BLOOM):
        requests.append("Есть размытая граница с кальцием (blooming). Эти воксели не включены в просвет; нужна локальная проверка.")
    if grown.fov_exit:
        for zyx in grown.accepted:
            z, y, x = zyx
            if z in (0, array.shape[0] - 1) or y in (0, array.shape[1] - 1) or x in (0, array.shape[2] - 1):
                uncertainty[zyx] |= FOV
    if grown.volume_capped:
        uncertainty[component.astype(bool)] |= VOLUME_CAP

    if not connected:
        for voxel in (cca_voxel, ica_voxel):
            uncertainty[voxel] |= GAP
        requests.insert(0, "Просвет между точками ОСА и ВСА не связен. Разрыв не соединяется. Нужна локальная коррекция.")
        qa = summarize(component.astype(bool), np.zeros_like(component), uncertainty, [], image, False)
        return SegmentationResult(
            "gap", requests,
            _blank(image, np.uint8),
            _blank(image, np.uint8),
            image_from_array(uncertainty, image),
            image_from_array(component, image),
            qa, models, [],
        )

    if params.use_vesselness:
        _annotate_vesselness(array, component.astype(bool), uncertainty, params)

    labeled = label_branches(image, array, component.astype(bool), cca_voxel, ica_voxel, eca_point, models, params)
    uncertainty |= labeled.uncertainty
    requests.extend(labeled.requests)
    lumen = (labeled.labels > 0).astype(np.uint8)
    ref = None if reference is None else sitk.GetArrayFromImage(reference).astype(bool)
    qa = summarize(component.astype(bool), labeled.labels, uncertainty, labeled.samples, image, True, ref)
    qa["status_notes"] = labeled.stats
    status = "ok" if labeled.labels[cca_voxel] == 1 and labeled.labels[ica_voxel] == 2 else "needs_review"
    if any("НСА" in item or "канал" in item or "субокклюз" in item or "bloom" in item.lower() for item in requests):
        if status == "ok":
            status = "needs_review"
    # A missing canal or a requested ECA click is an incomplete anatomical end,
    # not a failed lumen boundary. Keep needs_review rather than ok.
    return SegmentationResult(
        status, requests,
        image_from_array(lumen, image),
        image_from_array(labeled.labels.astype(np.uint8), image),
        image_from_array(uncertainty, image),
        image_from_array(component, image),
        qa, models, labeled.samples,
    )


def _annotate_vesselness(array, mask, uncertainty, params: LumenParams):
    """Low Frangi response flags an already marginal boundary voxel. It does not edit the mask."""
    from skimage.filters import frangi

    coords = np.argwhere(mask)
    if len(coords) == 0:
        return
    z0, y0, x0 = np.maximum(coords.min(axis=0) - 2, 0)
    z1, y1, x1 = np.minimum(coords.max(axis=0) + 3, np.array(array.shape))
    crop = array[z0:z1, y0:y1, x0:x1]
    if crop.size > 3_000_000:
        return
    response = frangi(crop, sigmas=(1.0, 2.0, 3.0), black_ridges=False)
    peak = float(response.max()) if response.size else 0.0
    if peak <= 0:
        return
    response /= peak
    boundary = uncertainty[z0:z1, y0:y1, x0:x1] & BOUNDARY
    weak = (response < params.vesselness_min) & (boundary > 0)
    uncertainty[z0:z1, y0:y1, x0:x1][weak] |= BOUNDARY


def _failed(image, status, requests, models):
    return SegmentationResult(
        status, requests,
        _blank(image, np.uint8), _blank(image, np.uint8), _blank(image, np.uint16), _blank(image, np.uint8),
        {"connected_seeds": False, "dice_is_not_a_pass_criterion": True}, models, [],
    )


def review_slice(result: SegmentationResult, source: sitk.Image, axis: int, index: int) -> np.ndarray:
    ct = sitk.GetArrayFromImage(source)
    labels = sitk.GetArrayFromImage(result.labels)
    uncertainty = sitk.GetArrayFromImage(result.uncertainty)
    return overlay_slice(ct, labels, uncertainty, result.models, axis, index)
