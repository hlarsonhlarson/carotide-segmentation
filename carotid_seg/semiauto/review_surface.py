"""Compare one semi-automatic segmentation with an existing lumen surface.

The surface is a prior reconstruction in the same physical frame, not an
independent clinical reference. The script drops slices that break a uniform
DICOM stack, crops around the surface, places the two lumen points on that
surface's own center, and writes overlays next to the output directory.
"""
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import vtk
from vtk.util.numpy_support import vtk_to_numpy

from .geometry import index_to_physical, physical_to_voxel, spacing_zyx
from .params import LumenParams
from .qa import surface_voxels
from .segment import review_slice, segment_lumen


def load_contiguous_ct(folder: Path) -> sitk.Image:
    """Load the largest run of CT slices whose spacing stays under 3 mm."""
    import pydicom

    rows = []
    for path in Path(folder).iterdir():
        if not path.is_file():
            continue
        try:
            ds = pydicom.dcmread(str(path), stop_before_pixels=True)
        except Exception:
            continue
        if str(getattr(ds, "Modality", "")) != "CT":
            continue
        ipp = getattr(ds, "ImagePositionPatient", None)
        iop = getattr(ds, "ImageOrientationPatient", None)
        if ipp is None or iop is None:
            continue
        orientation = np.array(iop, dtype=float)
        normal = np.cross(orientation[:3], orientation[3:])
        position = float(np.dot(np.array(ipp, dtype=float), normal))
        rows.append((position, str(path)))
    rows.sort()
    if not rows:
        raise RuntimeError(f"В {folder} нет CT-срезов.")
    runs = []
    start = 0
    for index in range(1, len(rows)):
        if rows[index][0] - rows[index - 1][0] > 3.0:
            runs.append(rows[start:index])
            start = index
    runs.append(rows[start:])
    run = max(runs, key=len)
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames([path for _, path in run])
    image = reader.Execute()
    image.dropped_slices = len(rows) - len(run)
    return image


def read_stl(path: Path):
    reader = vtk.vtkSTLReader()
    reader.SetFileName(str(path))
    reader.Update()
    poly = reader.GetOutput()
    verts = vtk_to_numpy(poly.GetPoints().GetData())
    return poly, np.asarray(verts, dtype=float)


def center_at(verts: np.ndarray, fraction: float) -> np.ndarray:
    z = verts[:, 2]
    target = float(z.min() + fraction * (z.max() - z.min()))
    slab = verts[np.abs(z - target) < 3.0]
    if len(slab) < 20:
        slab = verts[np.argsort(np.abs(z - target))[:80]]
    return slab.mean(axis=0)


def snap_into_contrast(image: sitk.Image, rough: np.ndarray) -> np.ndarray:
    """Move a surface-center estimate into the nearest contrast core.

    The raw center of an STL slab often sits on the wall or between branches.
    This search is only for placing the two review points. It is not the
    segmentation rule, and it can still land in the wrong bright structure.
    """
    from scipy import ndimage as ndi

    array = sitk.GetArrayViewFromImage(image)
    index = np.array(image.TransformPhysicalPointToContinuousIndex(tuple(float(v) for v in rough)))
    spacing = np.array(image.GetSpacing(), dtype=float)
    radius_mm = 7.0
    reach = np.ceil(radius_mm / spacing).astype(int)
    center = np.rint(index).astype(int)
    low = np.maximum(center - reach, 0)
    high = np.minimum(center + reach + 1, np.array(array.shape[::-1]))
    # array is z,y,x; index is x,y,z
    x0, y0, z0 = low
    x1, y1, z1 = high
    sub = np.asarray(array[z0:z1, y0:y1, x0:x1])
    if sub.size == 0:
        return rough
    soft = float(np.percentile(sub, 20))
    candidate = (sub >= soft + 150.0) & (sub <= soft + 450.0)
    labels, count = ndi.label(candidate)
    best = None
    for label in range(1, count + 1):
        coords = np.argwhere(labels == label)
        if len(coords) < 12:
            continue
        # The component centroid often sits on partial volume between the
        # artery and a neighbouring enhanced vein. The seed is a bright
        # voxel of this component, the one nearest the rough centre.
        values = sub[coords[:, 0], coords[:, 1], coords[:, 2]]
        bright = coords[values >= np.percentile(values, 70)]
        nearest = None
        for coord in bright:
            point = np.array(image.TransformIndexToPhysicalPoint((
                int(x0 + coord[2]), int(y0 + coord[1]), int(z0 + coord[0])
            )))
            distance = float(np.linalg.norm(point - rough))
            if distance > radius_mm:
                continue
            if nearest is None or distance < nearest[0]:
                nearest = (distance, point)
        if nearest is not None and (best is None or nearest[0] < best[0]):
            best = nearest
    return rough if best is None else best[1]


def crop_around(image: sitk.Image, verts: np.ndarray, margin_mm: float) -> sitk.Image:
    low = verts.min(axis=0) - margin_mm
    high = verts.max(axis=0) + margin_mm
    i0 = np.array(image.TransformPhysicalPointToContinuousIndex(tuple(low.tolist())))
    i1 = np.array(image.TransformPhysicalPointToContinuousIndex(tuple(high.tolist())))
    start = np.floor(np.minimum(i0, i1)).astype(int)
    stop = np.ceil(np.maximum(i0, i1)).astype(int)
    size = np.array(image.GetSize())
    start = np.clip(start, 0, size - 1)
    stop = np.clip(stop, 0, size - 1)
    width = np.maximum(stop - start + 1, 1)
    return sitk.RegionOfInterest(image, width.tolist(), start.tolist())


def distances_to_surface(poly, points: np.ndarray) -> np.ndarray:
    implicit = vtk.vtkImplicitPolyDataDistance()
    implicit.SetInput(poly)
    distances = np.empty(len(points), dtype=float)
    for index, point in enumerate(points):
        distances[index] = abs(float(implicit.EvaluateFunction(point.tolist())))
    return distances


def compare(poly, verts, image, labels) -> dict:
    mask = labels > 0
    surface = surface_voxels(mask)
    coords = np.argwhere(surface)
    if len(coords) == 0:
        return {"pred_to_surface_mean_mm": None, "pred_to_surface_hd95_mm": None,
                "surface_to_pred_mean_mm": None, "surface_to_pred_hd95_mm": None}
    physical = np.vstack([index_to_physical(image, voxel) for voxel in coords])
    pred_to_mesh = distances_to_surface(poly, physical)
    from scipy import ndimage as ndi
    outside = ndi.distance_transform_edt(~mask, sampling=spacing_zyx(image))
    mesh_to_pred = []
    for point in verts[:: max(1, len(verts) // 8000)]:
        voxel = physical_to_voxel(image, point)
        if voxel is None:
            continue
        mesh_to_pred.append(float(outside[voxel]))
    mesh_to_pred = np.array(mesh_to_pred) if mesh_to_pred else np.array([np.nan])
    return {
        "pred_to_surface_mean_mm": float(np.mean(pred_to_mesh)),
        "pred_to_surface_hd95_mm": float(np.percentile(pred_to_mesh, 95)),
        "surface_to_pred_mean_mm": float(np.nanmean(mesh_to_pred)),
        "surface_to_pred_hd95_mm": float(np.nanpercentile(mesh_to_pred, 95)),
        "n_surface_voxels": int(len(coords)),
    }


def write_overlays(result, image, out_dir: Path, stem: str):
    labels = sitk.GetArrayFromImage(result.labels)
    present = np.argwhere(labels > 0)
    if len(present) == 0:
        return []
    z_values = np.unique(present[:, 0])
    chosen = [int(z) for z in np.linspace(z_values.min(), z_values.max(), 6)]
    written = []
    for z in chosen:
        rgb = review_slice(result, image, axis=0, index=z)
        path = out_dir / f"{stem}_z{z:03d}.png"
        sitk.WriteImage(sitk.GetImageFromArray(rgb, isVector=True), str(path))
        written.append(str(path))
    return written


def review_stl(ct_dir: Path, stl_path: Path, out_dir: Path, params: LumenParams | None = None) -> dict:
    image = load_contiguous_ct(ct_dir)
    poly, verts = read_stl(stl_path)
    cropped = crop_around(image, verts, margin_mm=18.0)
    cca = snap_into_contrast(cropped, center_at(verts, 0.22))
    ica = snap_into_contrast(cropped, center_at(verts, 0.72))
    result = segment_lumen(cropped, cca.tolist(), ica.tolist(), params=params or LumenParams())
    labels = sitk.GetArrayFromImage(result.labels)
    surface = compare(poly, verts, cropped, labels)
    out_dir.mkdir(parents=True, exist_ok=True)
    overlays = write_overlays(result, cropped, out_dir, stl_path.stem)
    sitk.WriteImage(result.labels, str(out_dir / f"{stl_path.stem}_labels.nii.gz"))
    sitk.WriteImage(result.uncertainty, str(out_dir / f"{stl_path.stem}_uncertainty.nii.gz"))
    report = {
        "stl": stl_path.name,
        "status": result.status,
        "requests": result.requests,
        "dropped_slices": int(getattr(image, "dropped_slices", 0)),
        "crop_size": list(cropped.GetSize()),
        "spacing_xyz": list(cropped.GetSpacing()),
        "seeds_xyz": {"cca": cca.tolist(), "ica": ica.tolist()},
        "qa": result.qa,
        "agreement_with_prior_surface": surface,
        "overlays": overlays,
        "prior_surface_is_not_an_independent_reference": True,
    }
    (out_dir / f"{stl_path.stem}_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json))
    return report


def _json(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    raise TypeError(type(value))
