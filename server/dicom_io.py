"""Unified DICOM loading for the interactive viewer.

Replaces dicom2nifti (used by the older batch scripts) for this new code
path: dicom2nifti.convert_directory swallows per-series conversion errors
internally (bare except, no re-raise — see run_stl_test.py's convert_patient
for the workaround that required), and it only handles the plain
multi-file-series case anyway. mra_pipeline.py already proved SimpleITK
reads both DICOM shapes we actually see in real data directly: a plain
per-slice-file series (ImageSeriesReader) and a Siemens single-file enhanced
multi-frame series (ReadImage).

A real study folder routinely mixes many series in one place — one real
patient's folder had ~160 files spanning a localizer, a head TOF run, a neck
TOF run, and more, all flat in one directory (not pre-sorted into
subfolders). Silently guessing "the largest series" picked the wrong one
here (a head series instead of the neck one actually wanted) — still a
valid, correctly-reconstructed image, just not the one anyone meant to open.
So this lists distinct series (grouped by SeriesInstanceUID, which groups a
classic multi-file series into many files and leaves a Siemens enhanced
file as its own one-file series) and lets the caller pick one explicitly,
rather than guessing.
"""
import tempfile
from pathlib import Path

import numpy as np
import pydicom
import SimpleITK as sitk


def _slice_sort_key(ds):
    """Sort by patient-space slice-normal position, including oblique scans.

    InstanceNumber is an acquisition identifier, not a geometric coordinate.
    Use it only when spatial metadata is unavailable.
    """
    position = getattr(ds, "ImagePositionPatient", None)
    orientation = getattr(ds, "ImageOrientationPatient", None)
    if position is not None and orientation is not None:
        position = np.asarray(position, dtype=float)
        orientation = np.asarray(orientation, dtype=float)
        if position.shape == (3,) and orientation.shape == (6,):
            normal = np.cross(orientation[:3], orientation[3:])
            length = np.linalg.norm(normal)
            if np.isfinite(position).all() and np.isfinite(length) and length > 1e-8:
                return (0, float(np.dot(position, normal / length)))
    instance_number = getattr(ds, "InstanceNumber", None)
    if instance_number is not None:
        return (1, int(instance_number))
    return (2, str(getattr(ds, "filename", "")))


def _group_series(path: Path) -> dict:
    """Returns {series_uid: {"description": str, "modality": str, "files": [str, ...]}}
    for every DICOM file found under `path`, each series' files in spatial
    acquisition order. Reads headers only (stop_before_pixels) — cheap even
    over ~160 files."""
    groups: dict[str, dict] = {}
    for f in sorted(path.rglob("*")):
        if not f.is_file():
            continue
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True)
        except Exception:
            continue
        uid = getattr(ds, "SeriesInstanceUID", None)
        if uid is None:
            continue
        if uid not in groups:
            groups[uid] = {
                "description": str(getattr(ds, "SeriesDescription", "") or "").strip(),
                "modality": str(getattr(ds, "Modality", "") or "").strip(),
                "files": [],
            }
        groups[uid]["files"].append((_slice_sort_key(ds), str(f)))

    for g in groups.values():
        g["files"].sort(key=lambda item: item[0])
        g["files"] = [f for _, f in g["files"]]
    return groups


def list_series(path_str: str):
    """Returns [{series_uid, description, modality, n_files}, ...] for the
    study folder at `path_str`, largest (by file count) first."""
    path = Path(path_str).expanduser()
    if not path.exists():
        raise FileNotFoundError(str(path))
    groups = _group_series(path)
    out = [
        {
            "series_uid": uid,
            "description": g["description"],
            "modality": g["modality"],
            "n_files": len(g["files"]),
        }
        for uid, g in groups.items()
    ]
    out.sort(key=lambda s: s["n_files"], reverse=True)
    return out


def _validate_stack_geometry(headers):
    """Reject stacks that cannot be represented by one regular orthogonal grid.

    DICOM decimal-string rounding is allowed (0.01 mm or 1% of slice step).
    Tilted/sheared stacks require resampling, not silently averaged spacing.
    """
    if len(headers) < 2:
        return
    positions, orientations, spacings, sizes = [], [], [], []
    for ds in headers:
        if int(getattr(ds, "NumberOfFrames", 1)) != 1:
            raise ValueError("Multiple multi-frame DICOM files are not a supported volume")
        try:
            position = np.asarray(ds.ImagePositionPatient, dtype=float)
            orientation = np.asarray(ds.ImageOrientationPatient, dtype=float)
            spacing = np.asarray(ds.PixelSpacing, dtype=float)
            size = (int(ds.Rows), int(ds.Columns))
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError("DICOM stack is missing spatial geometry metadata") from exc
        if (position.shape != (3,) or orientation.shape != (6,) or spacing.shape != (2,)
                or not all(np.isfinite(a).all() for a in (position, orientation, spacing))
                or np.any(spacing <= 0) or min(size) <= 0):
            raise ValueError("Invalid DICOM spatial geometry")
        basis = orientation.reshape(2, 3)
        if not np.allclose(basis @ basis.T, np.eye(2), atol=1e-4, rtol=0):
            raise ValueError("DICOM orientation must have orthonormal direction cosines")
        positions.append(position)
        orientations.append(orientation)
        spacings.append(spacing)
        sizes.append(size)
    if not np.allclose(orientations, orientations[0], atol=1e-4, rtol=0):
        raise ValueError("DICOM series mixes slice orientations; choose a consistent stack")
    if len(set(sizes)) != 1 or not np.allclose(spacings, spacings[0], atol=1e-5, rtol=1e-4):
        raise ValueError("DICOM series mixes image dimensions or pixel spacings")
    normal = np.cross(orientations[0][:3], orientations[0][3:])
    normal /= np.linalg.norm(normal)
    positions = np.asarray(positions)
    projected = positions @ normal
    order = np.argsort(projected)
    steps = np.diff(projected[order])
    if np.any(steps <= 1e-3):
        raise ValueError("DICOM series contains duplicate slice positions")
    step = float(np.median(steps))
    tolerance = max(.01, .01 * step)
    if np.any(abs(steps - step) > tolerance):
        raise ValueError("DICOM slice spacing is nonuniform or slices are missing")
    # Per-step checks alone permit accumulated drift; check the entire grid.
    expected = positions[order[0]] + np.arange(len(headers))[:, None] * step * normal
    if np.max(np.linalg.norm(positions[order] - expected, axis=1)) > tolerance:
        raise ValueError("DICOM positions form a tilted, sheared or irregular grid; resampling is required")
    if not (np.all(np.diff(projected) > 0) or np.all(np.diff(projected) < 0)):
        raise ValueError("DICOM slices are not in spatial order")


def _validate_multiframe_geometry(ds):
    """Validate enhanced DICOM functional groups before ITK flattens them."""
    count = int(getattr(ds, "NumberOfFrames", 1))
    if count <= 1:
        return
    frames = getattr(ds, "PerFrameFunctionalGroupsSequence", [])
    shared = getattr(ds, "SharedFunctionalGroupsSequence", [])
    if len(frames) != count:
        raise ValueError("Multi-frame DICOM requires per-frame spatial geometry")

    def value(frame, sequence, attribute):
        for group in (frame, *shared):
            items = getattr(group, sequence, [])
            if items and hasattr(items[0], attribute):
                return getattr(items[0], attribute)
        raise ValueError(f"Multi-frame DICOM is missing {attribute}")

    headers = []
    for frame in frames:
        header = pydicom.dataset.Dataset()
        header.Rows, header.Columns = ds.Rows, ds.Columns
        header.ImagePositionPatient = value(frame, "PlanePositionSequence", "ImagePositionPatient")
        header.ImageOrientationPatient = value(frame, "PlaneOrientationSequence", "ImageOrientationPatient")
        header.PixelSpacing = value(frame, "PixelMeasuresSequence", "PixelSpacing")
        headers.append(header)
    _validate_stack_geometry(headers)


def _read_image(files: list[str]):
    if len(files) > 1:
        _validate_stack_geometry([pydicom.dcmread(f, stop_before_pixels=True) for f in files])
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(files)
        return reader.Execute()
    _validate_multiframe_geometry(pydicom.dcmread(files[0], stop_before_pixels=True))
    return sitk.ReadImage(files[0])


def display_flips(direction):
    """Correct acquisition-axis reversals without changing stored voxel indices.

    ITK direction is row-major XYZ in patient LPS. PNG coronal/sagittal
    rows already reverse voxel Z; that assumes Z increases superiorly.
    Flip the display when an acquisition axis has the opposite direction.
    This does not resample oblique acquisitions into anatomical planes.
    """
    matrix = np.asarray(direction).reshape(3, 3)
    reversed_axes = [bool(matrix[i, i] < 0) for i in range(3)]
    x, y, z = reversed_axes
    return {
        "axial": {"x": x, "y": y},
        "coronal": {"x": x, "y": z},
        "sagittal": {"x": y, "y": z},
    }


def load_series(path_str: str, series_uid: str | None = None, *, include_display_flips=False,
                include_geometry=False):
    """Returns (array, spacing_mm, origin_mm, modality) for one series in the
    DICOM study at `path_str` (a folder). array axis order is (z, y, x),
    matching SimpleITK's GetArrayFromImage convention used throughout this
    codebase (see mra_pipeline.load_tof_volume); spacing_mm/origin_mm are
    reordered to match.

    If `series_uid` is omitted, falls back to the largest series by file
    count — fine for a folder holding one obvious series, but callers that
    care which series they get (any real multi-series study) should call
    list_series first and pass series_uid explicitly.
    """
    path = Path(path_str).expanduser()
    if not path.exists():
        raise FileNotFoundError(str(path))

    groups = _group_series(path)
    if not groups:
        raise FileNotFoundError(f"no DICOM files found under {path}")

    if series_uid is not None:
        if series_uid not in groups:
            raise KeyError(f"series_uid {series_uid!r} not found under {path}")
        chosen = groups[series_uid]
    else:
        chosen = max(groups.values(), key=lambda g: len(g["files"]))

    img = _read_image(chosen["files"])

    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    if img.GetDimension() != 3 or arr.ndim != 3:
        raise ValueError("Select a three-dimensional scalar DICOM volume")
    spacing_mm = np.asarray(img.GetSpacing())[::-1]
    origin_mm = np.asarray(img.GetOrigin())[::-1]
    direction = np.asarray(img.GetDirection()).reshape(3, 3)
    if (not np.isfinite(spacing_mm).all() or np.any(spacing_mm <= 0)
            or not np.isfinite(origin_mm).all()
            or not np.allclose(direction.T @ direction, np.eye(3), atol=1e-4, rtol=0)):
        raise ValueError("DICOM volume has invalid spacing, origin or direction")
    modality = chosen["modality"] or "MR"

    result = (arr, spacing_mm, origin_mm, modality)
    if include_display_flips:
        result = (*result, display_flips(img.GetDirection()))
    if include_geometry:
        result = (*result, np.asarray(img.GetDirection()).reshape(3, 3))
    return result


def to_nifti_tempfile(array: np.ndarray, spacing_mm, origin_mm) -> Path:
    """Writes `array` (z,y,x axis order, as returned by load_series) to a
    temporary .nii.gz for TotalSegmentator, which requires a NIfTI file
    path. Only called when auto-seeding is actually requested — the
    interactive viewer itself never touches NIfTI."""
    img = sitk.GetImageFromArray(array)
    img.SetSpacing(tuple(float(v) for v in spacing_mm[::-1]))
    img.SetOrigin(tuple(float(v) for v in origin_mm[::-1]))
    fd, tmp_path = tempfile.mkstemp(suffix=".nii.gz")
    import os
    os.close(fd)
    sitk.WriteImage(img, tmp_path)
    return Path(tmp_path)
