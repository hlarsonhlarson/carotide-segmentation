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
    """Spatial order, not filename order. VERIFIED this was a real bug on
    real data: files were previously appended in `sorted(path.rglob("*"))`
    (alphabetical by filename) order and handed straight to
    ImageSeriesReader.SetFileNames, which trusts the given order instead of
    re-sorting — unlike GetGDCMSeriesFileNames, which does sort spatially,
    but only works on a whole directory, not an arbitrary file list grouped
    by series UID the way this project needs (a real study folder mixes
    many series flat in one directory). One real patient's filenames
    happened to be alphabetically consistent with InstanceNumber, masking
    this; nothing guarantees that in general, and the symptom otherwise is
    exactly "slices jump around non-sequentially" when scrolling — prefer
    InstanceNumber (what actually defines acquisition order), falling back
    to the slice-normal position, then filename, for the rare file missing
    both.
    """
    instance_number = getattr(ds, "InstanceNumber", None)
    if instance_number is not None:
        return (0, int(instance_number))
    position = getattr(ds, "ImagePositionPatient", None)
    if position is not None:
        return (1, float(position[2]))
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


def _read_image(files: list[str]):
    if len(files) > 1:
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(files)
        return reader.Execute()
    return sitk.ReadImage(files[0])


def load_series(path_str: str, series_uid: str | None = None):
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
    spacing_mm = np.asarray(img.GetSpacing())[::-1]
    origin_mm = np.asarray(img.GetOrigin())[::-1]
    modality = chosen["modality"] or "MR"

    return arr, spacing_mm, origin_mm, modality


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
