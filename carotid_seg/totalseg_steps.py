from pathlib import Path
import os
import shutil

from totalsegmentator.python_api import totalsegmentator

CCA_CLASSES = ["common_carotid_artery_left", "common_carotid_artery_right"]
ICA_CLASSES = ["internal_carotid_artery_left", "internal_carotid_artery_right"]
LANDMARK_CLASSES = ["styloid_process_left", "styloid_process_right"]
NECK_VERTEBRAE = [f"vertebrae_C{i}" for i in range(1, 8)]


def _thread_limit() -> int:
    """Use every available CPU by default; an environment value can override it."""
    available = os.cpu_count() or 1
    try:
        requested = int(os.environ.get("CAROTID_AUTOSEED_THREADS", str(available)))
    except ValueError:
        requested = available
    return max(1, min(requested, available))


def run_bootstrap_segmentations(ct_path: Path, work_dir: Path, device: str = "gpu") -> dict:
    """Run the two open-source TotalSegmentator tasks that give us seed masks.

    Returns {class_name: Path} for every requested class that TotalSegmentator
    actually produced (a class can be missing if it's outside the scan's FOV).
    """
    work_dir.mkdir(parents=True, exist_ok=True)

    total_out = work_dir / "total"
    # This stage only supplies an editable CCA seed and vertebral landmarks.
    # The normal ``total`` task runs five 1.5-mm models; its native fast mode
    # uses one 3-mm model and is substantially quicker for a draft proposal.
    # ICA still comes from the dedicated high-resolution head/neck model below.
    threads = _thread_limit()
    totalsegmentator(
        str(ct_path), str(total_out),
        task="total", roi_subset=CCA_CLASSES + NECK_VERTEBRAE,
        device=device, fast=True, quiet=True,
        nr_thr_resamp=threads, nr_thr_saving=threads,
    )

    # ``headneck_bones_vessels`` has no Fast variant in TotalSegmentator. It is
    # therefore the sole high-resolution pass, required for ICA seed masks.
    headneck_out = work_dir / "headneck_bones_vessels"
    totalsegmentator(
        str(ct_path), str(headneck_out),
        task="headneck_bones_vessels",
        device=device, fast=False, quiet=True,
        nr_thr_resamp=threads, nr_thr_saving=threads,
    )

    paths = {}
    for name in CCA_CLASSES + NECK_VERTEBRAE:
        p = total_out / f"{name}.nii.gz"
        if p.exists():
            paths[name] = p
    for name in ICA_CLASSES + LANDMARK_CLASSES:
        p = headneck_out / f"{name}.nii.gz"
        if p.exists():
            paths[name] = p
    return paths


def cleanup_work_dir(work_dir: Path):
    shutil.rmtree(work_dir, ignore_errors=True)
