"""Native macOS worker for TotalSegmentator.

Run outside Docker so the web application stays lightweight:
    .venv/bin/python scripts/local_autoseed_worker.py
"""
from __future__ import annotations

import io
import json
import os
import tempfile
import threading
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from starlette.concurrency import run_in_threadpool

app = FastAPI(title="Carotid local autoseed worker")


def _cpu_threads() -> int:
    available = os.cpu_count() or 1
    try:
        threads = int(os.environ.get("CAROTID_AUTOSEED_THREADS", str(available)))
    except ValueError:
        threads = available
    return max(1, min(threads, available))


CPU_THREADS = _cpu_threads()
_model_lock = threading.Lock()
_torch_configured = False


def _json_default(value):
    """Convert NumPy QA scalars returned by the segmentation pipeline."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _configure_torch_threads() -> None:
    global _torch_configured
    if _torch_configured:
        return
    # TotalSegmentator imports PyTorch lazily. Configure it before inference
    # so the worker uses all cores unless explicitly overridden. This remains
    # lazy: importing PyTorch before Uvicorn starts makes /health unavailable
    # for several minutes on a cold macOS launch.
    import torch

    torch.set_num_threads(CPU_THREADS)
    torch.set_num_interop_threads(min(CPU_THREADS, torch.get_num_interop_threads()))
    _torch_configured = True


def align_labels(label_path, source_path):
    """Restore the source voxel grid including flips and oblique direction."""
    reference = sitk.ReadImage(str(source_path))
    label_image = sitk.ReadImage(str(label_path))
    aligned = sitk.Resample(label_image, reference, sitk.Transform(),
                            sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    return sitk.GetArrayFromImage(aligned)


@app.get("/health")
def health():
    return {
        "ok": True,
        "device": os.environ.get("CAROTID_AUTOSEED_DEVICE", "cpu"),
        "threads": CPU_THREADS,
        "total_task_mode": "fast",
        "busy": _model_lock.locked(),
    }


@app.post("/segment")
async def segment(request: Request):
    payload = await request.body()
    if not payload:
        raise HTTPException(400, "NIfTI payload is required")
    if not _model_lock.acquire(blocking=False):
        raise HTTPException(409, "A segmentation is already running")
    try:
        return await run_in_threadpool(_segment_payload, payload)
    finally:
        _model_lock.release()


def _segment_payload(payload):
    from carotid_seg.pipeline import process_case

    with tempfile.TemporaryDirectory(prefix="carotid-local-autoseed-") as temp:
        root = Path(temp)
        source, output, work = root / "input.nii.gz", root / "output", root / "work"
        source.write_bytes(payload)
        try:
            _configure_torch_threads()
            qa = process_case(source, output, work, device=os.environ.get("CAROTID_AUTOSEED_DEVICE", "cpu"))
            # The pipeline canonicalizes NIfTI (and may flip axes). Resample
            # its labels back onto the exact grid sent by the web app rather
            # than merely transposing array axes; otherwise a valid mask can
            # be drawn at the wrong anatomical location in the viewer.
            labels = align_labels(output / "carotid_segments.nii.gz", source)
        except Exception as exc:
            raise HTTPException(500, f"automatic segmentation failed: {exc}") from exc
    result = io.BytesIO()
    np.savez_compressed(result, labels=labels, qa=json.dumps(qa, ensure_ascii=False, default=_json_default))
    return Response(result.getvalue(), media_type="application/octet-stream")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("CAROTID_AUTOSEED_WORKER_PORT", "8011")))
