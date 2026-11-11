"""Native macOS worker for TotalSegmentator.

Run outside Docker so the web application stays lightweight:
    .venv/bin/python scripts/local_autoseed_worker.py
"""
from __future__ import annotations

import io
import json
import os
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response

from carotid_seg.pipeline import process_case

app = FastAPI(title="Carotid local autoseed worker")


@app.get("/health")
def health():
    return {"ok": True, "device": os.environ.get("CAROTID_AUTOSEED_DEVICE", "cpu")}


@app.post("/segment")
async def segment(request: Request):
    payload = await request.body()
    if not payload:
        raise HTTPException(400, "NIfTI payload is required")
    with tempfile.TemporaryDirectory(prefix="carotid-local-autoseed-") as temp:
        root = Path(temp)
        source, output, work = root / "input.nii.gz", root / "output", root / "work"
        source.write_bytes(payload)
        try:
            qa = process_case(source, output, work, device=os.environ.get("CAROTID_AUTOSEED_DEVICE", "cpu"))
            labels = np.asarray(nib.load(str(output / "carotid_segments.nii.gz")).dataobj, dtype=np.uint8)
        except Exception as exc:
            raise HTTPException(500, f"automatic segmentation failed: {exc}") from exc
    result = io.BytesIO()
    np.savez_compressed(result, labels=labels, qa=json.dumps(qa, ensure_ascii=False))
    return Response(result.getvalue(), media_type="application/octet-stream")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("CAROTID_AUTOSEED_WORKER_PORT", "8011")))
