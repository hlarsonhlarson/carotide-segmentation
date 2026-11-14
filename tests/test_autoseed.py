import io

import numpy as np
import pytest
from fastapi import HTTPException

from server import cases
from server.app import _require_autoseed_review
from server import autoseed
from server.autoseed import paths_from_labels
from server import workflow


def test_paths_from_labels_extracts_editable_carotid_branches():
    labels = np.zeros((50, 50, 50), dtype=np.uint8)
    # Bifurcation voxel follows the pipeline convention: it belongs to CCA.
    labels[5:26, 25, 25] = 1
    labels[26:43, 25, 25] = 3
    labels[26, 26:43, 25] = 5

    paths = paths_from_labels(labels, (1.0, 1.0, 1.0))

    assert set(paths) == {("right", "cca_ica"), ("right", "cca_eca")}
    assert paths["right", "cca_ica"][0] == [5, 25, 25]
    assert paths["right", "cca_ica"][-1] == [42, 25, 25]
    assert paths["right", "cca_eca"][0] == [5, 25, 25]
    assert paths["right", "cca_eca"][-1] == [26, 42, 25]


def test_completed_automatic_draft_requires_explicit_review():
    case = cases.create(np.zeros((4, 4, 4)), np.ones(3), np.zeros(3), "CT", "synthetic")
    case.autoseed_job = {"id": "draft-1", "state": "done"}
    case.autoseed_review = {"job_id": "draft-1", "accepted": False}
    with pytest.raises(HTTPException, match="confirm"):
        _require_autoseed_review(case)
    case.autoseed_review["accepted"] = True
    _require_autoseed_review(case)
    cases._CASES.clear()


def test_source_geometry_edit_revokes_automatic_review_confirmation():
    case = cases.create(np.zeros((4, 4, 4)), np.ones(3), np.zeros(3), "CT", "synthetic")
    case.autoseed_job = {"id": "draft-1", "state": "done", "review_accepted": True}
    case.autoseed_review = {"job_id": "draft-1", "accepted": True}
    workflow.invalidate(case, "right", "surface")
    assert case.autoseed_review["accepted"] is False
    assert case.autoseed_job["review_accepted"] is False
    cases._CASES.clear()


def test_automatic_draft_keeps_generated_contours(monkeypatch):
    """A successful automatic path must retain its paired editable contour."""
    case = cases.create(np.zeros((4, 4, 4)), np.ones(3), np.zeros(3), "CT", "synthetic")
    job = {
        "id": "draft-1",
        "state": "queued",
        "revision": {side: dict(values) for side, values in case.revisions.items()},
        "review_required": True,
        "review_accepted": False,
    }
    case.autoseed_job = job
    labels = np.zeros(case.array.shape, dtype=np.uint8)
    payload = io.BytesIO()
    np.savez_compressed(payload, labels=labels, qa="{}")

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return payload.getvalue()

    key = ("right", "cca_ica")
    draft = {key: [{"point_mm": [1.0, 1.0, 1.0], "contour_px": [[1.0, 1.0], [2.0, 2.0]]}]}
    monkeypatch.setattr(autoseed.urllib.request, "urlopen", lambda *args, **kwargs: Response())
    monkeypatch.setattr(autoseed, "paths_from_labels", lambda *args: {key: [[1, 1, 1], [2, 2, 2]]})
    monkeypatch.setattr(autoseed, "draft_series", lambda *args: (draft, {}))

    autoseed._run(case, job)

    assert case.segmentation_series[key] == draft[key]
    cases._CASES.clear()
