from pathlib import Path

import pytest
from fastapi import HTTPException

from server import storage, projects, cases
import numpy as np


def setup_roots(monkeypatch, tmp_path):
    root = tmp_path / "dicom"
    root.mkdir()
    monkeypatch.setenv("CAROTID_DICOM_ROOT", str(root))
    monkeypatch.setenv("CAROTID_DICOM_ROOT_ALIAS", "/data/dicom")
    return root


def test_native_paths_roundtrip_to_container_alias(monkeypatch, tmp_path):
    root = setup_roots(monkeypatch, tmp_path)
    original = root / "study"
    original.mkdir()
    assert storage.allowed_path("/data/dicom/study", "dicom") == original
    assert storage.portable_path(original, "dicom") == "/data/dicom/study"
    assert storage.allowed_path(str(original), "dicom") == original


@pytest.mark.parametrize("path", ["/data/dicom/../secret", "/data/dicom-other/study", "/etc/passwd"])
def test_alias_does_not_escape_data_root(monkeypatch, tmp_path, path):
    setup_roots(monkeypatch, tmp_path)
    with pytest.raises(HTTPException) as exc:
        storage.allowed_path(path, "dicom")
    assert exc.value.status_code == 403


def test_alias_symlink_escape_is_rejected(monkeypatch, tmp_path):
    root = setup_roots(monkeypatch, tmp_path)
    (root / "outside").symlink_to(tmp_path)
    with pytest.raises(HTTPException):
        storage.allowed_path("/data/dicom/outside/secret", "dicom")


def test_snapshot_keeps_portable_source(monkeypatch, tmp_path):
    root = setup_roots(monkeypatch, tmp_path)
    case = cases.create(np.zeros((2, 2, 2)), np.ones(3), np.zeros(3), "CT", str(root / "study"))
    try:
        assert projects.snapshot(case)["source_path"] == "/data/dicom/study"
    finally:
        cases._CASES.pop(case.case_id, None)
