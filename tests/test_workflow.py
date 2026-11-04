import json
import threading
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from vtk import vtkPolyData

from server import app as backend, cases, projects, workflow, su2_run
from server.coordinates import voxel_to_world, world_to_voxel, surface_xyz
from server.reslice import sample_oblique_slice


@pytest.fixture
def client():
    cases._CASES.clear()
    with TestClient(backend.app) as client:
        yield client
    cases._CASES.clear()


@pytest.fixture
def case():
    c = cases.create(np.ones((10, 10, 10), dtype=np.float32), np.array([2., 1., .5]),
                     np.array([20., 30., 40.]), 'CT', '/synthetic', series_uid='1.2.3')
    c.centerline_points['right', 'cca_ica'] = [[1, 2, 3], [6, 2, 3]]
    c.control_points['right', 'cca_ica'] = [[1, 2, 3], [6, 2, 3]]
    c.segmentation_series['right', 'cca_ica'] = [
        dict(point_mm=[22., 32., 41.5], tangent_mm=[1., 0., 0.], contour_mm=[[22., 32., 41.5]] * 3)
    ]
    return c


def fill_outputs(case):
    case.built_surfaces['right'] = b'stl'
    case.su2_mesh_jobs['right'] = dict(state='done', vtu=b'vtu', vtp=b'vtp', face_map={'1': 'wall'}, diag={})
    case.su2_run_jobs['right'] = dict(state='done')
    case.su2_postprocess_jobs['right'] = dict(state='done', result={'value': 1}, frames={
        'wss_mag_pa': np.ones((2, 3)), 'pressure_pa': np.ones((2, 3))})


def test_edit_invalidates_and_autosaves_without_touching_other_side(client, case, tmp_path):
    assert client.post(f'/cases/{case.case_id}/project', json={'path': str(tmp_path)}).status_code == 200
    fill_outputs(case)
    case.built_surfaces['left'] = b'other'
    assert client.post(f'/cases/{case.case_id}/save/project').status_code == 200
    r = client.delete(f'/cases/{case.case_id}/centerline/right/cca_ica')
    assert r.status_code == 200, r.text
    assert ('right', 'cca_ica') not in case.segmentation_series
    assert 'right' not in case.built_surfaces
    assert not case.su2_mesh_jobs and not case.su2_run_jobs and not case.su2_postprocess_jobs
    assert case.built_surfaces['left'] == b'other'
    saved = json.loads((tmp_path / 'project.json').read_text())
    assert not any(key.startswith('right:') for key in saved['artifacts'])
    assert saved['sides']['right']['branches']['cca_ica']['centerline_points'] == []
    assert client.get(f'/cases/{case.case_id}/workflow/right').json()['meshing'] == 'stale'


def test_contour_and_params_invalidate_correct_descendants(client, case):
    fill_outputs(case)
    r = client.post(f'/cases/{case.case_id}/segment/right/cca_ica/point/0/manual',
                    json={'contour_px': [[70, 70], [75, 70], [75, 75]]})
    assert r.status_code == 200, r.text
    assert not case.built_surfaces and not case.su2_mesh_jobs and not case.su2_postprocess_jobs
    fill_outputs(case)
    r = client.post(f'/cases/{case.case_id}/segment/right/su2_params', json=backend.SU2_PARAM_DEFAULTS)
    assert r.status_code == 200, r.text
    assert case.built_surfaces and case.su2_mesh_jobs
    assert not case.su2_run_jobs and not case.su2_postprocess_jobs


def test_round_trip_and_reopen_reuses_case(client, case, tmp_path, monkeypatch):
    assert client.post(f'/cases/{case.case_id}/project', json={'path': str(tmp_path)}).status_code == 200
    fill_outputs(case)
    assert client.post(f'/cases/{case.case_id}/save/project').status_code == 200
    before = projects.snapshot(case)
    cases._CASES.clear()
    monkeypatch.setattr(backend.dicom_io, 'load_series', lambda *args: (case.array, case.spacing_mm, case.origin_mm, 'CT'))
    result = client.post('/projects/open', json={'path': str(tmp_path)})
    assert result.status_code == 200, result.text
    restored = cases.get(result.json()['case_id'])
    assert restored.project_id == case.project_id
    assert projects.snapshot(restored)['sides'] == before['sides']
    assert restored.built_surfaces == case.built_surfaces
    assert restored.su2_postprocess_jobs['right']['result'] == {'value': 1}
    assert client.post('/projects/open', json={'path': str(tmp_path)}).json()['case_id'] == restored.case_id


def test_refuses_different_project_and_unrelated_folder(client, case, tmp_path):
    (tmp_path / 'notes.txt').write_text('keep me')
    assert client.post(f'/cases/{case.case_id}/project', json={'path': str(tmp_path)}).status_code == 409
    folder = tmp_path / 'study'
    assert client.post(f'/cases/{case.case_id}/project', json={'path': str(folder)}).status_code == 200
    other = cases.create(case.array, case.spacing_mm, case.origin_mm, 'CT', '/other', '9')
    original = (folder / 'project.json').read_bytes()
    assert client.post(f'/cases/{other.case_id}/project', json={'path': str(folder)}).status_code == 409
    assert (folder / 'project.json').read_bytes() == original


def test_atomic_write_failure_preserves_snapshot(tmp_path, monkeypatch):
    path = tmp_path / 'project.json'
    path.write_bytes(b'original')
    monkeypatch.setattr(projects.os, 'replace', lambda *a: (_ for _ in ()).throw(OSError('disk error')))
    with pytest.raises(OSError):
        projects.atomic_write(path, b'new')
    assert path.read_bytes() == b'original'
    assert list(tmp_path.iterdir()) == [path]


def test_cancel_before_solver_launch(monkeypatch):
    job = {'state': 'running'}
    su2_run.cancel_run(job)
    monkeypatch.setattr(su2_run.subprocess, 'Popen', lambda *a, **k: pytest.fail('cancelled solver launched'))
    su2_run.run_in_background(job, None, None, None)
    assert job['state'] == 'cancelled'


def test_old_mesh_completion_cannot_resurrect_results(client, case, monkeypatch):
    started, finish = threading.Event(), threading.Event()
    threads = []
    real_thread = threading.Thread
    def track(*args, **kwargs):
        thread = real_thread(*args, **kwargs)
        threads.append(thread)
        return thread
    monkeypatch.setattr(backend.threading, 'Thread', track)
    monkeypatch.setattr(backend.build_surface, 'build_surface', lambda *a: (vtkPolyData(), {}))
    def slow_mesh(*args, **kwargs):
        started.set()
        assert finish.wait(3)
        return b'vtu', b'vtp', {1: 'wall'}, {}
    monkeypatch.setattr(backend.su2_export, 'build_su2_mesh', slow_mesh)
    assert client.post(f'/cases/{case.case_id}/segment/right/su2_mesh').status_code == 200
    assert started.wait(3)
    client.delete(f'/cases/{case.case_id}/centerline/right/cca_ica')
    finish.set()
    for t in threads: t.join(3)
    assert 'right' not in case.su2_mesh_jobs


def test_coordinates_and_oblique_pixel_origin():
    p = np.array([[1., 2., 3.], [4., 5., 6.]])
    spacing, origin = np.array([2., 3., 4.]), np.array([10., 20., 30.])
    np.testing.assert_allclose(world_to_voxel(voxel_to_world(p, spacing, origin), spacing, origin), p)
    assert surface_xyz([1, 2, 3]) == [3, 2, 1]
    z, y, x = np.indices((10, 10, 10))
    array = (100 * z + 10 * y + x).astype(float)
    plane, u, v, start = sample_oblique_slice(array, [1, 1, 1], [0, 0, 0], [5, 5, 5], [1, 0, 0], 4, .5)
    for row in range(4):
        for col in range(4):
            point = start + col * .5 * u + row * .5 * v
            assert plane[row, col] == pytest.approx(np.dot(point, [100, 10, 1]))


@pytest.mark.parametrize('field,value', [('cardiac_cycle_s', 0), ('ica_fraction', 1), ('psv_cms', -1), ('edv_cms', 100)])
def test_invalid_params_rejected(client, case, field, value):
    params = dict(backend.SU2_PARAM_DEFAULTS, **{field: value})
    assert client.post(f'/cases/{case.case_id}/segment/right/su2_params', json=params).status_code == 422


def test_changed_geometry_refuses_restore(case, tmp_path):
    manifest = projects.snapshot(case)
    manifest['geometry']['origin_mm'][0] += 1
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as e:
        projects.restore(case, manifest, tmp_path)
    assert e.value.status_code == 409


def test_migration_preserves_original_and_requires_review(client, case, tmp_path, monkeypatch):
    from server.migrate_project import migrate
    source = tmp_path / 'old'
    source.mkdir()
    (source / 'project.json').write_text(json.dumps({'source_path': case.source_path, 'series_uid': case.series_uid}))
    (source / 'right_centerline.json').write_text(json.dumps({'cca_ica': {'control_points': [[1, 2, 3]], 'centerline_points': [[1, 2, 3], [6, 2, 3]]}}))
    original = (source / 'project.json').read_bytes()
    monkeypatch.setattr(backend.dicom_io, 'load_series', lambda *a: (case.array, case.spacing_mm, case.origin_mm, 'CT'))
    destination = tmp_path / 'new'
    migrate(source, destination)
    assert (source / 'project.json').read_bytes() == original
    data = json.loads((destination / 'project.json').read_text())
    assert data['warnings'] and data['migration']['review_required']
    assert not data['artifacts']
    with pytest.raises(ValueError):
        migrate(source, source)


def test_stale_artifact_is_not_restored(case, tmp_path):
    data = projects.snapshot(case)
    data['artifacts'] = {'right:cfd': {'revisions': {'geometry': 99, 'params': 0}}}
    projects.restore(case, data, tmp_path)
    assert not case.su2_postprocess_jobs
    assert case.project_warnings


def test_no_path_traversal_in_artifacts(tmp_path):
    with pytest.raises(ValueError):
        projects.artifact_path(tmp_path, '../outside')


def test_actual_surface_and_tetrahedral_mesh():
    from server import build_surface, su2_export
    angles = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    branch = []
    for z in np.arange(8., 29., 2.):
        branch.append({'point_mm': [z, 16., 16.], 'tangent_mm': [1., 0., 0.],
                       'radius_mm': 3., 'contour_mm': [[z, 16 + 3 * np.cos(a), 16 + 3 * np.sin(a)] for a in angles]})
    series = {'cca_ica': branch, 'cca_eca': []}
    surface, diag = build_surface.build_surface((40, 32, 32), np.ones(3), np.zeros(3), series)
    assert surface is not None, diag
    vtu, vtp, faces, mesh_diag = su2_export.build_su2_mesh(surface, series)
    assert mesh_diag['n_volume_elements'] > 0
    assert set(faces.values()) == {'wall', 'cca_inlet', 'ica_outlet'}
    text = su2_export.vtu_vtp_bytes_to_su2_mesh_text(vtu, vtp, faces)
    assert 'NDIME= 3' in text


def test_invalid_saved_coordinates_are_rejected(case, tmp_path):
    data = projects.snapshot(case)
    data['sides']['right']['branches']['cca_ica']['control_points'] = [[100, 0, 0]]
    with pytest.raises(ValueError, match='outside'):
        projects.restore(case, data, tmp_path)
