"""Atomic project snapshots. Large artifacts are immutable; project.json commits them."""
import io
import json
import os
import tempfile
import uuid
from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from . import cases, workflow, storage

router = APIRouter()
FORMAT_VERSION = 2


def atomic_write(path, data):
    path = Path(path)
    fd, name = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def snapshot(case):
    return {
        'format_version': FORMAT_VERSION, 'project_id': case.project_id,
        'source_path': case.source_path, 'series_uid': case.series_uid,
        'geometry': {'shape': list(case.array.shape), 'spacing_mm': case.spacing_mm.tolist(),
                     'origin_mm': case.origin_mm.tolist(), 'coordinate_system': 'zyx-mm'},
        'revisions': case.revisions,
        'stale': {s: sorted(v) for s, v in case.stale.items()},
        'sides': {side: {
            'branches': {branch: {
                'control_points': case.control_points.get((side, branch), []),
                'centerline_points': case.centerline_points.get((side, branch), []),
                'segmentation': case.segmentation_series.get((side, branch), []),
            } for branch in workflow.BRANCHES},
            'params': case.su2_params.get(side),
        } for side in workflow.SIDES},
        'artifacts': case.saved_artifacts, 'warnings': case.project_warnings,
    }


def autosave(case):
    if case.project_dir:
        folder = Path(case.project_dir)
        current = json.loads((folder / 'project.json').read_text())
        if current.get('project_id') != case.project_id:
            raise HTTPException(409, 'project ownership changed; reopen the project')
        try:
            atomic_write(folder / 'project.json', json.dumps(snapshot(case), allow_nan=False, indent=2).encode())
        except OSError as e:
            raise HTTPException(507, f'Autosave failed; edits remain in memory. Save again before closing: {e}') from e


def artifact_path(folder, name):
    path = (folder / name).resolve()
    if not path.is_relative_to(folder.resolve()) or path == folder.resolve():
        raise ValueError('artifact path must stay inside the project folder')
    return path


def save_artifact(case, side, stage):
    folder = Path(case.project_dir)
    def write(suffix, data):
        name = f'{side}_{stage}_{uuid.uuid4().hex}{suffix}'
        atomic_write(folder / name, data)
        return name
    record = {'revisions': dict(case.revisions[side])}
    if stage == 'surface':
        data = case.built_surfaces.get(side)
        if data is None:
            raise HTTPException(409, 'build a current surface first')
        record['stl'] = write('.stl', data)
    elif stage == 'mesh':
        job = case.su2_mesh_jobs.get(side, {})
        if job.get('state') != 'done':
            raise HTTPException(409, 'build a current mesh first')
        record.update(vtu=write('.vtu', job['vtu']), vtp=write('.vtp', job['vtp']),
                      face_map=job['face_map'], diag=job.get('diag', {}))
    else:
        job = case.su2_postprocess_jobs.get(side, {})
        if job.get('state') != 'done':
            raise HTTPException(409, 'extract current CFD results first')
        record['result'] = write('.json', json.dumps(job['result'], allow_nan=False).encode())
        buf = io.BytesIO()
        np.savez_compressed(buf, **job['frames'])
        record['frames'] = write('.npz', buf.getvalue())
        record['provenance'] = job.get('provenance', {})
    case.saved_artifacts[f'{side}:{stage}'] = record
    autosave(case)


class ProjectRequest(BaseModel):
    path: str


@router.post('/cases/{case_id}/project')
@workflow.locked
def set_project_dir(case_id: str, req: ProjectRequest):
    case = cases.get(case_id)
    folder = storage.allowed_path(req.path, 'projects')
    folder.mkdir(parents=True, exist_ok=True)
    manifest = folder / 'project.json'
    if manifest.exists():
        existing = json.loads(manifest.read_text())
        if existing.get('project_id') != case.project_id:
            raise HTTPException(409, 'This folder already contains a project. Open it, or choose a new empty folder.')
    elif any(folder.iterdir()):
        raise HTTPException(409, 'Choose a dedicated empty folder for this study.')
    old = case.project_dir
    if old and Path(old).resolve() != folder:
        raise HTTPException(409, 'Project already has a folder; copy the closed project folder to move it.')
    case.project_dir = str(folder)
    try:
        atomic_write(manifest, json.dumps(snapshot(case), allow_nan=False, indent=2).encode())
    except Exception:
        case.project_dir = old
        raise
    return {'project_dir': case.project_dir}


@router.get('/cases/{case_id}/project')
@workflow.locked
def get_project_dir(case_id: str):
    return {'project_dir': cases.get(case_id).project_dir}


@router.post('/cases/{case_id}/save/project')
@workflow.locked
def save_project(case_id: str):
    case = cases.get(case_id)
    if not case.project_dir:
        raise HTTPException(400, 'choose a project folder first')
    autosave(case)
    for side in workflow.SIDES:
        for stage, ready in [('surface', side in case.built_surfaces),
                             ('mesh', case.su2_mesh_jobs.get(side, {}).get('state') == 'done'),
                             ('cfd', case.su2_postprocess_jobs.get(side, {}).get('state') == 'done')]:
            if ready and f'{side}:{stage}' not in case.saved_artifacts:
                save_artifact(case, side, stage)
    return {'saved_to': case.project_dir}


@router.post('/cases/{case_id}/save/{step}/{side}')
@workflow.locked
def save_step(case_id: str, step: str, side: str):
    case = cases.get(case_id)
    if side not in workflow.SIDES or step not in ('centerline', 'segmentation', 'surface', 'mesh', 'su2_params', 'cfd_results'):
        raise HTTPException(400, 'unknown step or side')
    if not case.project_dir:
        raise HTTPException(400, 'choose a project folder first')
    if step in ('surface', 'mesh', 'cfd_results'):
        save_artifact(case, side, 'cfd' if step == 'cfd_results' else step)
    else:
        autosave(case)
    return {'saved_to': case.project_dir}


def validate_branch(data, shape):
    def points(values, size):
        if len(values) == 0:
            return np.empty((0, size))
        array = np.asarray(values, dtype=float)
        if array.ndim != 2 or array.shape[1] != size or not np.isfinite(array).all():
            raise ValueError('invalid annotation coordinates')
        return array
    for name in ('control_points', 'centerline_points'):
        coordinates = points(data[name], 3)
        if np.any(coordinates < 0) or np.any(coordinates >= np.asarray(shape)):
            raise ValueError('path lies outside source image')
    for point in data['segmentation']:
        points([point['point_mm'], point['tangent_mm']], 3)
        points(point.get('contour_mm', []), 3)
        points(point.get('contour_px', []), 2)
        radius = point.get('radius_mm')
        if radius is not None and (not np.isfinite(radius) or radius <= 0):
            raise ValueError('invalid contour radius')


def restore(case, manifest, folder):
    if manifest.get('format_version') != FORMAT_VERSION:
        raise HTTPException(400, 'unsupported project format')
    geometry = manifest['geometry']
    if (geometry['shape'] != list(case.array.shape)
            or not np.allclose(geometry['spacing_mm'], case.spacing_mm)
            or not np.allclose(geometry['origin_mm'], case.origin_mm)
            or geometry.get('coordinate_system') != 'zyx-mm'):
        raise HTTPException(409, 'Source image geometry has changed; original project was not modified.')
    case.project_warnings = list(manifest.get('warnings', []))
    case.project_id = manifest['project_id']
    case.revisions = manifest['revisions']
    case.stale = {s: set(manifest.get('stale', {}).get(s, [])) for s in workflow.SIDES}
    for side in workflow.SIDES:
        values = manifest['sides'][side]
        for branch in workflow.BRANCHES:
            data = values['branches'][branch]
            validate_branch(data, case.array.shape)
            case.control_points[side, branch] = data['control_points']
            case.centerline_points[side, branch] = data['centerline_points']
            case.segmentation_series[side, branch] = data['segmentation']
        if values.get('params') is not None:
            from .app import SU2ParamsRequest
            case.su2_params[side] = SU2ParamsRequest(**values['params']).model_dump()
    for key, record in manifest.get('artifacts', {}).items():
        side, stage = key.split(':')
        if side not in workflow.SIDES or stage not in workflow.STAGES:
            continue
        expected = case.revisions[side]
        rev = record['revisions']
        if rev['geometry'] != expected['geometry'] or (stage == 'cfd' and rev['params'] != expected['params']):
            case.stale[side].add(stage)
            case.project_warnings.append(f'{side}/{stage}: outdated artifact skipped')
            continue
        try:
            def read(name):
                return artifact_path(folder, record[name]).read_bytes()
            if stage == 'surface':
                case.built_surfaces[side] = read('stl')
            elif stage == 'mesh':
                case.su2_mesh_jobs[side] = dict(state='done', vtu=read('vtu'), vtp=read('vtp'),
                                              face_map=record['face_map'], diag=record.get('diag', {}))
            else:
                result = json.loads(read('result'))
                with np.load(io.BytesIO(read('frames')), allow_pickle=False) as frames:
                    data = {k: frames[k].copy() for k in ('wss_mag_pa', 'pressure_pa')}
                case.su2_postprocess_jobs[side] = dict(state='done', result=result, frames=data,
                                                       provenance=record.get('provenance', {}))
            case.saved_artifacts[key] = record
        except Exception as e:
            case.stale[side].add(stage)
            case.project_warnings.append(f'{side}/{stage}: could not restore artifact: {e}')


@router.post('/projects/open')
def open_project(req: ProjectRequest):
    from .app import _load_dicom_case, get_meta
    folder = storage.allowed_path(req.path, 'projects')
    try:
        manifest = json.loads((folder / 'project.json').read_text())
    except (OSError, ValueError) as e:
        raise HTTPException(400, f'could not read project.json: {e}')
    if manifest.get('format_version') != FORMAT_VERSION:
        raise HTTPException(409, 'Legacy project: migrate a copy with python -m server.migrate_project SOURCE DESTINATION. The original is preserved.')
    # One live case per folder prevents two windows from overwriting each other's edits.
    for existing in list(cases._CASES.values()):
        if existing.project_dir == str(folder):
            return project_info(existing, get_meta(existing.case_id))
    if not manifest.get('source_path') or not manifest.get('series_uid'):
        raise HTTPException(400, 'project has no source path or DICOM series identifier')
    case, info = _load_dicom_case(manifest['source_path'], manifest['series_uid'])
    try:
        restore(case, manifest, folder)
    except HTTPException:
        cases._CASES.pop(case.case_id, None)
        raise
    except (KeyError, TypeError, ValueError) as e:
        cases._CASES.pop(case.case_id, None)
        raise HTTPException(400, f'invalid project: {e}')
    case.project_dir = str(folder)
    return project_info(case, info)


def project_info(case, info):
    info.update(project_dir=case.project_dir, warnings=case.project_warnings, restored={
        side: {'centerline': [b for b in workflow.BRANCHES if case.centerline_points.get((side, b))],
               'segmentation': [b for b in workflow.BRANCHES if case.segmentation_series.get((side, b))],
               'cfd_results': side in case.su2_postprocess_jobs,
               'surface_file_exists': side in case.built_surfaces,
               'mesh_files_exist': side in case.su2_mesh_jobs} for side in workflow.SIDES
    })
    return info
