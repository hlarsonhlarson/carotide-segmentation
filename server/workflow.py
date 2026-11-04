"""Dependency invalidation and serialization of edits to one study."""
from functools import wraps
from fastapi import HTTPException
from . import cases, su2_run

SIDES = ('left', 'right')
BRANCHES = ('cca_ica', 'cca_eca')
STAGES = ('surface', 'mesh', 'cfd')


def locked(fn):
    @wraps(fn)
    def call(case_id: str, *args, **kwargs):
        case = cases.get(case_id)
        if case is None:
            raise HTTPException(404, 'unknown case_id')
        with case.lock:
            return fn(case_id, *args, **kwargs)
    return call


def invalidate(case, side, stage='surface'):
    """Invalidate descendants, including persisted references and detached jobs."""
    if stage not in STAGES:
        raise ValueError(stage)
    downstream = STAGES[STAGES.index(stage):]
    if 'surface' in downstream:
        case.surface_cache.pop(side, None)
        case.built_surfaces.pop(side, None)
        case.revisions[side]['geometry'] += 1
    if 'mesh' in downstream:
        case.su2_mesh_jobs.pop(side, None)
    job = case.su2_run_jobs.pop(side, None)
    if job is not None:
        su2_run.cancel_run(job)
    case.su2_postprocess_jobs.pop(side, None)
    for name in downstream:
        case.saved_artifacts.pop(f'{side}:{name}', None)
        case.stale[side].add(name)


def status(case, side):
    if side not in SIDES:
        raise HTTPException(400, 'invalid side')
    paths = [case.centerline_points.get((side, b), []) for b in BRANCHES]
    series = [case.segmentation_series.get((side, b), []) for b in BRANCHES]
    active = [s for p, s in zip(paths, series) if p]
    segmented = bool(paths[0]) and bool(active) and all(s and all(p.get('contour_mm') for p in s) for s in active)
    def state(stage, job):
        if job:
            return job.get('state', 'idle')
        return 'stale' if stage in case.stale[side] else 'idle'
    return {
        'pathway': 'done' if len(paths[0]) > 1 else 'idle',
        'segmentation': 'done' if segmented else ('partial' if any(series) else 'idle'),
        'surface': 'done' if side in case.built_surfaces else state('surface', None),
        'meshing': state('mesh', case.su2_mesh_jobs.get(side)),
        'boundaryConditions': 'done' if side in case.su2_params else 'idle',
        'cfdComputation': state('cfd', case.su2_postprocess_jobs.get(side) or case.su2_run_jobs.get(side)),
    }


def edit(fn):
    @wraps(fn)
    def call(case_id: str, *args, **kwargs):
        from .projects import autosave
        result = fn(case_id, *args, **kwargs)
        autosave(cases.get(case_id))
        return result
    return locked(call)
