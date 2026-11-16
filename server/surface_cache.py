"""Reuse surface reconstruction without sharing mutable VTK outputs."""
import copy
import hashlib
import json

from . import build_surface


def get_surface(case, side, series_by_branch):
    # Include complete contours, image geometry and reconstruction settings:
    # edits, deleted branches and parameter changes must all cause a miss.
    inputs = {
        "shape": case.array.shape,
        "spacing": case.spacing_mm.tolist(),
        "origin": case.origin_mm.tolist(),
        "series": series_by_branch,
        "settings": {
            name: value for name, value in vars(build_surface).items()
            if name.isupper() and isinstance(value, (int, float, str, bool))
        },
    }
    key = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).digest()
    # Concurrent preview/meshing requests should reconstruct only once.
    with case.surface_cache_lock:
        cached = case.surface_cache.get(side)
        if cached is None or cached[0] != key:
            poly, diag = build_surface.build_surface(
                case.array.shape, case.spacing_mm, case.origin_mm, series_by_branch,
            )
            if poly is None:
                return None, diag
            case.surface_cache[side] = (key, poly, copy.deepcopy(diag))
        _, poly, diag = case.surface_cache[side]
        # Face tagging and concurrent consumers may mutate their input.
        result = poly.NewInstance()
        result.DeepCopy(poly)
        return result, copy.deepcopy(diag)
