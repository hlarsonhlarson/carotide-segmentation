"""Run four CFD steps on a synthetic duct with two outlet patches.

This verifies the local solver/wrapper/output chain, not patient hemodynamics.
Usage: .venv/bin/python scripts/smoke_su2.py --output NEW_DIRECTORY
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import threading
import sys

import numpy as np

from native_web_backend import environment, check_solver


def duct_mesh():
    nx, ny, nz = 8, 4, 4
    points = np.array([(i * .03 / nx, j * .004 / ny, k * .004 / nz)
                       for i in range(nx + 1) for j in range(ny + 1) for k in range(nz + 1)])
    def node(i, j, k):
        return (i * (ny + 1) + j) * (nz + 1) + k
    tets = []
    faces = Counter()
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                a, b, c, d = node(i, j, k), node(i+1, j, k), node(i+1, j+1, k), node(i, j+1, k)
                e, f, g, h = node(i, j, k+1), node(i+1, j, k+1), node(i+1, j+1, k+1), node(i, j+1, k+1)
                for tet in ([a, b, c, g], [a, c, d, g], [a, d, h, g],
                            [a, h, e, g], [a, e, f, g], [a, f, b, g]):
                    if np.linalg.det(points[tet[1:]] - points[tet[0]]) < 0:
                        tet[1], tet[2] = tet[2], tet[1]
                    tets.append(tet)
                    for indices in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)):
                        faces[tuple(sorted(tet[x] for x in indices))] += 1
    markers = {name: [] for name in ("wall", "cca_inlet", "ica_outlet", "eca_outlet")}
    for face, count in faces.items():
        if count != 1:
            continue
        coordinates = points[list(face)]
        if np.allclose(coordinates[:, 0], 0):
            name = "cca_inlet"
        elif np.allclose(coordinates[:, 0], .03):
            name = "ica_outlet" if coordinates[:, 1].mean() < .002 else "eca_outlet"
        else:
            name = "wall"
        markers[name].append(face)
    lines = ["NDIME= 3", f"NELEM= {len(tets)}"]
    lines += ["10 " + " ".join(map(str, tet)) for tet in tets]
    lines += [f"NPOIN= {len(points)}"]
    lines += [f"{x:.12g} {y:.12g} {z:.12g} {i}" for i, (x, y, z) in enumerate(points)]
    lines += [f"NMARK= {len(markers)}"]
    for name, triangles in markers.items():
        lines += [f"MARKER_TAG= {name}", f"MARKER_ELEMS= {len(triangles)}"]
        lines += ["5 " + " ".join(map(str, triangle)) for triangle in triangles]
    return "\n".join(lines) + "\n", points, markers


def wall_surface(output, points_m, markers):
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk
    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(points_m * 1000., deep=True))
    cells, face_ids = vtk.vtkCellArray(), []
    for face_id, triangles in enumerate(markers.values(), 1):
        for ids in triangles:
            triangle = vtk.vtkTriangle()
            for j, node in enumerate(ids):
                triangle.GetPointIds().SetId(j, node)
            cells.InsertNextCell(triangle)
            face_ids.append(face_id)
    surface = vtk.vtkPolyData()
    surface.SetPoints(points)
    surface.SetPolys(cells)
    nodes = numpy_to_vtk(np.arange(1, len(points_m) + 1), deep=True)
    nodes.SetName("GlobalNodeID")
    surface.GetPointData().AddArray(nodes)
    faces = numpy_to_vtk(np.asarray(face_ids), deep=True)
    faces.SetName("ModelFaceID")
    surface.GetCellData().AddArray(faces)
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(output / "mesh_wall.vtp"))
    writer.SetInputData(surface)
    writer.Write()
    return (output / "mesh_wall.vtp").read_bytes()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = environment()
    check_solver(env)
    os.environ.update(env)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    # Import after the solver environment is configured.
    from server import su2_run
    mesh, points_m, markers = duct_mesh()
    (output / "mesh.su2").write_text(mesh)
    vtp_bytes = wall_surface(output, points_m, markers)
    cfg = output / "run.cfg"
    cfg.write_text(su2_run._cfg_text("mesh.su2", 4, 20., [1., 0., 0.])
                   .replace("OUTPUT_WRT_FREQ= 10", "OUTPUT_WRT_FREQ= 1"))
    params = output / "run_params.json"
    params.write_text(json.dumps({
        "case_name": "synthetic_duct_smoke", "psv_cms": 60., "edv_cms": 20.,
        "cardiac_cycle_s": 1., "ica_fraction": .5,
        "ica_map_pa": 80 * 133.322, "eca_map_pa": 80 * 133.322,
        "inlet_direction": [1., 0., 0.],
    }))
    job = {"state": "running"}
    thread = threading.Thread(target=su2_run.run_in_background,
                              args=(job, output, cfg, params), daemon=True)
    thread.start()
    thread.join(timeout=90)
    if thread.is_alive():
        su2_run.cancel_run(job)
        thread.join(timeout=10)
        raise SystemExit(f"Solver smoke test timed out; inspect {output / 'solver_stdout.log'}")
    print(json.dumps({k: v for k, v in job.items() if k != "process"}, indent=2))
    if job.get("state") != "done" or not list(output.glob("flow_*.vtu")):
        raise SystemExit(f"Solver smoke test failed; inspect {output / 'solver_stdout.log'}")
    from server import su2_postprocess
    series = su2_postprocess.extract_wall_time_series(output, vtp_bytes, su2_run.TIME_STEP_S)
    fields = su2_postprocess.compute_derived_fields(series, 1.)
    print("Postprocessing:", json.dumps(fields["temporal_quality"]))
    print("Synthetic CFD smoke test completed. This is not a validated patient simulation.")


if __name__ == "__main__":
    main()
