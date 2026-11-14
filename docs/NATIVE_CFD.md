# Native macOS CFD with the Docker web interface

The web interface remains at `http://localhost:8080`. In this mode nginx
forwards `/api/` to the native API on `127.0.0.1:8012`, which can load the macOS
SU2 Python extension and launch OpenMPI. The Linux Docker API remains available
as the default configuration but cannot load a macOS `.dylib`.

## Start

From the repository root, check the local installation:

```sh
.venv/bin/python scripts/native_web_backend.py --check
```

Start the native API in a terminal:

```sh
.venv/bin/python scripts/native_web_backend.py
```

In another terminal, start/switch the web interface:

```sh
docker compose -f compose.yaml -f compose.native.yaml up -d --build frontend
```

Keep using both compose files when rebuilding this mode. Reload the browser
and reopen the saved project after switching backends: in-memory case IDs
belong to the process that loaded them.

`/api/health` should return `"solver_errors": []`. This means the solver
installation is available, not that the study has a ready mesh or calibrated
boundary conditions. Build the mesh and save the boundary conditions before
starting a patient run.

## Configuration and files

- `CAROTID_SU2_BIN`: defaults to `~/su2-project/install/bin`.
- `CAROTID_MPI_PYTHON`: defaults to `python3.14` found on PATH, including
  `/opt/homebrew/bin`. It must match the Python used to compile `pysu2`.
- The startup probe runs a single MPI rank and checks the custom inlet/outlet
  wrapper methods used by the pulsatile runner.
- Existing macOS MPI handling defaults to one rank because multi-rank runs
  have hung in MPI window initialization on this machine. `CAROTID_MPI_N`
  can override it; multi-rank performance/stability is not established by
  the single-rank smoke test.
- DICOM/project roots come from the existing Docker bind mounts in
  `compose.yaml` and `.env`; explicit `CAROTID_DICOM_ROOT` and
  `CAROTID_PROJECTS_ROOT` values can override them.
- `/data/dicom` and `/data/projects` remain accepted path aliases. Saved
  manifests retain portable `/data/dicom/...` source paths so they reopen
  in Docker. Files are neither moved nor copied by this switch.
- TotalSegmentator remains on its separate native MPS worker at port 8011.
  The native API uses `http://127.0.0.1:8011/segment` by default.

Save active studies before switching backends. Avoid editing the same project
from two backend processes at once.

## Solver smoke test

Use a new output directory:

```sh
.venv/bin/python scripts/smoke_su2.py --output /absolute/path/to/new-smoke-directory
```

The script creates a synthetic duct with two outlet patches, runs four
physical steps through the normal pulsatile runner, and checks VTU extraction,
wall geometry matching and derived fields. Runtime is bounded to 90 seconds.
It retains configuration, output frames and solver logs for inspection.

This verifies execution and output compatibility only. The run is far shorter
than a cardiac cycle and does not validate the Windkessel coupling, mass
balance, physiological results, mesh convergence or patient segmentation.

## Return to the Docker API

Save the project, then:

```sh
docker compose -f compose.yaml up -d --build frontend
```

Reload the browser and reopen the project. Stop the native API with Ctrl-C in
its terminal when it is no longer needed.
