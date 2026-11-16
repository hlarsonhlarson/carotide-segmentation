"""Run the web API on macOS with SU2, retaining Docker-compatible projects.

Uses the data roots already configured in compose.yaml/.env. The web UI is
proxied with compose.native.yaml; no DICOM or project files are relocated.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def environment():
    env = os.environ.copy()
    env["PATH"] = "/opt/homebrew/bin:" + env.get("PATH", "")
    env.setdefault("CAROTID_SU2_BIN", str(Path.home() / "su2-project/install/bin"))
    env.setdefault("CAROTID_MPI_PYTHON", shutil.which("python3.14", path=env["PATH"]) or "")
    env["PYTHONPATH"] = os.pathsep.join((str(ROOT), env["CAROTID_SU2_BIN"], env.get("PYTHONPATH", "")))
    for key, value in {
        "OMPI_MCA_btl": "self,tcp", "OMPI_MCA_pml": "ob1",
        "OMPI_MCA_osc": "sm", "PMIX_MCA_gds": "hash",
        "CAROTID_AUTOSEED_WORKER_URL": "http://127.0.0.1:8011/segment",
    }.items():
        env.setdefault(key, value)
    result = subprocess.run(
        ["docker", "compose", "-f", str(ROOT / "compose.yaml"), "config", "--format", "json"],
        cwd=ROOT, env=env, capture_output=True, text=True, check=True, timeout=30,
    )
    volumes = json.loads(result.stdout)["services"]["backend"]["volumes"]
    for kind in ("DICOM", "PROJECTS"):
        target = f"/data/{kind.lower()}"
        matches = [v for v in volumes if v["target"] == target and v["type"] == "bind"]
        if len(matches) != 1:
            raise ValueError(f"Expected a configured bind mount for {target}")
        variable = f"CAROTID_{kind}_ROOT"
        env.setdefault(variable, matches[0]["source"])
        env[variable + "_ALIAS"] = target
        if not Path(env[variable]).is_dir():
            raise ValueError(f"Data root does not exist: {env[variable]}")
    return env


def check_solver(env):
    python = env["CAROTID_MPI_PYTHON"]
    if not python or not Path(python).is_file():
        raise ValueError("Set CAROTID_MPI_PYTHON to the Python used to build pysu2")
    probe = """
import pysu2
from mpi4py import MPI
required = ['SetMarkerCustomInletFlowVar1', 'SetMarkerCustomInletFlowDirection',
            'SetMarkerCustomOutletPressure', 'BoundaryConditionsUpdate',
            'GetMarkerVertexNormals', 'MarkerPrimitives']
missing = [name for name in required if not hasattr(pysu2.CSinglezoneDriver, name)]
if missing:
    raise RuntimeError('Incompatible pysu2 API: ' + ', '.join(missing))
print('SU2/pysu2 and MPI ready; ranks:', MPI.COMM_WORLD.Get_size())
"""
    subprocess.run(["mpirun", "-n", "1", python, "-c", probe],
                   env=env, check=True, timeout=30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Probe paths and MPI without starting the API")
    args = parser.parse_args()
    try:
        env = environment()
        check_solver(env)
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        print(f"Native backend setup failed: {exc}", file=sys.stderr)
        return 1
    print("DICOM root:", env["CAROTID_DICOM_ROOT"], flush=True)
    print("Projects root:", env["CAROTID_PROJECTS_ROOT"], flush=True)
    if args.check:
        return 0
    os.chdir(ROOT)
    os.execve(sys.executable, [sys.executable, "-m", "uvicorn", "server.app:app",
                              "--host", "127.0.0.1", "--port", "8012"], env)


if __name__ == "__main__":
    raise SystemExit(main())
