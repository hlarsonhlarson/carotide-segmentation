#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Keep this demo isolated from any local .env and patient-data mounts.
export DICOM_DIR="$PWD/data/codespaces-dicom"
export PROJECTS_DIR="$PWD/data/codespaces-projects"
export WEB_PORT=8080
mkdir -p "$DICOM_DIR" "$PROJECTS_DIR"
for attempt in $(seq 1 60); do
  if docker info >/dev/null 2>&1; then break; fi
  if [ "$attempt" = 60 ]; then echo "Docker did not start" >&2; exit 1; fi
  sleep 1
done
docker compose build
docker compose run --rm --no-deps -v "$DICOM_DIR:/seed" backend python -c '
from pathlib import Path
import runpy
create = runpy.run_path("tests/create_demo.py")["create"]
folder = Path("/seed/synthetic-demo")
if not folder.exists():
    create(folder)
elif len(list(folder.glob("*.dcm"))) != 40:
    raise SystemExit("Incomplete synthetic demo; choose a new demo directory instead of overwriting data")
'
docker compose up -d --wait --wait-timeout 180
docker compose exec -T backend python tests/prepare_demo.py
printf '\nDemo ready on port 8080. Open project: /data/projects/demo\n'
printf 'To share: Ports -> 8080 -> Port Visibility -> Public -> Copy Forwarded Address.\n'
