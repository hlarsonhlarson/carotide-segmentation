#!/usr/bin/env bash
# Native macOS backend with the local SU2/pysu2 install.
# Docker cannot load the macOS pysu2 dylib; use this instead of the
# container backend when you need CFD.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export CAROTID_SU2_BIN="${CAROTID_SU2_BIN:-/Users/egor/su2-project/install/bin}"
export CAROTID_MPI_PYTHON="${CAROTID_MPI_PYTHON:-/opt/homebrew/bin/python3.14}"
export PATH="/opt/homebrew/bin:${PATH}"
export PYTHONPATH="${CAROTID_SU2_BIN}${PYTHONPATH:+:$PYTHONPATH}"
export OMPI_MCA_btl="${OMPI_MCA_btl:-self,tcp}"
export OMPI_MCA_pml="${OMPI_MCA_pml:-ob1}"
export OMPI_MCA_osc="${OMPI_MCA_osc:-sm}"
export PMIX_MCA_gds="${PMIX_MCA_gds:-hash}"
export CAROTID_DICOM_ROOT="${CAROTID_DICOM_ROOT:-/Users/egor/Desktop/CT Data}"
export CAROTID_PROJECTS_ROOT="${CAROTID_PROJECTS_ROOT:-/Users/egor/Desktop/CT Data}"
cd "$ROOT"
exec .venv/bin/python -m uvicorn server.app:app --host 127.0.0.1 --port "${PORT:-8000}"
