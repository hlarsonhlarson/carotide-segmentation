"""Machine-specific paths, configured without editing source code."""
import os
import shutil
from pathlib import Path

SU2_BIN_DIR = os.environ.get('CAROTID_SU2_BIN', '')
MPI_PYTHON = os.environ.get('CAROTID_MPI_PYTHON', '')


def solver_errors():
    errors = []
    if not SU2_BIN_DIR or not Path(SU2_BIN_DIR).is_dir():
        errors.append('Set CAROTID_SU2_BIN to the compatible SU2/pysu2 installation directory')
    if not MPI_PYTHON or not Path(MPI_PYTHON).is_file() or not os.access(MPI_PYTHON, os.X_OK):
        errors.append('Set CAROTID_MPI_PYTHON to the Python executable used to build pysu2')
    if shutil.which('mpirun') is None:
        errors.append('mpirun is not available on PATH')
    return errors
