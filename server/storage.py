"""Browse only the data directories explicitly mounted into the container."""
import os
from pathlib import Path
from typing import Literal
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()


def root(kind):
    return Path(os.environ.get('CAROTID_DICOM_ROOT' if kind == 'dicom' else 'CAROTID_PROJECTS_ROOT', f'/data/{kind}')).resolve()


def allowed_path(value, kind):
    variable = 'CAROTID_DICOM_ROOT' if kind == 'dicom' else 'CAROTID_PROJECTS_ROOT'
    path = Path(value).expanduser()
    alias = os.environ.get(variable + '_ALIAS')
    if alias and path.is_relative_to(Path(alias)):
        path = root(kind) / path.relative_to(Path(alias))
    path = path.resolve()
    if os.environ.get(variable) and not path.is_relative_to(root(kind)):
        raise HTTPException(403, 'Folder is outside the mounted data directory')
    return path


def portable_path(value, kind):
    """Store container-compatible source paths when the backend runs natively."""
    variable = 'CAROTID_DICOM_ROOT' if kind == 'dicom' else 'CAROTID_PROJECTS_ROOT'
    alias = os.environ.get(variable + '_ALIAS')
    path = Path(value).expanduser().resolve()
    if alias and path.is_relative_to(root(kind)):
        return str(Path(alias) / path.relative_to(root(kind)))
    return str(value)


@router.get('/storage')
def browse(kind: Literal['dicom', 'projects'], path: str = ''):
    base = root(kind)
    folder = (base / path).resolve()
    if not folder.is_relative_to(base):
        raise HTTPException(403, 'Folder is outside the mounted data directory')
    if not folder.is_dir():
        raise HTTPException(404, 'Data folder is not available; check the Docker volume mount')
    children = []
    for item in sorted(folder.iterdir(), key=lambda p: p.name.lower()):
        if item.name.startswith('.') or not item.is_dir() or not item.resolve().is_relative_to(base):
            continue
        children.append({'name': item.name, 'path': str(item.relative_to(base)),
                         'project': (item / 'project.json').is_file(), 'files': _file_count(item)})
    return {'path': str(folder.relative_to(base)), 'absolute_path': str(folder), 'folders': children}


def _file_count(folder):
    try:
        return sum(1 for entry in os.scandir(folder) if entry.is_file() and not entry.name.startswith('.'))
    except OSError:
        return None


class NewFolderRequest(BaseModel):
    path: str = ''
    name: str


@router.post('/storage/folder')
def create_project_folder(req: NewFolderRequest):
    """A new empty subfolder in the projects root (projects need their own folder)."""
    name = req.name.strip()
    if not name or name.startswith('.') or any(c in name for c in '/\\:') or name in ('.', '..'):
        raise HTTPException(422, 'Folder name must be a plain name without slashes')
    base = root('projects')
    parent = (base / req.path).resolve()
    if not parent.is_relative_to(base) or not parent.is_dir():
        raise HTTPException(403, 'Folder is outside the projects directory')
    target = parent / name
    if target.exists():
        raise HTTPException(409, 'A folder with this name already exists')
    target.mkdir()
    return {'name': name, 'path': str(target.relative_to(base)), 'absolute_path': str(target)}
