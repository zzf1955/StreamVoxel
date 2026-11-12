"""Code and weight identity checks for elastic workers (a joiner must run the same code and weights as the coordinator)."""
from __future__ import annotations

import hashlib
from pathlib import Path

import streamvoxel
from streamvoxel.common.io import digest


class IntegrityError(RuntimeError):
    pass


def package_root():
    return Path(streamvoxel.__file__).resolve().parent


def package_files(root=None):
    """Sorted posix names (`streamvoxel/...`) of every Python file in the package; `root` is the package folder."""
    root = Path(root) if root is not None else package_root()
    return sorted(f'{root.name}/{path.relative_to(root).as_posix()}' for path in root.rglob('*.py') if '__pycache__' not in path.parts)


def package_digest(root=None):
    """One sha256 over the sorted file names and the sha256 of each file's content."""
    root = Path(root) if root is not None else package_root()
    total = hashlib.sha256()
    for name in package_files(root):
        total.update(name.encode() + b'\0' + digest(root.parent / name).encode() + b'\n')
    return total.hexdigest()


def verify_weights(paths, expected):
    """`paths`/`expected`: {'base': ..., 'control': ...}. An expected hash of None is not checked. Returns the actual hashes."""
    actual = {}
    for name, path in paths.items():
        actual[name] = digest(path)
        want = expected.get(name)
        if want is not None and actual[name] != want:
            raise IntegrityError(f'{name} weights {path} have sha256 {actual[name][:12]}, the coordinator expects {want[:12]}')
    return actual
