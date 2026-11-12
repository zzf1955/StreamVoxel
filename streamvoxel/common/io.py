"""Small file helpers shared by every stage (no torch dependency)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256_file(path) -> str:
    """SHA256 of a file, streamed in 8 MiB chunks."""
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


digest = sha256_file


def append(path, value) -> None:
    """Append one JSON line."""
    with Path(path).open("a") as handle:
        handle.write(json.dumps(value) + "\n")


def atomic_json(path, value) -> None:
    """Write indented JSON through a temporary file and an atomic replace."""
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)
