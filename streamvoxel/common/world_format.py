"""Canonical World-array validation shared by remote tooling and local preview."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

AXIS_ORDER = "YZX"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def validate_world_array(array: np.ndarray, *, label: str = "blocks") -> np.ndarray:
    value = np.asarray(array)
    if value.ndim not in (3, 4):
        raise ValueError(f"{label} must have shape [Y,Z,X] or [N,Y,Z,X], got {value.shape}")
    if value.dtype != np.uint16:
        raise ValueError(f"{label} must be uint16 global tokens, got {value.dtype}")
    if any(size <= 0 for size in value.shape):
        raise ValueError(f"{label} has an empty dimension: {value.shape}")
    return np.ascontiguousarray(value)


def load_world(path: Path) -> tuple[np.ndarray, dict[str, Any] | None]:
    blocks = validate_world_array(np.load(path), label=str(path))
    metadata_path = path.with_name("metadata.json")
    metadata = load_json(metadata_path) if metadata_path.is_file() else None
    if metadata is not None and metadata.get("axis_order", AXIS_ORDER) != AXIS_ORDER:
        raise ValueError(f"{metadata_path} is not {AXIS_ORDER} ordered")
    return blocks, metadata
