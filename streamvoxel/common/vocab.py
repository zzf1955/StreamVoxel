"""Global vocabulary loader."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_global_vocab(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("token_to_block"), dict):
        raise ValueError(f"invalid global vocabulary: {path}")
    return value
