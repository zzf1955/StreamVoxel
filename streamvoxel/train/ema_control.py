"""Toggle EMA and its decay while an unconditional training process is running."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from streamvoxel.common.io import atomic_json


def read_ema_control(run_dir, *, fallback):
    """Absent or unreadable control falls back to the launch settings."""
    result = dict(enabled=bool(fallback["enabled"]), decay=float(fallback["decay"]))
    path = Path(run_dir) / "ema_control.json"
    if not path.is_file():
        return result
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError, UnicodeError):
        return result
    if not isinstance(payload, dict):
        return result
    if "enabled" in payload:
        if not isinstance(payload["enabled"], bool):
            return result
        result["enabled"] = payload["enabled"]
    if "decay" in payload:
        try:
            decay = float(payload["decay"])
        except (TypeError, ValueError):
            return dict(enabled=bool(fallback["enabled"]), decay=float(fallback["decay"]))
        if not math.isfinite(decay) or not 0 < decay < 1:
            return dict(enabled=bool(fallback["enabled"]), decay=float(fallback["decay"]))
        result["decay"] = decay
    return result


def update_ema_control(run_dir, *, enabled=None, decay=None):
    if enabled is None and decay is None:
        raise ValueError("Set EMA enabled and/or decay")
    run = Path(run_dir)
    if not (run / "config.json").is_file():
        raise FileNotFoundError("Run directory has no config.json")
    path = run / "ema_control.json"
    payload = {}
    if path.is_file():
        current = json.loads(path.read_text())
        if isinstance(current, dict):
            payload.update(current)
    if enabled is not None:
        payload["enabled"] = bool(enabled)
    if decay is not None:
        decay = float(decay)
        if not math.isfinite(decay) or not 0 < decay < 1:
            raise ValueError("EMA decay must be in (0, 1)")
        payload["decay"] = decay
    payload["updated_unix"] = time.time()
    atomic_json(path, payload)
    return {key: payload[key] for key in ("enabled", "decay") if key in payload}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--enable", action="store_true")
    group.add_argument("--disable", action="store_true")
    parser.add_argument("--decay", type=float)
    args = parser.parse_args()
    enabled = True if args.enable else False if args.disable else None
    print(json.dumps(update_ema_control(args.run_dir, enabled=enabled, decay=args.decay)), flush=True)


if __name__ == "__main__":
    main()
