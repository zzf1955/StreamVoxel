"""Change the learning rate of a running unconditional training job."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from streamvoxel.common.io import atomic_json


def read_lr_control(run_dir, *, fallback):
    """Absent or unreadable control keeps the launch learning rate."""
    fallback = float(fallback)
    path = Path(run_dir) / "lr_control.json"
    if not path.is_file():
        return fallback
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError, UnicodeError):
        return fallback
    if not isinstance(payload, dict) or "lr" not in payload:
        return fallback
    try:
        lr = float(payload["lr"])
    except (TypeError, ValueError):
        return fallback
    if not math.isfinite(lr) or lr <= 0:
        return fallback
    return lr


def update_lr_control(run_dir, lr):
    lr = float(lr)
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("Learning rate must be positive and finite")
    run = Path(run_dir)
    if not (run / "config.json").is_file():
        raise FileNotFoundError("Run directory has no config.json")
    atomic_json(run / "lr_control.json", dict(lr=lr, updated_unix=time.time()))
    return dict(lr=lr, effect="the next optimizer step uses this learning rate; AdamW moments stay")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--lr", type=float, required=True)
    args = parser.parse_args()
    print(json.dumps(update_lr_control(args.run_dir, args.lr)), flush=True)


if __name__ == "__main__":
    main()
