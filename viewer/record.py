"""Fetch the whole generation process of a low-entropy-last run as one compressed recording for offline replay.

    python -m viewer.record --config <viewer_config.json> --out build/recordings/<name>.jsonl
    python -m viewer.start --config ... --java ... --server ... --replay build/recordings/<name>.jsonl [--speed 2]

The recording is the transport's compressed batch lines (every packet the session emitted, unmerged), so it replays at
the session's own frame cadence. The live bridge keeps the same file as `recording.jsonl` while it plays.
"""
import argparse
import json
from pathlib import Path
import shlex
import subprocess


def record(cfg, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix('.part')
    command = (f'cd {shlex.quote(cfg["remote_source"])} && {shlex.quote(cfg["remote_python"])} -u -m viewer.transport '
               f'--run {shlex.quote(cfg["remote_run"])} --after -1 --once')
    with tmp.open('wb') as handle:
        status = subprocess.run(['ssh', '-o', 'Compression=yes', '-o', 'ConnectTimeout=20', '-o', 'ServerAliveInterval=15',
                                 cfg['ssh'], command], stdout=handle).returncode
    if status:
        raise RuntimeError(f'ssh status={status}; partial recording kept at {tmp}')
    tmp.replace(out)
    return out.stat().st_size


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(record(json.loads(args.config.read_text()), args.out), 'bytes')


if __name__ == '__main__':
    main()
