"""SSH tail of window packets and the return path for player control.

The session's wire packets go out in order, unmerged, as bounded compressed lines (`MAX_PACKETS` / `MAX_RAW` bytes of
packet JSON each), so the viewer receives the whole generation process, every frame the session emitted, and can
start displaying after the first small line instead of waiting for a large one. There is no ack gating: the viewer
reads continuously into its own buffer and plays it back on a steady clock, and TCP back-pressure paces the sender.
After a dropped link the viewer reconnects with `--after <last packet it holds>`.

`--once` sends what the session has written so far and exits (used to fetch a full recording for offline replay).
"""
import argparse
import json
import os
from pathlib import Path
import sys
import threading
import time

from streamvoxel.common.live_wire import batch_line, packet_sha


MAX_PACKETS = 256
MAX_RAW = 384 * 1024
POLL_S = .05


def telemetry_line(run, last):
    gpus = []
    for path in sorted(Path(run).glob('telemetry_r*.json')):
        try:
            gpus.append(json.loads(path.read_text()))
        except (OSError, ValueError):
            continue
    stamp = [gpu.get('unix_ns') for gpu in gpus]
    if not gpus or stamp == last:
        return None, last
    try:
        applied = json.loads((Path(run) / 'applied_settings.json').read_text())
    except (OSError, ValueError):
        applied = {}
    return dict(kind='telemetry', gpus=gpus, applied=applied), stamp


def serve(run, after, once=False):
    run = Path(run)

    def receive():
        # A bad line or a failed write must not end this thread: the viewer's commands (reset, pause) would silently stop.
        for line in sys.stdin:
            try:
                state = json.loads(line)
                if state.get('kind') != 'control':
                    continue
                state['received_unix_ns'] = time.time_ns()
                tmp = run / f'control.{os.getpid()}.tmp'   # per-process name: a second transport must not race this one
                tmp.write_text(json.dumps(state))
                tmp.replace(run / 'control.json')
            except (OSError, ValueError):
                continue
        os._exit(0)   # stdin closed: the viewer is gone, do not linger as an orphan

    if not once:
        threading.Thread(target=receive, daemon=True).start()
    seq = after + 1
    last = None
    next_telemetry = 0.
    while True:
        tick = time.monotonic()
        if tick >= next_telemetry and not once:
            line, last = telemetry_line(run, last)
            if line:
                print(json.dumps(line, separators=(',', ':')), flush=True)
            next_telemetry = tick + .5
        packets, ended, size = [], False, 0
        while len(packets) < MAX_PACKETS and size < MAX_RAW:
            path = run / 'wire' / f'{seq:08d}.json'
            if not path.exists():
                break
            text = path.read_text()
            size += len(text)
            packets.append(json.loads(text))
            seq += 1
            if packets[-1]['kind'] in ('complete', 'failed'):
                ended = True
                break
        if not packets and (run / 'error.json').exists():
            packet = dict(seq=seq, kind='failed', error=json.loads((run / 'error.json').read_text()))
            packet['sha256'] = packet_sha(packet)
            packets, ended = [packet], True
        if packets:
            print(json.dumps(batch_line(packets), separators=(',', ':')), flush=True)
        if ended or (once and not packets):
            return
        if not packets:
            time.sleep(POLL_S)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--after', type=int, default=-1)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    serve(args.run, args.after, args.once)


if __name__ == '__main__':
    main()
