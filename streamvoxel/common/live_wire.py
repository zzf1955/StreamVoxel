"""Window-level preview packets for the low-entropy-last live session. Stdlib only."""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
import threading
import time
import zlib


KINDS = ('session', 'window_start', 'frame', 'window_commit', 'reset', 'complete', 'failed')
# `steps` is the low-entropy-last cap. `batch` is how many slots may be occupied; the CUDA graph
# is captured at the maximum, so a live change cannot exceed that maximum.
LIMITS = dict(steps=(1, 128), batch=(1, 64), speed=(1, 100))
MODES = ('dynamic', 'fixed')


def packet_sha(packet):
    content = {k: v for k, v in packet.items() if k != 'sha256'}
    return hashlib.sha256(json.dumps(content, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def verify_packet(packet, expected):
    if packet.get('kind') not in KINDS:
        raise ValueError(f'Unknown packet kind {packet.get("kind")}')
    if packet['seq'] != expected or packet_sha(packet) != packet['sha256']:
        raise ValueError('Preview sequence/hash mismatch')
    return packet


class Publisher:
    def __init__(self, run):
        self.folder = Path(run) / 'wire'
        self.folder.mkdir(exist_ok=False)
        self.seq = 0
        self.lock = threading.Lock()

    def publish(self, kind, **data):
        if kind not in KINDS:
            raise ValueError(f'Unknown packet kind {kind}')
        with self.lock:
            packet = dict(seq=self.seq, kind=kind, emitted_unix_ns=time.time_ns(), **data)
            packet['sha256'] = packet_sha(packet)
            # Compact output: `indent` forces the pure-Python encoder, which dominates large frame packets.
            path = self.folder / f'{self.seq:08d}.json'
            tmp = path.with_suffix('.tmp')
            tmp.write_text(json.dumps(packet))
            tmp.replace(path)
            self.seq += 1
            return packet['seq']


def batch_line(packets):
    """One transport line for consecutive packets; `unbatch` returns them unchanged."""
    return dict(kind='batch', count=len(packets), z=encoded(zlib.compress(json.dumps(packets, separators=(',', ':')).encode(), 6)))


def unbatch(line):
    packets = json.loads(zlib.decompress(decoded(line['z'])))
    if len(packets) != line['count']:
        raise ValueError('Batch length mismatch')
    return packets


def encoded(data):
    return base64.b64encode(data).decode('ascii')


def decoded(text):
    return base64.b64decode(text, validate=True)


def clamp_settings(request, defaults, max_batch=None):
    """Integer-valid settings. Keys absent from both `request` and `defaults` are left out."""
    out = {}
    for key, (low, high) in LIMITS.items():
        if key not in request and key not in defaults:
            continue
        try:
            value = int(request[key] if key in request else defaults[key])
        except (TypeError, ValueError):
            if key not in defaults:
                continue
            value = int(defaults[key])
        out[key] = min(max(value, low), high)
    if 'batch' in out and max_batch is not None:
        out['batch'] = min(out['batch'], int(max_batch))
    if 'mode' in request or 'mode' in defaults:
        mode = request.get('mode', defaults.get('mode', 'dynamic'))
        out['mode'] = mode if mode in MODES else defaults.get('mode', 'dynamic')
    if 'paused' in request or 'paused' in defaults:
        out['paused'] = bool(int(request.get('paused', defaults.get('paused', False))))
    return out
