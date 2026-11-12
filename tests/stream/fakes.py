"""CPU-only fixtures for driving `Session` with in-process fake workers (no CUDA, no model, no processes)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from streamvoxel.stream import session as session_module


class ListLink:
    """Records everything the coordinator sends to one fake worker."""

    def __init__(self):
        self.sent = []

    def put(self, message):
        self.sent.append(message)

    send = put

    def close(self):
        pass


class FakeStore:
    ids = [1, 2, 3]
    manifest = dict(air_compact_ids=[0])

    def __init__(self, run, mapping):
        self.mapping = mapping

    def resolve(self, current, request):
        if request is None:
            return current
        request = int(request)
        if request in (1, 2):
            return self.ids[(self.ids.index(current) + (1 if request == 2 else -1)) % len(self.ids)]
        if request not in self.ids:
            raise ValueError(f'World {request} is outside')
        return request

    def load(self, world, box):
        z0, x0, z1, x1 = box
        cells = {(z, x): np.full((4, 2, 2), world, np.int16) for z in range(z0 // 16, z1 // 16) for x in range(x0 // 16, x1 // 16)}
        return cells, np.zeros((1, 2, 2), np.int16)


class FakeMapping:
    @classmethod
    def from_json(cls, path):
        return cls()


def make_session(tmp_path, monkeypatch, gpus=2, **cfg_extra):
    monkeypatch.setattr(session_module, 'Store', FakeStore)
    monkeypatch.setattr(session_module, 'Mapping', FakeMapping)
    monkeypatch.setattr(session_module, 'preview_lookup', lambda *a, **k: np.arange(8, dtype=np.uint16))
    run = Path(tmp_path) / 'run'
    (run / 'panel').mkdir(parents=True)
    (run / 'panel' / 'compact_mapping.json').write_text('{}')
    cfg = dict(world=1, initial_box_zx=[0, 0, 64, 64], visible_gpus=list(range(gpus)), slots_per_gpu=4, air_bank={'1': [2000000, 0, 0]},
               max_windows=50, headless_windows=100, max_display_lag_windows=1000)
    cfg.update(cfg_extra)
    session = session_module.Session(run, cfg)
    session.noted = -1e9  # the first note() writes applied_settings.json regardless of the process clock
    return session, install_fake_workers(session, gpus)


def install_fake_workers(session, count):
    """Attach `count` recording links as the workers' command queues; return the list of links."""
    from streamvoxel.stream.channel import Member
    links = [ListLink() for _ in range(count)]
    session.members = {wid: Member(wid, link, session.slots_per) for wid, link in enumerate(links)}
    return links


def write_control(session, **fields):
    ctrl = dict(ready=True, ack=10 ** 9, player_present=True, player_zx=[16.0, 16.0], reset_seq=0, world_req=None, stop=False)
    ctrl.update(fields)
    (session.run / 'control.json').write_text(json.dumps(ctrl))


def norm(value):
    """JSON-comparable form of a message: arrays become (shape, dtype, sum)."""
    if isinstance(value, np.ndarray):
        return ['nd', list(value.shape), str(value.dtype), int(value.astype(np.int64).sum())]
    if isinstance(value, dict):
        return {str(k): norm(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (tuple, list)):
        return [norm(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted(norm(v) for v in value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def norm_message(message):
    """Normalized worker-bound message. A commit's optional trailing sequence field is not part of the golden."""
    message = tuple(message)
    if message[0] == 'commit':
        message = message[:4]
    return norm(message)


def wire_packets(session):
    out = []
    for path in sorted((session.run / 'wire').glob('*.json')):
        packet = json.loads(path.read_text())
        packet.pop('emitted_unix_ns', None)
        packet.pop('sha256', None)
        out.append(packet)
    return out


def applied(session):
    path = session.run / 'applied_settings.json'
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    data.pop('unix_ns', None)
    return data


def row(nfe=5, target=(0, 0)):
    return dict(target=list(target), nfe=nfe, last_frac=.5, cap=8, seconds=.1, started_unix_ns=1, finished_unix_ns=2, world=1)


def done_message(epoch, wid, target, nfe=5, value=3):
    values = np.full((4, 2, 2), value, np.int16)
    packet = dict(row(nfe, target), gpu=wid, slot=0, actual=nfe)
    return ('done', epoch, wid, tuple(target), values, row(nfe, target), packet)


def drain(link):
    sent, link.sent = link.sent, []
    return [norm_message(m) for m in sent]

