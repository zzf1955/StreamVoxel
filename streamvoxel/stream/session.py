"""Low-entropy-last live session: fixed slots, a new window fills a slot on the forward after the previous one finishes.

The main process is the coordinator (scheduler state, control, wire); each GPU runs in its own process (`worker.py`).

Entry points: `python -m streamvoxel.stream.session {prepare,gate,run} --run <dir>`.
`prepare` links the panel. `gate` checks the padded CUDA graph against the unpadded forward.
`run` serves until the window cap, the GPU deadline, or an idle timeout.
"""
from __future__ import annotations

import argparse
import collections
from dataclasses import dataclass, replace
import itertools
import json
import multiprocessing
import os
from pathlib import Path
import queue
import signal
import threading
import time

import numpy as np

from streamvoxel.common.io import atomic_json, digest
from streamvoxel.common.live_wire import Publisher, clamp_settings
from streamvoxel.common.live_rows import pack_row, states_digest
from streamvoxel.common.preview_states import load_registry, preview_lookup
from streamvoxel.common.colored_frontier import ready_cells
from streamvoxel.stream import channel as netchannel
from streamvoxel.stream.channel import ACTIVE, BUILDING, CLOSED, DRAINING, READY, Member, NetLink, ProcLink
from streamvoxel.stream.scheduler import member_shares, pick_window, plan_inserts, separated
from streamvoxel.stream.worker import run_worker
from streamvoxel.stream.condition import Mapping
from streamvoxel.stream.integrity import package_digest, package_files

EXTRA_SOURCE = ('viewer/bridge.py', 'viewer/start.py', 'viewer/transport.py')


def wid_free(owner, wid):
    return all(value != wid for value in owner.values())


def _cfg(run):
    return json.loads((Path(run) / 'config.json').read_text())


def _require(cfg):
    keys = ['world', 'parent_panel', 'mapping', 'vocabulary', 'base', 'control', 'initial_box_zx']
    if not cfg.get('listen'):
        keys.append('visible_gpus')  # elastic workers bring their own GPUs
    missing = [key for key in keys if key not in cfg]
    if missing:
        raise KeyError(f'config missing {missing}')


def local_gpu_env(physical, environ=None):
    """Environment of the local worker that runs on physical GPU `physical` (the nvidia-smi index).

    The child sees only that card, so inside it the device is cuda:0. PCI_BUS_ID makes CUDA count GPUs the way
    nvidia-smi does. If the coordinator itself runs under a CUDA_VISIBLE_DEVICES mask, the GPU must be in it.
    """
    physical = int(physical)
    if physical < 0:
        raise ValueError(f'visible_gpus entry {physical} is not a GPU index')
    mask = (environ if environ is not None else os.environ).get('CUDA_VISIBLE_DEVICES')
    if mask is not None and mask.strip():
        entries = [entry.strip() for entry in mask.split(',')]
        if str(physical) not in entries:
            raise ValueError(f'visible_gpus entry {physical} is not in CUDA_VISIBLE_DEVICES={mask!r}; '
                             'visible_gpus lists physical (nvidia-smi) indices, so unset the mask or include them')
    return {'CUDA_VISIBLE_DEVICES': str(physical), 'CUDA_DEVICE_ORDER': 'PCI_BUS_ID'}


class _environment:
    """Set environment variables for the duration of a block (a spawned child copies them at start)."""

    def __init__(self, values):
        self.values, self.saved = dict(values), {}

    def __enter__(self):
        for key, value in self.values.items():
            self.saved[key] = os.environ.get(key)
            os.environ[key] = value

    def __exit__(self, *exc):
        for key, value in self.saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def source_manifest():
    """Hash of every file of the package plus the viewer entry files (relative to the working directory). A missing file is an error."""
    return dict(files={name: digest(Path(name)) for name in (*package_files(), *EXTRA_SOURCE)})


def prepare(run):
    run = Path(run)
    cfg = _cfg(run)
    _require(cfg)
    import torch
    parent = Path(cfg['parent_panel'])
    manifest = json.loads((parent / 'manifest.json').read_text())
    if digest(parent / 'manifest.json') != (parent / 'COMPLETE').read_text().strip():
        raise RuntimeError('Parent panel manifest does not match COMPLETE')
    saved = torch.load(cfg['control'], map_location='cpu', mmap=True, weights_only=False)
    if digest(cfg['base']) != saved['config']['identity']['base']:
        raise RuntimeError('Base checkpoint does not match the ControlNet identity')
    panel = run / 'panel'
    panel.mkdir(exist_ok=False)
    files = {}
    for name, src in (('compact_mapping.json', Path(cfg['mapping'])), ('vocabulary.json', Path(cfg['vocabulary']))):
        (panel / name).symlink_to(src.resolve())
        files[name] = digest(panel / name)
    for case in manifest['cases']:
        folder = panel / case['directory']
        folder.mkdir()
        for leaf in ('gt_global.npy', 'biomes.npy'):
            src = parent / case['directory'] / leaf
            if digest(src) != manifest['files'][case['directory'] + '/' + leaf]:
                raise RuntimeError(f'Panel file changed: {src}')
            (folder / leaf).symlink_to(src.resolve())
            files[case['directory'] + '/' + leaf] = digest(folder / leaf)
    own = dict(format='live_low_entropy_last_panel_v1', parent=str(parent), parent_manifest_sha256=digest(parent / 'manifest.json'),
               cases=manifest['cases'], files=files, air_compact_ids=list(saved['config']['air_compact_ids']),
               control_step=int(saved['step']), base_sha256=digest(cfg['base']), control_sha256=digest(cfg['control']))
    atomic_json(panel / 'manifest.json', own)
    (panel / 'COMPLETE').write_text(digest(panel / 'manifest.json') + '\n')
    frozen = source_manifest()
    atomic_json(run / 'frozen_source_manifest.json', frozen)
    print(json.dumps(dict(prepared=str(panel), worlds=len(manifest['cases']), air=own['air_compact_ids'])), flush=True)


GENERATED = ('wire', 'results', 'session.log', 'session.pid', 'control.json', 'applied_settings.json', 'error.json',
             'status.json', 'numerical_gate.json', 'slot_log_g*.jsonl', 'gpu_telemetry_r*.csv', 'telemetry_r*.json', 'timing_r*',
             'crashed_*', 'failed_gate_*', 'prev_*')


def fresh(run):
    """Delete everything a previous session left in the run folder; config and panel stay."""
    import shutil
    run = Path(run)
    for pattern in GENERATED:
        for path in run.glob(pattern):
            shutil.rmtree(path) if path.is_dir() else path.unlink()
    if (run / 'panel').exists():
        atomic_json(run / 'frozen_source_manifest.json', source_manifest())
    print(json.dumps(dict(fresh=str(run))), flush=True)


class Store:
    def __init__(self, run, mapping):
        panel = Path(run) / 'panel'
        self.manifest = json.loads((panel / 'manifest.json').read_text())
        if digest(panel / 'manifest.json') != (panel / 'COMPLETE').read_text().strip():
            raise RuntimeError('Run panel was modified')
        self.panel = panel
        self.cases = {case['world']: case for case in self.manifest['cases']}
        self.ids = sorted(self.cases)
        self.mapping = mapping

    def resolve(self, current, request):
        if request is None:
            return current
        request = int(request)
        if request in (1, 2):
            return self.ids[(self.ids.index(current) + (1 if request == 2 else -1)) % len(self.ids)]
        if request not in self.cases:
            raise ValueError(f'World {request} is outside {self.ids[0]}..{self.ids[-1]}')
        return request

    def load(self, world, box):
        case = self.cases[world]
        folder = self.panel / case['directory']
        for leaf in ('gt_global.npy', 'biomes.npy'):
            if digest(folder / leaf) != self.manifest['files'][case['directory'] + '/' + leaf]:
                raise RuntimeError(f'{world} {leaf} hash changed')
        gt = np.load(folder / 'gt_global.npy', mmap_mode='r')
        biomes = np.load(folder / 'biomes.npy')
        z0, x0, z1, x1 = box
        canvas = case.get('gt_canvas_box_zx')
        if gt.shape[1] < z1 or gt.shape[2] < x1:
            if not canvas:
                raise ValueError(f'Ground truth {gt.shape} does not cover the initial box {box}')
            oz, ox = int(canvas[0]), int(canvas[1])
            src = gt[:, z0 - oz:z1 - oz, x0 - ox:x1 - ox]
        else:
            src = gt[:, z0:z1, x0:x1]
        if src.shape[1:] != (z1 - z0, x1 - x0):
            raise ValueError(f'Initial box {box} sliced to {src.shape} from ground truth {gt.shape}')
        compact = self.mapping.to_compact[src]
        if (compact < 0).any():
            raise ValueError(f'World {world} initial box has blocks outside the compact vocabulary')
        cells = {}
        for z in range(z0 // 16, z1 // 16):
            for x in range(x0 // 16, x1 // 16):
                view = compact[:, (z * 16 - z0):(z * 16 - z0 + 16), (x * 16 - x0):(x * 16 - x0 + 16)]
                cells[(z, x)] = np.ascontiguousarray(view, dtype=np.int16)
        return cells, biomes


def _control(run, fallback):
    path = Path(run) / 'control.json'
    if not path.exists():
        return dict(ready=True, ack=-1, player_present=False, player_zx=list(fallback), reset_seq=0, world_req=None, stop=False)
    data = json.loads(path.read_text())
    data.setdefault('player_zx', list(fallback))
    data.setdefault('reset_seq', 0)
    data.setdefault('ack', -1)
    data.setdefault('ready', False)
    return data


def _allow_insert(ctrl, generated, headless, commit_seqs, lag):
    if ctrl.get('stop') or ctrl.get('paused'):
        return False
    if not ctrl.get('player_present') and generated < headless:
        return True
    if not (ctrl.get('ready') and ctrl.get('player_present')):
        return False
    barrier = commit_seqs[-lag] if len(commit_seqs) >= lag else -1
    return ctrl.get('ack', -1) >= barrier


def _rows_for(cells, raw_map):
    rows = []
    for index, (target, values) in enumerate(sorted(cells.items())):
        flat = raw_map[np.asarray(values).reshape(-1)]
        positions = np.flatnonzero(flat != 0).astype(np.int32)
        rows.append(dict(row=index, step=0, **pack_row(target, positions, flat[positions])))
    return rows


def _air_bank(cfg, states):
    if cfg.get('air_bank'):
        return {str(k): list(v) for k, v in cfg['air_bank'].items()}
    return {str(states.index('minecraft:cave_air')): [2000000, 0, 0], str(states.index('minecraft:void_air')): [2000002, 0, 0]}


@dataclass(frozen=True)
class ElasticSettings:
    """Everything that tunes elastic mode (config key `elastic`; seconds unless noted)."""
    lost_after_s: float = 15.          # a ready or draining worker silent for this long is lost
    building_lost_after_s: float = 60.  # same while it is still loading the model
    drain_timeout_s: float = 60.
    no_worker_timeout_s: float = 120.  # no worker at all for this long ends the session
    hb_interval_s: float = 2.
    snapshot_chunk: int = 64           # cells per snapshot message
    max_pending_bytes: int = 512 << 20  # send backlog above which a slow peer is dropped
    min_workers: int = 1               # ready workers needed before the first `go`
    max_slots_per_worker: int = 64
    max_slots_total: int = 64          # ceiling of the batch a player can request
    hello_timeout_s: float = 10.


class Session:
    """Coordinator. Workers ask for windows with `want`; it answers with `pick` after sending them every commit.

    Elastic mode (`start_listener`: network workers that join, drain and get lost) is switched on explicitly; local mode
    spawns one process per configured GPU.
    """

    def __init__(self, run, cfg, clock=time.monotonic):
        self.clock = clock
        self.elastic = False
        self.poll_s = .05
        self.alive_check_s = 2.  # local mode: how often the worker processes are checked for being alive
        self.run, self.cfg = Path(run), cfg
        self.errors = []
        self.box = tuple(cfg['initial_box_zx'])
        self.fallback = [(self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2]
        self.mapping = Mapping.from_json(self.run / 'panel' / 'compact_mapping.json')
        self.store = Store(self.run, self.mapping)
        self.world = cfg['world']
        self.cells, self.biomes = self.store.load(self.world, self.box)
        self.initial = set(self.cells)
        self.inflight = set()
        self.batch_open = False
        self.batch_joined = set()
        self.epoch = 0
        self.handled_reset = None
        self.generated = 0
        self.commit_seqs = []
        self.finished_nfe = []
        self.usage = {}
        self.pub = Publisher(self.run)
        self.out = self.run / 'results'
        self.out.mkdir(exist_ok=False)
        (self.out / 'trace').mkdir()
        self.windows = []
        self.slots_per = int(cfg.get('slots_per_gpu', 8))
        self.devices = list(cfg.get('visible_gpus') or [])
        self.l1_cap = int(cfg.get('l1_cap', 8))
        self.l2_cap = int(cfg.get('l2_cap', 256))
        self.diagonal = bool(cfg.get('diagonal', False))
        self.last_eps = float(cfg.get('last_eps', 1e-4))
        self.frame_interval = float(cfg.get('frame_interval_s', .3))
        # False: no intermediate frames; each window reaches the viewer once, whole, when it commits.
        self.progressive = bool(cfg.get('progressive_frames', True))
        self.noted = 0.
        self.defaults = dict(steps=32, batch=self.slots_per, mode='dynamic', paused=False)
        self.defaults.update(cfg.get('defaults') or {})
        self.requested_batch = int(self.defaults['batch'])  # the live request after clamping (want and tick keep it current)
        self.headless = int(cfg.get('headless_windows', 0))
        self.max_windows = int(cfg.get('max_windows', 200))
        self.lag = max(1, int(cfg.get('max_display_lag_windows', self.slots_per)))
        self.deadline = self.clock() + float(cfg.get('max_gpu_seconds', 3600))
        self.idle_limit = float(cfg.get('idle_timeout_seconds', 300))
        self.idle_since = None
        states = load_registry()['idToState']
        self.states = states
        self.raw_map = preview_lookup('compact', json.loads((self.run / 'panel' / 'compact_mapping.json').read_text()))
        self.bank = _air_bank(cfg, states)
        self.members = {}
        self.owner = {}
        self.cseq = 0
        self.up = None
        self.procs = []
        self.gone, self.wire = set(), collections.deque()
        self.built, self.bye = set(), set()
        self.started = False
        self.stopping = None
        self.signalled = False
        self.retry = collections.deque()
        self.batch_expected = set()
        self.next_wid = 0
        self.stats = collections.Counter()
        self.listener = None
        self.authkey = None
        self.code_digest = None
        self.no_worker_since = None
        self.tick_at = 0.
        self.alive_checked = self.clock()
        self.build_failures = 0
        self.opts = ElasticSettings()
        self.unpublished = {}  # elastic: wid -> {target: packet} for windows committed by `done` whose `job` has not arrived

    def publish(self, kind, **data):
        return self.pub.publish(kind, **data)

    def send(self, wid, message):
        self.members[wid].link.send(message)

    def broadcast(self, message):
        for wid in sorted(self.members):
            if self.members[wid].state != CLOSED:
                self.send(wid, message)

    def active_members(self):
        """Members that take part in batch shares, ascending wid (locally: every worker, built or not)."""
        states = (READY,) if self.elastic else ACTIVE  # an elastic worker still building takes no share
        return [self.members[wid] for wid in sorted(self.members) if self.members[wid].state in states]

    def capacity_total(self):
        if not self.members and not self.elastic:
            return self.slots_per * len(self.devices)
        return sum(member.capacity for member in self.active_members())

    def max_batch(self):
        """Upper clamp for the requested batch: locally the live capacity, elastic the configured ceiling (never the current
        capacity, or a scaler would never see the demand)."""
        if not self.elastic:
            return self.capacity_total()
        return min(64, self.opts.max_slots_total)

    def reset(self, ctrl):
        request = int(ctrl.get('reset_seq') or 0)
        if self.handled_reset is None:
            self.handled_reset = request
            return False
        if request <= self.handled_reset:
            return False
        try:
            world = self.store.resolve(self.world, ctrl.get('world_req'))
        except ValueError as error:
            atomic_json(self.out / 'status.json', dict(state='reset_rejected', reason=str(error)))
            self.handled_reset = request
            return False
        cleared = sorted(self.cells)
        self.world = world
        self.cells, self.biomes = self.store.load(world, self.box)
        self.initial = set(self.cells)
        self.inflight.clear()
        self.owner.clear()
        self.cseq = 0
        self.batch_open = False
        self.batch_joined = set()
        self.unpublished.clear()
        if self.elastic:
            self.retry.clear()
            self.batch_expected = set()
        self.generated = 0
        self.epoch += 1
        self.handled_reset = request
        self.broadcast(('reset', self.epoch, world, dict(self.cells), self.biomes))
        frames = _rows_for(self.cells, self.raw_map)
        self.publish('reset', world=world, epoch=self.epoch, clear=[list(cell) for cell in cleared], frames=frames)
        return True

    def note(self, wid, active, idle):
        self.usage[wid] = dict(active=active, idle=idle)
        if self.clock() - self.noted < .5:
            return
        self.noted = self.clock()
        recent = self.finished_nfe[-32:]
        ctrl = dict(self.defaults)
        ctrl.update(_control(self.run, self.fallback))
        settings = clamp_settings(ctrl, self.defaults, self.max_batch())
        atomic_json(self.run / 'applied_settings.json', dict(
            steps=settings.get('steps', 32), batch=settings.get('batch', self.slots_per),
            mode=settings.get('mode', 'dynamic'), paused=settings.get('paused', False), world=self.world,
            windows=self.generated, inflight=len(self.inflight), slots=self.capacity_total(),
            idle=sum(item['idle'] for item in self.usage.values()), mean_nfe=(sum(recent) / len(recent) if recent else 0),
            epoch=self.epoch, unix_ns=time.time_ns()))

    def want(self, wid, idle_indexes, active_now):
        """Admission for one worker. Every commit already went to its queue, so the reply sees them all."""
        ctrl = _control(self.run, self.fallback)
        if self.reset(ctrl):
            return
        asked = dict(self.defaults)
        asked.update(ctrl)
        settings = clamp_settings(asked, self.defaults, self.max_batch())
        self.requested_batch = settings['batch']
        mode, paused = settings['mode'], settings['paused']
        room = self.max_windows - self.generated - len(self.inflight)
        active_members = self.active_members()
        ngpus = len(active_members)
        share = member_shares(settings['batch'], [(m.wid, m.capacity, m.capacity) for m in active_members]).get(wid, 0)
        # An elastic member that is draining or still building gets an empty pick (the worker blocks on the reply).
        eligible = not self.elastic or self.members[wid].state == READY
        if not eligible or paused or room <= 0 or not _allow_insert(dict(ctrl, paused=paused), self.generated, self.headless, self.commit_seqs, self.lag):
            chosen = []
        elif mode == 'fixed':
            if not self.inflight and not self.batch_open:
                self.batch_open = True
                self.batch_joined = set()
                if self.elastic:
                    self.batch_expected = {m.wid for m in active_members}
            if self.batch_open and wid not in self.batch_joined and (not self.elastic or wid in self.batch_expected):
                self.batch_joined.add(wid)
                chosen = list(idle_indexes)[:share]
                if self.elastic:
                    self._recheck_batch()
                elif len(self.batch_joined) >= ngpus:
                    self.batch_open = False
            else:
                chosen = []
        else:
            chosen = plan_inserts(idle_indexes, active_now, share, 'dynamic')
        picks = []
        for index in chosen:
            if room <= 0:
                break
            cell = self._take_retry() if self.retry else None
            if cell is None:
                cell = pick_window(set(self.cells), self.inflight, ctrl['player_zx'], diagonal=self.diagonal)
            if cell is None:
                break
            if self.elastic:
                self.trace_pick(wid, cell)
            self.inflight.add(cell)
            self.owner[cell] = wid
            picks.append((index, cell))
            room -= 1
        active = active_now + len(picks)
        if eligible:
            self.note(wid, active, self.members[wid].capacity - active)
        sleep = .02
        if active:
            self.idle_since = None
        else:
            waiting = paused or self.generated < self.headless
            quiet = (not waiting) and not self.inflight and (self.generated >= self.max_windows or (
                self.generated >= self.headless and not ctrl.get('player_present')))
            if quiet:
                if self.idle_since is None:
                    self.idle_since = self.clock()
                if self.clock() - self.idle_since > self.idle_limit or self.generated >= self.max_windows:
                    self.halt()
                sleep = .05
            else:
                self.idle_since = None
        self.send(wid, ('pick', self.epoch, picks, dict(world=self.world, cap=settings['steps'], mode=mode,
                                                         batch=settings['batch'], sleep=sleep)))

    def done(self, target, values, row):
        self.cells[target] = values
        self.inflight.discard(target)
        self.owner.pop(target, None)
        self.generated += 1
        self.finished_nfe.append(row['nfe'])
        self.windows.append(row)
        self.cseq += 1
        self.broadcast(('commit', self.epoch, target, values, self.cseq))
        with (self.out / 'window_times.jsonl').open('a') as times:
            times.write(json.dumps(row) + '\n')

    def job(self, wid, forward, cap, rows, packets):
        pending = self.unpublished.get(wid)
        if pending:
            for packet in packets:
                pending.pop(tuple(packet['target']), None)
        if rows:
            self.publish('frame', forward=forward, gpu=wid, world=self.world, cap=cap, inflight=len(self.inflight),
                         slots=self.capacity_total(), committed=self.generated, frames=rows)
        for packet in packets:
            self.commit_seqs.append(self.publish('window_commit', **packet))
        if packets:
            atomic_json(self.out / 'status.json', dict(state='running', world=self.world, generated=self.generated,
                                                       inflight=len(self.inflight), epoch=self.epoch))

    def handle(self, message):
        kind = message[0]
        if self.elastic and self.handle_membership(message):
            return
        if kind == 'error':
            self.errors.append(message[2])
            self.halt()
        elif kind == 'bye':
            self.bye.add(message[1])
            if self.elastic:
                self.on_bye(message[1])
        elif kind == 'built':
            self.built.add(message[1])
            if message[1] in self.members and self.members[message[1]].state == BUILDING:
                self.members[message[1]].state = READY
            if self.elastic:
                self.on_built(message[1], message[2])
        elif message[1] != self.epoch or (self.stopping is not None and kind in ('want', 'start')):
            return
        elif kind == 'want':
            self.want(*message[2:])
        elif kind == 'start':
            self.publish('window_start', **message[3])
        elif kind == 'done':
            _, _, wid, target, values, row, _ = message
            if self.elastic and self.owner.get(target) != wid:
                self.stats['stale_done'] += 1  # a lost worker's late result, or a window nobody owns: commit exactly once
                return
            self.done(target, values, row)
            if self.elastic:
                self.unpublished.setdefault(wid, {})[tuple(target)] = message[6]
                self.finish_drains()
        elif kind == 'job':
            self.job(*message[2:])

    # ---- elastic membership (network workers) -------------------------------------------------
    # Everything below runs on the coordinator's main thread. Network threads only put messages on `self.up`.

    def start_listener(self, address, authkey, settings=None, **overrides):
        """Listen for network workers. `settings` is an `ElasticSettings`; keyword overrides replace single fields."""
        self.opts = replace(settings or ElasticSettings(), **overrides)
        self.elastic = True
        self.up = queue.Queue()
        self.batch_expected = set()
        self.authkey = authkey
        self.listener = netchannel.Listener(tuple(address))  # authentication is done per connection, not in accept()
        self.address = self.listener.address
        self.closing = False
        raw = getattr(getattr(self.listener, '_listener', None), '_socket', None)
        if raw is not None:
            raw.settimeout(.2)  # so the accept loop can notice `closing`
        self.no_worker_since = self.clock()
        threading.Thread(target=self._accept_loop, name='sv-accept', daemon=True).start()
        return self.address

    def stop_listener(self):
        self.closing = True
        if self.listener is not None:
            try:
                self.listener.close()
            except OSError:
                pass

    def _accept_loop(self):
        while not self.closing:
            try:
                conn = self.listener.accept()
            except OSError:
                continue
            except Exception:
                if self.closing:
                    return
                continue
            threading.Thread(target=self._handshake, args=(conn,), name='sv-hello', daemon=True).start()

    def _handshake(self, conn):
        try:
            netchannel.deliver_challenge(conn, self.authkey)
            netchannel.answer_challenge(conn, self.authkey)
        except Exception:
            self.up.put(('rejected', 'auth'))
            conn.close()
            return
        try:
            if not conn.poll(self.opts.hello_timeout_s):
                raise TimeoutError
            hello = netchannel.loads(conn.recv_bytes(1 << 20))
            if not (isinstance(hello, tuple) and hello[0] == 'hello' and isinstance(hello[1], dict)):
                raise ValueError
        except Exception:
            self.up.put(('rejected', 'hello'))
            conn.close()
            return
        self.up.put(('join', conn, hello))

    def event(self, name, **fields):
        with (self.run / 'worker_events.jsonl').open('a') as log:
            log.write(json.dumps(dict(t=time.time(), event=name, **fields), default=str) + '\n')

    def trace_pick(self, wid, cell):
        with (self.run / 'pick_trace.jsonl').open('a') as log:
            log.write(json.dumps(dict(cseq=self.cseq, epoch=self.epoch, wid=wid, cell=list(cell),
                                      inflight=sorted(list(c) for c in self.inflight))) + '\n')

    def init_remote(self, wid, capacity):
        """What a joining worker needs besides the snapshot (no cells: those come as the snapshot stream)."""
        return dict(cfg=self.cfg, run=None, world=self.world, epoch=self.epoch, l1_cap=self.l1_cap, l2_cap=self.l2_cap,
                    last_eps=self.last_eps, air_ids=list(self.store.manifest['air_compact_ids']), frame_interval=self.frame_interval,
                    progressive=self.progressive, raw_map=self.raw_map, fallback=self.fallback,
                    gate_steps=int(self.defaults['steps']), parent=None, remote=True, wid=wid, capacity=capacity,
                    torch_device=0, nvsmi_index=None, run_gate=True, gate_cells=sorted(self.initial),
                    base_sha256=self.store.manifest.get('base_sha256'), control_sha256=self.store.manifest.get('control_sha256'))

    def snapshot_stream(self, snapshot, cseq0):
        """The join snapshot as a lazy message stream (compressed chunks). Tests replace this to hold the stream back."""
        return netchannel.snapshot_messages(self.epoch, cseq0, self.world, self.biomes, snapshot, chunk=self.opts.snapshot_chunk)

    def reject(self, conn, reason, detail=None):
        self.stats['rejected'] += 1
        self.event('reject', reason=reason, detail=detail)
        try:
            conn.send_bytes(netchannel.dumps(('reject', reason)))
        except OSError:
            pass
        try:
            conn.close()
        except OSError:
            pass

    def on_join(self, conn, hello):
        fields = hello[1]
        opts = self.opts
        if self.stopping is not None:
            return self.reject(conn, 'stopping')
        if fields.get('proto') != netchannel.PROTO:
            return self.reject(conn, 'protocol')
        if self.code_digest is not None and fields.get('code_digest') != self.code_digest:
            return self.reject(conn, 'code_digest', f'worker {str(fields.get("code_digest"))[:12]}, coordinator {self.code_digest[:12]}')
        for name in ('base', 'control'):
            got, expected = fields.get(f'{name}_sha256'), self.store.manifest.get(f'{name}_sha256')
            if got is not None and expected is not None and got != expected:
                return self.reject(conn, 'weights', f'{name} weights differ from the panel manifest')
        if fields.get('states_sha256') not in (None, states_digest(self.states)):
            return self.reject(conn, 'states')
        capacity = fields.get('capacity')
        if not isinstance(capacity, int) or isinstance(capacity, bool) or not 1 <= capacity <= 64:
            return self.reject(conn, 'capacity')
        capacity = min(capacity, int(opts.max_slots_per_worker))
        wid, self.next_wid = self.next_wid, self.next_wid + 1  # never reused
        # One main-thread step: snapshot point, broadcast-table entry. No commit can slip in between.
        snapshot, cseq0 = dict(self.cells), self.cseq
        head = [('welcome', wid, self.init_remote(wid, capacity), capacity)]
        prefix = itertools.chain(head, self.snapshot_stream(snapshot, cseq0))
        link = NetLink(conn, wid, self.up, prefix, hb_interval=opts.hb_interval_s, clock=lambda: self.clock(),
                       max_pending_bytes=opts.max_pending_bytes)
        member = Member(wid, link, capacity, state=BUILDING)
        member.extra['joined'] = self.clock()
        self.members[wid] = member
        self.stats['joined'] += 1
        self.stats['snapshot_bytes_last'] = sum(netchannel.approx_bytes(v) for v in snapshot.values())
        link.start()
        self.event('join', wid=wid, capacity=capacity, cseq0=cseq0, epoch=self.epoch, cells=len(snapshot))

    def on_built(self, wid, report):
        member = self.members.get(wid)
        if member is None or member.state != READY:
            return
        if report:
            atomic_json(self.out / f'numerical_gate_w{wid}.json', report)
            self.event('gate', wid=wid, report=report)
        self.build_failures = 0
        self.stats['join_to_ready_ms_last'] = int(1000 * (self.clock() - member.extra['joined']))
        self.event('ready', wid=wid, shares=self.shares())
        for other in sorted(self.members):
            if other != wid and self.members[other].extra.get('leave_pending'):
                self.on_leave(other, retry=True)
        if not self.started:
            ready = [m for m in self.members.values() if m.state == READY]
            if len(ready) >= self.opts.min_workers:
                self.started = True
                for m in ready:
                    self.send_go(m)
        else:
            self.send_go(member)

    def send_go(self, member):
        if not member.extra.get('go'):
            member.extra['go'] = True
            member.link.send(('go',))

    def shares(self):
        ready = self.active_members()
        return member_shares(self.requested_batch, [(m.wid, m.capacity, m.capacity) for m in ready])

    def on_leave(self, wid, retry=False):
        """Graceful drain request (SIGTERM on the worker, or an admin call). Returns whether it was accepted.
        A refusal (last ready worker) is remembered and approved when another worker becomes ready (`on_built`)."""
        member = self.members.get(wid)
        if member is None or member.state not in (READY, BUILDING):
            return False
        others = [m for m in self.members.values() if m is not member and m.state == READY]
        if member.state == READY and not others and self.stopping is None:
            member.extra['leave_pending'] = True
            if not retry:
                self.stats['leave_denied'] += 1
                self.event('leave_denied', wid=wid, reason='last_worker')
                member.link.send(('leave_denied', 'last_worker'))
            return False
        member.extra.pop('leave_pending', None)
        member.state = DRAINING
        member.extra['drain_t0'] = self.clock()
        self.batch_expected.discard(wid)
        self._recheck_batch()
        self.event('drain', wid=wid, owned=sum(1 for w in self.owner.values() if w == wid), shares=self.shares())
        self.finish_drains()
        return True

    def request_drain(self, wid):
        """Thread-safe: ask the main loop to drain a worker."""
        self.up.put(('leave', wid))

    def finish_drains(self):
        """Tell a draining member to stop once it owns no window; its `bye` closes it."""
        for member in self.members.values():
            if member.state == DRAINING and not member.extra.get('stop_sent') and wid_free(self.owner, member.wid):
                member.extra['stop_sent'] = True
                member.link.send(('stop',))

    def _recheck_batch(self):
        if self.batch_open and self.batch_expected <= self.batch_joined:
            self.batch_open = False

    def _publish_unpublished(self, wid):
        """A lost worker's finished windows are committed but their frame/commit packets rode a `job` that never came:
        publish each whole cell and its commit packet from the coordinator's copy."""
        pending = self.unpublished.pop(wid, None)
        if not pending:
            return
        rows, packets = [], []
        for target, packet in sorted(pending.items()):
            values = self.cells.get(target)
            if values is None:
                continue
            flat = self.raw_map[np.asarray(values).reshape(-1)]
            row = pack_row(target, np.arange(flat.size, dtype=np.int32), flat)
            if row['visible']:
                rows.append(dict(row, row=int(packet.get('gpu', wid)) * 64 + int(packet.get('slot', 0)), step=int(packet.get('nfe', 0))))
            packets.append(packet)
        self.job(wid, -1, int(packets[0].get('cap', 0)) if packets else 0, rows, packets)

    def _close_member(self, member, reason, hard):
        wid = member.wid
        if hard:
            member.link.close()  # fence first: nothing from this connection is read after this point
            self._publish_unpublished(wid)
        cells = sorted(cell for cell, owner in self.owner.items() if owner == wid)
        for cell in cells:
            self.inflight.discard(cell)
            del self.owner[cell]
            self.retry.append(cell)
        member.state = CLOSED
        member.extra['lost' if hard else 'graceful'] = reason
        self.usage.pop(wid, None)
        self.batch_expected.discard(wid)
        self._recheck_batch()
        if not hard:
            member.link.finish()
        self.stats['lost' if hard else 'closed'] += 1
        self.stats['reclaimed'] += len(cells)
        self.event('lost' if hard else 'closed', wid=wid, reason=reason, reclaimed=len(cells), shares=self.shares())
        self.finish_drains()

    def on_lost(self, wid, reason):
        member = self.members.get(wid)
        if member is not None and member.state != CLOSED:
            self._close_member(member, reason, hard=True)

    def on_bye(self, wid):
        member = self.members.get(wid)
        if member is None or member.state == CLOSED:
            return
        self._close_member(member, 'bye' if member.state == DRAINING or self.stopping is not None else 'bye_unexpected', hard=False)

    def _take_retry(self):
        """First reclaimed window that can start now. Committed or no longer ready ones are dropped, blocked ones wait."""
        keep, found = [], None
        ready = None
        while self.retry:
            cell = self.retry.popleft()
            if cell in self.cells or cell in self.inflight:
                continue
            if ready is None:
                ready = set(ready_cells(self.cells))
            if cell not in ready:
                continue
            if found is None and separated(cell, self.inflight, diagonal=self.diagonal):
                found = cell
            else:
                keep.append(cell)
        self.retry.extend(keep)
        return found

    def handle_membership(self, message):
        """Messages about who is connected. Returns True when consumed (they carry no epoch, so they come first)."""
        kind = message[0]
        if kind == 'join':
            self.on_join(message[1], message[2])
        elif kind == 'lost':
            self.on_lost(message[1], message[2])
        elif kind == 'leave':
            self.on_leave(message[1])
        elif kind == 'rejected':
            # Failed before `join` (wrong authkey, bad hello): counted and logged like a rejected join.
            self.stats['rejected'] += 1
            self.stats['rejected_' + message[1]] += 1
            self.event('reject', reason=message[1])
        elif kind == 'error':
            member = self.members.get(message[1])
            if member is None or member.state == CLOSED:
                return True
            if member.state == BUILDING:
                self.build_failures += 1
                self.event('build_failed', wid=member.wid, error=str(message[2])[:200], failures=self.build_failures)
                serving = any(m.state == READY for m in self.members.values())
                if self.build_failures >= 3 and not serving:
                    self.errors.append(message[2])
                    self.halt()
            self._close_member(member, 'error: ' + str(message[2])[:200], hard=True)
        elif kind in ('want', 'start', 'done', 'job', 'built', 'bye'):
            wid = message[2] if kind in ('want', 'start', 'done', 'job') else message[1]
            member = self.members.get(wid)
            if member is None or (member.state == CLOSED and (kind in ('want', 'built') or 'lost' in member.extra)):
                if kind == 'done':
                    self.stats['stale_done'] += 1
                return True
            return False
        else:
            return False
        return True

    def tick(self):
        now = self.clock()
        for member in list(self.members.values()):
            if member.state == CLOSED:
                continue
            limit = self.opts.building_lost_after_s if member.state == BUILDING else self.opts.lost_after_s
            if now - member.link.last_seen > limit:
                self.on_lost(member.wid, 'heartbeat')
            elif member.state == DRAINING and now - member.extra['drain_t0'] > self.opts.drain_timeout_s:
                self.on_lost(member.wid, 'drain_timeout')
        self.finish_drains()
        if any(m.state in (READY, BUILDING) for m in self.members.values()):
            self.no_worker_since = now
        elif self.stopping is None and now - self.no_worker_since > self.opts.no_worker_timeout_s:
            self.errors.append('no workers')
            self.halt()
        if self.clock() - self.tick_at >= .5:
            self.tick_at = self.clock()
            asked = dict(self.defaults)
            asked.update(_control(self.run, self.fallback))
            self.requested_batch = clamp_settings(asked, self.defaults, self.max_batch())['batch']
            atomic_json(self.run / 'metrics.json', self.metrics())

    def metrics(self):
        counts = collections.Counter(m.state for m in self.members.values())
        batch = self.requested_batch
        out = {f'sv_workers_{state}': counts.get(state, 0) for state in (BUILDING, READY, DRAINING, CLOSED)}
        out.update(sv_slots_total=self.capacity_total(), sv_requested_batch=batch, sv_effective_batch=min(batch, self.capacity_total()),
                   sv_inflight=len(self.inflight), sv_generated_total=self.generated, sv_wire_backlog=len(self.wire),
                   sv_lost_total=self.stats['lost'], sv_reclaimed_windows_total=self.stats['reclaimed'],
                   sv_stale_done_total=self.stats['stale_done'], sv_joined_total=self.stats['joined'],
                   sv_rejected_total=self.stats['rejected'], sv_leave_denied_total=self.stats['leave_denied'],
                   sv_build_failures_total=self.build_failures, sv_epoch=self.epoch, sv_cseq=self.cseq,
                   sv_snapshot_bytes_last=self.stats['snapshot_bytes_last'], sv_join_to_ready_ms_last=self.stats['join_to_ready_ms_last'],
                   sv_shares={str(k): v for k, v in self.shares().items()})
        return out

    def metrics_text(self):
        lines = []
        for key, value in self.metrics().items():
            if isinstance(value, dict):
                lines += [f'{key}{{wid="{wid}"}} {share}' for wid, share in value.items()]
            else:
                lines.append(f'{key} {value}')
        return '\n'.join(lines) + '\n'

    def halt(self):
        if self.stopping is None:
            self.stopping = self.clock()
            self.broadcast(('stop',))

    def init_for(self, rank, gate_steps):
        """Worker start-up payload. Worker `rank` runs on physical GPU `visible_gpus[rank]`: its process sees only that card
        (`cuda_env`), so `torch_device` is 0 and telemetry reads nvidia-smi row `visible_gpus[rank]`. The worker id, the
        frame rows (`rank * 64 + slot`) and the gate (rank 0) stay keyed by rank."""
        physical = int(self.devices[rank])
        return dict(cfg=self.cfg, run=str(self.run), world=self.world, cells=dict(self.cells), biomes=self.biomes,
                    l1_cap=self.l1_cap, l2_cap=self.l2_cap, last_eps=self.last_eps,
                    air_ids=list(self.store.manifest['air_compact_ids']), frame_interval=self.frame_interval, progressive=self.progressive,
                    raw_map=self.raw_map, fallback=self.fallback,
                    gate_steps=gate_steps, parent=os.getpid(),
                    wid=rank, torch_device=0, nvsmi_index=physical, cuda_env=local_gpu_env(physical),
                    run_gate=(rank == 0), capacity=self.slots_per)

    def start_local_workers(self, context=None):
        """One spawn process per configured GPU, each with its own command queue; all share the `up` queue.
        Each child is started with CUDA_VISIBLE_DEVICES set to its one physical GPU, before Python (and CUDA) starts there."""
        if len(set(self.devices)) != len(self.devices):
            raise ValueError(f'visible_gpus has a repeated GPU: {self.devices}')
        ctx = context or multiprocessing.get_context('spawn')
        self.up = ctx.Queue()
        self.procs = []
        inits = []
        for rank in range(len(self.devices)):
            commands = ctx.Queue()
            init = self.init_for(rank, int(self.defaults['steps']))
            proc = ctx.Process(target=run_worker, args=(rank, init, commands, self.up), name=f'live-gpu-{rank}')
            self.members[rank] = Member(rank, ProcLink(commands, proc), self.slots_per, state=BUILDING)
            self.procs.append(proc)
            inits.append(init)
        for proc, init in zip(self.procs, inits):
            with _environment(init['cuda_env']):
                proc.start()

    def pump_once(self):
        """One coordinator iteration. Returns False once the session is over."""
        procs = self.procs
        if self.stopping is None and (self.clock() > self.deadline or self.signalled):
            self.halt()
        if self.elastic:
            self.tick()
            over = all(member.state == CLOSED for member in self.members.values())
            if self.stopping is not None and ((over and not self.wire) or self.clock() - self.stopping > 60):
                return False
        else:
            if not self.started and len(self.built) == len(procs):
                self.started = True
                self.broadcast(('go',))
            if self.stopping is not None and ((len(self.bye | self.gone) == len(procs) and not self.wire) or self.clock() - self.stopping > 60):
                return False
        control = []
        try:
            message = self.up.get(timeout=0 if self.wire else self.poll_s)
            while True:
                (self.wire if message[0] in ('start', 'job') else control).append(message)
                message = self.up.get_nowait()
        except queue.Empty:
            pass
        if not self.elastic and self.clock() - self.alive_checked >= self.alive_check_s:
            self.alive_checked = self.clock()  # on a fixed interval, whatever traffic there is
            for rank, proc in enumerate(procs):
                if rank not in self.bye and rank not in self.gone and not proc.is_alive():
                    self.gone.add(rank)
                    self.errors.append(f'gpu{rank}: worker exited with code {proc.exitcode}')
                    self.halt()
        if not control and not self.wire:
            return True
        # Scheduling answers go first; frames and window packets (file writes) wait one message at a time.
        for message in control:
            self.handle(message)
        if self.wire:
            self.handle(self.wire.popleft())
        return True

    def shutdown(self):
        if self.elastic:
            self.stop_listener()
            for member in self.members.values():
                if member.state != CLOSED:
                    member.link.finish()
            atomic_json(self.run / 'metrics.json', self.metrics())  # final counts, after the last bye
            return
        for proc in self.procs:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=5)
            if proc.is_alive():
                proc.kill()
        self.up.cancel_join_thread()
        self.up.close()
        for member in self.members.values():
            member.link.close()

    def coordinate(self):
        """Run the workers until they stop. Errors, the deadline, a signal, or the idle/window limits end it."""
        if self.cfg.get('listen'):
            self.start_listener(tuple(self.cfg['listen']), netchannel.read_authkey(), ElasticSettings(**(self.cfg.get('elastic') or {})))
            self.code_digest = package_digest()  # joiners must run exactly this code
        else:
            self.start_local_workers()
        try:
            while self.pump_once():
                pass
        finally:
            self.shutdown()


def serve(run):
    run = Path(run)
    cfg = _cfg(run)
    _require(cfg)
    _serve(Session(run, cfg))


def _serve(session):
    try:
        frozen = json.loads((session.run / 'frozen_source_manifest.json').read_text())
        for name, hashed in frozen['files'].items():
            if digest(name) != hashed:
                raise RuntimeError(f'Source changed after prepare: {name}')
        for number in (signal.SIGTERM, signal.SIGINT):
            signal.signal(number, lambda *_: setattr(session, 'signalled', True))
        session.publish('session', world=session.world, worlds=session.store.ids, initial_box_zx=list(session.box),
                        initial_cells=[list(cell) for cell in sorted(session.initial)], slots_per_gpu=session.slots_per,
                        gpus=len(session.devices), defaults=session.defaults, air_bank=session.bank,
                        states_sha256=states_digest(session.states))
        session.coordinate()
    except Exception as error:  # the viewer must see a failure and the run folder must say why
        session.errors.append(f'{type(error).__name__}: {error}')
    if session.errors:
        try:
            session.publish('failed', error=session.errors)
        except Exception:
            pass
        atomic_json(session.run / 'error.json', dict(errors=session.errors))
        raise RuntimeError(session.errors)
    session.publish('complete', new_windows=session.generated, world=session.world)
    atomic_json(session.out / 'windows.json', session.windows)
    summary = dict(state='session_ended', generated=session.generated, world=session.world,
                   nfe_mean=(sum(session.finished_nfe) / len(session.finished_nfe) if session.finished_nfe else 0),
                   nfe_min=(min(session.finished_nfe) if session.finished_nfe else 0),
                   nfe_max=(max(session.finished_nfe) if session.finished_nfe else 0),
                   nfe=list(session.finished_nfe))
    atomic_json(session.out / 'summary.json', summary)
    print(json.dumps({k: v for k, v in summary.items() if k != 'nfe'}), flush=True)


def gate_config(cfg):
    """The config of a gate run: one local GPU, no listener (a gate never waits for network workers), no windows."""
    local = {key: value for key, value in cfg.items() if key not in ('listen', 'elastic')}
    return dict(local, visible_gpus=list(cfg.get('visible_gpus') or [0])[:1], headless_windows=0, max_windows=0,
                max_gpu_seconds=900, idle_timeout_seconds=1, defaults=dict(cfg.get('defaults') or {}, paused=False))


def gate(run):
    """Load one GPU and run the numerical gate, then exit before the live loop."""
    run = Path(run)
    cfg = _cfg(run)
    _require(cfg)
    _serve(Session(run, gate_config(cfg)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'gate', 'run', 'fresh'))
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare(args.run)
    elif args.mode == 'gate':
        gate(args.run)
    elif args.mode == 'fresh':
        fresh(args.run)
    else:
        serve(args.run)


if __name__ == '__main__':
    main()
