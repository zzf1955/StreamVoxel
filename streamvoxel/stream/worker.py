"""One GPU of the low-entropy-last session in its own process, so two GPUs never share a GIL.

The coordinator (`session.py`) owns the scheduler state and the wire. Messages are tuples whose second field
(after the kind) is the epoch, and anything from an older epoch is dropped.

Coordinator -> worker, one FIFO queue per worker:
  ('go',) ('stop',) ('reset', epoch, world, cells, biomes) ('commit', epoch, target, values[, cseq]) ('pick', epoch, picks, plan)
Worker -> coordinator, one shared queue:
  ('built', wid, report) ('want', epoch, wid, idle, active) ('start', epoch, wid, packet) ('done', epoch, wid, target, values, row, packet)
  ('job', epoch, wid, forward, cap, rows, packets) ('bye', wid) ('error', wid, text)
`wid` is the worker id (locally the rank); `report` is None locally. Which cuda device, which nvidia-smi row and who runs the
numerical gate come from `init` (`torch_device`, `nvsmi_index`, `run_gate`), not from the id. A local worker of rank r is
started with CUDA_VISIBLE_DEVICES=visible_gpus[r] (`init['cuda_env']`), so its device is cuda:0 and its nvidia-smi row
is visible_gpus[r]; frame rows stay keyed by the worker id (`wid * 64 + slot`).

The coordinator sends every commit to every worker before it answers a `want`, so the `pick` reply finds the
worker's replica holding every commit the coordinator knew when it chose the windows.
"""
from __future__ import annotations

import json
from pathlib import Path
import os
import queue
import time
import zlib

import numpy as np

from streamvoxel.common.io import atomic_json
from streamvoxel.common.live_rows import pack_row
from streamvoxel.stream.channel import loads


class Phases:
    """Wall time per worker-loop phase and CUDA-event time of each forward, written as means every `every` forwards.

    `gpu_busy` is forward GPU time over wall time; the rest of the wall is CPU work, queue waits and idle sleeps.
    """

    def __init__(self, path, every=50):
        self.path, self.every = Path(path), int(every)
        self.reset()

    def reset(self):
        self.total = {}
        self.events = []
        self.forwards = 0
        self.active = 0
        self.started = self.mark = time.perf_counter()

    def lap(self, name):
        now = time.perf_counter()
        self.total[name] = self.total.get(name, 0.) + now - self.mark
        self.mark = now

    def forward(self, start, end, active):
        self.events.append((start, end))
        self.forwards += 1
        self.active += active
        if self.forwards >= self.every:
            self.flush()

    def flush(self):
        if not self.forwards:
            return
        end = self.events[-1][1]
        end.synchronize()
        gpu = sum(a.elapsed_time(b) for a, b in self.events)
        wall = time.perf_counter() - self.started
        n = self.forwards
        row = dict(forwards=n, wall_s=round(wall, 3), forward_ms=round(wall * 1e3 / n, 2), gpu_ms=round(gpu / n, 2),
                   gpu_busy=round(gpu / 1e3 / wall, 3), active_mean=round(self.active / n, 2),
                   phases_ms={k: round(v * 1e3 / n, 2) for k, v in sorted(self.total.items(), key=lambda kv: -kv[1])},
                   unix_ns=time.time_ns())
        atomic_json(self.path, row)
        with self.path.with_suffix('.jsonl').open('a') as handle:
            handle.write(json.dumps(row) + '\n')
        self.reset()


def _frame_row(commit, raw_map, slot, gpu):
    row = pack_row(commit['target'], commit['positions'], raw_map[commit['values']])
    if not row['visible']:
        return None
    row.update(row=gpu * 64 + slot, step=commit['step'])
    return row


def _merge_commits(commits):
    """One frame commit from several forwards of one window; later steps win on a repeated position."""
    if len(commits) == 1:
        return commits[0]
    positions = np.concatenate([c['positions'] for c in commits])
    values = np.concatenate([c['values'] for c in commits])
    order = np.argsort(positions, kind='stable')
    positions, values = positions[order], values[order]
    keep = np.append(positions[1:] != positions[:-1], True)
    return dict(positions=positions[keep], values=values[keep], step=commits[-1]['step'], target=commits[-1]['target'])


def encode_rows(merged, rank, raw_map):
    rows = []
    for index, commit in sorted(merged.items()):
        row = _frame_row(commit, raw_map, index, rank)
        if row is not None:
            rows.append(row)
    return rows


class ReplicaGap(RuntimeError):
    """A commit skipped a sequence number: the replica can no longer be trusted (re-join)."""


class Replica:
    """The worker's copy of the committed cells and biomes, kept current by the coordinator's broadcasts."""

    def __init__(self, world, cells, biomes):
        self.epoch = 0
        self.world = world
        self.cells = dict(cells)
        self.biomes = biomes
        self.last_cseq = 0
        self.snapshot = None  # None: no snapshot in progress; else the number of cells announced by the head
        self.snapshot_done = False  # a network worker is ready for the gate once its snapshot has arrived

    def apply(self, message):
        """Apply a state message. Returns 'reset' or 'commit' when it changed the replica, else None (stale or not state)."""
        kind = message[0]
        if kind == 'reset':
            _, epoch, world, cells, biomes = message
            if epoch <= self.epoch:
                return None
            self.epoch, self.world, self.cells, self.biomes = epoch, world, dict(cells), biomes
            self.last_cseq = 0
            return 'reset'
        if kind == 'snapshot':
            # Join: the full committed state at sequence number cseq0; commits cseq0+1.. follow on the same stream.
            _, epoch, cseq0, world, biomes, count = message
            self.epoch, self.world, self.biomes, self.cells = epoch, world, biomes, {}
            self.last_cseq, self.snapshot, self.snapshot_done = cseq0, count, False
            return 'snapshot'
        if kind == 'snapshot_chunk':
            if self.snapshot is None:
                return None
            for target, values in loads(zlib.decompress(message[2])):
                self.cells[tuple(target)] = values
            return 'snapshot'
        if kind == 'snapshot_end':
            if self.snapshot is not None and len(self.cells) != self.snapshot:
                raise ReplicaGap(f'snapshot has {len(self.cells)} of {self.snapshot} cells')
            self.snapshot, self.snapshot_done = None, True
            return 'snapshot'
        if kind == 'commit':
            _, epoch, target, values, cseq = message
            if epoch != self.epoch:
                return None
            if cseq <= self.last_cseq:
                return None  # already applied (idempotent)
            if cseq != self.last_cseq + 1:
                raise ReplicaGap(f'commit {cseq} after {self.last_cseq}')
            self.last_cseq = cseq
            self.cells[tuple(target)] = values
            return 'commit'
        return None


class Inbox:
    """Reads the coordinator's queue in order, applies state messages to the replica, and returns the rest."""

    def __init__(self, source, replica):
        self.source, self.replica = source, replica
        self.stopped = False

    def next(self, timeout=None):
        """The next non-state message (or None on timeout). A reset is returned as ('reset',) after it was applied."""
        try:
            message = self.source.get(timeout=timeout) if timeout else self.source.get_nowait()
        except queue.Empty:
            return None
        if message[0] == 'stop':
            self.stopped = True
            return message
        applied = self.replica.apply(message)
        if applied == 'reset':
            return ('reset',)
        if applied == 'commit':
            return ('commit',)
        if applied == 'snapshot':
            return ('snapshot',)
        return message

    def reply(self, epoch, alive=lambda: True):
        """Wait for the `pick` reply of `epoch`. Returns it, or None when a reset/stop supersedes it or `alive()` turns false.

        It never gives up on a timeout: the coordinator has already counted the picks as in flight.
        """
        while alive():
            message = self.next(.2)
            if message is None:
                continue
            if message[0] in ('stop', 'reset'):
                return None
            if message[0] == 'pick' and message[1] == epoch:
                return message
        return None


class Worker:
    def __init__(self, rank, init, commands, up):
        self.init, self.up = init, up
        self.wid = init['wid']
        self.torch_device = init['torch_device']
        self.run_gate = init['run_gate']
        self.cfg = init['cfg']
        self.run = Path(init['run'])
        self.replica = Replica(init['world'], init['cells'], init['biomes'])
        self.inbox = Inbox(commands, self.replica)
        self.parent = init['parent']
        self.leave_event = None   # network worker: set by SIGTERM; the loop (not the signal handler) sends `leave`
        self.leave_sent = False

    def maybe_leave(self):
        if self.leave_event is not None and self.leave_event.is_set() and not self.leave_sent:
            self.leave_sent = True
            self.up.put(('leave', self.wid))

    def nvsmi_index(self):
        """Physical GPU for telemetry (None: sample every GPU)."""
        return self.init['nvsmi_index']

    def orphaned(self):
        if self.init.get('remote'):
            return bool(getattr(self.inbox.source, 'dead', False))  # network worker: the coordinator connection is gone
        return os.getppid() != self.parent

    def _await_snapshot(self):
        """Network worker: the join snapshot streams in while the engine builds; the gate needs all of it."""
        while not self.replica.snapshot_done:
            self.maybe_leave()
            if self.inbox.stopped or self.orphaned():
                return False
            self.inbox.next(.2)
        return True

    def run_loop(self):
        import torch
        from streamvoxel.stream.telemetry import Telemetry
        from streamvoxel.stream.engine import SlotForward, open_model
        from streamvoxel.stream.scheduler import make_item
        from streamvoxel.stream.slot import LowEntropyLastSlot
        wid, cfg = self.wid, self.cfg
        device = torch.device(f'cuda:{self.torch_device}')
        torch.cuda.set_device(device)
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = True
        telemetry = Telemetry(self.run, wid, gpu_index=self.nvsmi_index())
        log = (self.run / f'slot_log_g{wid}.jsonl').open('a')
        try:
            if not cfg.get('reserve_memory'):
                self._limit_memory(device)
            model, spec, _ = open_model(cfg['base'], cfg['control'], device)
            if self.init.get('remote'):
                from streamvoxel.stream.integrity import verify_weights
                verify_weights(dict(base=cfg['base'], control=cfg['control']),
                               dict(base=self.init.get('base_sha256'), control=self.init.get('control_sha256')))
            slots_per, l2_cap = self.init['capacity'], self.init['l2_cap']
            fast = cfg.get('fast_ops', True)
            if fast:
                # Packed cached K/V + FlexAttention, gather patchify, compiled; sampling bitwise equal to LowEntropyLastSlot.
                from streamvoxel.stream.fast_ops import RECOMMENDED, FastLowEntropyLastSlot, FastSlotForward, SelectGraphs
                engine = FastSlotForward(model, spec, device, slots_per, self.init['l1_cap'], l2_cap, **RECOMMENDED)
            else:
                engine = SlotForward(model, spec, device, slots_per, self.init['l1_cap'], l2_cap)
            torch.cuda.set_device(device)
            air = tuple(self.init['air_ids'])
            slots = [(FastLowEntropyLastSlot if fast else LowEntropyLastSlot)(engine.shape, spec.num_classes, air, air_token=0, last_eps=self.init['last_eps'],
                                                       block_size=int(cfg.get('block_size', 4)), device=device)
                     for _ in range(slots_per)]
            self.select = SelectGraphs(engine, slots) if fast else None
            if self.init.get('remote') and not self._await_snapshot():
                return
            report = self._gate(engine, spec) if self.run_gate else None
            # Locally the gate result is only written to results/numerical_gate.json; a network worker also reports it.
            self.up.put(('built', wid, report if self.init.get('remote') else None))
            if not self._await_go():
                return
            reserve = self._reserve_memory(device)
            self._loop(torch, engine, spec, slots, reserve, log, make_item)
        finally:
            log.close()
            telemetry.close()

    def _await_go(self):
        while not self.inbox.stopped:
            self.maybe_leave()
            message = self.inbox.next(.2)
            if message is not None and message[0] == 'go':
                return True
            if self.orphaned():
                return False
        return False

    def _loop(self, torch, engine, spec, slots, reserve, log, make_item):
        rank, init = self.wid, self.init  # `rank` below is the worker id (locally equal to the rank)
        if self.select is not None:
            from streamvoxel.stream.fast_ops import consume_batch
        replica = self.replica
        phases = Phases(self.run / f'timing_r{rank}.json')
        out = self.run / 'results' / 'trace'
        frame_interval = init['frame_interval']
        progressive = init['progressive']
        pending, deferred = {}, None
        last_frame = time.monotonic()
        seen = replica.epoch
        forwards = 0

        def flush(job):
            if job is None:
                return
            epoch, forward, cap, merged, traces, packets = job
            rows = encode_rows(merged, rank, init['raw_map']) if merged else []
            for target, finished in traces:
                np.savez_compressed(out / f'{target[0]}_{target[1]}.npz', commit_step=finished['step_map'].numpy(),
                                    values=finished['values'].numpy(), target=np.asarray(target), nfe=finished['nfe'],
                                    last_frac=finished['last_frac'], cap=finished['cap'])
            if rows or packets:
                self.up.put(('job', epoch, rank, forward, cap, rows, packets))

        while not self.inbox.stopped:
            if self.orphaned():
                break
            self.maybe_leave()
            if reserve:
                reserve.ensure()
            phases.lap('reserve')
            while self.inbox.next() is not None:
                pass  # ('leave_denied', ...) needs no action: the coordinator keeps the request and drains this worker later
            if self.inbox.stopped:
                break
            if seen != replica.epoch:
                for slot in slots:
                    slot.drop()
                engine.clear_tokens()
                pending.clear()
                deferred = None
                seen = replica.epoch
            epoch = replica.epoch
            idle_indexes = [index for index, slot in enumerate(slots) if slot.idle]
            self.up.put(('want', epoch, rank, idle_indexes, len(slots) - len(idle_indexes)))
            answer = self.inbox.reply(epoch, lambda: not self.orphaned())
            phases.lap('schedule')
            if self.inbox.stopped:
                break
            if answer is None:
                continue
            _, _, picks, plan = answer
            if replica.epoch != epoch:
                continue
            world, cap = plan['world'], plan['cap']
            for index, cell in picks:
                item = make_item(replica.cells, cell, replica.biomes, spec, world, cap, init['l2_cap'])
                phases.lap('item')
                reserve.retry(lambda: engine.load_slot(index, item)) if reserve else engine.load_slot(index, item)
                phases.lap('encode')
                slots[index].arm(item.seed, cap, cell)
                slots[index].t0 = time.perf_counter()
                slots[index].t0_ns = time.time_ns()
                self.up.put(('start', epoch, rank, dict(target=list(cell), slot=index, gpu=rank, world=world, cap=cap,
                                                        seed=item.seed, mode=plan['mode'], batch=plan['batch'])))
                phases.lap('publish')
            if not any(not slot.idle for slot in slots):
                flush(deferred)
                deferred = None
                time.sleep(plan['sleep'])
                phases.lap('idle')
                continue
            engine.wait_loads()
            count = 1 + max(index for index, slot in enumerate(slots) if not slot.idle)
            timesteps = [slot.time(engine.tokens[index]) if index < count else 1.0 for index, slot in enumerate(slots)]
            phases.lap('timestep')
            began, ended = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            began.record()
            logits = engine.replay(timesteps, count)
            ended.record()
            phases.lap('launch')
            # The previous forward's frame encode and trace writes run while this forward is on the GPU.
            flush(deferred)
            deferred = None
            phases.lap('encode_rows')
            ended.synchronize()
            phases.lap('gpu_wait')
            forwards += 1
            done = []
            targets = {index: slots[index].target for index in range(count)}
            if self.select is not None:
                results = consume_batch(slots, logits, engine.tokens, count, self.select)
            else:
                results = {index: slots[index].consume(logits[index], engine.tokens[index]) for index in range(count) if not slots[index].idle}
            for index, (finished, commit) in results.items():
                target = targets[index]
                if commit is not None and progressive:
                    pending.setdefault(index, []).append(dict(commit, target=target))
                if finished is not None:
                    engine.tokens[index].fill_(spec.num_classes)
                    done.append((index, finished))
            phases.lap('consume')
            traces, packets = [], []
            for index, finished in done:
                target = tuple(finished['target'])
                seconds = round(time.perf_counter() - getattr(slots[index], 't0', time.perf_counter()), 4)
                row = dict(target=list(target), nfe=finished['nfe'], last_frac=finished['last_frac'], cap=finished['cap'],
                           seconds=seconds, started_unix_ns=int(getattr(slots[index], 't0_ns', time.time_ns())),
                           finished_unix_ns=time.time_ns(), world=replica.world)
                packet = dict(row, gpu=rank, slot=index, actual=finished['actual'])
                self.up.put(('done', epoch, rank, target, np.ascontiguousarray(finished['values'].numpy(), np.int16), row, packet))
                traces.append((target, finished))
                packets.append(packet)
            merged = {}
            if not progressive:
                # One row per finished window: its whole cell, sent with the window's commit.
                merged = {index: dict(positions=np.arange(finished['values'].numel(), dtype=np.int32),
                                      values=finished['values'].numpy().reshape(-1), step=finished['nfe'], target=tuple(finished['target']))
                          for index, finished in done}
            elif pending and (done or time.monotonic() - last_frame >= frame_interval):
                merged = {index: _merge_commits(commits) for index, commits in pending.items()}
                pending.clear()
                last_frame = time.monotonic()
            phases.lap('merge')
            log.write(json.dumps(dict(forward=forwards, bucket=engine.bucket(count), slots=[
                dict(phase=slot.phase, target=None if slot.target is None else list(slot.target), nfe=slot.nfe,
                     left=(None if slot.quotas is None else len(slot.quotas) - slot.solid_calls))
                for slot in slots])) + '\n')
            log.flush()
            phases.lap('log')
            if merged or packets:
                deferred = (epoch, forwards, cap, merged, traces, packets)
            phases.forward(began, ended, count)
        flush(deferred)

    def _limit_memory(self, device):
        """Cap this process so the last `leave_free_gb` cannot be consumed by later allocations."""
        import torch
        leave = float(self.cfg.get('leave_free_gb') or 0)
        if leave <= 0:
            return
        total = torch.cuda.get_device_properties(device).total_memory
        fraction = min(0.98, max(0.5, (total - leave * 1024 ** 3) / total))
        torch.cuda.set_per_process_memory_fraction(fraction, device)
        print(json.dumps(dict(event='memory_fraction', fraction=round(fraction, 4), leave_free_gb=leave)), flush=True)

    def _reserve_memory(self, device):
        """Hold leftover GPU memory so another job cannot start on this card; `ensure` hands it back on demand."""
        from streamvoxel.stream.reserve import GpuReserve
        if not self.cfg.get('reserve_memory'):
            return None
        reserve = GpuReserve(device, int(float(self.cfg.get('leave_free_gb', 2)) * 1024 ** 3), rank=self.wid)
        reserve.hold()
        return reserve

    def _gate(self, engine, spec):
        from streamvoxel.stream.scheduler import make_item, pick_window
        cells, biomes, world = self.replica.cells, self.replica.biomes, self.replica.world
        if self.init.get('gate_cells') is not None:
            # Joiner: gate on the session's initial cells only (their content never changes), not on mid-run generated tiles.
            cells = {c: cells[c] for c in map(tuple, self.init['gate_cells']) if c in cells}
        fallback = self.init['fallback']
        first = pick_window(set(cells), set(), fallback)
        second = pick_window(set(cells), {first}, fallback)
        if first is None or second is None:
            raise RuntimeError('Initial frontier cannot fill two separated windows')
        cap, l2_cap = int(self.init['gate_steps']), self.init['l2_cap']
        item = make_item(cells, first, biomes, spec, world, cap, l2_cap)
        other = make_item(cells, second, biomes, spec, world, cap, l2_cap)
        report = engine.numerical_gate(item, other)
        atomic_json(self.run / 'results' / 'numerical_gate.json', report)
        print(json.dumps(dict(gate=report)), flush=True)
        if (report['graph_used'] and not report['graph_equal']) or not report['padded_close'] or not report['bucket_close']:
            raise RuntimeError(f'Numerical gate failed: {report}')
        return report


def run_worker(rank, init, commands, up):
    """Process entry point (spawn)."""
    import traceback
    wid = init['wid']
    # A local worker sees only its physical GPU (visible_gpus[rank]); set again here, before torch touches CUDA.
    os.environ.update(init.get('cuda_env') or {})
    try:
        worker = Worker(rank, init, commands, up)
        wid = worker.wid
        worker.run_loop()
        up.put(('bye', wid))
    except BaseException as error:
        traceback.print_exc()
        up.put(('error', wid, f'gpu{wid}: {type(error).__name__}: {error}'))
