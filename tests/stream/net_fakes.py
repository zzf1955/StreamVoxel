"""Fake network workers (no torch, no GPU) that speak the real worker protocol over real `multiprocessing.connection` channels."""
from __future__ import annotations

import threading
import time

import numpy as np

from streamvoxel.stream import channel as netchannel
from streamvoxel.stream.scheduler import separated
from streamvoxel.stream.worker import Inbox, Replica

KEY = b'k' * 32


class Tee:
    """Records everything the coordinator sent to a worker, in order, before the Inbox consumes it."""

    def __init__(self, channel):
        self.channel, self.log = channel, []

    @property
    def dead(self):
        return self.channel.dead

    def get(self, timeout=None):
        message = self.channel.get(timeout=timeout)
        self.log.append(message)
        return message

    def get_nowait(self):
        message = self.channel.get_nowait()
        self.log.append(message)
        return message


class FakeNetWorker:
    """Connects in a background thread (the coordinator's main thread is the test thread and must keep pumping)."""

    def __init__(self, address, *, capacity=4, key=KEY, hb_interval=.05, lost_after=30., ticks=(1, 3), rng=None, code_digest=None, auto=True, base_sha256=None, control_sha256=None):
        self.capacity, self.ticks, self.rng, self.auto = capacity, ticks, rng, auto
        self.channel = self.tee = self.inbox = self.replica = self.wid = None
        self.error = None
        self.connected = threading.Event()
        self.active = {}          # slot -> [cell, ticks left]
        self.waiting = False
        self.go = self.built = self.left = self.closed = False
        self.picks_seen = []
        self.done_sent = []
        self.thread = threading.Thread(target=self._connect, args=(address, key, hb_interval, lost_after, code_digest, base_sha256, control_sha256), daemon=True)
        self.thread.start()

    def _connect(self, address, key, hb_interval, lost_after, code_digest, base_sha256=None, control_sha256=None):
        try:
            hello = dict(capacity=self.capacity, code_digest=code_digest, base_sha256=base_sha256, control_sha256=control_sha256)
            self.channel, welcome = netchannel.connect(address, key, hello, hb_interval=hb_interval, lost_after=lost_after)
            self.wid = welcome[1]
            self.replica = Replica(welcome[2]['world'], {}, None)
            self.tee = Tee(self.channel)
            self.inbox = Inbox(self.tee, self.replica)
        except BaseException as error:  # noqa: BLE001 - surfaced to the test
            self.error = error
        self.connected.set()

    @property
    def log(self):
        return self.tee.log if self.tee else []

    @property
    def stopped(self):
        return bool(self.inbox and self.inbox.stopped)

    def pump(self):
        """Apply whatever arrived; remember picks and go."""
        if self.inbox is None or self.closed:
            return
        while True:
            message = self.inbox.next()
            if message is None:
                break
            if message[0] == 'go':
                self.go = True
            elif message[0] == 'leave_denied':
                self.left = False
            elif message[0] == 'pick' and message[1] == self.replica.epoch:
                self.waiting = False
                self.picks_seen.append(message)
                for index, cell in message[2]:
                    self.active[index] = [tuple(cell), self.rng.randint(*self.ticks) if self.rng else 1]
        if self.stopped and not self.closed:
            self.channel.put(('bye', self.wid))
            self.channel.close()
            self.closed = True

    def synced(self):
        return self.replica is not None and self.replica.snapshot_done

    def step(self, finish=True):
        """One scheduling step of the fake GPU loop: build, ask for work, tick the active windows, send results."""
        self.pump()
        if self.inbox is None or self.closed:
            return
        if self.synced() and not self.built:
            self.built = True
            self.channel.put(('built', self.wid, None))
            return
        if not self.go or not self.auto:
            return
        if finish:
            for index in sorted(self.active):
                self.active[index][1] -= 1
                if self.active[index][1] <= 0:
                    self.finish(index)
        if not self.waiting:
            self.want()

    def want(self):
        idle = [i for i in range(self.capacity) if i not in self.active]
        self.waiting = True
        self.channel.put(('want', self.replica.epoch, self.wid, idle, self.capacity - len(idle)))

    def finish(self, index):
        cell, _ = self.active.pop(index)
        values = np.full((4, 2, 2), 3, np.int16)
        row = dict(target=list(cell), nfe=5, last_frac=.5, cap=8, seconds=.1, started_unix_ns=1, finished_unix_ns=2, world=self.replica.world)
        self.channel.put(('done', self.replica.epoch, self.wid, cell, values, row, dict(row, gpu=self.wid, slot=index, actual=5)))
        self.done_sent.append(cell)

    def leave(self):
        self.left = True
        self.channel.put(('leave', self.wid))

    def kill(self):
        self.closed = True
        self.channel.kill()


def settle(session, workers, cond, timeout=10., finish=True):
    """Pump the coordinator and step the workers until `cond()`."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        session.pump_once()
        for worker in workers:
            worker.step(finish=finish)
        if cond():
            return
    raise AssertionError('condition not reached in time')


def check_invariants(session):
    assert set(session.owner) == session.inflight
    assert not (session.inflight & set(session.cells))
    cells = sorted(session.inflight)
    assert all(separated(cell, set(cells) - {cell}) for cell in cells)
    targets = [tuple(w['target']) for w in session.windows]
    assert session.generated == len(session.windows) == len(set(targets))
    assert not (set(targets) & session.initial)
    assert set(targets) | session.initial == set(session.cells)
    wids = list(session.members)
    assert wids == sorted(set(wids))
