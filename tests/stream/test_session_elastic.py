"""Elastic membership over real `multiprocessing.connection` channels with fake (CPU, in-process) workers.

Covers join with a consistent snapshot, graceful drain, abrupt loss with reclaim, last-worker protection, authkey and
unpickler hardening, heartbeats, and a seeded soak of random joins/leaves/kills checking the commit invariants.
"""
import multiprocessing
import pickle
import random
import threading
import time

import numpy as np
import pytest

from streamvoxel.stream import channel as netchannel
from streamvoxel.stream.channel import CLOSED, DRAINING, READY
from streamvoxel.stream.worker import Replica, ReplicaGap
from tests.stream.fakes import done_message, make_session, write_control
from tests.stream.net_fakes import KEY, FakeNetWorker, check_invariants, settle


@pytest.fixture
def elastic(tmp_path, monkeypatch):
    workers, sessions = [], []

    def build(batch=4, **opts):
        session, _ = make_session(tmp_path, monkeypatch, gpus=0, max_windows=10 ** 6, headless_windows=10 ** 6)
        session.members = {}
        session.poll_s = .002
        opts.setdefault('snapshot_chunk', 4)
        session.start_listener(('127.0.0.1', 0), KEY, **opts)
        write_control(session, batch=batch)
        sessions.append(session)
        return session

    def worker(session, **kw):
        w = FakeNetWorker(session.address, **kw)
        workers.append(w)
        return w

    build.worker = worker
    yield build
    for w in workers:
        if w.channel is not None and not w.closed:
            w.channel.kill()
    for session in sessions:
        session.stop_listener()


def ready_up(session, *workers):
    settle(session, workers, lambda: all(w.go for w in workers) and all(session.members[w.wid].state == READY for w in workers))


def boot(elastic, session, count=1, auto=False, **kw):
    kw['auto'] = auto
    workers = [elastic.worker(session, **kw) for _ in range(count)]
    settle(session, workers, lambda: all(w.connected.is_set() for w in workers))
    assert all(w.error is None for w in workers), [repr(w.error) for w in workers]
    workers.sort(key=lambda w: w.wid)  # connections race; the tests name workers by wid
    ready_up(session, *workers)
    return workers


def ask(session, workers, worker, idle_count=None):
    """One want/pick round for `worker`, without finishing anything."""
    before = len(worker.picks_seen)
    worker.want()
    settle(session, workers, lambda: len(worker.picks_seen) > before, finish=False)
    return worker.picks_seen[-1][2]


def complete(session, workers, worker, count):
    for index in sorted(worker.active)[:count]:
        worker.finish(index)
    n = session.generated + count
    settle(session, workers, lambda: session.generated >= n, finish=False)


# ---- (a) join mid-run --------------------------------------------------------------------------

def test_join_second_worker_mid_run_snapshot_consistent(elastic):
    session = elastic(batch=6)
    w0, = boot(elastic, session)
    assert session.shares() == {0: 4}                       # one worker: its whole capacity
    picks = ask(session, [w0], w0)
    assert len(picks) == 4
    complete(session, [w0], w0, 3)                         # cseq 1..3 before the join
    assert session.cseq == 3
    gate = threading.Event()                                 # hold the snapshot chunks so commits pile up behind them
    stream = session.snapshot_stream

    def held(snapshot, cseq0):
        for message in stream(snapshot, cseq0):
            if message[0] == 'snapshot_chunk':
                gate.wait()
            yield message
    session.snapshot_stream = held
    w1 = elastic.worker(session, auto=False)
    settle(session, [w0, w1], lambda: 1 in session.members and w1.connected.is_set(), finish=False)
    assert session.members[1].state == netchannel.BUILDING and session.shares() == {0: 4}   # building: no share yet
    ask(session, [w0, w1], w0)
    complete(session, [w0, w1], w0, 1)                     # cseq 4 after the snapshot point (cseq0 == 3)
    complete(session, [w0, w1], w0, 1)                     # cseq 5
    assert session.cseq == 5
    gate.set()
    settle(session, [w0, w1], lambda: w1.go and session.members[1].state == READY, finish=False)
    assert session.shares() == {0: 3, 1: 3}                 # rebalanced on the membership change
    kinds = [m[0] for m in w1.log]
    assert kinds[0] == 'snapshot' and kinds.count('snapshot') == 1 and kinds[-1] in ('commit', 'snapshot_end', 'go')
    assert w1.log[0][2] == 3                                # the snapshot point
    commits = [m[4] for m in w1.log if m[0] == 'commit']
    assert commits == [4, 5]                                # nothing lost, nothing repeated
    assert w1.replica.cells.keys() == session.cells.keys() and w1.replica.last_cseq == session.cseq
    for cell, values in session.cells.items():
        assert np.array_equal(w1.replica.cells[cell], values)
    # idempotent duplicate, and a gap is refused
    assert w1.replica.apply(('commit', session.epoch, (9, 9), 'x', 5)) is None and (9, 9) not in w1.replica.cells
    with pytest.raises(ReplicaGap):
        w1.replica.apply(('commit', session.epoch, (9, 9), 'x', 9))
    # the new worker takes work, and the pair keeps the separation and exactly-once invariants
    p0, p1 = ask(session, [w0, w1], w0), ask(session, [w0, w1], w1)
    assert len(w1.active) == 3 and len(p1) == 3
    complete(session, [w0, w1], w1, 3)
    check_invariants(session)
    settle(session, [w0, w1], lambda: w0.replica.last_cseq == session.cseq == w1.replica.last_cseq, finish=False)


# ---- (b) graceful drain ------------------------------------------------------------------------

def test_graceful_drain_of_one_of_two(elastic):
    session = elastic(batch=6)
    w0, w1 = boot(elastic, session, 2)
    ask(session, [w0, w1], w0), ask(session, [w0, w1], w1)
    assert len(w1.active) == 3
    w1.leave()
    settle(session, [w0, w1], lambda: session.members[1].state == DRAINING, finish=False)
    assert session.shares() == {0: 4}                       # w1 no longer counts
    assert ask(session, [w0, w1], w1) == []                 # draining: only empty picks
    w1.pump()
    assert not any(m[0] == 'stop' for m in w1.log)           # still owns windows
    owned = sorted(w1.active.values())
    complete(session, [w0, w1], w1, 2)
    settle(session, [w0, w1], lambda: session.stats['stale_done'] == 0 and session.generated == 2, finish=False)
    w1.pump()
    assert not w1.stopped
    complete(session, [w0, w1], w1, 1)                      # last owned window accepted
    settle(session, [w0, w1], lambda: w1.stopped and session.members[1].state == CLOSED, finish=False)
    assert session.stats['closed'] == 1 and session.stats['lost'] == 0 and session.stats['stale_done'] == 0
    assert session.generated == 3 and not [c for c, o in session.owner.items() if o == 1]
    assert len(owned) == 3
    picks = ask(session, [w0], w0)
    assert len(w0.active) == 4                              # w0 now carries the whole batch share
    check_invariants(session)


# ---- (c) abrupt loss ---------------------------------------------------------------------------

def test_abrupt_loss_reclaims_and_commits_exactly_once(elastic):
    session = elastic(batch=6)
    w0, w1 = boot(elastic, session, 2)
    ask(session, [w0, w1], w1)
    held = sorted(cell for cell, _ in w1.active.values())
    assert len(held) == 3 and set(held) == {c for c, o in session.owner.items() if o == 1}
    w1.kill()
    settle(session, [w0], lambda: session.members[1].state == CLOSED, finish=False)
    assert session.members[1].extra['lost'] == 'eof'
    assert not (set(held) & (session.inflight | set(session.owner) | set(session.cells)))
    assert session.stats['reclaimed'] == 3 and sorted(session.retry) == held
    # a late result from the dead worker (already read off its socket) is refused
    session.handle(done_message(session.epoch, 1, held[0]))
    assert session.stats['stale_done'] == 1 and held[0] not in session.cells and session.generated == 0
    picks = ask(session, [w0], w0)
    assert [cell for _, cell in picks][:len(held)] == held          # reclaimed windows are re-picked first
    complete(session, [w0], w0, len(w0.active))
    targets = [tuple(w['target']) for w in session.windows]
    assert len(targets) == len(set(targets)) == session.generated
    check_invariants(session)


def test_done_before_lost_commits_and_reclaims_the_rest(elastic):
    session = elastic(batch=6)
    w0, w1 = boot(elastic, session, 2)
    ask(session, [w0, w1], w1)
    held = sorted(cell for cell, _ in w1.active.values())
    session.up.put(done_message(session.epoch, 1, held[0]))   # queued ahead of the loss
    session.up.put(('lost', 1, 'eof'))
    settle(session, [w0], lambda: session.members[1].state == CLOSED, finish=False)
    assert session.generated == 1 and held[0] in session.cells, (dict(session.stats), session.errors, session.members[1].extra, held, dict(session.owner), session.epoch, session.stopping)
    assert session.stats['reclaimed'] == 2 and sorted(session.retry) == held[1:]
    check_invariants(session)


# ---- (d) last worker -------------------------------------------------------------------------

def test_last_worker_cannot_drain(elastic):
    session = elastic(batch=4)
    w0, = boot(elastic, session)
    w0.leave()
    settle(session, [w0], lambda: ('leave_denied', 'last_worker') in w0.log, finish=False)
    assert session.members[0].state == READY and session.stats['leave_denied'] == 1
    w1, = boot(elastic, session)
    w0.leave()
    settle(session, [w0, w1], lambda: session.members[0].state in (DRAINING, CLOSED), finish=False)
    w1.leave()
    settle(session, [w0, w1], lambda: session.stats['leave_denied'] == 2 or w1.stopped, finish=False)
    assert session.members[1].state == READY and session.stats['leave_denied'] == 2


# ---- (e) restricted unpickler ------------------------------------------------------------------

class Evil:
    def __reduce__(self):
        import os
        return (os.system, ('echo pwned > /dev/null',))


@pytest.mark.parametrize('payload', [Evil(), (lambda: 0).__class__, __import__('subprocess').Popen, __import__('builtins').eval])
def test_restricted_unpickler_refuses(payload):
    with pytest.raises((pickle.UnpicklingError, pickle.PicklingError, AttributeError)):
        netchannel.loads(pickle.dumps(payload, protocol=5))


def test_restricted_unpickler_allows_protocol_payloads():
    message = ('commit', 0, (1, 2), np.arange(8, dtype=np.int16).reshape(2, 2, 2), 7)
    back = netchannel.loads(netchannel.dumps(message))
    assert back[:3] == message[:3] and np.array_equal(back[3], message[3]) and back[4] == 7
    rich = dict(a={1, 2}, b=np.float32(1.5), c=np.int64(3), d=[(0, 1)], e=np.zeros((2, 3), np.uint16), f=frozenset({1}), g=b'x', h=None)
    out = netchannel.loads(netchannel.dumps(rich))
    assert out['a'] == {1, 2} and out['c'] == 3 and out['e'].dtype == np.uint16 and out['f'] == frozenset({1})


def test_malicious_message_from_authenticated_peer_is_lost_not_executed(elastic, tmp_path):
    session = elastic(batch=4)
    w0, w1 = boot(elastic, session, 2)
    marker = tmp_path / 'pwned'

    class Boom:
        def __reduce__(self):
            return (marker.write_text, ('x',))
    w1.channel.pipe.conn.send_bytes(pickle.dumps(('want', 0, 1, Boom(), 0)))
    settle(session, [w0], lambda: session.members[1].state == CLOSED, finish=False)
    assert session.members[1].extra['lost'] == 'protocol' and not marker.exists()


def test_malicious_hello_is_refused(elastic, tmp_path):
    session = elastic(batch=4)
    conn = multiprocessing.connection.Client(session.address, authkey=KEY)
    marker = tmp_path / 'pwned2'

    class Boom:
        def __reduce__(self):
            return (marker.write_text, ('x',))
    conn.send_bytes(pickle.dumps(('hello', Boom())))
    settle(session, [], lambda: session.stats['rejected_hello'] == 1)
    assert not session.members and not marker.exists()


# ---- (f) authkey -------------------------------------------------------------------------------

def test_bad_authkey_is_rejected(elastic):
    session = elastic(batch=4)
    w = elastic.worker(session, key=b'wrong' * 8)
    settle(session, [], lambda: w.connected.is_set() and session.stats['rejected_auth'] == 1)
    assert isinstance(w.error, multiprocessing.AuthenticationError) and not session.members


def test_bad_authkey_is_logged_counted_and_final_metrics_written(elastic):
    import json
    session = elastic(batch=4)
    w = elastic.worker(session, key=b'wrong' * 8)
    settle(session, [], lambda: w.connected.is_set() and session.stats['rejected_auth'] == 1)
    events = [json.loads(line) for line in (session.run / 'worker_events.jsonl').read_text().splitlines()]
    assert [(e['event'], e['reason']) for e in events] == [('reject', 'auth')]
    assert session.stats['rejected'] == 1 and session.metrics()['sv_rejected_total'] == 1
    (session.run / 'metrics.json').unlink(missing_ok=True)
    session.shutdown()
    assert json.loads((session.run / 'metrics.json').read_text())['sv_rejected_total'] == 1


def test_bad_hello_values_are_rejected_and_wids_never_reused(elastic):
    session = elastic(batch=4, max_slots_per_worker=8)
    session.code_digest = 'abc'
    bad = elastic.worker(session, code_digest='zzz')
    settle(session, [], lambda: bad.connected.is_set())
    assert isinstance(bad.error, netchannel.Rejected) and str(bad.error) == 'code_digest'
    for capacity in (0, 65):
        w = elastic.worker(session, capacity=capacity, code_digest='abc')
        settle(session, [], lambda: w.connected.is_set())
        assert str(w.error) == 'capacity'
    a = elastic.worker(session, capacity=64, code_digest='abc')
    settle(session, [a], lambda: a.connected.is_set())
    assert a.wid == 0 and session.members[0].capacity == 8     # clamped to max_slots_per_worker
    session.on_lost(0, 'test')
    b = elastic.worker(session, code_digest='abc', auto=False)
    settle(session, [b], lambda: b.connected.is_set())
    assert b.wid == 1                                          # not reused


# ---- heartbeats --------------------------------------------------------------------------------

def test_silent_worker_is_lost_but_heartbeating_one_is_not(elastic):
    session = elastic(batch=4, lost_after_s=.6, building_lost_after_s=.6)
    quiet = elastic.worker(session, hb_interval=100.)
    alive = elastic.worker(session, hb_interval=.05, auto=False)
    settle(session, [alive], lambda: quiet.connected.is_set() and alive.connected.is_set())
    settle(session, [alive], lambda: session.members[quiet.wid].state == CLOSED, timeout=5)
    assert session.members[quiet.wid].extra['lost'] == 'heartbeat' and session.members[alive.wid].state != CLOSED, (session.run / 'worker_events.jsonl').read_text() + str(session.errors) + str(alive.log[-5:])


def test_no_worker_timeout_halts(elastic):
    session = elastic(batch=4, no_worker_timeout_s=.3)
    settle(session, [], lambda: session.stopping is not None, timeout=3)
    assert session.errors == ['no workers']
    assert not session.pump_once()


# ---- metrics -----------------------------------------------------------------------------------

def test_metrics_names_and_values(elastic):
    session = elastic(batch=6)
    w0, w1 = boot(elastic, session, 2)
    text = session.metrics_text()
    for name in ('sv_workers_ready 2', 'sv_slots_total 8', 'sv_joined_total 2', 'sv_lost_total 0', 'sv_epoch 0', 'sv_cseq 0'):
        assert name in text
    assert 'sv_shares{wid="0"} 3' in text and session.stats['join_to_ready_ms_last'] >= 0


# ---- (g) soak ----------------------------------------------------------------------------------

@pytest.mark.parametrize('seed', [1, 2, 3])
def test_soak_random_join_leave_kill(elastic, seed):
    rng = random.Random(seed)
    session = elastic(batch=7, snapshot_chunk=3, lost_after_s=30.)
    fleet = [elastic.worker(session, rng=rng, ticks=(1, 4))]
    settle(session, fleet, lambda: fleet[0].go)
    for step in range(160):
        roll = rng.random()
        live = [w for w in fleet if not w.closed and w.wid is not None and session.members.get(w.wid) is not None
                and session.members[w.wid].state != CLOSED]
        if roll < .10 and len(fleet) < 9:
            fleet.append(elastic.worker(session, rng=rng, ticks=(1, 4), capacity=rng.choice((2, 4, 8))))
        elif roll < .18 and live:
            victim = rng.choice(live)
            if not victim.left:
                victim.leave()
        elif roll < .24 and len(live) > 1:
            victim = rng.choice(live)
            victim.kill()
        for _ in range(3):
            session.pump_once()
            for w in fleet:
                w.step()
        check_invariants(session)
    # wind down: drain everything but one READY worker, then let the survivor finish its windows
    settle(session, fleet, lambda: all(w.connected.is_set() for w in fleet))
    live = lambda: [w for w in fleet if w.wid is not None and session.members[w.wid].state != CLOSED]  # noqa: E731
    if not [w for w in live() if session.members[w.wid].state == READY and not w.left]:
        fleet.append(elastic.worker(session, rng=rng, ticks=(1, 4)))
        settle(session, fleet, lambda: fleet[-1].go and session.members[fleet[-1].wid].state == READY)
    survivor = [w for w in live() if session.members[w.wid].state == READY and not w.left][0]
    for w in live():
        if w is not survivor and not w.left:
            w.leave()
    try:
        settle(session, fleet, lambda: all(w is survivor or session.members[w.wid].state == CLOSED for w in fleet if w.wid is not None), timeout=8)
    except AssertionError:
        raise AssertionError([(w.wid, session.members[w.wid].state, w.left, w.closed, w.waiting, len(w.active), w.go, w.stopped,
                               [o for o in session.owner.values()].count(w.wid), session.members[w.wid].extra.get('stop_sent')) for w in fleet if w.wid is not None])
    for _ in range(60):
        session.pump_once()
        survivor.step()
    check_invariants(session)
    # every window was committed exactly once; every replica still connected equals the coordinator
    targets = [tuple(w['target']) for w in session.windows]
    assert len(targets) == len(set(targets)) == session.generated > 0
    assert sorted(session.members) == list(range(len(session.members))) and session.stats['joined'] == len(session.members)
    survivor.pump()
    settle(session, [survivor], lambda: survivor.replica.last_cseq == session.cseq, finish=False)
    assert survivor.replica.cells.keys() == session.cells.keys()
    assert session.stats['lost'] + session.stats['closed'] <= len(session.members)


# ---- stage 2b: telemetry index, joiner gate input, connect retry ----

def test_telemetry_index_is_the_explicit_init_field():
    from streamvoxel.stream.worker import Worker
    base = dict(cfg=dict(visible_gpus=[4, 5]), run='.', world=1, cells={}, biomes=None, parent=None, wid=7, torch_device=0, run_gate=False,
                capacity=2)
    assert Worker(7, dict(base, nvsmi_index=None), None, None).nvsmi_index() is None   # a wid beyond visible_gpus: no index, no error
    assert Worker(7, dict(base, nvsmi_index=3), None, None).nvsmi_index() == 3


def test_init_remote_has_no_hardcoded_gpu_and_fixed_gate_cells(elastic):
    session = elastic()
    init = session.init_remote(9, 2)
    assert init['nvsmi_index'] is None and init['run_gate'] is True
    assert sorted(map(tuple, init['gate_cells'])) == sorted(session.initial)
    from streamvoxel.stream import worker_main
    assert worker_main.local_gpu_index({'CUDA_VISIBLE_DEVICES': '3,4'}) == 3
    assert worker_main.local_gpu_index({'CUDA_VISIBLE_DEVICES': 'GPU-abc'}) is None
    assert worker_main.local_gpu_index({}) is None


def test_joiner_gate_input_independent_of_midrun_cells():
    from streamvoxel.stream.worker import Worker
    captured = []

    class Engine:
        def numerical_gate(self, item, other):
            captured.append((item, other))
            return dict(graph_used=False, graph_equal=False, padded_close=True, bucket_close=True)

    class Spec:
        height, tile, biome_halo = 2, 2, 4

    initial = {(z, x): np.full((2, 2, 2), 1 + z + x, np.int16) for z in range(3) for x in range(3)}
    results = []
    import streamvoxel.stream.scheduler as sched
    orig = sched.make_item
    sched.make_item = lambda cells, target, *a, **k: (target, sorted(cells))
    try:
        for extra in ({}, {(5, 5): np.zeros((2, 2, 2), np.int16), (3, 0): np.ones((2, 2, 2), np.int16)}):
            captured.clear()
            w = Worker(7, dict(cfg={}, run='/tmp', world=1, cells={}, biomes=None, parent=None, fallback=[24., 24.], gate_steps=1, l2_cap=4,
                               gate_cells=sorted(initial), wid=7, torch_device=0, run_gate=True, capacity=2, nvsmi_index=None), None, None)
            w.replica.cells.update(initial)
            w.replica.cells.update(extra)
            import streamvoxel.stream.worker as wmod
            orig_atomic = wmod.atomic_json
            wmod.atomic_json = lambda *a, **k: None
            try:
                w._gate(Engine(), Spec)
            finally:
                wmod.atomic_json = orig_atomic
            results.append(list(captured))
    finally:
        sched.make_item = orig
    assert results[0] == results[1] and results[0]


def _in_thread(session, call):
    out = {}

    def target():
        try:
            out['value'] = call()
        except BaseException as error:
            out['error'] = error
        out['done'] = True

    threading.Thread(target=target, daemon=True).start()
    settle(session, [], lambda: out.get('done'))
    return out


def test_connect_retries_until_listening_then_connects(elastic):
    from streamvoxel.stream import worker_main
    session = elastic()
    calls, delays = [], []
    real = netchannel.connect

    def flaky(*a, **k):
        calls.append(1)
        if len(calls) < 4:
            raise ConnectionRefusedError()
        return real(*a, **k)

    netchannel_connect, netchannel.connect = netchannel.connect, flaky
    try:
        out = _in_thread(session, lambda: worker_main.connect_with_retry(
            session.address, KEY, worker_main.hello_fields(2), connect_timeout=60, sleep=delays.append, clock=lambda: 0.))
        out['value'][0].close()
    finally:
        netchannel.connect = netchannel_connect
    assert len(calls) == 4 and delays == [.5, 1., 2.]


def test_connect_backoff_caps_and_times_out():
    from streamvoxel.stream import worker_main
    now, delays = [0.], []

    def sleep(d):
        delays.append(d)
        now[0] += d

    with pytest.raises(OSError):
        worker_main.connect_with_retry(('127.0.0.1', 1), KEY, {}, connect_timeout=20, sleep=sleep, clock=lambda: now[0])
    assert delays[:5] == [.5, 1., 2., 4., 5.] and set(delays[4:]) == {5.} and sum(delays) <= 20


def test_connect_does_not_retry_authentication_failure(elastic):
    from streamvoxel.stream import worker_main
    session = elastic()
    delays = []
    out = _in_thread(session, lambda: worker_main.connect_with_retry(
        session.address, b'wrong', worker_main.hello_fields(2), sleep=delays.append))
    assert isinstance(out['error'], multiprocessing.AuthenticationError)
    assert delays == []
