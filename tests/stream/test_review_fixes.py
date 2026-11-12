"""Regression tests for the code-review fixes (elastic coordinator, channel, worker entry)."""
import json

import numpy as np
import pytest

from streamvoxel.stream.channel import CLOSED, READY
from tests.stream.fakes import wire_packets
from tests.stream.test_session_elastic import ask, boot, elastic  # noqa: F401 - fixture
from tests.stream.net_fakes import settle


def kinds(session):
    return [p['kind'] for p in wire_packets(session)]


# ---- 1. committed windows of a hard-killed worker still reach the viewer ----

def test_lost_worker_committed_windows_are_published(elastic):  # noqa: F811
    session = elastic(batch=6)
    w0, w1 = boot(elastic, session, 2)
    ask(session, [w0, w1], w1)
    index = sorted(w1.active)[0]
    cell = w1.active[index][0]
    w1.finish(index)                       # `done` goes out; the `job` that carries frames and the commit packet never does
    settle(session, [w0], lambda: session.generated == 1, finish=False)
    assert 'window_commit' not in kinds(session)
    w1.kill()
    settle(session, [w0], lambda: session.members[1].state == CLOSED, finish=False)
    packets = wire_packets(session)
    commits = [p for p in packets if p['kind'] == 'window_commit']
    frames = [p for p in packets if p['kind'] == 'frame']
    assert [tuple(p['target']) for p in commits] == [cell]
    assert len(frames) == 1 and [tuple(r['target']) for r in frames[0]['frames']] == [cell]
    assert frames[0]['frames'][0]['visible'] == frames[0]['frames'][0]['committed'] == 16
    assert session.commit_seqs == [commits[0]['seq']]
    # the late job of the dead worker must not publish a second time
    session.handle(('job', session.epoch, 1, 1, 8, [], [commits[0]]))
    assert kinds(session).count('window_commit') == 1


def test_job_before_loss_leaves_nothing_to_republish(elastic):  # noqa: F811
    session = elastic(batch=6)
    w0, w1 = boot(elastic, session, 2)
    ask(session, [w0, w1], w1)
    index = sorted(w1.active)[0]
    cell = w1.active[index][0]
    w1.finish(index)
    settle(session, [w0], lambda: session.generated == 1, finish=False)
    packet = session.windows[0]
    session.handle(('job', session.epoch, 1, 1, 8, [], [dict(packet, gpu=1, slot=index, actual=5)]))
    assert kinds(session).count('window_commit') == 1
    session.on_lost(1, 'test')
    assert kinds(session).count('window_commit') == 1


# ---- 2. failed joiners do not stop a session that has ready workers ----

def test_build_failures_do_not_halt_while_a_worker_is_ready(elastic):  # noqa: F811
    session = elastic(batch=4)
    w0, = boot(elastic, session)
    joiners = [elastic.worker(session, auto=False) for _ in range(3)]
    settle(session, [], lambda: all(j.connected.is_set() for j in joiners) and len(session.members) == 4)
    for wid in (1, 2, 3):
        session.handle(('error', wid, 'out of memory'))
    assert session.build_failures == 3 and session.stopping is None and not session.errors
    assert session.members[0].state != CLOSED and all(session.members[w].state == CLOSED for w in (1, 2, 3))
    events = [json.loads(line)['event'] for line in (session.run / 'worker_events.jsonl').read_text().splitlines()]
    assert events.count('build_failed') == 3


def test_build_failures_halt_when_nothing_is_ready(elastic):  # noqa: F811
    session = elastic(batch=4)
    joiners = [elastic.worker(session, auto=False) for _ in range(3)]
    settle(session, [], lambda: all(j.connected.is_set() for j in joiners) and len(session.members) == 3)
    for wid in (0, 1, 2):
        session.handle(('error', wid, 'out of memory'))
    assert session.stopping is not None and session.errors == ['out of memory']


# ---- 3. code and weight identity ----

def test_package_digest_covers_every_file_and_changes_with_content(tmp_path):
    from streamvoxel.stream.integrity import package_digest, package_files
    root = tmp_path / 'streamvoxel'
    (root / 'models').mkdir(parents=True)
    (root / 'a.py').write_text('x = 1\n')
    (root / 'models' / 'm.py').write_text('y = 2\n')
    (root / '__pycache__').mkdir()
    (root / '__pycache__' / 'junk.py').write_text('z')
    assert package_files(root) == ['streamvoxel/a.py', 'streamvoxel/models/m.py']
    first = package_digest(root)
    assert package_digest(root) == first
    (root / 'models' / 'm.py').write_text('y = 3\n')
    second = package_digest(root)
    assert second != first
    (root / 'b.py').write_text('')
    assert package_digest(root) != second
    from streamvoxel.stream.integrity import package_digest as real
    assert len(real()) == 64
    names = package_files()
    assert 'streamvoxel/models/compact_controlnet.py' in names and 'streamvoxel/common/live_wire.py' in names


def test_source_manifest_includes_models_and_fails_on_missing_file(tmp_path, monkeypatch):
    from streamvoxel.stream import session as module
    root = __import__('pathlib').Path(module.__file__).resolve().parents[2]
    monkeypatch.chdir(root)
    files = module.source_manifest()['files']
    assert 'streamvoxel/models/compact_controlnet.py' in files and 'viewer/bridge.py' in files
    monkeypatch.setattr(module, 'EXTRA_SOURCE', ('viewer/does_not_exist.py',))
    with pytest.raises(FileNotFoundError):
        module.source_manifest()


def test_join_rejects_wrong_weights_and_accepts_matching(elastic):  # noqa: F811
    session = elastic(batch=4)
    session.store.manifest = dict(session.store.manifest, base_sha256='a' * 64, control_sha256='b' * 64)
    results = []
    for base, control in (('a' * 64, 'c' * 64), ('d' * 64, 'b' * 64), ('a' * 64, 'b' * 64), (None, None)):
        w = elastic.worker(session, auto=False, base_sha256=base, control_sha256=control)
        settle(session, [], lambda: w.connected.is_set())
        results.append(str(w.error) if w.error else 'joined')
    assert results == ['weights', 'weights', 'joined', 'joined']
    assert session.init_remote(5, 2)['base_sha256'] == 'a' * 64


def test_coordinate_sets_code_digest_and_joiner_hello_carries_it(tmp_path, monkeypatch):
    from streamvoxel.stream.integrity import package_digest
    from streamvoxel.stream import worker_main
    from tests.stream.fakes import make_session
    monkeypatch.setenv('SV_AUTHKEY', 'k' * 32)
    session, _ = make_session(tmp_path, monkeypatch, gpus=0, listen=['127.0.0.1', 0])
    session.pump_once = lambda: False
    session.coordinate()
    assert session.code_digest == package_digest()
    seen = {}
    monkeypatch.setattr(worker_main, 'connect_with_retry', lambda address, key, hello, **kw: seen.update(hello=hello) or (
        None, ('welcome', 3, dict(cfg={}), 2)))
    worker_main.join_remote(('x', 1), b'k' * 32, 2)
    assert seen['hello']['code_digest'] == package_digest()


def test_worker_verifies_weight_hashes_after_loading(tmp_path):
    from streamvoxel.stream.integrity import IntegrityError, verify_weights
    from streamvoxel.common.io import digest
    base = tmp_path / 'base.pt'
    base.write_bytes(b'weights')
    ok = digest(base)
    assert verify_weights(dict(base=base), dict(base=ok)) == dict(base=ok)
    assert verify_weights(dict(base=base), dict(base=None)) == dict(base=ok)
    with pytest.raises(IntegrityError, match='base weights'):
        verify_weights(dict(base=base), dict(base='0' * 64))


# ---- 4. message validation, failure reporting ----

from tests.stream.test_session_elastic import settle  # noqa: E402,F401


@pytest.mark.parametrize('bad', [('want', 0, 1), ('zzz', 1), ('done', 0, 1, (0, 0)), 5, 'want', (), ('built', 1, None, 'extra'),
                                 ('job', 0, 1, 1, 8, [])])
def test_malformed_message_from_a_peer_ends_only_that_connection(elastic, bad):  # noqa: F811
    session = elastic(batch=4)
    w0, w1 = boot(elastic, session, 2)
    w1.channel.pipe.conn.send_bytes(__import__('pickle').dumps(bad))
    settle(session, [w0], lambda: session.members[1].state == CLOSED, finish=False)
    assert session.members[1].extra['lost'] == 'protocol' and session.members[0].state != CLOSED and not session.errors


def test_valid_message_table_matches_what_the_worker_sends():
    from streamvoxel.stream.channel import valid_message
    assert valid_message(('want', 0, 3, [0], 1), 3) and not valid_message(('want', 0, 4, [0], 1), 3)
    assert valid_message(('built', 3, None), 3) and valid_message(('hb', 3, 1.0), 3) and valid_message(('bye', 3), 3)
    assert valid_message(('leave', 3), 3) and valid_message(('error', 3, 'x'), 3)
    assert not valid_message(['want', 0, 3, [0], 1], 3)


def test_serve_writes_error_file_and_publishes_failed_on_exception(tmp_path, monkeypatch):
    import signal
    from streamvoxel.stream import session as module
    from tests.stream.fakes import make_session
    session, _ = make_session(tmp_path, monkeypatch, gpus=1)
    (session.run / 'frozen_source_manifest.json').write_text(json.dumps(dict(files={})))

    def boom():
        raise ValueError('boom')
    session.coordinate = boom
    old = {n: signal.getsignal(n) for n in (signal.SIGTERM, signal.SIGINT)}
    try:
        with pytest.raises(RuntimeError):
            module._serve(session)
    finally:
        for n, h in old.items():
            signal.signal(n, h)
    assert 'boom' in json.loads((session.run / 'error.json').read_text())['errors'][0]
    assert wire_packets(session)[-1]['kind'] == 'failed'


# ---- 5. gate with an elastic config ----

def test_gate_config_drops_listener_and_elastic_needs_no_visible_gpus():
    from streamvoxel.stream import session as module
    base = dict(world=1, parent_panel='p', mapping='m', vocabulary='v', base='b', control='c', initial_box_zx=[0, 0, 32, 32])
    elastic_cfg = dict(base, listen=['0.0.0.0', 7788], elastic=dict(drain_timeout_s=5))
    module._require(elastic_cfg)                                   # no visible_gpus needed when listening
    with pytest.raises(KeyError):
        module._require(base)                                      # local mode still needs them
    gate = module.gate_config(elastic_cfg)
    assert 'listen' not in gate and 'elastic' not in gate and gate['visible_gpus'] == [0] and gate['max_windows'] == 0
    assert module.gate_config(dict(base, visible_gpus=[3, 4]))['visible_gpus'] == [3]


def test_elastic_session_without_visible_gpus_reports_zero_local_gpus(tmp_path, monkeypatch):
    from tests.stream.fakes import make_session
    session, _ = make_session(tmp_path, monkeypatch, gpus=0, listen=['127.0.0.1', 0])
    assert session.devices == []


# ---- 7. batch used for shares and metrics is the live request ----

def test_shares_and_metrics_follow_the_live_requested_batch(elastic):  # noqa: F811
    from tests.stream.fakes import write_control
    session = elastic(batch=2)
    assert session.defaults['batch'] != 2
    w0, = boot(elastic, session)
    ask(session, [w0], w0)
    assert session.shares() == {0: 2} and len(w0.active) == 2
    metrics = session.metrics()
    assert metrics['sv_requested_batch'] == 2 and metrics['sv_effective_batch'] == 2 and metrics['sv_shares'] == {'0': 2}
    write_control(session, batch=3)
    ask(session, [w0], w0)
    assert session.shares() == {0: 3} and session.metrics()['sv_requested_batch'] == 3
    write_control(session, batch=1000)   # clamped to the configured ceiling, never above 64
    ask(session, [w0], w0)
    assert session.requested_batch == 64


# ---- 6. a refused leave is remembered ----

def test_refused_last_worker_leave_is_approved_when_another_becomes_ready(elastic):  # noqa: F811
    from streamvoxel.stream.channel import DRAINING
    session = elastic(batch=4)
    w0, = boot(elastic, session)
    w0.leave()
    settle(session, [w0], lambda: session.stats['leave_denied'] == 1, finish=False)
    assert session.members[0].state == READY
    w1, = boot(elastic, session)                       # nobody asks again
    settle(session, [w0, w1], lambda: session.members[0].state in (DRAINING, CLOSED), finish=False)
    assert session.members[1].state == READY and session.stats['leave_denied'] == 1


# ---- 8. SIGTERM handler only sets a flag ----

def test_sigterm_handler_does_not_touch_the_channel_and_the_loop_sends_leave():
    import signal
    import threading
    from streamvoxel.stream import worker_main
    from streamvoxel.stream.worker import Worker
    event = threading.Event()
    old = signal.getsignal(signal.SIGTERM)
    try:
        handler = worker_main.install_drain_handler(event)
        assert signal.getsignal(signal.SIGTERM) is handler
        handler(signal.SIGTERM, None)
    finally:
        signal.signal(signal.SIGTERM, old)
    assert event.is_set()

    class Up:
        def __init__(self):
            self.sent = []

        def put(self, message):
            self.sent.append(message)

    up = Up()
    worker = Worker(5, dict(cfg={}, run='.', world=1, cells={}, biomes=None, parent=None, wid=5, torch_device=0, run_gate=False,
                            capacity=2, nvsmi_index=None), None, up)
    worker.maybe_leave()
    assert up.sent == []
    worker.leave_event = event
    worker.maybe_leave()
    worker.maybe_leave()
    assert up.sent == [('leave', 5)]


# ---- 10. a dead local worker is noticed even while messages keep arriving ----

def test_local_crash_detected_on_interval_despite_traffic(tmp_path, monkeypatch):
    import queue
    from tests.stream.fakes import make_session
    session, _ = make_session(tmp_path, monkeypatch, gpus=2)

    class Proc:
        def __init__(self, alive, code=None):
            self.alive, self.exitcode = alive, code

        def is_alive(self):
            return self.alive

    now = [100.]
    session.clock = lambda: now[0]
    session.alive_checked = now[0]
    session.procs = [Proc(True), Proc(False, 3)]
    session.up = queue.Queue()
    for _ in range(6):
        session.up.put(('job', 99, 0, 0, 0, [], []))     # stale epoch: traffic that is ignored
        session.pump_once()
        now[0] += 1.
    assert session.errors == ['gpu1: worker exited with code 3'] and session.stopping is not None and session.gone == {1}


# ---- 9. channel threads ----

class FakeConn:
    """A connection whose reads fail with a chosen error, or whose writes block until it is closed."""

    def __init__(self, read_error=None, block_writes=False):
        import threading
        self.read_error, self.block_writes = read_error, block_writes
        self.shut = threading.Event()
        self.sent = []

    def recv_bytes(self, limit=None):
        if self.read_error is not None:
            raise self.read_error
        self.shut.wait(5)
        raise EOFError

    def send_bytes(self, data):
        if self.block_writes:
            self.shut.wait(5)
            raise OSError('closed')
        self.sent.append(data)

    def fileno(self):
        raise ValueError('fake')

    def close(self):
        self.shut.set()


@pytest.mark.parametrize('error', [TypeError('an integer is required'), ValueError('x'), RuntimeError('y'), OSError('z')])
def test_reader_thread_survives_any_exception_type(error):
    import threading
    from streamvoxel.stream.channel import Pipe
    ended, seen = [], []
    caught = []
    old, threading.excepthook = threading.excepthook, lambda args: caught.append(args)
    try:
        pipe = Pipe(FakeConn(read_error=error), 1, seen.append, ended.append, send_hb=False).start()
        pipe.reader.join(2)
        pipe.kill()
    finally:
        threading.excepthook = old
    assert ended == ['eof'] and not caught and not pipe.reader.is_alive()


def test_handler_error_ends_the_connection_with_protocol():
    from streamvoxel.stream.channel import Pipe, dumps

    class OneMessage(FakeConn):
        def recv_bytes(self, limit=None):
            if not self.sent:
                self.sent.append(1)
                return dumps(('x',))
            return super().recv_bytes(limit)

    def boom(message):
        raise KeyError('bad')
    ended = []
    pipe = Pipe(OneMessage(), 1, boom, ended.append, send_hb=False).start()
    pipe.reader.join(2)
    pipe.kill()
    assert ended == ['protocol']


def test_graceful_finish_never_blocks_the_caller_and_cuts_a_stuck_sender():
    import time
    from streamvoxel.stream.channel import Pipe
    conn = FakeConn(block_writes=True)
    pipe = Pipe(conn, 1, lambda m: None, lambda r: None, send_hb=False).start()
    pipe.put(('hello',))
    time.sleep(.05)                       # the sender is now stuck inside send_bytes
    start = time.monotonic()
    pipe.finish(timeout=.3)
    assert time.monotonic() - start < .1
    assert conn.shut.wait(2) and not pipe.sender.join(2) and not pipe.sender.is_alive()


def test_pending_counter_is_consistent_under_concurrent_put_and_send():
    import threading
    from streamvoxel.stream.channel import Pipe
    conn = FakeConn()
    pipe = Pipe(conn, 1, lambda m: None, lambda r: None, send_hb=False, hb_interval=.01).start()
    blob = np.zeros(100, np.uint8)

    def spam():
        for _ in range(500):
            pipe.put(('blob', blob))
    threads = [threading.Thread(target=spam) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    pipe.finish(wait=True)
    assert pipe.pending == 0 and len(conn.sent) == 2000


# ---- 16. the joiner's gate result reaches the coordinator's event log ----

def test_joiner_gate_report_is_recorded_in_worker_events(elastic):  # noqa: F811
    session = elastic(batch=4)
    w = elastic.worker(session, auto=False)
    settle(session, [], lambda: w.connected.is_set() and 0 in session.members)
    report = dict(graph_used=True, graph_equal=True, padded_close=True, bucket_close=True)
    session.handle(('built', 0, report))
    events = [json.loads(line) for line in (session.run / 'worker_events.jsonl').read_text().splitlines()]
    assert [e['report'] for e in events if e['event'] == 'gate'] == [report]
    assert json.loads((session.out / 'numerical_gate_w0.json').read_text()) == report
