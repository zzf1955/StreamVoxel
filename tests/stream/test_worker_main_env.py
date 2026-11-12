"""worker_main container settings: SV_* environment variables, CUDA device order, ready file, exit codes (CPU only)."""
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from streamvoxel.stream import worker_main
from tests.stream.fakes import make_session, write_control
from tests.stream.net_fakes import KEY, settle

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def no_signal(monkeypatch):
    """`run` installs a SIGTERM handler, which only works in the main thread; tests drive it from a helper thread."""
    monkeypatch.setattr(worker_main, 'install_drain_handler', lambda event: None)


@pytest.fixture
def listening(tmp_path, monkeypatch):
    sessions = []

    def build():
        session, _ = make_session(tmp_path, monkeypatch, gpus=0, max_windows=10 ** 6, headless_windows=10 ** 6)
        session.members = {}
        session.poll_s = .002
        session.start_listener(('127.0.0.1', 0), KEY)
        write_control(session, batch=4)
        sessions.append(session)
        return session

    yield build
    for session in sessions:
        session.stop_listener()


def _run_in_thread(session, call):
    out = {}

    def target():
        out['value'] = call()

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    settle(session, [], lambda: 'value' in out)
    thread.join(5)
    return out['value']


# ---- CUDA device order ----

def test_cuda_order_defaults_to_pci_bus_and_keeps_an_explicit_choice():
    env = {}
    assert worker_main.cuda_order_default(env) == 'PCI_BUS_ID' and env == {'CUDA_DEVICE_ORDER': 'PCI_BUS_ID'}
    env = {'CUDA_DEVICE_ORDER': 'FASTEST_FIRST'}
    assert worker_main.cuda_order_default(env) == 'FASTEST_FIRST'


def test_main_sets_cuda_order_before_parsing_and_before_torch(monkeypatch):
    monkeypatch.setenv('CUDA_DEVICE_ORDER', 'x')
    monkeypatch.delenv('CUDA_DEVICE_ORDER')
    seen = {}

    def parse_args(argv=None):
        seen['order'] = os.environ.get('CUDA_DEVICE_ORDER')
        raise SystemExit(2)

    monkeypatch.setattr(worker_main, 'parse_args', parse_args)
    with pytest.raises(SystemExit):
        worker_main.main([])
    assert seen['order'] == 'PCI_BUS_ID'


def test_importing_worker_main_does_not_load_torch():
    # main() sets CUDA_DEVICE_ORDER first; that is early enough only if the import chain does not load torch (CUDA).
    code = 'import sys; import streamvoxel.stream.worker_main; print("torch" in sys.modules)'
    out = subprocess.run([sys.executable, '-c', code], cwd=ROOT, capture_output=True, text=True, check=True,
                         env=dict(os.environ, PYTHONPATH=str(ROOT)))
    assert out.stdout.strip() == 'False'


# ---- settings from flags and SV_* variables ----

def test_flags_still_work_without_environment():
    args = worker_main.parse_args(['--connect', 'coord:7788', '--capacity', '8', '--nvsmi-index', '2',
                                   '--connect-timeout', '30', '--base', 'b.pt', '--control', 'c.pt'], environ={})
    assert args.address == ('coord', 7788) and args.capacity == 8 and args.nvsmi_index == 2
    assert args.connect_timeout == 30. and args.base == 'b.pt' and args.control == 'c.pt' and args.ready_file is None


def test_environment_fills_every_setting():
    env = dict(SV_CONNECT='sv-coord-a.default.svc:7788', SV_CAPACITY='8', SV_CONNECT_TIMEOUT='600', SV_NVSMI_INDEX='0',
               SV_BASE='/w/base.pt', SV_CONTROL='/w/control.pt', SV_READY_FILE='/scratch/ready')
    args = worker_main.parse_args([], environ=env)
    assert args.address == ('sv-coord-a.default.svc', 7788) and args.capacity == 8 and args.connect_timeout == 600.
    assert args.nvsmi_index == 0 and args.base == '/w/base.pt' and args.control == '/w/control.pt'
    assert args.ready_file == '/scratch/ready'


def test_flags_win_over_environment_and_defaults_apply():
    env = dict(SV_CONNECT='a:1', SV_CAPACITY='4', SV_NVSMI_INDEX='3')
    args = worker_main.parse_args(['--connect', 'b:2', '--capacity', '16', '--nvsmi-index', '0'], environ=env)
    assert args.address == ('b', 2) and args.capacity == 16 and args.nvsmi_index == 0
    args = worker_main.parse_args([], environ=dict(SV_CONNECT='a:1', SV_CAPACITY='4', SV_NVSMI_INDEX=''))
    assert args.connect_timeout == 120. and args.nvsmi_index is None


@pytest.mark.parametrize('env', [dict(SV_CAPACITY='8'), dict(SV_CONNECT='nohost', SV_CAPACITY='8'), dict(SV_CONNECT='a:1'),
                                 dict(SV_CONNECT='a:1', SV_CAPACITY='eight'), dict(SV_CONNECT='a:port', SV_CAPACITY='8')])
def test_missing_or_bad_settings_exit_with_usage_code(env):
    with pytest.raises(SystemExit) as error:
        worker_main.parse_args([], environ=env)
    assert error.value.code == worker_main.EXIT_USAGE


def test_main_without_a_key_exits_with_usage_code(monkeypatch):
    for key in ('SV_AUTHKEY', 'SV_AUTHKEY_FILE'):
        monkeypatch.delenv(key, raising=False)
    assert worker_main.main(['--connect', 'a:1', '--capacity', '8']) == worker_main.EXIT_USAGE
    monkeypatch.setenv('SV_AUTHKEY_FILE', '/nonexistent/sv-authkey')
    assert worker_main.main(['--connect', 'a:1', '--capacity', '8']) == worker_main.EXIT_USAGE


# ---- exit codes of run() ----

def test_unreachable_coordinator_exits_5(no_signal):
    assert worker_main.run(('127.0.0.1', 1), KEY, 2, connect_timeout=0.) == worker_main.EXIT_UNREACHABLE


def test_drain_before_join_exits_0(monkeypatch):
    monkeypatch.setattr(worker_main, 'install_drain_handler', lambda event: event.set())  # SIGTERM already arrived
    assert worker_main.run(('127.0.0.1', 1), KEY, 2, connect_timeout=60.) == worker_main.EXIT_OK


def test_wrong_key_exits_3_and_is_not_retried(listening, no_signal):
    session = listening()
    code = _run_in_thread(session, lambda: worker_main.run(session.address, b'w' * 32, 2, connect_timeout=60.))
    assert code == worker_main.EXIT_AUTH
    assert session.members == {}


def test_refused_hello_exits_4(listening, no_signal):
    session = listening()
    code = _run_in_thread(session, lambda: worker_main.run(session.address, KEY, 0, connect_timeout=60.))  # capacity 0
    assert code == worker_main.EXIT_REJECTED
    assert session.stats['rejected'] == 1


# ---- ready file and the serving path ----

class Up:
    def __init__(self):
        self.sent, self.closed = [], False

    def put(self, message):
        self.sent.append(message)

    def close(self):
        self.closed = True


def test_ready_file_follows_built_and_leave(tmp_path):
    up, path = Up(), tmp_path / 'probe' / 'ready'
    ready = worker_main.ReadyFile(up, path)
    ready.put(('want', 0, 7, [], 0))
    assert not path.exists()
    ready.put(('built', 7, None))
    assert path.read_text() == '7\n'
    ready.put(('leave', 7))
    assert not path.exists()
    ready.clear()  # idempotent
    assert [m[0] for m in up.sent] == ['want', 'built', 'leave']
    plain = worker_main.ReadyFile(up)
    plain.put(('built', 1, None))
    assert up.sent[-1] == ('built', 1, None)


def _fake_join(up):
    def join_remote(address, authkey, capacity, **kwargs):
        return up, 3, dict(cfg={}, wid=3)
    return join_remote


@pytest.mark.parametrize('fail', [False, True])
def test_run_serves_reports_ready_and_cleans_up(tmp_path, monkeypatch, no_signal, fail):
    up, path, seen = Up(), tmp_path / 'ready', {}

    class FakeWorker:
        def __init__(self, wid, init, commands, upstream):
            self.wid, self.init, self.up = wid, init, upstream
            self.leave_event = None

        def run_loop(self):
            self.up.put(('built', self.wid, None))
            seen['ready_while_serving'] = path.exists()
            seen['scratch'] = Path(self.init['run'])
            if fail:
                raise RuntimeError('boom')

    monkeypatch.setattr(worker_main, 'join_remote', _fake_join(up))
    monkeypatch.setattr(worker_main, 'Worker', FakeWorker)
    code = worker_main.run(('h', 1), KEY, 8, scratch=tmp_path / 'scratch', ready_file=path)
    assert seen['ready_while_serving'] and (seen['scratch'] / 'results' / 'trace').is_dir()
    assert not path.exists() and up.closed
    if fail:
        assert code == worker_main.EXIT_ERROR and up.sent[-1][0] == 'error' and 'boom' in up.sent[-1][2]
    else:
        assert code == worker_main.EXIT_OK and up.sent[-1] == ('bye', 3)


def test_retry_is_logged_once_and_success_after_retries(capsys, monkeypatch):
    calls = []

    def flaky(*a, **k):
        calls.append(1)
        if len(calls) < 3:
            raise ConnectionRefusedError()
        return 'channel', ('welcome', 0, {}, 2)

    monkeypatch.setattr(worker_main.netchannel, 'connect', flaky)
    out = worker_main.connect_with_retry(('c', 1), KEY, {}, connect_timeout=60, sleep=lambda d: None, clock=lambda: 0.)
    err = capsys.readouterr().err
    assert out[0] == 'channel'
    assert err.count('not reachable yet') == 1 and 'after 2 failed attempts' in err
