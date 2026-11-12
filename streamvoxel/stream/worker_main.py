"""Remote worker entry point: connect to a coordinator, join with a snapshot, run the GPU loop, leave on SIGTERM.

    SV_AUTHKEY_FILE=... python -m streamvoxel.stream.worker_main --connect host:7788 --capacity 8

Every flag can also come from the environment (containers): SV_CONNECT (host:port), SV_CAPACITY, SV_CONNECT_TIMEOUT,
SV_NVSMI_INDEX, SV_BASE, SV_CONTROL, SV_READY_FILE; the key comes from SV_AUTHKEY_FILE (or SV_AUTHKEY). A flag wins over
its variable.

The worker keeps its scratch files (telemetry, slot logs, trace npz) in a local directory (`SV_WORKER_RUN`), see
DESIGN_ELASTIC D8. SIGTERM only sets a flag; the worker loop then asks the coordinator to drain it. If it is the last
ready worker the coordinator keeps it until another one is ready. SIGTERM before the join ends the process with code 0.

`SV_READY_FILE` (optional): created when the model is built and the worker has reported ready, removed when it asks to
leave and at exit (a readiness probe can test for it).

Exit codes: 0 finished (drained, stopped by the coordinator, or SIGTERM before joining); 1 error in the worker loop;
2 bad arguments or settings; 3 wrong authkey; 4 refused by the coordinator (code digest, weights, capacity, ...);
5 coordinator not reachable within the connect timeout.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import sys
import tempfile
import threading
import time

from streamvoxel.common.io import digest
from streamvoxel.stream import channel as netchannel
from streamvoxel.stream.integrity import package_digest
from streamvoxel.stream.worker import Worker

EXIT_OK, EXIT_ERROR, EXIT_USAGE, EXIT_AUTH, EXIT_REJECTED, EXIT_UNREACHABLE = 0, 1, 2, 3, 4, 5


class Aborted(Exception):
    """The drain flag was set while still trying to reach the coordinator."""


def cuda_order_default(environ=None):
    """CUDA numbers devices fastest-first by default, nvidia-smi by PCI bus. Make CUDA use the PCI order (unless the
    caller chose otherwise) so a CUDA index and an nvidia-smi index name the same card. Must run before torch loads CUDA."""
    env = os.environ if environ is None else environ
    env.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')
    return env['CUDA_DEVICE_ORDER']


def hello_fields(capacity, code_digest=None, states_sha256=None, base_sha256=None, control_sha256=None):
    return dict(capacity=int(capacity), code_digest=code_digest, states_sha256=states_sha256,
                base_sha256=base_sha256, control_sha256=control_sha256)


def local_gpu_index(environ=None):
    """First numeric entry of CUDA_VISIBLE_DEVICES (the physical index nvidia-smi knows), else None."""
    first = (environ if environ is not None else os.environ).get('CUDA_VISIBLE_DEVICES', '').split(',')[0].strip()
    return int(first) if first.isdigit() else None


def connect_with_retry(address, authkey, hello, *, connect_timeout=120., first=.5, cap=5., sleep=time.sleep, clock=time.monotonic,
                       stop=None, **channel_args):
    """`netchannel.connect` that waits for a coordinator which is not listening yet (OSError, including a name that does
    not resolve yet, exponential backoff). Authentication failures and rejections are final and never retried.
    `stop` (an Event): once set, give up with `Aborted` instead of retrying."""
    from multiprocessing import AuthenticationError
    start = clock()
    deadline, delay, failures = start + connect_timeout, first, 0
    while True:
        if stop is not None and stop.is_set():
            raise Aborted('drain requested before the worker joined')
        try:
            out = netchannel.connect(tuple(address), authkey, hello, **channel_args)
            if failures:
                print(f'worker: connected to {address[0]}:{address[1]} after {failures} failed attempts '
                      f'({clock() - start:.1f} s)', file=sys.stderr, flush=True)
            return out
        except (AuthenticationError, netchannel.Rejected):
            raise
        except OSError as error:
            failures += 1
            if failures == 1:
                print(f'worker: coordinator {address[0]}:{address[1]} not reachable yet ({type(error).__name__}), '
                      f'retrying for up to {connect_timeout:.0f} s', file=sys.stderr, flush=True)
            if clock() + delay > deadline:
                raise
            sleep(delay)
            delay = min(delay * 2, cap)


def join_remote(address, authkey, capacity, *, code_digest=None, states_sha256=None, base=None, control=None,
                nvsmi_index=None, hb_interval=2., lost_after=15., connect_timeout=120., stop=None):
    """Register with the coordinator. Returns (channel, wid, init) where `init` is ready for `Worker` (empty replica).

    The hello carries the digest of this process's code and, when local weight files are given, their sha256. `base` and
    `control` replace the weight paths of the coordinator's config on this machine."""
    hello = hello_fields(capacity, code_digest=code_digest or package_digest(), states_sha256=states_sha256,
                         base_sha256=digest(base) if base else None, control_sha256=digest(control) if control else None)
    channel, (_, wid, init_remote, granted) = connect_with_retry(address, authkey, hello, connect_timeout=connect_timeout,
                                                                 hb_interval=hb_interval, lost_after=lost_after, stop=stop)
    init = dict(init_remote, cells={}, biomes=None, wid=wid, capacity=granted)
    init['nvsmi_index'] = nvsmi_index if nvsmi_index is not None else (local_gpu_index() or 0)
    local = {key: value for key, value in (('base', base), ('control', control)) if value}
    if local:
        init['cfg'] = dict(init['cfg'], **local)
    return channel, wid, init


def install_drain_handler(event):
    """SIGTERM handler that only sets `event` (it runs between bytecodes of the main thread, which may hold queue locks)."""
    handler = lambda *_: event.set()  # noqa: E731
    signal.signal(signal.SIGTERM, handler)
    return handler


class ReadyFile:
    """Proxy of the worker's upstream channel that mirrors readiness into a file: created when `built` is sent, removed
    when `leave` is sent. Without a path it only forwards."""

    def __init__(self, up, path=None):
        self.up, self.path = up, Path(path) if path else None

    def put(self, message):
        self.up.put(message)
        if self.path is None:
            return
        if message[0] == 'built':
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(f'{message[1]}\n')  # the worker id
        elif message[0] == 'leave':
            self.clear()

    def clear(self):
        if self.path is not None and self.path.exists():
            self.path.unlink()


def run(address, authkey, capacity, scratch=None, ready_file=None, **kwargs):
    """Join, serve, leave. Returns the process exit code (see the module docstring)."""
    from multiprocessing import AuthenticationError
    drain = threading.Event()
    install_drain_handler(drain)  # before connecting, so SIGTERM while waiting for the coordinator ends cleanly
    try:
        channel, wid, init = join_remote(address, authkey, capacity, stop=drain, **kwargs)
    except Aborted as error:
        print(f'worker: {error}', file=sys.stderr, flush=True)
        return EXIT_OK
    except AuthenticationError as error:
        print(f'worker: authentication with {address[0]}:{address[1]} failed (wrong authkey?): {error}', file=sys.stderr, flush=True)
        return EXIT_AUTH
    except netchannel.Rejected as error:
        print(f'worker: the coordinator refused this worker: {error}', file=sys.stderr, flush=True)
        return EXIT_REJECTED
    except OSError as error:
        print(f'worker: coordinator {address[0]}:{address[1]} not reachable: {type(error).__name__}: {error}', file=sys.stderr, flush=True)
        return EXIT_UNREACHABLE
    scratch = Path(scratch or os.environ.get('SV_WORKER_RUN') or tempfile.mkdtemp(prefix=f'sv_worker{wid}_'))
    (scratch / 'results' / 'trace').mkdir(parents=True, exist_ok=True)
    init['run'] = str(scratch)
    ready = ReadyFile(channel, ready_file)
    code = EXIT_OK
    try:
        worker = Worker(wid, init, channel, ready)
        worker.leave_event = drain  # the worker loop sends `leave` once the flag is set
        worker.run_loop()
        channel.put(('bye', wid))
    except BaseException as error:
        import traceback
        traceback.print_exc()
        channel.put(('error', wid, f'worker{wid}: {type(error).__name__}: {error}'))
        code = EXIT_ERROR
    finally:
        ready.clear()
        channel.close()
    return code


def _env(name, cast, environ):
    value = environ.get(name, '').strip()
    return cast(value) if value else None


def parse_args(argv=None, environ=None):
    """Flags, with SV_* environment variables as defaults (a flag wins)."""
    env = os.environ if environ is None else environ
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--connect', default=None, help='host:port of the coordinator (env SV_CONNECT)')
    parser.add_argument('--capacity', type=int, default=None, help='slots on this GPU (env SV_CAPACITY)')
    parser.add_argument('--connect-timeout', type=float, default=None,
                        help='seconds to keep retrying while the coordinator is not listening (env SV_CONNECT_TIMEOUT, default 120)')
    parser.add_argument('--nvsmi-index', type=int, default=None,
                        help='physical GPU index for telemetry (env SV_NVSMI_INDEX; default: first CUDA_VISIBLE_DEVICES, else 0)')
    parser.add_argument('--base', default=None, help='local path of the base checkpoint (env SV_BASE; default: the path in the coordinator config)')
    parser.add_argument('--control', default=None, help='local path of the ControlNet checkpoint (env SV_CONTROL; default: the path in the coordinator config)')
    parser.add_argument('--ready-file', default=None, help='file that exists while the worker is ready (env SV_READY_FILE)')
    args = parser.parse_args(argv)
    try:
        args.connect = args.connect or _env('SV_CONNECT', str, env)
        args.capacity = args.capacity if args.capacity is not None else _env('SV_CAPACITY', int, env)
        timeout = args.connect_timeout if args.connect_timeout is not None else _env('SV_CONNECT_TIMEOUT', float, env)
        args.connect_timeout = 120. if timeout is None else timeout
        args.nvsmi_index = args.nvsmi_index if args.nvsmi_index is not None else _env('SV_NVSMI_INDEX', int, env)
    except ValueError as error:
        parser.error(f'bad SV_* setting: {error}')
    args.base = args.base or _env('SV_BASE', str, env)
    args.control = args.control or _env('SV_CONTROL', str, env)
    args.ready_file = args.ready_file or _env('SV_READY_FILE', str, env)
    if not args.connect or ':' not in args.connect:
        parser.error('--connect host:port (or SV_CONNECT) is required')
    if args.capacity is None:
        parser.error('--capacity (or SV_CAPACITY) is required')
    host, port = args.connect.rsplit(':', 1)
    if not port.isdigit():
        parser.error(f'bad port in {args.connect!r}')
    args.address = (host, int(port))
    return args


def main(argv=None):
    cuda_order_default()  # before anything can load CUDA (torch is first imported inside Worker.run_loop)
    args = parse_args(argv)
    try:
        authkey = netchannel.read_authkey()
    except (OSError, RuntimeError) as error:
        print(f'worker: {error}', file=sys.stderr, flush=True)
        return EXIT_USAGE
    return run(args.address, authkey, args.capacity, connect_timeout=args.connect_timeout, nvsmi_index=args.nvsmi_index,
               base=args.base, control=args.control, ready_file=args.ready_file)


if __name__ == '__main__':
    sys.exit(main())
