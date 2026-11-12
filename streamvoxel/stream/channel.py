"""Coordinator-side view of a worker: a link that carries messages to it, and the member record the session keeps.

The default link is a `multiprocessing` queue plus the worker process, exactly the transport the session always used.
Any other link only has to provide `send(message)` and `close()`.

Worker side, a channel is any object with `put(message)` (worker -> coordinator) and, on the command side,
`get(timeout=None)` / `get_nowait()` that raise `queue.Empty` (coordinator -> worker); a `multiprocessing.Queue` qualifies.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

BUILDING, READY, DRAINING, CLOSED = 'building', 'ready', 'draining', 'closed'
ACTIVE = (BUILDING, READY)


class ProcLink:
    """Commands to one local worker process through its own FIFO queue."""

    def __init__(self, commands, process=None):
        self.commands, self.process = commands, process

    def send(self, message):
        self.commands.put(message)

    def close(self):
        self.commands.cancel_join_thread()
        self.commands.close()


@dataclass
class Member:
    wid: int
    link: Any
    capacity: int
    state: str = READY
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------------------------
# Stage 2: network transport (multiprocessing.connection, authkey HMAC, restricted unpickling).
# The default local mode never touches anything below this line.
# ---------------------------------------------------------------------------------------------
import io
import os
import pickle
import queue
import socket
import threading
import time
import zlib
from pathlib import Path
from multiprocessing.connection import Client, Listener, answer_challenge, deliver_challenge  # noqa: F401

PROTO = 1
MAX_MESSAGE = 1 << 30
_NUMPY_MODULES = ('numpy.core.multiarray', 'numpy._core.multiarray')
_ALLOWED = {('numpy', 'ndarray'), ('numpy', 'dtype')}
_ALLOWED |= {(module, name) for module in _NUMPY_MODULES for name in ('_reconstruct', 'scalar')}
_ALLOWED |= {(module, '_frombuffer') for module in ('numpy.core.numeric', 'numpy._core.numeric')}
_ALLOWED |= {('builtins', name) for name in ('set', 'frozenset', 'bytearray', 'complex', 'range', 'slice')}


class RestrictedUnpickler(pickle.Unpickler):
    """Only numpy arrays/scalars and builtin containers may be rebuilt; anything else is refused."""

    def find_class(self, module, name):
        if (module, name) in _ALLOWED:
            return super().find_class(module, name)
        if module.startswith('numpy') and name.startswith('dtype') or (module == 'numpy.dtypes' and name.endswith('DType')):
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f'refused to unpickle {module}.{name}')


def dumps(message):
    return pickle.dumps(message, protocol=5)


def loads(data):
    return RestrictedUnpickler(io.BytesIO(data)).load()


def approx_bytes(message, depth=3):
    """Cheap size estimate of a message (array payloads dominate)."""
    if hasattr(message, 'nbytes'):
        return int(message.nbytes)
    if isinstance(message, (bytes, bytearray)):
        return len(message)
    if depth and isinstance(message, (tuple, list)):
        return 64 + sum(approx_bytes(item, depth - 1) for item in message)
    if depth and isinstance(message, dict):
        return 64 + sum(approx_bytes(v, depth - 1) for v in message.values())
    return 16


def shutdown_connection(conn):
    """Close a Connection so that a thread blocked in recv on it wakes up (a plain close does not on Linux)."""
    try:
        sock = socket.socket(fileno=os.dup(conn.fileno()))
        try:
            sock.shutdown(socket.SHUT_RDWR)
        finally:
            sock.close()
    except Exception:
        pass
    try:
        conn.close()
    except Exception:
        pass


# Worker -> coordinator messages and their exact field counts (kind included). Anything else is a protocol violation.
FIELD_COUNTS = dict(built=3, want=5, start=4, done=7, job=7, bye=2, error=3, leave=2, hb=3)
_WID_INDEX = dict(want=2, start=2, done=2, job=2)


def valid_message(message, wid):
    """A known kind, the exact field count, and this connection's worker id in the id field."""
    if not isinstance(message, tuple) or not message or not isinstance(message[0], str):
        return False
    count = FIELD_COUNTS.get(message[0])
    if count is None or len(message) != count:
        return False
    return message[_WID_INDEX.get(message[0], 1)] == wid


_CLOSE = object()


class Pipe:
    """One reader thread and one sender thread around a Connection; heartbeats ride the sender thread.

    `on_message(message)` is called on the reader thread, `on_end(reason)` once when the stream ends or fails
    (not when this side closed it). Nothing here touches session state.
    """

    def __init__(self, conn, wid, on_message, on_end, *, prefix=None, hb_interval=2., send_hb=True,
                 max_pending_bytes=512 << 20, clock=time.monotonic, name='pipe'):
        self.conn, self.wid, self.on_message, self.on_end = conn, wid, on_message, on_end
        self.prefix, self.hb_interval, self.send_hb = prefix, hb_interval, send_hb
        self.max_pending, self.clock = max_pending_bytes, clock
        self.out = queue.Queue()
        self.pending = 0
        self.last_seen = clock()
        self.bytes_sent = self.bytes_received = 0
        self.closed = False       # we closed it (graceful or hard)
        self.ended = False
        self._lock = threading.Lock()
        self._pending_lock = threading.Lock()   # `pending` is changed by the sender thread and by whoever calls put()
        self.reader = threading.Thread(target=self._read, name=f'{name}-r', daemon=True)
        self.sender = threading.Thread(target=self._send, name=f'{name}-s', daemon=True)

    def start(self):
        self.reader.start()
        self.sender.start()
        return self

    def _end(self, reason):
        with self._lock:
            if self.ended or self.closed:
                return
            self.ended = True
        self.on_end(reason)

    def _read(self):
        while True:
            try:
                data = self.conn.recv_bytes(MAX_MESSAGE)
            except Exception:  # EOF, reset, or a connection closed under us from another thread (any error type)
                return self._end('eof')
            self.last_seen = self.clock()
            self.bytes_received += len(data)
            try:
                self.on_message(loads(data))
            except Exception:
                return self._end('protocol')

    def _send(self):
        try:
            if self.prefix is not None:
                for message in self.prefix:
                    if self.closed:
                        return
                    self._write(message)
            while True:
                try:
                    item = self.out.get(timeout=self.hb_interval)
                except queue.Empty:
                    if self.send_hb:
                        item = ('hb', self.wid, time.monotonic())
                        self._write(item)
                    continue
                if item is _CLOSE:
                    break
                with self._pending_lock:
                    self.pending -= approx_bytes(item)
                self._write(item)
        except Exception:  # a broken connection surfaces as OSError/ValueError/EOFError/TypeError depending on the race
            if not self.closed:
                self._end('send_error')
            return
        finally:
            if self.closed == 'graceful':
                shutdown_connection(self.conn)

    def _write(self, message):
        data = dumps(message)
        self.conn.send_bytes(data)
        self.bytes_sent += len(data)

    def put(self, message):
        if self.closed or self.ended:
            return
        size = approx_bytes(message)
        with self._pending_lock:
            self.pending += size
            over = self.pending > self.max_pending
        self.out.put(message)
        if over:
            self._end('slow_peer')

    def finish(self, timeout=5., wait=False):
        """Flush everything queued, then close (graceful). The sender thread closes the connection after the flush; a
        sender stuck on a peer that does not read is cut after `timeout`. `wait=True` also waits (only for the end of a
        worker process, never on the coordinator's main thread)."""
        if self.closed:
            return
        self.closed = 'graceful'
        self.out.put(_CLOSE)
        if wait:
            self.sender.join(timeout)
            if self.sender.is_alive():
                shutdown_connection(self.conn)
        else:
            guard = threading.Timer(timeout, lambda: shutdown_connection(self.conn) if self.sender.is_alive() else None)
            guard.daemon = True
            guard.start()

    def kill(self):
        """Fence: close now, queued output is discarded."""
        self.closed = 'hard'
        shutdown_connection(self.conn)
        self.out.put(_CLOSE)


class NetLink:
    """Coordinator-side link to one remote worker. `send` only enqueues; the sender thread pickles and writes."""

    def __init__(self, conn, wid, up, prefix, *, hb_interval=2., clock=time.monotonic, max_pending_bytes=512 << 20):
        def on_message(message):
            if not valid_message(message, wid):
                raise ValueError('protocol')  # the reader thread ends the connection with reason 'protocol'
            if message[0] != 'hb':  # a heartbeat only refreshes `last_seen`, which the reader thread already did
                up.put(message)

        self.wid = wid
        self.pipe = Pipe(conn, wid, on_message, lambda reason: up.put(('lost', wid, reason)), prefix=prefix,
                         hb_interval=hb_interval, max_pending_bytes=max_pending_bytes, clock=clock, name=f'net-w{wid}')

    def start(self):
        self.pipe.start()
        return self

    def send(self, message):
        self.pipe.put(message)

    @property
    def last_seen(self):
        return self.pipe.last_seen

    def close(self):
        """Hard close (the fence used on loss)."""
        self.pipe.kill()

    def finish(self):
        self.pipe.finish()


class NetChannel:
    """Worker-side channel: `put` (to the coordinator) and `get/get_nowait` (from it), like a pair of queues."""

    def __init__(self, conn, wid, *, hb_interval=2., lost_after=15., clock=time.monotonic):
        self.wid, self.lost_after, self.clock = wid, lost_after, clock
        self.inbox = queue.Queue()
        self._ended = False

        def on_message(message):
            if not isinstance(message, tuple) or not message:
                raise ValueError('protocol')
            if message[0] != 'hb':
                self.inbox.put(message)

        def on_end(reason):
            self._ended = True

        self.pipe = Pipe(conn, wid, on_message, on_end, hb_interval=hb_interval, max_pending_bytes=64 << 20,
                         clock=clock, name=f'worker{wid}').start()

    @property
    def dead(self):
        return self._ended or bool(self.pipe.closed) or self.clock() - self.pipe.last_seen > self.lost_after

    def put(self, message):
        self.pipe.put(message)

    def get(self, timeout=None):
        return self.inbox.get(timeout=timeout)

    def get_nowait(self):
        return self.inbox.get_nowait()

    def close(self):
        self.pipe.finish()

    def kill(self):
        self.pipe.kill()


def read_authkey(environ=None):
    """The shared secret: the file named by SV_AUTHKEY_FILE, else SV_AUTHKEY. At least 16 bytes."""
    env = os.environ if environ is None else environ
    path = env.get('SV_AUTHKEY_FILE')
    key = Path(path).read_bytes().strip() if path else env.get('SV_AUTHKEY', '').encode()
    if len(key) < 16:
        raise RuntimeError('the authkey (SV_AUTHKEY_FILE, or SV_AUTHKEY) must be at least 16 bytes')
    return key


class Rejected(RuntimeError):
    pass


def connect(address, authkey, hello, *, timeout=30., **channel_args):
    """Worker side of the registration: HMAC handshake, `hello`, then `welcome` or `reject`.

    Returns (NetChannel, welcome) where welcome = ('welcome', wid, init_remote, capacity). A wrong authkey raises
    multiprocessing.AuthenticationError; a refused hello raises Rejected(reason).
    """
    conn = Client(address, authkey=authkey)
    try:
        conn.send_bytes(dumps(('hello', dict(hello, proto=PROTO))))
        if not conn.poll(timeout):
            raise TimeoutError('no welcome from the coordinator')
        reply = loads(conn.recv_bytes(MAX_MESSAGE))
    except BaseException:
        conn.close()
        raise
    if reply[0] != 'welcome':
        conn.close()
        raise Rejected(reply[1] if len(reply) > 1 else 'rejected')
    return NetChannel(conn, reply[1], **channel_args), reply


def snapshot_messages(epoch, cseq0, world, biomes, cells, chunk=64):
    """The snapshot as a lazy message stream: head, zlib-compressed chunks of (target, values), end."""
    items = sorted(cells.items())
    yield ('snapshot', epoch, cseq0, world, biomes, len(items))
    for k in range(0, max(1, len(items)), chunk):
        yield ('snapshot_chunk', k // chunk, zlib.compress(dumps(items[k:k + chunk]), 1))
    yield ('snapshot_end', epoch, cseq0)
