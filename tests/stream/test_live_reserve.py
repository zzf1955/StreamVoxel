"""CPU check that the GPU reserve hands chunks back when free memory drops under the margin."""
import sys
import types

from streamvoxel.stream.reserve import GpuReserve

MB = 1024 ** 2


class FakeCuda:
    def __init__(self, free):
        self.free = free
        self.emptied = 0

    def mem_get_info(self, device):
        return self.free, 24 * 1024 * MB

    def empty_cache(self):
        self.emptied += 1


def install(monkeypatch, cuda):
    torch = types.SimpleNamespace(cuda=cuda)
    cuda.OutOfMemoryError = type('OutOfMemoryError', (RuntimeError,), {})
    monkeypatch.setitem(sys.modules, 'torch', torch)
    return torch


def test_ensure_releases_only_the_shortfall(monkeypatch):
    cuda = FakeCuda(free=3000 * MB)
    install(monkeypatch, cuda)
    reserve = GpuReserve(0, 2048 * MB, chunk=256 * MB)
    reserve.chunks = [object()] * 8
    reserve.ensure()
    assert len(reserve.chunks) == 8 and not cuda.emptied
    cuda.free = 1000 * MB
    reserve.ensure()
    assert len(reserve.chunks) == 8 - 5 and cuda.emptied == 1
    cuda.free = 0
    reserve.chunks = []
    reserve.ensure()


def test_retry_releases_margin_after_oom(monkeypatch):
    cuda = FakeCuda(free=3000 * MB)
    torch = install(monkeypatch, cuda)
    reserve = GpuReserve(0, 512 * MB, chunk=256 * MB)
    reserve.chunks = [object()] * 6
    calls = []

    def call():
        calls.append(len(reserve.chunks))
        if len(calls) == 1:
            raise torch.cuda.OutOfMemoryError('oom')
        return 'ok'

    assert reserve.retry(call) == 'ok'
    assert calls == [6, 3]
