"""Local workers run on the physical GPU listed in `visible_gpus` (CPU only: no CUDA, no real processes)."""
import os

import numpy as np
import pytest

from streamvoxel.stream import session as session_module
from streamvoxel.stream import worker as worker_module
from streamvoxel.stream.worker import Worker, encode_rows
from tests.stream.fakes import make_session


@pytest.fixture(autouse=True)
def clean_cuda_env(monkeypatch):
    # setenv records the original value so teardown restores it; then start from "no mask".
    for key in ('CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER'):
        monkeypatch.setenv(key, 'sentinel')
        monkeypatch.delenv(key)


class FakeQueue:
    def put(self, message):
        pass


class FakeProcess:
    started = []

    def __init__(self, target, args, name):
        self.target, self.args, self.name = target, args, name

    def start(self):
        FakeProcess.started.append(dict(name=self.name, rank=self.args[0], init=self.args[1],
                                        env={k: os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER')}))

    def is_alive(self):
        return False


class FakeContext:
    Queue = FakeQueue
    Process = FakeProcess


def spawn(tmp_path, monkeypatch, devices):
    session, _ = make_session(tmp_path, monkeypatch, gpus=len(devices), visible_gpus=list(devices))
    session.members = {}
    FakeProcess.started = []
    session.start_local_workers(context=FakeContext)
    return session, FakeProcess.started


@pytest.mark.parametrize('devices', [[1, 2], [0, 1], [3]])
def test_init_selects_physical_gpu_per_rank(tmp_path, monkeypatch, devices):
    session, _ = make_session(tmp_path, monkeypatch, gpus=len(devices), visible_gpus=list(devices))
    for rank, physical in enumerate(devices):
        init = session.init_for(rank, 8)
        assert (init['wid'], init['torch_device'], init['nvsmi_index'], init['run_gate']) == (rank, 0, physical, rank == 0)
        assert init['cuda_env'] == {'CUDA_VISIBLE_DEVICES': str(physical), 'CUDA_DEVICE_ORDER': 'PCI_BUS_ID'}
        worker = Worker(rank, init, None, None)
        assert worker.torch_device == 0 and worker.nvsmi_index() == physical and worker.wid == rank


@pytest.mark.parametrize('devices', [[1, 2], [0, 1], [3]])
def test_spawn_sets_child_environment_and_restores_parent(tmp_path, monkeypatch, devices):
    session, started = spawn(tmp_path, monkeypatch, devices)
    assert [s['rank'] for s in started] == list(range(len(devices)))
    assert [s['name'] for s in started] == [f'live-gpu-{rank}' for rank in range(len(devices))]
    assert [s['env'] for s in started] == [{'CUDA_VISIBLE_DEVICES': str(p), 'CUDA_DEVICE_ORDER': 'PCI_BUS_ID'} for p in devices]
    assert [s['init']['nvsmi_index'] for s in started] == list(devices)
    assert all(s['init']['torch_device'] == 0 for s in started)
    assert 'CUDA_VISIBLE_DEVICES' not in os.environ and 'CUDA_DEVICE_ORDER' not in os.environ
    assert sorted(session.members) == list(range(len(devices)))


def test_spawn_restores_an_existing_parent_mask(tmp_path, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '1,2,3')
    _, started = spawn(tmp_path, monkeypatch, [1, 2])
    assert [s['env']['CUDA_VISIBLE_DEVICES'] for s in started] == ['1', '2']
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '1,2,3' and 'CUDA_DEVICE_ORDER' not in os.environ


def test_gpu_outside_parent_mask_or_repeated_is_refused(tmp_path, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '2,3')
    with pytest.raises(ValueError, match='not in CUDA_VISIBLE_DEVICES'):
        spawn(tmp_path / 'a', monkeypatch, [0, 1])
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES')
    with pytest.raises(ValueError, match='repeated'):
        spawn(tmp_path / 'b', monkeypatch, [1, 1])
    assert FakeProcess.started == []


def test_local_gpu_env():
    assert session_module.local_gpu_env(3, {}) == {'CUDA_VISIBLE_DEVICES': '3', 'CUDA_DEVICE_ORDER': 'PCI_BUS_ID'}
    assert session_module.local_gpu_env(3, {'CUDA_VISIBLE_DEVICES': ''})['CUDA_VISIBLE_DEVICES'] == '3'
    assert session_module.local_gpu_env(3, {'CUDA_VISIBLE_DEVICES': '1, 3'})['CUDA_VISIBLE_DEVICES'] == '3'
    with pytest.raises(ValueError):
        session_module.local_gpu_env(3, {'CUDA_VISIBLE_DEVICES': 'GPU-abc'})
    with pytest.raises(ValueError):
        session_module.local_gpu_env(-1, {})


def test_run_worker_applies_cuda_env_before_the_worker_starts(monkeypatch):
    seen = {}

    class Recorder:
        def __init__(self, rank, init, commands, up):
            self.wid = init['wid']

        def run_loop(self):
            seen['env'] = os.environ.get('CUDA_VISIBLE_DEVICES'), os.environ.get('CUDA_DEVICE_ORDER')

    class Up:
        messages = []

        def put(self, message):
            self.messages.append(message)

    monkeypatch.setattr(worker_module, 'Worker', Recorder)
    up = Up()
    worker_module.run_worker(1, dict(wid=1, cuda_env={'CUDA_VISIBLE_DEVICES': '2', 'CUDA_DEVICE_ORDER': 'PCI_BUS_ID'}), None, up)
    assert seen['env'] == ('2', 'PCI_BUS_ID') and up.messages == [('bye', 1)]
    seen.clear()
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES')
    worker_module.run_worker(7, dict(wid=7), None, up)    # a network-style init has no cuda_env: environment untouched
    assert seen['env'][0] is None


@pytest.mark.parametrize('devices', [[1, 2], [0, 1], [3]])
def test_frame_rows_stay_keyed_by_rank(tmp_path, monkeypatch, devices):
    session, _ = make_session(tmp_path, monkeypatch, gpus=len(devices), visible_gpus=list(devices))
    raw_map = np.arange(8, dtype=np.uint16)
    for rank in range(len(devices)):
        wid = session.init_for(rank, 8)['wid']
        merged = {slot: dict(target=(0, slot), positions=np.array([0], np.int32), values=np.array([1]), step=1) for slot in (0, 3)}
        assert [row['row'] for row in encode_rows(merged, wid, raw_map)] == [rank * 64, rank * 64 + 3]


def test_gate_config_runs_on_the_first_listed_physical_gpu(tmp_path, monkeypatch):
    gate = session_module.gate_config(dict(visible_gpus=[3, 4]))
    session, _ = make_session(tmp_path, monkeypatch, gpus=1, visible_gpus=gate['visible_gpus'])
    init = session.init_for(0, 8)
    assert (init['torch_device'], init['nvsmi_index'], init['cuda_env']['CUDA_VISIBLE_DEVICES'], init['run_gate']) == (0, 3, '3', True)
