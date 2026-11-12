"""CPU checks for low-entropy-last slot counts, separated admission, and the window wire."""
import json
from itertools import combinations

import numpy as np
import torch

from streamvoxel.common.live_wire import LIMITS, Publisher, clamp_settings, verify_packet
from streamvoxel.sampling.low_entropy_last import low_entropy_last_quotas
from streamvoxel.stream.scheduler import make_item, pick_window, plan_inserts, rank_share, separated
from streamvoxel.stream.slot import ForwardBudget, LowEntropyLastSlot
from streamvoxel.stream.condition import ModelSpec


def test_clamp_settings_and_wire_sequence(tmp_path):
    assert clamp_settings({'steps': 'nope'}, dict(steps=32)) == dict(steps=32)
    assert clamp_settings({'steps': 999}, dict(steps=32))['steps'] == LIMITS['steps'][1]
    assert clamp_settings({'speed': 0}, dict(speed=100))['speed'] == 1
    assert clamp_settings({}, dict(speed=100))['speed'] == 100
    assert clamp_settings({'speed': 30, 'steps': 8}, dict(steps=32)) == dict(speed=30, steps=8)
    assert clamp_settings({'speed': 'x'}, dict(steps=32)) == dict(steps=32)
    publisher = Publisher(tmp_path)
    first = publisher.publish('session', world=128)
    second = publisher.publish('window_start', target=[1, 2], slot=0, gpu=0)
    assert (first, second) == (0, 1)
    packet = json.loads((tmp_path / 'wire' / '00000000.json').read_text())
    verify_packet(packet, 0)
    try:
        verify_packet(packet, 1)
    except ValueError:
        pass
    else:
        raise AssertionError('A replayed sequence was accepted')


def test_rank_share_splits_a_batch_across_two_gpus():
    assert rank_share(4, 2, 0) == 2 and rank_share(4, 2, 1) == 2
    assert rank_share(8, 2, 0) == 4 and rank_share(8, 2, 1) == 4
    assert (rank_share(1, 2, 0), rank_share(1, 2, 1)) == (1, 0)
    assert plan_inserts([0, 1, 2, 3], 0, 4, 'dynamic') == [0, 1, 2, 3]
    assert plan_inserts([1, 3], 2, 4, 'dynamic') == [1, 3]
    assert plan_inserts([2, 3], 2, 2, 'dynamic') == []
    assert plan_inserts([0, 1, 2, 3], 1, 4, 'fixed') == []
    assert plan_inserts([0, 1, 2, 3], 0, 2, 'fixed') == [0, 1]


def test_pick_keeps_live_windows_apart():
    committed = {(5, 5)}
    player = (5.5 * 16, 5.5 * 16)
    picked = []
    inflight = set()
    while len(picked) < 8:
        cell = pick_window(committed, inflight, player)
        if cell is None:
            break
        assert cell not in committed and separated(cell, inflight)
        picked.append(cell)
        inflight.add(cell)
    # One committed cell has eight candidates, and the gap leaves only a separated pair.
    assert picked[0] == (4, 5) and len(picked) >= 2
    for left, right in combinations(picked, 2):
        assert max(abs(left[0] - right[0]), abs(left[1] - right[1])) >= 2


def test_diagonal_admission_only_excludes_edge_neighbours():
    committed = {(5, 5)}
    player = (5.5 * 16, 5.5 * 16)
    picked = []
    inflight = set()
    while len(picked) < 8:
        cell = pick_window(committed, inflight, player, diagonal=True)
        if cell is None:
            break
        assert cell not in committed and separated(cell, inflight, diagonal=True)
        picked.append(cell)
        inflight.add(cell)
    # Nearest first; (4, 5) blocks its edge neighbours only, so (5, 4), (5, 6) and (6, 5) still fit.
    assert picked[0] == (4, 5) and set(picked) == {(4, 5), (5, 4), (5, 6), (6, 5)}
    for left, right in combinations(picked, 2):
        assert abs(left[0] - right[0]) + abs(left[1] - right[1]) >= 2
    assert not separated((4, 5), {(4, 4)}, diagonal=True)
    assert separated((4, 4), {(5, 5)}, diagonal=True)
    assert separated((1, 2), {(0, 0)}, diagonal=True)
    assert not separated((4, 4), {(4, 4)}, diagonal=True)


def test_refill_follows_the_20_21_22_20_example():
    slots = [ForwardBudget() for _ in range(4)]
    jobs = iter([20, 21, 22, 20, 21, 21])

    def fill():
        for slot in slots:
            if slot.phase == 'idle':
                slot.arm(next(jobs))

    fill()
    for _ in range(20):
        [slot.forward() for slot in slots]
    assert [slot.phase for slot in slots] == ['idle', 'solid', 'solid', 'idle']
    assert [slot.nfe for slot in slots] == [20, 20, 20, 20]
    fill()
    assert [slot.phase for slot in slots] == ['marking', 'solid', 'solid', 'marking']
    finished = [slot.forward() for slot in slots]
    # The 21-step window ends on this forward. The two replacements have only been marked.
    assert finished == [False, True, False, False]
    assert [slot.phase for slot in slots] == ['solid', 'idle', 'solid', 'solid']


def test_slot_forward_count_matches_the_quota_prefix():
    shape = (16, 8, 8)
    classes, cap, seed = 6, 8, 3
    slot = LowEntropyLastSlot(shape, classes, air_ids=(0,), air_token=0, last_eps=1e-3, block_size=4, device='cpu')
    tokens = torch.full(shape, classes, dtype=torch.long)
    slot.arm(seed, cap, (1, 2))
    forwards = 0
    while slot.phase != 'idle':
        phase = slot.phase
        logits = _logits(forwards)
        finished, commit = slot.consume(logits, tokens)
        forwards += 1
        if phase == 'solid':
            assert commit is not None
        assert forwards < cap + 3
    expect = 1 + len(low_entropy_last_quotas(int(np.prod(shape)), cap, slot.last_count))
    assert forwards == expect
    assert int((tokens == classes).sum()) == 0
    assert int(slot.step_map.min()) >= 1


def test_make_item_is_relative_and_ignores_uncommitted_neighbors():
    spec = ModelSpec(height=256, tile=16, num_classes=304, biome_classes=66, biome_halo=4, y_origin=-64.)
    cells = {(0, 0): np.zeros((256, 16, 16), np.int16)}
    biomes = np.zeros((64, 66, 66), np.int64)
    item = make_item(cells, (0, 1), biomes, spec, world=128, cap=32, l2_cap=8)
    assert item.l1_xz.tolist() == [[-16, 0]]
    assert item.l2_xz.tolist() == [[-16, 0]]
    assert item.biomes.shape == (64, 12, 12)
    assert int(item.biomes[:, 3:9, 3:9].sum()) == 0
    assert item.steps == 32


def _logits(step, ground=8):
    generator = torch.Generator().manual_seed(step + 1)
    logits = torch.randn((6, 16, 8, 8), generator=generator)
    logits[0, ground:] += 30.
    return logits


def test_phase_timer_writes_means(tmp_path):
    import json
    from streamvoxel.stream.worker import Phases

    class Event:
        def synchronize(self):
            pass

        def elapsed_time(self, other):
            return 10.0

    phases = Phases(tmp_path / 'timing_r0.json', every=2)
    for _ in range(2):
        phases.lap('schedule')
        phases.lap('gpu_wait')
        phases.forward(Event(), Event(), active=4)
    row = json.loads((tmp_path / 'timing_r0.json').read_text())
    assert row['forwards'] == 2 and row['gpu_ms'] == 10.0 and row['active_mean'] == 4
    assert set(row['phases_ms']) == {'schedule', 'gpu_wait'}
    assert len((tmp_path / 'timing_r0.jsonl').read_text().splitlines()) == 1


def test_merge_commits_keeps_latest_value_per_position():
    import numpy as np
    from streamvoxel.stream.worker import _merge_commits

    first = dict(positions=np.array([5, 1], np.int32), values=np.array([7, 3], np.int16), step=2, target=(0, 1))
    second = dict(positions=np.array([2, 5], np.int32), values=np.array([4, 9], np.int16), step=3, target=(0, 1))
    merged = _merge_commits([first, second])
    assert merged['positions'].tolist() == [1, 2, 5]
    assert merged['values'].tolist() == [3, 4, 9]
    assert merged['step'] == 3 and merged['target'] == (0, 1)


def test_slot_time_matches_masked_token_count():
    import torch
    from streamvoxel.sampling.schedule import actual_time
    from streamvoxel.stream.slot import LowEntropyLastSlot

    classes, shape = 6, (16, 8, 8)
    slot = LowEntropyLastSlot(shape, classes, (0,), air_token=0, last_eps=1e-4, block_size=4, device='cpu')
    gen = torch.Generator().manual_seed(0)
    logits = torch.randn(classes, *shape, generator=gen)
    logits[0] += torch.linspace(-12, 12, shape[0])[:, None, None]
    tokens = torch.full(shape, classes, dtype=torch.long)
    slot.arm(3, 12, (0, 0))
    while not slot.idle:
        assert slot.time() == actual_time(int((tokens == classes).sum()) / tokens.numel())
        slot.consume(logits, tokens)


def test_replica_follows_commits_and_drops_stale_epochs():
    from streamvoxel.stream.worker import Replica

    first = np.zeros((2, 2, 2), np.int16)
    replica = Replica(7, {(0, 0): first}, 'b0')
    assert replica.apply(('commit', 0, (0, 1), first + 1, 1)) == 'commit'
    assert set(replica.cells) == {(0, 0), (0, 1)}
    assert replica.apply(('reset', 1, 8, {(5, 5): first}, 'b1')) == 'reset'
    assert (replica.epoch, replica.world, set(replica.cells), replica.biomes) == (1, 8, {(5, 5)}, 'b1')
    assert replica.apply(('commit', 0, (9, 9), first, 1)) is None
    assert replica.apply(('reset', 1, 9, {}, 'b2')) is None
    assert replica.apply(('pick', 1, [], {})) is None and set(replica.cells) == {(5, 5)}


def test_inbox_applies_commits_before_the_pick_reply_and_ignores_old_picks():
    import queue
    from streamvoxel.stream.worker import Inbox, Replica

    channel = queue.Queue()
    replica = Replica(1, {(0, 0): np.zeros(1, np.int16)}, None)
    inbox = Inbox(channel, replica)
    channel.put(('commit', 0, (0, 1), np.ones(1, np.int16), 1))
    channel.put(('pick', 5, [(0, (9, 9))], {}))
    channel.put(('commit', 0, (0, 2), np.ones(1, np.int16), 2))
    channel.put(('pick', 0, [(0, (0, 3))], {}))
    answer = inbox.reply(0)
    assert answer[2] == [(0, (0, 3))]
    assert set(replica.cells) == {(0, 0), (0, 1), (0, 2)}
    channel.put(('commit', 0, (1, 1), np.ones(1, np.int16), 3))
    channel.put(('reset', 1, 2, {}, None))
    channel.put(('pick', 0, [(0, (4, 4))], {}))
    assert inbox.reply(0) is None and replica.epoch == 1 and not replica.cells
    channel.put(('stop',))
    assert inbox.reply(1) is None and inbox.stopped


def _stub_session(tmp_path, cells, monkeypatch):
    import queue
    from streamvoxel.stream.channel import Member, ProcLink
    from tests.stream.fakes import make_session

    s, _ = make_session(tmp_path, monkeypatch, gpus=2, slots_per_gpu=4, world=3, defaults=dict(steps=8, batch=4),
                        headless_windows=100, max_windows=100, max_display_lag_windows=4, idle_timeout_seconds=60)
    s.fallback = [8., 8.]
    s.cells = dict(cells)
    s.commands = [queue.Queue(), queue.Queue()]
    s.members = {rank: Member(rank, ProcLink(s.commands[rank]), 4) for rank in (0, 1)}
    s.published = []
    s.publish = lambda kind, **data: s.published.append((kind, data)) or len(s.published) - 1
    return s


def test_coordinator_sends_commits_to_every_worker_before_the_pick_and_keeps_admission_global(tmp_path, monkeypatch):
    cells = {(0, 0): np.zeros(1, np.int16)}
    s = _stub_session(tmp_path, cells, monkeypatch)
    s.handle(('want', 0, 0, [0, 1], 0))
    s.handle(('want', 0, 1, [0, 1], 0))
    first = s.commands[0].get_nowait()
    second = s.commands[1].get_nowait()
    assert first[0] == second[0] == 'pick'
    taken = [cell for _, cell in first[2] + second[2]]
    assert len(taken) == len(set(taken)) == len(s.inflight)
    assert all(max(abs(a[0] - b[0]), abs(a[1] - b[1])) >= 2 for a in taken for b in taken if a != b)
    target = first[2][0][1]
    row = dict(target=list(target), nfe=5, seconds=1., world=3)
    s.handle(('done', 0, 0, target, np.ones(1, np.int16), row, dict(row)))
    assert target not in s.inflight and s.generated == 1
    s.handle(('want', 0, 0, [0], 0))
    kinds = [[s.commands[rank].get_nowait()[0] for _ in range(s.commands[rank].qsize())] for rank in (0, 1)]
    assert kinds[0].index('commit') < kinds[0].index('pick') and kinds[1] == ['commit']
    s.handle(('done', 5, 0, (9, 9), np.ones(1, np.int16), row, dict(row)))
    assert s.generated == 1 and (9, 9) not in s.cells
    s.handle(('job', 0, 1, 3, 8, [dict(row=1)], [dict(row, gpu=1, slot=0, actual=4)]))
    assert [kind for kind, _ in s.published] == ['frame', 'window_commit'] and s.commit_seqs == [1]


def test_transport_batches_and_unbatch_round_trip(tmp_path):
    from streamvoxel.common.live_wire import batch_line, unbatch
    publisher = Publisher(tmp_path)
    for i in range(5):
        publisher.publish('window_start', target=[i, 0], slot=i, gpu=0)
    packets = [json.loads(path.read_text()) for path in sorted((tmp_path / 'wire').glob('*.json'))]
    line = json.loads(json.dumps(batch_line(packets)))
    assert line['kind'] == 'batch' and unbatch(line) == packets
    for expected, packet in enumerate(unbatch(line)):
        verify_packet(packet, expected)


def test_coalesce_merges_each_cells_rows_into_its_last_frame_without_crossing_windows(tmp_path):
    from streamvoxel.common.live_rows import coalesce, pack_row, unpack_row
    publisher = Publisher(tmp_path)
    rng = np.random.default_rng(0)
    cell = {}
    for target in ((0, 0), (0, 2)):
        positions = rng.choice(65536, 600, replace=False).astype(np.int32)
        cell[target] = (positions, rng.integers(0, 5, 600).astype(np.uint16))
    chunk = lambda target, i: pack_row(target, cell[target][0][i * 200:(i + 1) * 200], cell[target][1][i * 200:(i + 1) * 200])
    publisher.publish('window_start', target=[0, 0])
    publisher.publish('frame', frames=[dict(row=0, step=1, **chunk((0, 0), 0))])
    publisher.publish('window_start', target=[0, 2])
    publisher.publish('frame', frames=[dict(row=0, step=2, **chunk((0, 0), 1)), dict(row=1, step=1, **chunk((0, 2), 0))])
    publisher.publish('frame', frames=[dict(row=0, step=3, **chunk((0, 0), 2)), dict(row=1, step=2, **chunk((0, 2), 1))])
    publisher.publish('window_commit', target=[0, 0])
    publisher.publish('window_start', target=[0, 0])
    publisher.publish('frame', frames=[dict(row=0, step=1, **chunk((0, 0), 0))])
    packets = [json.loads(path.read_text()) for path in sorted((tmp_path / 'wire').glob('*.json'))]
    out = coalesce(packets)
    assert [p['seq'] for p in out] == [p['seq'] for p in packets]
    for expected, packet in enumerate(out):
        verify_packet(packet, expected)
    assert out[1]['frames'] == [] and out[3]['frames'] == []
    merged = {tuple(row['target']): row for row in out[4]['frames']}
    for target, count in (((0, 0), 600), ((0, 2), 400)):
        positions, raw = cell[target]
        keep = raw[:count] != 0
        order = np.argsort(positions[:count][keep])
        pos, val = unpack_row(merged[target])
        assert (pos == positions[:count][keep][order]).all() and (val == raw[:count][keep][order]).all()
        assert merged[target]['committed'] == count
    assert merged[(0, 0)]['step'] == 3 and out[7] == packets[7]
