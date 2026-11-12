"""Characterization tests of the session <-> worker protocol, driven with fake in-process workers (CPU only).

`test_golden_transcripts` compares scripted scenarios against `golden_session_transcript.json`, which was recorded on
the code before the stage-1 refactor (`SV_RECORD_GOLDEN=1` re-records it; review the diff by hand).
"""
import json
import os
from pathlib import Path
import queue
import time

import numpy as np
import pytest

from streamvoxel.stream.scheduler import pick_window, rank_share, separated
from streamvoxel.stream.worker import Inbox, Replica, encode_rows, _frame_row
from tests.stream.fakes import (applied, done_message, drain, make_session, norm, wire_packets, write_control)

GOLDEN = Path(__file__).with_name('golden_session_transcript.json')


def want(session, wid, idle, active=0, epoch=None):
    session.handle(('want', session.epoch if epoch is None else epoch, wid, list(idle), active))


# ---- scheduler pieces the protocol relies on -------------------------------------------------

def test_rank_share_pinned():
    assert [rank_share(5, 2, r) for r in range(2)] == [3, 2]
    assert [rank_share(7, 3, r) for r in range(3)] == [3, 2, 2]
    assert [rank_share(1, 2, r) for r in range(2)] == [1, 0]
    assert [rank_share(8, 2, r) for r in range(2)] == [4, 4]
    assert rank_share(5, 0, 0) == 5 and rank_share(0, 2, 0) == 0
    for batch in range(0, 20):
        for ranks in (1, 2, 3, 4):
            shares = [rank_share(batch, ranks, r) for r in range(ranks)]
            assert sum(shares) == batch and max(shares) - min(shares) <= 1 and shares == sorted(shares, reverse=True)


def test_row_is_gpu_times_64_plus_slot():
    raw_map = np.arange(8, dtype=np.uint16)
    commit = dict(target=(0, 2), positions=np.array([0, 5], np.int32), values=np.array([1, 2], np.int64), step=3)
    row = _frame_row(commit, raw_map, slot=5, gpu=1)
    assert row['row'] == 1 * 64 + 5 and row['step'] == 3 and row['target'] == [0, 2] and row['visible'] == 2
    merged = {0: commit, 3: dict(commit, target=(1, 1))}
    rows = encode_rows(merged, 1, raw_map)
    assert [r['row'] for r in rows] == [64, 67]
    invisible = dict(commit, values=np.array([0, 0], np.int64))
    assert _frame_row(invisible, raw_map, 0, 0) is None


# ---- worker-side message handling -------------------------------------------------------------

def test_replica_apply_epochs():
    cells = {(0, 0): np.zeros((1,), np.int16)}
    replica = Replica(1, cells, 'b0')
    assert replica.apply(('commit', 0, (0, 1), 'v', 1)) == 'commit' and (0, 1) in replica.cells
    assert replica.apply(('commit', 1, (0, 2), 'v', 1)) is None and (0, 2) not in replica.cells
    assert replica.apply(('reset', 0, 2, {}, 'b1')) is None and replica.world == 1
    assert replica.apply(('reset', 1, 2, {(5, 5): 'x'}, 'b1')) == 'reset'
    assert replica.epoch == 1 and replica.world == 2 and set(replica.cells) == {(5, 5)} and replica.biomes == 'b1'
    assert replica.apply(('commit', 0, (9, 9), 'v', 1)) is None
    assert replica.apply(('commit', 1, [6, 6], 'v', 1)) == 'commit' and (6, 6) in replica.cells
    assert replica.apply(('go',)) is None and replica.apply(('pick', 1, [], {})) is None


def test_inbox_order_and_reply():
    source = queue.Queue()
    replica = Replica(1, {}, None)
    inbox = Inbox(source, replica)
    assert inbox.next() is None
    for message in (('commit', 0, (0, 0), 'a', 1), ('commit', 0, (0, 1), 'b', 2), ('pick', 0, [(0, (0, 2))], dict(cap=4)), ('go',)):
        source.put(message)
    assert inbox.next() == ('commit',)
    answer = inbox.reply(0, lambda: True)
    assert answer == ('pick', 0, [(0, (0, 2))], dict(cap=4)) and set(replica.cells) == {(0, 0), (0, 1)}
    assert inbox.next() == ('go',)
    source.put(('reset', 1, 2, {}, None))
    assert inbox.reply(1) is None and replica.epoch == 1
    source.put(('stop',))
    assert inbox.next(.01) == ('stop',) and inbox.stopped
    assert inbox.reply(1, lambda: False) is None


def test_inbox_reply_ignores_other_epoch_pick():
    source = queue.Queue()
    inbox = Inbox(source, Replica(1, {}, None))
    source.put(('pick', 5, [], {}))
    source.put(('pick', 0, [], {}))
    assert inbox.reply(0)[1] == 0


# ---- coordinator: admission, shares, commits ---------------------------------------------------

def test_dynamic_two_gpu_picks_and_separation(tmp_path, monkeypatch):
    session, links = make_session(tmp_path, monkeypatch)
    write_control(session, batch=5)
    want(session, 0, range(4))
    sent = links[0].sent
    assert links[1].sent == [] and len(sent) == 1
    kind, epoch, picks, plan = sent[0]
    assert (kind, epoch) == ('pick', 0) and len(picks) == 3
    assert [i for i, _ in picks] == [0, 1, 2]
    assert plan == dict(world=1, cap=32, mode='dynamic', batch=5, sleep=.02)
    cells = [c for _, c in picks]
    assert len(set(cells)) == 3 and session.inflight == set(cells)
    assert all(separated(c, set(cells) - {c}) for c in cells)
    want(session, 1, range(4))
    picks1 = links[1].sent[0][2]
    assert len(picks1) == 2
    assert session.inflight == set(cells) | {c for _, c in picks1}
    assert all(c not in session.initial for _, c in picks1)
    allc = list(session.inflight)
    assert all(separated(c, set(allc) - {c}) for c in allc)


def test_pick_matches_scheduler_and_nearest_player(tmp_path, monkeypatch):
    session, links = make_session(tmp_path, monkeypatch, gpus=1)
    write_control(session, batch=1, player_zx=[100.0, 8.0])
    expected = pick_window(set(session.cells), set(), [100.0, 8.0])
    want(session, 0, [3])
    assert links[0].sent[0][2] == [(3, expected)]


def test_share_applies_per_rank_and_active_counts(tmp_path, monkeypatch):
    session, links = make_session(tmp_path, monkeypatch)
    write_control(session, batch=3)
    want(session, 1, range(4), active=0)
    assert len(links[1].sent[0][2]) == 1  # rank 1 share of 3 over 2 ranks
    want(session, 0, range(4), active=1)
    assert len(links[0].sent[1 - 1][2]) == 1  # share 2, one already active
    want(session, 0, range(4), active=2)
    assert links[0].sent[1][2] == []


def test_fixed_mode_batches(tmp_path, monkeypatch):
    session, links = make_session(tmp_path, monkeypatch)
    write_control(session, batch=4, mode='fixed')
    want(session, 0, range(4))
    assert session.batch_open and session.batch_joined == {0} and len(links[0].sent[0][2]) == 2
    want(session, 0, range(4), active=2)
    assert links[0].sent[1][2] == []
    want(session, 1, range(4))
    assert not session.batch_open and session.batch_joined == {0, 1} and len(links[1].sent[0][2]) == 2
    assert links[1].sent[0][3]['mode'] == 'fixed'
    want(session, 1, range(4), active=2)
    assert links[1].sent[1][2] == []  # batch already closed, windows in flight


def test_paused_and_room_and_player_gates_still_reply_pick(tmp_path, monkeypatch):
    session, links = make_session(tmp_path, monkeypatch, gpus=1)
    write_control(session, paused=True)
    want(session, 0, range(4))
    assert links[0].sent[0][0] == 'pick' and links[0].sent[0][2] == []
    write_control(session, paused=False, ready=False)
    session.headless = 0
    want(session, 0, range(4))
    assert links[0].sent[1][2] == []
    write_control(session, ack=-1)
    session.commit_seqs = [5, 6, 7]
    session.lag = 2
    want(session, 0, range(4))
    assert links[0].sent[2][2] == []  # display ack behind the barrier
    write_control(session, ack=6)
    want(session, 0, range(4))
    assert len(links[0].sent[3][2]) == 4
    session.generated = session.max_windows
    want(session, 0, range(4), active=0)
    assert links[0].sent[4][2] == []


def test_headless_without_player(tmp_path, monkeypatch):
    session, links = make_session(tmp_path, monkeypatch, gpus=1, headless_windows=2, max_windows=2)
    want(session, 0, range(4))
    assert len(links[0].sent[0][2]) == 2  # room is capped by max_windows
    session.handle(done_message(0, 0, links[0].sent[0][2][0][1]))
    session.handle(done_message(0, 0, links[0].sent[0][2][1][1]))
    assert session.generated == 2
    want(session, 0, range(4))
    assert links[0].sent[-1][2] == []
    assert session.stopping is not None  # quiet at max_windows halts at once
    assert links[0].sent[-1] == ('pick', 0, [], dict(world=1, cap=32, mode='dynamic', batch=4, sleep=.05)) or links[0].sent[-2] == ('stop',)


def test_done_broadcasts_commit_before_pick_reply(tmp_path, monkeypatch):
    session, links = make_session(tmp_path, monkeypatch)
    write_control(session, batch=2)
    want(session, 0, range(4))
    target = links[0].sent[0][2][0][1]
    drain(links[0]); drain(links[1])
    session.handle(done_message(0, 0, target, nfe=7, value=4))
    for link in links:
        (message,) = link.sent
        assert message[:3] == ('commit', 0, target) and int(message[3].sum()) == 4 * 16
    assert target in session.cells and target not in session.inflight
    assert session.generated == 1 and session.finished_nfe == [7]
    lines = (session.out / 'window_times.jsonl').read_text().splitlines()
    assert len(lines) == 1 and json.loads(lines[0])['nfe'] == 7
    # the next pick reply to either rank is queued after the commit
    want(session, 1, range(4))
    assert [m[0] for m in links[1].sent] == ['commit', 'pick']


def test_handle_filters_and_lifecycle(tmp_path, monkeypatch):
    session, links = make_session(tmp_path, monkeypatch)
    session.handle(('built', 0, None))
    session.handle(('built', 1, None))
    assert session.built == {0, 1}
    write_control(session, batch=2)
    want(session, 0, range(4), epoch=7)
    session.handle(('start', 3, 0, dict(target=[0, 2])))
    session.handle(('done', 9, 0, (0, 2), np.zeros((4, 2, 2), np.int16), {}, {}))
    session.handle(('job', 9, 0, 1, 8, [dict(row=1)], []))
    assert links[0].sent == [] and wire_packets(session) == [] and session.generated == 0
    session.handle(('bye', 1))
    assert session.bye == {1}
    session.handle(('error', 0, 'gpu0: boom'))
    assert session.errors == ['gpu0: boom'] and session.stopping is not None
    assert [m for m in links[0].sent] == [('stop',)] and [m for m in links[1].sent] == [('stop',)]
    session.handle(('error', 1, 'again'))
    assert session.errors == ['gpu0: boom', 'again'] and links[0].sent == [('stop',)]  # halt is idempotent
    session.handle(('want', 0, 0, [0], 0))
    session.handle(('start', 0, 0, dict(target=[0, 2])))
    assert links[0].sent == [('stop',)] and wire_packets(session) == []
    session.handle(('job', 0, 0, 1, 8, [dict(row=1)], []))
    assert [p['kind'] for p in wire_packets(session)] == ['frame']


def test_start_and_job_publish(tmp_path, monkeypatch):
    session, links = make_session(tmp_path, monkeypatch)
    packet = dict(target=[0, 2], slot=1, gpu=0, world=1, cap=8, seed=1, mode='dynamic', batch=4)
    session.handle(('start', 0, 0, packet))
    commit = dict(target=[0, 2], nfe=3, gpu=0, slot=1)
    session.handle(('job', 0, 1, 12, 8, [dict(row=65)], [commit]))
    session.handle(('job', 0, 1, 13, 8, [], [commit]))
    packets = wire_packets(session)
    assert [p['kind'] for p in packets] == ['window_start', 'frame', 'window_commit', 'window_commit']
    assert packets[0]['target'] == [0, 2] and packets[0]['seq'] == 0
    frame = packets[1]
    assert (frame['forward'], frame['gpu'], frame['world'], frame['cap'], frame['slots'], frame['committed']) == (12, 1, 1, 8, 8, 0)
    assert session.commit_seqs == [2, 3]
    status = json.loads((session.out / 'status.json').read_text())
    assert status['state'] == 'running' and status['epoch'] == 0


def test_reset_flow(tmp_path, monkeypatch):
    session, links = make_session(tmp_path, monkeypatch)
    write_control(session, batch=2, reset_seq=3)
    want(session, 0, range(4))  # first sight of reset_seq only records it
    assert session.epoch == 0 and session.handled_reset == 3
    target = links[0].sent[0][2][0][1]
    session.handle(done_message(0, 0, target))
    drain(links[0]); drain(links[1])
    write_control(session, batch=2, reset_seq=4, world_req=2)
    want(session, 1, range(4))
    assert session.epoch == 1 and session.world == 2 and session.inflight == set() and session.generated == 0
    assert target not in session.cells and set(session.cells) == session.initial
    for link in links:
        (message,) = link.sent
        assert message[0] == 'reset' and message[1:3] == (1, 2) and set(message[3]) == session.initial
    packets = wire_packets(session)
    assert packets[-1]['kind'] == 'reset' and packets[-1]['epoch'] == 1 and packets[-1]['world'] == 2
    assert [0, 0] in packets[-1]['clear'] and list(target) in packets[-1]['clear']
    # a stale-epoch done is now dropped
    session.handle(done_message(0, 0, (0, 5)))
    assert (0, 5) not in session.cells
    # rejected world keeps the epoch
    write_control(session, reset_seq=5, world_req=99)
    want(session, 0, range(4))
    assert session.epoch == 1 and json.loads((session.out / 'status.json').read_text())['state'] == 'reset_rejected'


def test_applied_settings_fields(tmp_path, monkeypatch):
    session, links = make_session(tmp_path, monkeypatch)
    write_control(session, batch=99, steps=16, mode='fixed')
    want(session, 0, range(4))
    data = applied(session)
    assert data == dict(steps=16, batch=8, mode='fixed', paused=False, world=1, windows=0, inflight=len(session.inflight), slots=8,
                        idle=4 - len(links[0].sent[0][2]), mean_nfe=0, epoch=0)
    assert links[0].sent[0][3]['batch'] == 8 and links[0].sent[0][3]['cap'] == 16


def test_init_payload_default_identity(tmp_path, monkeypatch):
    session, _ = make_session(tmp_path, monkeypatch)
    init = session.init_for(1, 12)
    for key in ('cfg', 'run', 'world', 'cells', 'biomes', 'capacity', 'l1_cap', 'l2_cap', 'last_eps', 'air_ids', 'frame_interval',
                'progressive', 'raw_map', 'fallback', 'gate_steps', 'parent'):
        assert key in init
    assert init['capacity'] == 4 and init['gate_steps'] == 12 and init['parent'] == os.getpid()
    # identity: the rank is the worker id and the position in visible_gpus; the child sees only visible_gpus[rank] as cuda:0
    assert (init['wid'], init['torch_device'], init['nvsmi_index'], init['run_gate'], init['capacity']) == (1, 0, 1, False, 4)
    assert session.init_for(0, 1)['run_gate'] is True


def test_session_publish_gpu_count_unchanged(tmp_path, monkeypatch):
    session, _ = make_session(tmp_path, monkeypatch)
    assert session.slots_per == 4 and session.devices == [0, 1] and session.lag == 1000


# ---- scripted golden transcripts ---------------------------------------------------------------

def scenario_dynamic(session, links):
    write_control(session, batch=5)
    want(session, 0, range(4))
    want(session, 1, range(4))
    for wid in (0, 1):
        for index, target in links[wid].sent[-1][2]:
            session.handle(('start', 0, wid, dict(target=list(target), slot=index, gpu=wid)))
    first = links[0].sent[-1][2][0][1]
    session.handle(done_message(0, 0, first, nfe=6))
    session.handle(('job', 0, 0, 1, 8, [dict(row=0)], [dict(target=list(first), nfe=6)]))
    want(session, 1, [0, 1, 2, 3], active=2)
    want(session, 0, [0, 1, 2, 3], active=2)


def scenario_fixed(session, links):
    write_control(session, batch=4, mode='fixed')
    want(session, 0, range(4))
    want(session, 1, range(4))
    want(session, 0, range(4), active=2)
    for wid in (0, 1):
        for _, target in links[wid].sent[0][2]:
            session.handle(done_message(0, wid, target))
    want(session, 1, range(4))
    want(session, 0, range(4))


def scenario_reset(session, links):
    write_control(session, batch=2)
    want(session, 0, range(4))
    session.handle(done_message(0, 0, links[0].sent[0][2][0][1]))
    write_control(session, batch=2, reset_seq=1)
    want(session, 1, range(4))
    write_control(session, batch=2, reset_seq=2, world_req=3)
    want(session, 1, range(4))
    want(session, 0, range(4))


def scenario_paused_limits(session, links):
    write_control(session, paused=True, batch=2)
    want(session, 0, range(4))
    write_control(session, paused=False, batch=2)
    session.max_windows = 3
    want(session, 0, range(4))
    want(session, 1, range(4))
    session.handle(('error', 1, 'x'))
    want(session, 0, range(4))
    session.handle(('bye', 0))


def scenario_headless(session, links):
    session.headless = 2
    session.max_windows = 2
    write_control(session, player_present=False, ready=False)
    want(session, 0, range(4))
    for _, target in links[0].sent[0][2]:
        session.handle(done_message(0, 0, target))
    want(session, 0, range(4))
    session.idle_limit = 0
    want(session, 1, range(4))


SCENARIOS = dict(dynamic=(scenario_dynamic, {}), fixed=(scenario_fixed, {}), reset=(scenario_reset, {}),
                 paused_limits=(scenario_paused_limits, {}), headless=(scenario_headless, dict(headless_windows=0)))


def record(name, tmp_path, monkeypatch):
    script, extra = SCENARIOS[name]
    session, links = make_session(tmp_path, monkeypatch, **extra)
    script(session, links)
    return dict(
        sent=[[norm_msg for norm_msg in drain_all(link)] for link in links],
        wire=wire_packets(session), applied=applied(session),
        state=dict(epoch=session.epoch, world=session.world, generated=session.generated, inflight=sorted(map(list, session.inflight)),
                   cells=sorted(map(list, session.cells)), batch_open=session.batch_open, batch_joined=sorted(session.batch_joined),
                   commit_seqs=session.commit_seqs, errors=session.errors, stopping=session.stopping is not None))


def drain_all(link):
    return drain(link)


@pytest.mark.parametrize('name', sorted(SCENARIOS))
def test_golden_transcripts(name, tmp_path, monkeypatch):
    result = json.loads(json.dumps(record(name, tmp_path, monkeypatch)))
    if os.environ.get('SV_RECORD_GOLDEN'):
        golden = json.loads(GOLDEN.read_text()) if GOLDEN.exists() else {}
        golden[name] = result
        GOLDEN.write_text(json.dumps(golden, indent=1, sort_keys=True) + '\n')
        return
    assert result == json.loads(GOLDEN.read_text())[name]


# ---- stage-1 additions (new bookkeeping that must not change the message flow) -----------------

def test_cseq_owner_and_capacity(tmp_path, monkeypatch):
    session, links = make_session(tmp_path, monkeypatch)
    assert session.capacity_total() == 8
    write_control(session, batch=2)
    want(session, 0, range(4))
    want(session, 1, range(4))
    assert set(session.owner.values()) == {0, 1} and len(session.owner) == len(session.inflight)
    targets = {wid: links[wid].sent[-1][2][0][1] for wid in (0, 1)}
    for wid in (0, 1):
        target = targets[wid]
        session.handle(done_message(0, wid, target))
        assert target not in session.owner
    commits = [m for m in links[0].sent if m[0] == 'commit']
    assert [m[4] for m in commits] == [1, 2] and [m[4] for m in links[1].sent if m[0] == 'commit'] == [1, 2]
    write_control(session, batch=2, reset_seq=1)
    session.handle(('want', 0, 0, [0], 0))
    write_control(session, batch=2, reset_seq=2)
    session.handle(('want', 0, 0, [0], 0))
    assert session.cseq == 0 and session.owner == {}


def test_replica_tracks_cseq_ignores_duplicates_and_refuses_gaps():
    from streamvoxel.stream.worker import ReplicaGap
    replica = Replica(1, {}, None)
    assert replica.apply(('commit', 0, (0, 0), 'a', 1)) == 'commit' and replica.last_cseq == 1
    assert replica.apply(('commit', 0, (0, 0), 'a', 1)) is None and replica.last_cseq == 1
    with pytest.raises(ReplicaGap):
        replica.apply(('commit', 0, (0, 1), 'b', 4))
    assert replica.apply(('commit', 0, (0, 1), 'b', 2)) == 'commit' and replica.last_cseq == 2


def test_pump_once_drives_handle_with_a_plain_queue(tmp_path, monkeypatch):
    session, links = make_session(tmp_path, monkeypatch)
    session.up = queue.Queue()

    class Proc:
        def is_alive(self):
            return True
    session.procs = [Proc(), Proc()]
    write_control(session, batch=2)
    session.up.put(('built', 0, None))
    session.up.put(('built', 1, None))
    session.up.put(('want', 0, 1, [0, 1, 2, 3], 0))
    session.up.put(('start', 0, 1, dict(target=[0, 2])))
    assert session.pump_once() is True
    assert session.built == {0, 1} and not session.started
    assert links[0].sent == [] and [m[0] for m in links[1].sent] == ['pick']
    assert [p['kind'] for p in wire_packets(session)] == ['window_start']  # control first, then one wire message
    assert session.pump_once() is True and session.started
    assert [m[0] for m in links[0].sent] == ['go'] and [m[0] for m in links[1].sent] == ['pick', 'go']
    session.handle(('error', 0, 'x'))
    session.bye = {0, 1}
    assert session.pump_once() is False
