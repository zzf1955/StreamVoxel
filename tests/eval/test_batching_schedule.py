from streamvoxel.eval import batching_schedule as bs
from streamvoxel.eval.metrics import stale_edges

BATCH, STEPS, WINDOWS = 4, 3, 12


def run(arm, batch=BATCH, windows=WINDOWS):
    return bs.simulate(bs.initial_cells(), arm, batch, STEPS, windows)


def test_every_arm_commits_each_window_once_with_a_committed_neighbour():
    for arm in bs.ARMS:
        schedule, sizes, stale = run(arm)
        cells = [w['cell'] for w in schedule]
        assert len(cells) == WINDOWS == len(set(cells)) and not set(cells) & bs.initial_cells()
        assert sum(sizes) == WINDOWS * STEPS and len(sizes) == len(stale)
        assert max(sizes) <= (1 if arm == 'serial' else BATCH)
        for w in sorted(schedule, key=lambda w: w['start']):
            done = {v['cell'] for v in schedule if v['end'] < w['start']} | bs.initial_cells()
            assert any((w['cell'][0] + a, w['cell'][1] + b) in done for a in (-1, 0, 1) for b in (-1, 0, 1))
            assert w['committed_at_start'] == len(done)


def test_serial_is_one_window_per_forward_and_never_stale():
    schedule, sizes, stale = run('serial')
    assert set(sizes) == {1} and len(sizes) == WINDOWS * STEPS and not any(stale)
    assert [w['start'] for w in schedule] == [i * STEPS for i in range(WINDOWS)]


def test_dependency_safe_arms_have_no_stale_edges_and_naive_does():
    for arm in ('safe_fixed', 'safe_continuous'):
        schedule, sizes, stale = run(arm)
        assert not any(stale)
        for f in range(len(sizes)):
            assert stale_edges([w['cell'] for w in schedule if w['start'] <= f <= w['end']]) == 0
    assert sum(run('naive_fixed')[2]) > 0


def test_fixed_batches_start_together_and_wait_for_the_slowest():
    schedule, sizes, _ = run('naive_fixed')
    assert sizes[:STEPS] == [BATCH] * STEPS
    starts = sorted({w['start'] for w in schedule})
    assert all(s % STEPS == 0 for s in starts)


def test_summary_counts_idle_slots():
    s = bs.summarize([4, 4, 2, 2], [0, 1, 0, 0], windows=3, batch=4)
    assert s['forwards'] == 4 and s['mean_batch'] == 3 and s['empty_slot_rate'] == 0.25
    assert s['batch_hist'] == {'2': 2, '4': 2} and s['stale_edges_total'] == 1 and s['forwards_with_stale'] == 1 and s['stale_edges_max'] == 1
    assert bs.summarize([1, 1], [0, 0], windows=1, batch=1)['empty_slot_rate'] == 0
