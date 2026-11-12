"""CPU checks for window admission: `separated`, `pick_window`, `plan_inserts`, `rank_share`."""
from streamvoxel.stream.scheduler import pick_window, plan_inserts, rank_share, separated


def test_separated_chebyshev_and_diagonal():
    assert separated((0, 0), [])
    assert not separated((0, 0), [(1, 1)])
    assert separated((0, 0), [(2, 1), (-2, -2)])
    assert separated((0, 0), [(1, 1)], diagonal=True)
    assert not separated((0, 0), [(0, 1)], diagonal=True)
    assert not separated((3, 3), [(3, 3)], diagonal=True)
    assert separated((0, 0), [(0, 3)], gap=3)
    assert not separated((0, 0), [(0, 2)], gap=3)


def test_pick_window_nearest_first_and_blocked_frontier():
    committed = {(0, 0)}
    # The player stands right of the committed cell: the nearest ready cell is (0, 1), ties break on (z, x).
    assert pick_window(committed, set(), (8., 40.)) == (0, 1)
    assert pick_window(committed, set(), (8., 8.)) == (-1, 0)
    # An in-flight neighbour blocks every ready cell of a single committed cell except those two cells away.
    assert pick_window(committed, {(0, 1)}, (8., 40.)) == (0, -1)
    assert pick_window(committed, {(0, 1), (-1, -1), (1, -1)}, (8., 8.)) is None
    assert pick_window(set(), set(), (0., 0.)) is None


def test_pick_window_skips_inflight_cells_themselves():
    committed = {(0, 0), (0, 3)}
    for diagonal in (False, True):
        cell = pick_window(committed, {(0, 1)}, (8., 24.), diagonal=diagonal)
        assert cell is not None and cell != (0, 1) and separated(cell, {(0, 1)}, diagonal=diagonal)


def test_plan_inserts_dynamic_and_fixed():
    assert plan_inserts([3, 5], 6, 8, 'dynamic') == [3, 5]
    assert plan_inserts([3, 5, 6], 7, 8, 'dynamic') == [3]
    assert plan_inserts([3], 8, 8, 'dynamic') == []
    assert plan_inserts([], 0, 8, 'dynamic') == []
    assert plan_inserts([0, 1, 2], 0, 2, 'fixed') == [0, 1]
    assert plan_inserts([0, 1, 2], 1, 2, 'fixed') == []
    assert plan_inserts([0, 1], 0, -3, 'dynamic') == []


def test_rank_share_gives_the_remainder_to_early_ranks():
    for batch in range(0, 20):
        for ranks in (1, 2, 3):
            shares = [rank_share(batch, ranks, rank) for rank in range(ranks)]
            assert sum(shares) == batch
            assert max(shares) - min(shares) <= 1
            assert shares == sorted(shares, reverse=True)
    assert rank_share(5, 3, 0) == 2 and rank_share(5, 3, 2) == 1
    assert rank_share(5, 0, 0) == 5
