from streamvoxel.stream.scheduler import member_shares, rank_share


def test_equal_capacity_matches_rank_share():
    for capacity in (1, 4, 8, 16):
        for n in (1, 2, 3, 4):
            members = [(w, capacity, capacity) for w in range(n)]
            for batch in range(0, capacity * n + 1):
                got = member_shares(batch, members)
                assert [got[w] for w in range(n)] == [rank_share(batch, n, w) for w in range(n)], (capacity, n, batch)


def test_heterogeneous_examples_from_design():
    assert member_shares(12, [(0, 16, 16), (1, 8, 8)]) == {0: 8, 1: 4}
    assert member_shares(5, [(0, 16, 16), (1, 8, 8)]) == {0: 3, 1: 2}
    assert member_shares(5, []) == {}


def test_share_never_exceeds_capacity_and_sums_to_batch():
    members = [(0, 2, 2), (1, 8, 8), (3, 4, 4)]
    for batch in range(0, 15):
        got = member_shares(batch, members)
        assert sum(got.values()) == batch
        assert all(got[w] <= c for w, c, _ in members)
