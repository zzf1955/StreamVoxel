import numpy as np
import pytest
import torch

from streamvoxel.eval import metrics, samplers, state_probe
from streamvoxel.sampling.schedule import counts

SHAPE = (8, 8, 8)
CLASSES = 5


def random_forward(seed=0):
    state = torch.Generator().manual_seed(seed)

    def forward(tokens, t):
        return torch.randn(1, CLASSES, *SHAPE, generator=state)
    return forward


def confident_half_forward(tokens, t):
    """Voxels in the lower half (flat index < half) are near-certain, the rest uniform."""
    logits = torch.zeros(1, CLASSES, *SHAPE)
    flat = torch.arange(logits[0, 0].numel()).reshape(SHAPE)
    logits[0, 1] = torch.where(flat < flat.numel() // 2, 20., 0.)
    return logits


def test_every_selection_fills_the_window_in_the_planned_steps():
    for sel in (dict(selection='random'), dict(selection='entropy'), dict(selection='mixed', candidate_multiplier=1.5),
                dict(selection='block', block_size=2)):
        spec = dict(sel, schedule='quartercos', steps=8)
        result, meta = samplers.decode(random_forward(), spec, 7, 'cpu', shape=SHAPE, classes=CLASSES, trace=True)
        assert meta['complete'] and meta['nfe'] == 8
        assert result['compact'].min() >= 0 and result['compact'].max() < CLASSES
        assert (result['commit_step'] >= 1).all() and result['commit_step'].max() == 8
        counts_per_step = np.bincount(result['commit_step'].ravel(), minlength=9)[1:]
        assert counts_per_step.tolist() == counts(512, spec)


def test_decode_is_deterministic_per_seed():
    spec = dict(selection='block', block_size=2, schedule='quartercos', steps=4)
    a = samplers.decode(random_forward(1), spec, 3, 'cpu', shape=SHAPE, classes=CLASSES)[0]['compact']
    b = samplers.decode(random_forward(1), spec, 3, 'cpu', shape=SHAPE, classes=CLASSES)[0]['compact']
    c = samplers.decode(random_forward(1), spec, 4, 'cpu', shape=SHAPE, classes=CLASSES)[0]['compact']
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_global_entropy_reveals_confident_voxels_first_and_random_does_not():
    spec = dict(selection='entropy', schedule='linear', steps=8)
    commit = samplers.decode(confident_half_forward, spec, 0, 'cpu', shape=SHAPE, classes=CLASSES, trace=True)[0]['commit_step']
    flat = commit.ravel()
    assert (flat[:256] <= 4).all() and (flat[256:] >= 4).all()
    spec = dict(selection='random', schedule='linear', steps=8)
    flat = samplers.decode(confident_half_forward, spec, 0, 'cpu', shape=SHAPE, classes=CLASSES, trace=True)[0]['commit_step'].ravel()
    assert (flat[256:] <= 4).any()


def test_block_selection_balances_commits_over_blocks():
    spec = dict(selection='block', block_size=2, schedule='linear', steps=8)
    commit = samplers.decode(random_forward(), spec, 5, 'cpu', shape=SHAPE, classes=CLASSES, trace=True)[0]['commit_step']
    blocks = commit.reshape(4, 2, 4, 2, 4, 2).transpose(0, 2, 4, 1, 3, 5).reshape(-1, 8)
    for step in range(1, 9):
        per_block = (blocks <= step).sum(1)
        assert per_block.max() - per_block.min() <= 1


def test_low_entropy_last_defers_certain_air_to_the_final_forward():
    air_logits = torch.zeros(1, CLASSES, *SHAPE)
    air_logits[0, 0, 4:] = 30.
    air_logits[0, 1, :4] = 0.

    def forward(tokens, t):
        return air_logits.clone()
    spec = dict(selection='low_entropy_last', schedule='quartercos', steps=4, block_size=2, last_eps=1e-4, air_ids=[0], air_token=0)
    result, meta = samplers.decode(forward, spec, 1, 'cpu', shape=SHAPE, classes=CLASSES)
    assert meta['last_count'] == 4 * 64 and meta['complete']
    assert (result['compact'][4:] == 0).all()
    assert (result['compact'] < CLASSES).all()


def test_voxel_scores_pool_rings_and_count_unknown_gt_as_errors():
    gt = np.zeros((4, 32, 32), np.uint16)
    pred = gt.copy()
    pred[:, :16, :16] = 1
    order = [(1, 0, 0), (2, 0, 1)]
    scores = metrics.voxel_scores(gt, pred, order, air_ids=(0,), classes=3)
    assert scores['ring1']['acc'] == 0. and scores['ring2']['acc'] == 1.
    assert scores['all']['acc'] == pytest.approx(.5) and scores['all']['voxels'] == 2 * 4 * 256
    assert scores['all']['nonair_acc'] == 0.


def test_stale_edges_counts_touching_cells_only():
    assert metrics.stale_edges([(0, 0), (0, 2), (2, 0), (2, 2)]) == 0
    assert metrics.stale_edges([(0, 0), (0, 1)]) == 1
    assert metrics.stale_edges([(0, 0), (1, 1), (0, 1)]) == 3
    assert metrics.stale_edges([(3, 3), (3, 3)]) == 0


def test_array_metrics_seam_is_zero_for_a_perfect_window():
    pytest.importorskip('scipy')
    rng = np.random.default_rng(0)
    gt = rng.integers(0, 3, (16, 16, 16))
    context = np.full((16, 32, 32), -1)
    context[:, :16, 16:] = rng.integers(0, 3, (16, 16, 16))
    r = metrics.array_metrics(gt, gt.copy(), [0], context=context, target=(0, 0))
    assert r['acc'] == 1. and r['seam_occupancy_error'] == 0. and r['seam_height_jump_mae'] == 0.
    wrong = np.zeros_like(gt)
    assert metrics.array_metrics(gt, wrong, [0], context=context, target=(0, 0))['seam_occupancy_error'] > 0


def test_state_probe_masks_are_nested_and_reproducible():
    rng = np.random.default_rng(0)
    gt = rng.integers(0, 3, (16, 4, 4))
    picks = {'m': rng.integers(1, 5, gt.shape)}
    a = state_probe.build_masks(gt, np.asarray([0]), 11, picks)
    b = state_probe.build_masks(gt, np.asarray([0]), 11, picks)
    assert all(np.array_equal(a[k], b[k]) for k in a)
    for name in ('iid', 'air_last', 'band_topdown', 'entropy_m'):
        assert (a[(name, .3)] <= a[(name, .5)]).all() and a[(name, .5)].sum() == round(.5 * gt.size)


def test_state_probe_strata_probabilities_hit_the_mean_and_ece_auroc_behave():
    p = state_probe.strata_probabilities(np.random.default_rng(0), np.array([True, False] * 16), .3)
    assert p.mean() == pytest.approx(.7, abs=1e-6)
    confidence = np.array([.95, .95, .55, .55])
    correct = np.array([True, True, True, False])
    assert state_probe.ece(confidence, correct) == pytest.approx(abs(.95 - 1.) * .5 + abs(.55 - .5) * .5)
    assert state_probe.rank_auroc(np.array([1., 2., 3., 4.]), np.array([False, False, True, True])) == 1.
    assert np.isnan(state_probe.rank_auroc(np.array([1., 2.]), np.array([True, True])))


def test_surface_render_uses_highest_non_air_block_and_black_for_empty_columns():
    from streamvoxel.render.topdown import surface_rgb
    compact = np.zeros((256, 2, 2), np.int64)
    compact[10, 0, 0] = 1
    compact[20, 0, 0] = 2
    colors = np.array([[0, 0, 0], [255, 0, 0], [0, 255, 0]], np.float32)
    air = np.array([True, False, False])
    rgb, height, top = surface_rgb(compact, np.arange(3), colors, air)
    assert height[0, 0] == 20 and top[0, 0] == 2 and rgb[0, 0, 1] > rgb[0, 0, 0]
    assert (rgb[1, 1] == 0).all()
