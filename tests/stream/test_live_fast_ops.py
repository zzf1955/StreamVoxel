"""CPU contracts of `streamvoxel.stream.fast_ops` (GPU numerics and timing: `python scripts/bench_ops.py`)."""
import numpy as np
import pytest
import torch

from streamvoxel.sampling.block_quota import top_mask
from streamvoxel.sampling.block_quota import block_view
from streamvoxel.sampling.block_quota import block_quota_select
from streamvoxel.stream.fast_ops import (FastLowEntropyLastSlot, FastSlotForward, GatherPatch, consume_batch, conv_patch,
                                      quota_select_nosync, top_mask_nosync)
from streamvoxel.stream.slot import LowEntropyLastSlot


def test_gather_patch_equals_onehot_conv():
    torch.manual_seed(0)
    conv = torch.nn.Conv3d(6, 10, 4, stride=4)
    tokens = torch.randint(6, (2, 8, 12, 16))
    reference = conv_patch(conv, 5)(tokens)
    gathered = GatherPatch(conv, torch.float32)(tokens)
    assert gathered.shape == reference.shape == (2, 2 * 3 * 4, 10)
    torch.testing.assert_close(gathered, reference, rtol=1e-5, atol=1e-5)


def test_top_mask_nosync_matches_top_mask_with_ties():
    g = torch.Generator().manual_seed(1)
    for count in (0, 1, 7, 40, 500):
        scores = torch.randint(-4, 1, (8, 8, 8), generator=g).float()   # many exact ties
        eligible = torch.rand((8, 8, 8), generator=g) < .3
        assert torch.equal(top_mask_nosync(scores, eligible, count), top_mask(scores, eligible, count))


def _state(seed, shape=(16, 8, 8), b=4):
    g = torch.Generator().manual_seed(seed)
    last = torch.zeros(shape, dtype=torch.bool)
    tops = torch.randint(4, shape[0], shape[1:], generator=g)
    for z in range(shape[1]):
        for x in range(shape[2]):
            last[tops[z, x]:, z, x] = True
    non_last = ~last
    committed_mask = non_last & (torch.rand(shape, generator=g) < torch.rand((), generator=g))
    selectable = non_last & ~committed_mask
    k = int(torch.randint(1, max(2, int(selectable.sum())), (), generator=g))
    committed = int(committed_mask.sum())
    frac = (committed + k) / int(non_last.sum())
    blocks = int(np.prod(shape)) // b ** 3
    order = torch.randperm(blocks, generator=g)
    n_total = block_view(non_last.contiguous(), b).sum(1)
    score = (torch.randn(shape, generator=g) * 2).round() / 2     # ties inside blocks
    return score, selectable, k, b, order, frac, n_total, committed


@pytest.mark.parametrize('seed', range(25))
def test_quota_select_nosync_matches_block_quota_select(seed):
    score, selectable, k, b, order, frac, n_total, committed = _state(seed)
    expect = block_quota_select(score, selectable, k, b, order, frac, n_total)
    pos = torch.empty_like(order)
    pos[order] = torch.arange(len(order))
    got = quota_select_nosync(score, selectable, k, b, pos, frac, n_total, committed)
    assert torch.equal(got, expect)
    assert int(got.sum()) == k


def _logits(step, ground=8):
    generator = torch.Generator().manual_seed(step + 1)
    logits = torch.randn((6, 16, 8, 8), generator=generator)
    logits[0, ground:] += 30.
    return logits


def _run(kind, seeds, cap=8, batched=False):
    shape, classes = (16, 8, 8), 6
    make = FastLowEntropyLastSlot if kind == 'fast' else LowEntropyLastSlot
    slots = [make(shape, classes, air_ids=(0,), air_token=0, last_eps=1e-3, block_size=4, device='cpu') for _ in seeds]
    tokens = torch.full((len(seeds), *shape), classes, dtype=torch.long)
    for i, seed in enumerate(seeds):
        slots[i].arm(seed, cap + i, (i, 0))
    finished, commits, step = {}, [], 0
    while len(finished) < len(seeds):
        logits = torch.stack([_logits(step + 31 * i, ground=6 + i) for i in range(len(seeds))])
        if batched:
            results = consume_batch(slots, logits, tokens, len(seeds))
        else:
            results = {i: slots[i].consume(logits[i], tokens[i]) for i in range(len(seeds)) if not slots[i].idle}
        for i, (done, commit) in results.items():
            if commit is not None:
                commits.append((i, commit['step'], commit['positions'].tolist(), commit['values'].tolist()))
            if done is not None:
                finished[i] = done
        step += 1
        assert step < 64
    return finished, commits, tokens


@pytest.mark.parametrize('batched', [False, True])
def test_fast_slot_reproduces_slot(batched):
    seeds = [3, 11, 12345]
    ref, ref_commits, ref_tokens = _run('old', seeds)
    got, commits, tokens = _run('fast', seeds, batched=batched)
    assert commits == ref_commits
    assert torch.equal(tokens, ref_tokens)
    for i in ref:
        for key in ('nfe', 'actual', 'last_count'):
            assert got[i][key] == ref[i][key]
        assert torch.equal(got[i]['values'], ref[i]['values'])
        assert torch.equal(got[i]['step_map'], ref[i]['step_map'])


def _tiny(timed):
    from streamvoxel.models.compact_controlnet import CompactStreamingControlNet
    from streamvoxel.stream.condition import ModelSpec
    torch.manual_seed(5)
    if timed:
        from streamvoxel.models.unconditional_md4_timefree import UnconditionalMD4DiT
        base = UnconditionalMD4DiT(num_classes=3, voxel_shape=(16, 16, 16), hidden_dim=48, depth=5, num_heads=4, time_free=True)
    else:
        from streamvoxel.models.unconditional_md4 import UnconditionalMD4DiT
        base = UnconditionalMD4DiT(num_classes=3, voxel_shape=(16, 16, 16), hidden_dim=48, depth=5, num_heads=4)
    model = CompactStreamingControlNet(base, control_depth=3, control_hidden_dim=24, checkpointing=False,
                                       biome_halo=4, biome_patch=2, struct_classes=0)
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(std=.05)
    model.eval().requires_grad_(False)
    spec = ModelSpec(height=16, tile=16, num_classes=3, biome_classes=66, biome_halo=4, y_origin=-64.)
    return model, spec


def _item(seed, n1, n2):
    from streamvoxel.stream.condition import Item
    rng = np.random.default_rng(seed)
    l1 = rng.integers(0, 3, (n1, 16, 16, 16)).astype(np.int16)
    l2 = rng.integers(0, 3, (n2, 16, 16, 16)).astype(np.int16)
    xz1 = np.asarray([[-16, 0], [16, 0], [0, 16]][:n1], np.int64)
    xz2 = np.asarray([[-32, 0], [32, 16], [0, -32]][:n2], np.int64)
    biomes = np.zeros((4, 12, 12), np.int64)
    biomes[:, 3:9, 3:9] = rng.integers(0, 66, (4, 6, 6))
    return Item(f'i{seed}', seed, 8, 1.0, l1, xz1, l2, xz2, biomes, None)


@pytest.mark.parametrize('timed', [False, True])
def test_fast_slot_forward_matches_unpadded_reference(timed):
    from streamvoxel.stream.engine import unpadded_logits
    model, spec = _tiny(timed)
    engine = FastSlotForward(model, spec, 'cpu', 2, 3, 3, patchify='gather', attention='sdpa', capture=False)
    items = [_item(1, 1, 2), _item(2, 3, 3)]
    for i, item in enumerate(items):
        engine.load_slot(i, item)
    tokens = torch.randint(4, (2, 16, 16, 16))
    engine.tokens.copy_(tokens)
    got = engine.eager(torch.ones(2))
    assert engine.packed.lengths.tolist() == [2 + 64 + 18, 3 + 3 * 64 + 18]
    for i, item in enumerate(items):
        want = unpadded_logits(model, item, spec, tokens[i:i + 1], torch.ones(1))[0]
        torch.testing.assert_close(got[i], want, rtol=1e-4, atol=1e-4)
    single = engine._call(1)[0]
    torch.testing.assert_close(single, got[0], rtol=1e-4, atol=1e-4)
