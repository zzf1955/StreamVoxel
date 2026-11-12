"""Decoders that only evaluation uses: the random and global low-entropy baselines and the block/mixed variants.

The product sampler (4^3 blocks + low-entropy-last) is `streamvoxel.stream.slot.LowEntropyLastSlot`; `decode` drives it
with the same slot code the live engine runs, so E3 measures the shipped sampler. All other selections are baselines.

`forward(tokens[1,Y,Z,X], t) -> logits[1,C,Y,Z,X]`. `spec` is a dict:
  selection  random | entropy | mixed (candidate_multiplier) | block (block_size) | low_entropy_last (last_eps, block_size, air_ids, air_token)
  schedule   linear | quartercos | halfcos | reversequarter | power<p> | front<p> | native_quarter (1024-step random reference)
  steps      forward cap
RNG: one generator on `seed` for draws and categorical samples, block order on `seed + 9173`.
"""
import math
import time

import numpy as np
import torch

from streamvoxel.sampling.block_quota import block_view, top_mask
from streamvoxel.sampling.entropy import entropy_scores
from streamvoxel.sampling.schedule import actual_time, counts


def mixed_select(score, mask, position_draw, k, c):
    if not math.isfinite(c) or c < 1:
        raise ValueError('candidate multiplier must be finite and >=1')
    candidate_count = min(int(mask.sum()), math.ceil(c * k))
    candidates = top_mask(position_draw, mask, candidate_count)
    return top_mask(score, candidates, k), dict(candidate_count=candidate_count, candidate_multiplier=c)


def block_select(score, mask, k, b, order, committed_before):
    """Apportion the global linear cumulative quota over equal blocks (seeded order breaks remainder ties),
    then reveal inside each block the lowest-entropy positions its cumulative quota requires."""
    mb = block_view(mask, b)
    nb = len(order)
    after = committed_before + k
    q, rem = divmod(after, nb)
    target = torch.full((nb,), q, dtype=torch.long, device=mask.device)
    target[order[:rem]] += 1
    already = (~mb).sum(1)
    quota = target - already
    assert (quota >= 0).all() and (quota <= mb.sum(1)).all() and int(quota.sum()) == k
    sm = block_view(score, b).masked_fill(~mb, -float('inf'))
    ranks = torch.argsort(sm, dim=1, descending=True, stable=True)
    bc = torch.zeros_like(mb)
    bc.scatter_(1, ranks, torch.arange(b ** 3, device=mask.device)[None, :] < quota[:, None])
    y, z, x = mask.shape
    chosen = bc.reshape(y // b, z // b, x // b, b, b, b).permute(0, 3, 1, 4, 2, 5).reshape(mask.shape)
    assert int(chosen.sum()) == k and not (chosen & ~mask).any()
    return chosen, dict(block_size=b, global_cumulative_target=after, per_block_quota_min=int(quota.min()), per_block_quota_max=int(quota.max()))


def choose(score, mask, draw, q, spec, order=None, committed=0):
    mode = spec['selection']
    q = min(q, int(mask.sum()))
    info = {}
    if mode == 'random':
        chosen = top_mask(draw, mask, q)
    elif mode == 'entropy':
        chosen = top_mask(score, mask, q)
    elif mode == 'mixed':
        chosen, info = mixed_select(score, mask, draw, q, spec['candidate_multiplier'])
    elif mode == 'block':
        chosen, info = block_select(score, mask, q, spec['block_size'], order, committed)
    else:
        raise ValueError(mode)
    assert not (chosen & ~mask).any()
    return chosen, info


@torch.inference_mode()
def decode(forward, spec, seed, device, *, shape=(256, 16, 16), classes=304, trace=False):
    """One window from an all-MASK canvas. Returns `(result, meta)`: result has `compact` and, with `trace`, `commit_step` (0 = never)."""
    device = torch.device(device)
    if spec['selection'] == 'low_entropy_last':
        return decode_low_entropy_last(forward, spec, seed, device, shape, classes)
    tokens = torch.full((1, *shape), classes, device=device, dtype=torch.long)
    rng = torch.Generator(device=device).manual_seed(seed)
    native = spec['schedule'] == 'native_quarter'
    quotas = counts(tokens.numel(), spec)
    order = None
    if spec['selection'] == 'block':
        bg = torch.Generator(device=device).manual_seed(seed + 9173)
        order = torch.randperm(tokens.numel() // spec['block_size'] ** 3, device=device, generator=bg)
    commit = torch.zeros(shape, device=device, dtype=torch.int32)
    events = []
    remaining = tokens.numel()
    began = time.perf_counter()
    for s in range(spec['steps']):
        if remaining == 0:
            break
        mask = tokens[0] == classes
        t = (spec['steps'] - s) / spec['steps'] if native else actual_time(remaining / tokens.numel())
        logits = forward(tokens, t)
        draw = torch.rand(shape, device=device, generator=rng)
        score = entropy_scores(logits[0]) if spec['selection'] != 'random' else None
        if native:
            i = spec['steps'] - s
            p = 1. if i == 1 else (math.sin(math.pi * i / (2 * spec['steps'])) - math.sin(math.pi * (i - 1) / (2 * spec['steps']))) / math.sin(math.pi * i / (2 * spec['steps']))
            chosen, info = mask & (draw < p), {}
        else:
            chosen, info = choose(score, mask, draw, quotas[s], spec, order=order, committed=tokens.numel() - remaining)
        k = int(chosen.sum())
        if k:
            prob = logits[0].permute(1, 2, 3, 0)[chosen].float().softmax(-1)
            tokens[0][chosen] = torch.multinomial(prob, 1, generator=rng).squeeze(-1)
            commit[chosen] = s + 1
        remaining -= k
        events.append(dict(step=s + 1, model_t=t, committed=k, remaining=remaining, **info))
    if device.type == 'cuda':
        torch.cuda.synchronize()
    result = dict(compact=tokens[0].cpu().numpy().astype(np.int16))
    if trace:
        result['commit_step'] = commit.cpu().numpy()
    return result, dict(spec=spec, complete=remaining == 0, remaining=remaining, nfe=len(events), seconds=time.perf_counter() - began, events=events)


@torch.inference_mode()
def decode_low_entropy_last(forward, spec, seed, device, shape, classes):
    """The shipped sampler through the live slot: one marking forward, then the solid prefix of a `steps`-long quartercos schedule."""
    from streamvoxel.stream.slot import LowEntropyLastSlot
    slot = LowEntropyLastSlot(shape, classes, spec['air_ids'], spec['air_token'], spec['last_eps'], spec['block_size'], device)
    tokens = torch.full(shape, classes, device=device, dtype=torch.long)
    slot.arm(seed, spec['steps'], ())
    began = time.perf_counter()
    record = None
    while record is None:
        logits = forward(tokens[None], slot.time())[0]
        record, _ = slot.consume(logits, tokens)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    result = dict(compact=tokens.cpu().numpy().astype(np.int16), commit_step=record['step_map'].numpy())
    return result, dict(spec=spec, complete=True, remaining=0, nfe=record['nfe'], last_count=record['last_count'],
                        last_frac=record['last_frac'], seconds=time.perf_counter() - began)
