"""E3b: teacher-forced state-shift probe. How well does a model predict masked voxels in partially revealed states of different families?

For every window, revealed fraction f and mask type the state holds the ground truth at revealed positions and MASK elsewhere;
one forward pass; statistics over the masked voxels (NLL of the truth, argmax accuracy, mean entropy, AUROC of entropy for
predicting argmax errors, ECE with 15 equal-width bins on max probability), overall, for truth-air / truth-solid voxels and
for 4 groups of 64 y-layers. Masks come from seeds independent of the model, so every model sees identical states.

Mask types (reveal order is nested in f except 'strata'):
  iid                 uniformly random reveal order.
  strata              training-style strata_mask: one noise level t_b ~ U(0,1) per 8-layer band; masked probability sin(pi t/2) for bands
                      that are all air in the truth, t otherwise; voxels iid inside a band. The band probabilities are multiplied by a
                      common scale (clipped to [0,1], bisection) so that the mean masked probability equals 1-f.
  entropy_<name>      revealed = voxels with the smallest (commit_step, random tie-break) of a recorded decode trajectory, cut at f*total.
  above_surface_first truth-air voxels strictly above the column's highest truth-solid voxel first (random order), then the rest random.
  band_topdown        whole 8-layer bands from the top down, the last band partially (random order inside a band).
  band_bottomup       the same from the bottom up.
  air_last            (uses ground truth: upper bound for a predicted-air-last sampler) truth-solid voxels first, then truth-air.
  air_last_soft       reveal order by score 2*1[solid] + Gumbel (descending): solid preferred, some air early.

python -m streamvoxel.eval.state_probe --case <case dir> --mapping compact_mapping.json --vocabulary vocabulary.json \\
    --base base.pt --control control.pt --output <new dir> [--target-cell 6 8] [--initial-box 7 7 9 9] \\
    [--picks global_entropy=<sampler_grid output>/<id>/commit_step.npy ...]
Run it once per checkpoint pair on the same case and compare `state_probe_summary.json`. Needs a GPU.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

FRACTIONS = (0.1, 0.3, 0.5, 0.7, 0.9)
BAND = 8
GROUPS = 4
ECE_BINS = 15


def rank_auroc(score, positive):
    """AUROC of `score` for predicting `positive` (bool); ties get average ranks. nan if a class is empty."""
    n_pos = int(positive.sum())
    n_neg = len(positive) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    try:
        from scipy.stats import rankdata
        ranks = rankdata(score)
    except ImportError:   # no ties handling
        ranks = np.empty(len(score))
        ranks[np.argsort(score, kind="stable")] = np.arange(1, len(score) + 1)
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def ece(confidence, correct, bins=ECE_BINS):
    index = np.minimum((confidence * bins).astype(np.int64), bins - 1)
    count = np.bincount(index, minlength=bins).astype(np.float64)
    conf = np.bincount(index, weights=confidence, minlength=bins)
    acc = np.bincount(index, weights=correct.astype(np.float64), minlength=bins)
    used = count > 0
    return float((np.abs(conf[used] - acc[used])).sum() / count.sum())


def summarise(nll, correct, entropy, confidence):
    if len(nll) == 0:
        return dict(n=0)
    return dict(n=int(len(nll)), nll=float(nll.mean()), acc=float(correct.mean()), entropy=float(entropy.mean()),
                auroc=rank_auroc(entropy, ~correct), ece=ece(confidence, correct))


def strata_probabilities(rng, all_air, f):
    """Per-band masked probability [bands]: training-style draw, common scale so that the mean is 1-f."""
    t = rng.uniform(1e-4, 1 - 1e-4, size=len(all_air))
    p = np.where(all_air, np.sin(t * math.pi / 2), t)
    target = 1 - f
    lo, hi = 0., 1e4
    for _ in range(60):
        mid = (lo + hi) / 2
        if np.clip(p * mid, 0, 1).mean() < target:
            lo = mid
        else:
            hi = mid
    return np.clip(p * hi, 0, 1)


def reveal_orders(gt, air_ids, rng, picks):
    """Reveal keys (smaller = revealed earlier) per order family for one window gt [Y,Z,X]. Returns dict name -> float key."""
    Y, Z, X = gt.shape
    u = rng.random(gt.shape)
    keys = {"iid": u}
    solid = ~np.isin(gt, air_ids)
    top = np.where(solid.any(0), Y - 1 - np.argmax(solid[::-1], axis=0), -1)[None]
    above = np.arange(Y)[:, None, None] > top
    keys["above_surface_first"] = u + (~above).astype(np.float64)
    band = np.arange(Y)[:, None, None] // BAND
    keys["band_topdown"] = (Y // BAND - 1 - band) + u * .999
    for name, step in picks.items():
        keys["entropy_" + name] = step.astype(np.float64) + u * .5
    return keys


def reveal_from_key(key, f):
    n = int(round(f * key.size))
    reveal = np.zeros(key.size, bool)
    reveal[np.argpartition(key.ravel(), n - 1)[:n]] = True
    return reveal.reshape(key.shape)


def build_masks(gt, air_ids, seed, picks, fractions=FRACTIONS):
    """{(mask_type, f): revealed bool [Y,Z,X]} for one window; depends only on gt, picks and seed."""
    rng = np.random.default_rng(seed)
    keys = reveal_orders(gt, air_ids, rng, picks)
    extra = np.random.default_rng(seed + 777_777)       # separate stream: keeps the older mask types unchanged
    u2 = extra.random(gt.shape)
    solid = ~np.isin(gt, air_ids)
    Y = gt.shape[0]
    keys["air_last"] = u2 + (~solid).astype(np.float64)
    keys["band_bottomup"] = (np.arange(Y)[:, None, None] // BAND) + u2 * .999
    gumbel = -np.log(-np.log(np.clip(extra.random(gt.shape), 1e-12, 1 - 1e-12)))
    keys["air_last_soft"] = -(2. * solid + gumbel)
    out = {(name, f): reveal_from_key(key, f) for name, key in keys.items() for f in fractions}
    Y, Z, X = gt.shape
    all_air = np.isin(gt, air_ids).reshape(Y // BAND, -1).all(1)
    u = rng.random(gt.shape)
    for f in fractions:
        p = strata_probabilities(rng, all_air, f)
        masked = u < np.repeat(p, BAND)[:, None, None]
        out[("strata", f)] = ~masked
    return out


def summarize_records(store, realised, Y):
    result = {}
    for key, parts in store.items():
        cat = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        subsets = {"all": np.ones(len(cat["y"]), bool), "air_truth": cat["air"], "solid_truth": ~cat["air"]}
        for g in range(GROUPS):
            subsets[f"y{g * Y // GROUPS}-{(g + 1) * Y // GROUPS}"] = (cat["y"] // (Y // GROUPS)) == g
        entry = {name: summarise(cat["nll"][m], cat["correct"][m], cat["entropy"][m], cat["conf"][m]) for name, m in subsets.items()}
        entry["revealed_fraction_realised"] = float(np.mean(realised[key]))
        result.setdefault(key[0], {})[f"{key[1]:g}"] = entry
    return result


def evaluate(forward, gt, picks, air_ids, classes, seed, time_fn, device, fractions=FRACTIONS, log=print):
    """forward(tokens[1,Y,Z,X], t) -> logits[1,C,Y,Z,X]; gt [Y,Z,X] compact ids. Returns the nested summary dict."""
    import torch
    Y = gt.shape[0]
    air = np.asarray(sorted(air_ids))
    store, realised = {}, {}
    gt_t = torch.from_numpy(gt.astype(np.int64)).to(device)[None]
    masks = build_masks(gt, air, seed, picks, fractions)
    for key, revealed in masks.items():
        reveal = torch.from_numpy(revealed).to(device)[None]
        tokens = torch.where(reveal, gt_t, torch.full_like(gt_t, classes))
        masked = ~reveal
        fraction = float(masked.float().mean())
        with torch.inference_mode():
            logits = forward(tokens, time_fn(fraction)).float()
            logp = logits.log_softmax(1)
            prob = logp.exp()
            nll = -logp.gather(1, gt_t[:, None])[:, 0]
            entropy = -(prob * logp).sum(1)
            top, arg = prob.max(1)
        del logits, logp, prob
        yy = torch.arange(Y, device=device)[None, :, None, None].expand_as(gt_t)[masked]
        isair = torch.isin(gt_t, torch.as_tensor(air, device=device))[masked]
        rec = dict(nll=nll[masked], correct=(arg == gt_t)[masked], entropy=entropy[masked], conf=top[masked], y=yy, air=isair)
        store.setdefault(key, []).append({k: v.cpu().numpy() for k, v in rec.items()})
        realised.setdefault(key, []).append(1 - fraction)
        log(json.dumps(dict(mask=key[0], f=key[1])))
    return summarize_records(store, realised, Y)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ('case', 'mapping', 'vocabulary', 'base', 'control', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--target-cell', type=int, nargs=2, default=(6, 8))
    p.add_argument('--initial-box', type=int, nargs=4, default=(7, 7, 9, 9))
    p.add_argument('--picks', nargs='*', default=[], help='name=path/to/commit_step.npy of a decode of this same window')
    p.add_argument('--l2-cap', type=int, default=256)
    p.add_argument('--seed', type=int, default=20261023)
    p.add_argument('--time', choices=('actual', 'zero'), default='actual', help='time input fed to the model (actual masked fraction)')
    p.add_argument('--device', default='cuda:0')
    a = p.parse_args()

    import torch
    from streamvoxel.common.io import atomic_json
    from streamvoxel.eval.window import initial_canvas, load_case, open_eval_model, window_forward, window_item
    from streamvoxel.sampling.schedule import actual_time

    gt, biomes, mapping, air_global, air_compact, air_token = load_case(a.case, a.mapping, a.vocabulary)
    device = torch.device(a.device)
    torch.backends.cuda.matmul.allow_tf32 = True
    model, spec, identity = open_eval_model(a.base, a.control, device)
    z, x = a.target_cell
    canvas, known = initial_canvas(gt, mapping, a.initial_box)
    item = window_item(canvas, known, (z, x), biomes, spec, a.seed, 1, a.l2_cap)
    forward = window_forward(model, item, spec, device)
    target = mapping.compact(gt[:, z * 16:(z + 1) * 16, x * 16:(x + 1) * 16], 'global', spec.num_classes)
    picks = {}
    for entry in a.picks:
        name, path = entry.split('=', 1)
        picks[name] = np.load(path).astype(np.int32)
    time_fn = actual_time if a.time == 'actual' else (lambda m: 0.)
    result = evaluate(forward, target, picks, set(air_compact), spec.num_classes, a.seed, time_fn, device)
    a.output.mkdir(parents=True, exist_ok=True)
    atomic_json(a.output / 'state_probe_summary.json', dict(config=dict(
        base=str(a.base), control=str(a.control), case=str(a.case), target_cell=[z, x], seed=a.seed, fractions=list(FRACTIONS),
        air_ids=sorted(air_compact), picks=a.picks, time=a.time, model=identity), results=result))
    print(json.dumps({k: v['0.5']['all'] for k, v in result.items()}))


if __name__ == '__main__':
    main()
