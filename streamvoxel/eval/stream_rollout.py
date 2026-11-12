"""E2: stream 32x32 out to 256x256 in clockwise rings, one window at a time, and score against GT.

python -m streamvoxel.eval.stream_rollout --case <case dir> --mapping compact_mapping.json --vocabulary vocabulary.json \\
    --base base.pt --control control.pt --output <new dir> [--spec spec.json] [--client-jar <minecraft client jar>]

Each of the 252 generated cells is conditioned on its committed 3x3 neighbours (L1) and the nearest `--l2-cap` committed cells (L2) plus
biomes; GT outside the central 32x32 is never read by the model. The default sampler is the shipped one (low_entropy_last, 16 forwards).
Scores: accuracy, non-air accuracy and macro-F1 pooled and per ring (global 981 ids, so out-of-vocabulary GT counts as an error), and, with
a client jar (or STREAMVOXEL_MC_JAR) and the `lpips` package, top-down LPIPS over the generated area. One case per call; needs a GPU.
Writes `blocks.npy`, `records.json`, `scores.json`, `gt_topdown.png`, `generated_topdown.png`.
"""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from streamvoxel.common.io import atomic_json
from streamvoxel.common.sequential_rollout import clockwise_ring
from streamvoxel.eval.metrics import voxel_scores
from streamvoxel.eval.samplers import decode
from streamvoxel.eval.window import initial_canvas, load_case, open_eval_model, window_forward, window_item


def target_order(rings=7, center=8):
    return [(ring, z, x) for ring in range(1, rings + 1) for z, x in clockwise_ring(center - 1 - ring, center + 1 + ring)]


def rollout(model, spec, gt, biomes, mapping, spec_dict, seed, l2_cap, device, log=None):
    """Returns the final compact canvas and the per-window records."""
    canvas, known = initial_canvas(gt, mapping, (7, 7, 9, 9))
    records = []
    for i, (ring, z, x) in enumerate(target_order()):
        began = time.perf_counter()
        window_seed = (seed + i) % (2 ** 63 - 1)
        item = window_item(canvas, known, (z, x), biomes, spec, window_seed, spec_dict['steps'], l2_cap)
        forward = window_forward(model, item, spec, device)
        result, meta = decode(forward, spec_dict, window_seed, device, shape=(spec.height, spec.tile, spec.tile), classes=spec.num_classes)
        canvas[:, z * 16:(z + 1) * 16, x * 16:(x + 1) * 16] = result['compact']
        known[z, x] = True
        records.append(dict(iteration=i, ring=ring, z=z, x=x, seed=window_seed, nfe=meta['nfe'], l1_tiles=len(item.l1_blocks),
                            l2_tiles=len(item.l2_blocks), seconds=time.perf_counter() - began))
        if log:
            print(json.dumps(records[-1]), flush=True)
    return canvas, records


def topdown_scores(gt, blocks, vocabulary, jar, device):
    from PIL import Image
    from streamvoxel.render.topdown import lpips_net, masked_lpips, palette, surface_rgb
    colors, air = palette(vocabulary, jar)
    identity = np.arange(len(colors), dtype=np.uint16)
    g_rgb, g_h, g_top = surface_rgb(gt, identity, colors, air)
    q_rgb, q_h, q_top = surface_rgb(blocks, identity, colors, air)
    z, x = np.indices(gt.shape[1:])
    rings = np.maximum.reduce((7 - z // 16, z // 16 - 8, 7 - x // 16, x // 16 - 8))
    net = lpips_net(device)
    scores = {}
    for ring in range(8):
        selected = rings > 0 if ring == 0 else rings == ring
        scores['all' if ring == 0 else f'ring{ring}'] = dict(
            topdown_lpips_generated=masked_lpips(net, g_rgb, q_rgb, selected, device),
            height_mae=float(np.abs(q_h.astype(float) - g_h)[selected].mean()),
            surface_block_acc=float((q_top == g_top)[selected].mean()))
    return scores, Image.fromarray(g_rgb), Image.fromarray(q_rgb)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ('case', 'mapping', 'vocabulary', 'base', 'control', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--spec', type=Path, help='sampler spec json; default low_entropy_last, 16 forwards, block 4')
    p.add_argument('--steps', type=int, default=16)
    p.add_argument('--last-eps', type=float, default=1e-4)
    p.add_argument('--l2-cap', type=int, default=256)
    p.add_argument('--seed', type=int, default=2026101603)
    p.add_argument('--client-jar', type=Path)
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    gt, biomes, mapping, air_global, air_compact, air_token = load_case(a.case, a.mapping, a.vocabulary)
    if gt.shape != (256, 256, 256):
        raise ValueError(f'E2 expects a 256x256x256 case, got {gt.shape}')
    device = torch.device(a.device)
    model, spec, identity = open_eval_model(a.base, a.control, device)
    spec_dict = json.loads(a.spec.read_text()) if a.spec else dict(
        selection='low_entropy_last', schedule='quartercos', steps=a.steps, block_size=4, last_eps=a.last_eps, air_ids=air_compact, air_token=air_token)
    a.output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    canvas, records = rollout(model, spec, gt, biomes, mapping, spec_dict, a.seed, a.l2_cap, device, log=True)
    blocks = mapping.export(canvas, 'global')
    if not np.array_equal(blocks[:, 112:144, 112:144], gt[:, 112:144, 112:144]):
        raise ValueError('Initial condition changed')
    np.save(a.output / 'blocks.npy', blocks)
    atomic_json(a.output / 'records.json', records)
    scores = dict(voxel=voxel_scores(gt, blocks, [(r, z, x) for r, z, x in target_order()], air_ids=air_global),
                  seconds_total=time.time() - started, sampler=spec_dict, model=identity)
    from streamvoxel.render.topdown import client_jar
    if a.client_jar or os.environ.get('STREAMVOXEL_MC_JAR'):
        scores['topdown'], g_img, q_img = topdown_scores(gt, blocks, a.vocabulary, client_jar(a.client_jar), device)
        g_img.save(a.output / 'gt_topdown.png')
        q_img.save(a.output / 'generated_topdown.png')
    atomic_json(a.output / 'scores.json', scores)


if __name__ == '__main__':
    main()
