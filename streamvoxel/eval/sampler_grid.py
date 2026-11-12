"""E3a/E3c/E3d: one window, one model, many samplers. Scores each against GT and exports the reveal order.

python -m streamvoxel.eval.sampler_grid --case <case dir> --mapping compact_mapping.json --vocabulary vocabulary.json \\
    --base base.pt --control control.pt --output <new dir> [--target-cell 6 8] [--initial-box 7 7 9 9] \\
    [--steps 4 8 16 32 64] [--block-sizes 1 2 4 8 16] [--schedule quartercos] [--native] [--specs specs.json]

The default grid per step count is: random, global low-entropy, block<b> for each --block-sizes, and low_entropy_last (block 4).
`--specs` replaces it with a json list of sampler specs (see streamvoxel.eval.samplers). `--native` adds the 1024-step random reference.
For E3a run it once per model (a model pair is two --base/--control invocations) and join the rows. Needs a GPU.
Writes `<output>/<id>/blocks.npy` (global ids), `commit_step.npy` (E3d: forward index that revealed each voxel, 0 = never) and `rows.json`.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from streamvoxel.common.io import atomic_json, digest
from streamvoxel.eval.metrics import array_metrics
from streamvoxel.eval.samplers import decode
from streamvoxel.eval.window import initial_canvas, load_case, open_eval_model, window_forward, window_item


def default_specs(steps, block_sizes, schedule, air_compact, air_token, last_eps, native):
    specs = []
    for n in steps:
        base = dict(steps=n, schedule=schedule)
        specs += [dict(base, selection='random'), dict(base, selection='entropy')]
        specs += [dict(base, selection='block', block_size=b) for b in block_sizes]
        specs.append(dict(base, selection='low_entropy_last', block_size=4, last_eps=last_eps, air_ids=air_compact, air_token=air_token))
    if native:
        specs.append(dict(steps=1024, schedule='native_quarter', selection='random'))
    return specs


def spec_id(spec):
    return hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(',', ':')).encode()).hexdigest()[:12]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ('case', 'mapping', 'vocabulary', 'base', 'control', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--target-cell', type=int, nargs=2, default=(6, 8))
    p.add_argument('--initial-box', type=int, nargs=4, default=(7, 7, 9, 9))
    p.add_argument('--steps', type=int, nargs='+', default=[16])
    p.add_argument('--block-sizes', type=int, nargs='+', default=[4])
    p.add_argument('--schedule', default='quartercos')
    p.add_argument('--last-eps', type=float, default=1e-4)
    p.add_argument('--l2-cap', type=int, default=256)
    p.add_argument('--seed', type=int, default=2026101701)
    p.add_argument('--specs', type=Path)
    p.add_argument('--native', action='store_true')
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    gt, biomes, mapping, air_global, air_compact, air_token = load_case(a.case, a.mapping, a.vocabulary)
    device = torch.device(a.device)
    model, spec, identity = open_eval_model(a.base, a.control, device)
    z, x = a.target_cell
    canvas, known = initial_canvas(gt, mapping, a.initial_box)
    item = window_item(canvas, known, (z, x), biomes, spec, a.seed, max(a.steps), a.l2_cap)
    forward = window_forward(model, item, spec, device)
    specs = json.loads(a.specs.read_text()) if a.specs else default_specs(a.steps, a.block_sizes, a.schedule, air_compact, air_token, a.last_eps, a.native)
    a.output.mkdir(parents=True, exist_ok=False)
    target = gt[:, z * 16:(z + 1) * 16, x * 16:(x + 1) * 16]
    context = np.where(canvas >= 0, mapping.to_global[np.maximum(canvas, 0)], -1)
    rows = []
    for sp in specs:
        result, meta = decode(forward, sp, a.seed, device, shape=(spec.height, spec.tile, spec.tile), classes=spec.num_classes, trace=True)
        blocks = mapping.export(result['compact'], 'global')
        folder = a.output / spec_id(sp)
        folder.mkdir()
        np.save(folder / 'blocks.npy', blocks)
        if 'commit_step' in result:
            np.save(folder / 'commit_step.npy', result['commit_step'])
        rows.append(dict(id=folder.name, spec=sp, nfe=meta['nfe'], complete=meta['complete'], seconds=meta['seconds'],
                         metrics=array_metrics(target, blocks, air_global, context=context, target=(z, x), gt_context=gt),
                         sha256=digest(folder / 'blocks.npy')))
        atomic_json(a.output / 'rows.json', rows)
        print(json.dumps(dict(id=folder.name, nfe=meta['nfe'], acc=rows[-1]['metrics']['acc'])), flush=True)
    atomic_json(a.output / 'identity.json', dict(case=str(a.case), target_cell=[z, x], initial_box=list(a.initial_box), seed=a.seed,
                                                  l2_cap=a.l2_cap, model=identity, device=torch.cuda.get_device_name() if device.type == 'cuda' else 'cpu'))


if __name__ == '__main__':
    main()
