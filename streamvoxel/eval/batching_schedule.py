"""E4: batching schedules, stale dependency edges and seam error.

python -m streamvoxel.eval.batching_schedule --case <case dir> --mapping compact_mapping.json --vocabulary vocabulary.json \\
    --base base.pt --control control.pt --output <new dir> [--arms serial naive_fixed safe_fixed safe_continuous] [--batch 4] [--windows 32]
python -m streamvoxel.eval.batching_schedule --schedule-only --batch 4 --windows 32 --steps 16     (CPU, no model: schedule statistics only)

Arms (one window needs `--steps` forwards; every forward serves all active windows):
  serial           one window at a time (dynamic, batch 1), dependency-safe admission
  naive_fixed      fixed batches (a new batch starts only when every window has finished), nearest ready cells with no separation rule
  safe_fixed       fixed batches with the dependency-safe admission of `stream.scheduler.pick_window`
  safe_continuous  dynamic slots (a freed slot is refilled at once) with dependency-safe admission: the streaming backend's policy
`simulate` is a pure schedule model: per forward the number of active windows and the cell pairs that touch (`metrics.stale_edges`; a touching
pair is generated from conditions that do not contain each other). With a model, `rollout` replays the schedule: each window is conditioned on
the cells committed when it started, and its seam error is measured against the neighbours committed when it finished.
Windows are sampled one after another (the model sees the same condition as in a batched forward); this evaluation does not time batched forwards.
Writes `schedule.json` (per arm), `blocks.npy` per arm, `scores.json`. One case per call; the model part needs a GPU.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
import torch

from streamvoxel.common.colored_frontier import ready_cells
from streamvoxel.common.io import atomic_json
from streamvoxel.eval.metrics import array_metrics, stale_edges, voxel_scores
from streamvoxel.eval.samplers import decode
from streamvoxel.eval.window import initial_canvas, load_case, open_eval_model, window_forward, window_item
from streamvoxel.stream.scheduler import pick_window, plan_inserts

ARMS = dict(serial=dict(mode='dynamic', safe=True), naive_fixed=dict(mode='fixed', safe=False),
            safe_fixed=dict(mode='fixed', safe=True), safe_continuous=dict(mode='dynamic', safe=True))
INITIAL_BOX = (7, 7, 9, 9)
CELL = 16


def nearest_ready(committed, inflight, player_zx):
    """Nearest ready cell not in flight, with no separation from in-flight cells (the naive batch)."""
    blocked = set(inflight)
    pz, px = player_zx
    keys = [(((z + .5) * CELL - pz) ** 2 + ((x + .5) * CELL - px) ** 2, z, x) for z, x in ready_cells(committed) if (z, x) not in blocked]
    return min(keys)[1:] if keys else None


def simulate(initial, arm, batch, steps, windows, player_zx=(128., 128.), extent=16, diagonal=False):
    """Run one arm to `windows` committed windows. `initial` is the set of committed cells. Returns `(schedule, sizes, stale)`:
    `schedule` lists the windows in commit order (cell, slot, start/end forward, committed cells at start), `sizes[f]` the active windows
    in forward f and `stale[f]` the touching pairs among them."""
    cfg = ARMS[arm]
    slots = 1 if arm == 'serial' else batch
    committed, active, schedule, sizes, stale = set(initial), {}, [], [], []
    started = f = 0
    while len(schedule) < windows:
        idle = [i for i in range(slots) if i not in active]
        for slot in plan_inserts(idle, len(active), slots, cfg['mode']):
            inflight = [w['cell'] for w in active.values()]
            cell = pick_window(committed, inflight, player_zx, cell_size=CELL, diagonal=diagonal) if cfg['safe'] else nearest_ready(committed, inflight, player_zx)
            if started >= windows or cell is None:
                break
            if not (0 <= cell[0] < extent and 0 <= cell[1] < extent):
                raise ValueError(f'Window {cell} leaves the {extent}x{extent} cell world')
            active[slot] = dict(cell=cell, slot=slot, start=f, left=steps, committed_at_start=len(committed))
            started += 1
        if not active:
            raise RuntimeError('Frontier blocked with nothing in flight')
        sizes.append(len(active))
        stale.append(stale_edges([w['cell'] for w in active.values()]))
        for slot in sorted(active):
            active[slot]['left'] -= 1
            if not active[slot]['left']:
                w = active.pop(slot)
                committed.add(w['cell'])
                schedule.append(dict(cell=w['cell'], slot=slot, start=w['start'], end=f, committed_at_start=w['committed_at_start']))
        f += 1
    return schedule, sizes, stale


def summarize(sizes, stale, windows, batch):
    """Batch-size histogram, empty-slot rate (idle slot-forwards / slot-forwards), windows per forward and stale edge counts over forwards."""
    forwards = len(sizes)
    return dict(windows=windows, forwards=forwards, mean_batch=sum(sizes) / forwards, batch_hist={str(k): v for k, v in sorted(Counter(sizes).items())},
                empty_slot_rate=1 - sum(sizes) / (forwards * batch), windows_per_forward=windows / forwards,
                stale_edges_total=sum(stale), forwards_with_stale=sum(1 for s in stale if s), stale_edges_max=max(stale))


def arm_batch(arm, batch):
    return 1 if arm == 'serial' else batch


def initial_cells(box=INITIAL_BOX):
    z0, x0, z1, x1 = box
    return {(z, x) for z in range(z0, z1) for x in range(x0, x1)}


def rollout(model, spec, gt, biomes, mapping, spec_dict, schedule, seed, l2_cap, air_global, device, log=None):
    """Replay a schedule: a window is decoded from the canvas of the cells committed at its start and written at its end.
    Returns the compact canvas and per-window records with array metrics (seam errors against the neighbours committed at the end)."""
    canvas, known = initial_canvas(gt, mapping, INITIAL_BOX)
    timeline = sorted([(w['start'], 1, i) for i, w in enumerate(schedule)] + [(w['end'] + 1, 0, i) for i, w in enumerate(schedule)])
    results, records = {}, {}
    for _, starting, i in timeline:
        z, x = schedule[i]['cell']
        if starting:
            window_seed = (seed + z * 1000 + x) % (2 ** 63 - 1)
            item = window_item(canvas, known, (z, x), biomes, spec, window_seed, spec_dict['steps'], l2_cap)
            began = time.perf_counter()
            forward = window_forward(model, item, spec, device)
            results[i], meta = decode(forward, spec_dict, window_seed, device, shape=(spec.height, spec.tile, spec.tile), classes=spec.num_classes)
            records[i] = dict(cell=[z, x], seed=window_seed, nfe=meta['nfe'], l1_tiles=len(item.l1_blocks), seconds=time.perf_counter() - began)
        else:
            canvas[:, z * 16:(z + 1) * 16, x * 16:(x + 1) * 16] = results.pop(i)['compact']
            known[z, x] = True
            blocks = mapping.export(canvas[:, z * 16:(z + 1) * 16, x * 16:(x + 1) * 16], 'global')
            context = np.where(canvas >= 0, mapping.to_global[np.maximum(canvas, 0)], -1)
            records[i]['metrics'] = array_metrics(gt[:, z * 16:(z + 1) * 16, x * 16:(x + 1) * 16], blocks, air_global, context=context, target=(z, x), gt_context=gt)
            if log:
                print(json.dumps(dict(cell=[z, x], acc=records[i]['metrics']['acc'])), flush=True)
    return canvas, [records[i] for i in sorted(records, key=lambda i: schedule[i]['end'])]


def seam_summary(records):
    """Mean over windows of the seam errors (windows with no committed neighbour at their end are skipped) and of accuracy."""
    metrics = [r['metrics'] for r in records]
    seam = [m for m in metrics if 'seam_occupancy_error' in m]
    return dict(acc=float(np.mean([m['acc'] for m in metrics])), seam_windows=len(seam),
                seam_occupancy_error=float(np.mean([m['seam_occupancy_error'] for m in seam])) if seam else None,
                seam_height_jump_mae=float(np.mean([m['seam_height_jump_mae'] for m in seam])) if seam else None)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ('case', 'mapping', 'vocabulary', 'base', 'control'):
        p.add_argument('--' + name, type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--schedule-only', action='store_true', help='CPU schedule statistics only; no case or model')
    p.add_argument('--arms', nargs='+', default=list(ARMS), choices=list(ARMS))
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--windows', type=int, default=32)
    p.add_argument('--steps', type=int, default=16)
    p.add_argument('--player', type=float, nargs=2, default=(128., 128.), metavar=('Z', 'X'), help='player position in voxels; admission is nearest-first')
    p.add_argument('--diagonal', action='store_true', help='safe admission excludes only edge neighbours')
    p.add_argument('--last-eps', type=float, default=1e-4)
    p.add_argument('--l2-cap', type=int, default=256)
    p.add_argument('--seed', type=int, default=2026102801)
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    if not a.schedule_only and not all((a.case, a.mapping, a.vocabulary, a.base, a.control)):
        p.error('--case --mapping --vocabulary --base --control are required without --schedule-only')
    a.output.mkdir(parents=True, exist_ok=False)
    schedules = {arm: simulate(initial_cells(), arm, a.batch, a.steps, a.windows, tuple(a.player), diagonal=a.diagonal) for arm in a.arms}
    scores = {arm: dict(schedule=summarize(sizes, stale, a.windows, arm_batch(arm, a.batch))) for arm, (_, sizes, stale) in schedules.items()}
    atomic_json(a.output / 'schedule.json', {arm: dict(windows=s, batch_sizes=z, stale_edges=e) for arm, (s, z, e) in schedules.items()})
    if not a.schedule_only:
        gt, biomes, mapping, air_global, air_compact, air_token = load_case(a.case, a.mapping, a.vocabulary)
        device = torch.device(a.device)
        model, spec, identity = open_eval_model(a.base, a.control, device)
        spec_dict = dict(selection='low_entropy_last', schedule='quartercos', steps=a.steps, block_size=4, last_eps=a.last_eps, air_ids=air_compact, air_token=air_token)
        for arm, (schedule, _, _) in schedules.items():
            canvas, records = rollout(model, spec, gt, biomes, mapping, spec_dict, schedule, a.seed, a.l2_cap, air_global, device, log=True)
            blocks = mapping.export(canvas, 'global')
            np.save(a.output / f'blocks_{arm}.npy', blocks)
            scores[arm].update(quality=seam_summary(records), voxel=voxel_scores(gt, blocks, [(1, *w['cell']) for w in schedule], air_ids=air_global),
                               records=records)
        scores['sampler'], scores['model'] = spec_dict, identity
    atomic_json(a.output / 'scores.json', scores)


if __name__ == '__main__':
    main()
