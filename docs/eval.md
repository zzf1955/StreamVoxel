# Evaluation

All evaluation entries run on one GPU, take one case directory (`gt_global.npy`, `biomes.npy`; see data_format.md), the compact mapping, the vocabulary, and a base/control checkpoint pair. The wrappers in `scripts/` forward to the `python -m` entries:

```bash
scripts/eval_<name>.sh <case> <mapping.json> <vocabulary.json> <base.pt> <control.pt> <out_dir> [extra args]
```

`<out_dir>` must not exist. Run from the repository root. Shared code: `streamvoxel/eval/window.py` (case loading, conditioned forward of one window), `eval/samplers.py` (baselines and the product sampler), `eval/metrics.py`.

The product sampler (4^3 blocks plus low-entropy-last) is called through `streamvoxel.stream.slot.LowEntropyLastSlot`, the same slot code the streaming engine runs, so the evaluation measures the shipped sampler. Baselines in `eval/samplers.py`: `random`, `entropy` (global low-entropy), `mixed` (candidate multiplier), `block` (block-wise quota, `block_size`). Schedules: `linear`, `quartercos`, `halfcos`, `reversequarter`, `power<p>`, `front<p>`, `native_quarter` (1024-step random reference).

## Included entries

### E2: streaming outward growth (`streamvoxel.eval.stream_rollout`)

Starts from the central 32x32-voxel region of the case (the initial box 7 7 9 9 in 16-voxel cells) and generates the surrounding 252 cells in 7 clockwise rings, one window at a time, each conditioned on its committed 3x3 neighbours (L1), the nearest `--l2-cap` committed cells (L2) and biomes. Ground truth outside the initial region is never read by the model.

```bash
scripts/eval_stream_rollout.sh <case> <mapping> <vocabulary> <base> <control> <out_dir> --steps 16 [--client-jar <minecraft client jar>]
```

Arguments: `--case --mapping --vocabulary --base --control --output` (required), `--spec <json>` (sampler spec; default low_entropy_last, block 4), `--steps` (16), `--last-eps` (1e-4), `--l2-cap` (256), `--seed`, `--client-jar` (or env `STREAMVOXEL_MC_JAR`), `--device`.
Outputs: `blocks.npy`, `records.json` (252 entries: ring, cell, seed, nfe, tile counts, seconds), `scores.json` (accuracy, non-air accuracy and macro-F1, pooled and per ring, over global ids), and, with a client jar plus the `lpips` package, top-down LPIPS in `scores.json` and `gt_topdown.png`, `generated_topdown.png`. The initial region of the output is asserted to equal the ground truth.

### E3a / E3c / E3d: sampler comparison (`streamvoxel.eval.sampler_grid`)

One window, one model, many samplers; each sampler is scored against the ground truth and its reveal order is exported. E3a (main table) is one run per model pair, with the rows joined afterwards. E3c is the block-size by step-count sweep (`--steps`, `--block-sizes`). E3d is the reveal-order map (`commit_step.npy`).

```bash
scripts/eval_sampler_grid.sh <case> <mapping> <vocabulary> <base> <control> <out_dir> --steps 8 16 --block-sizes 1 4
```

Default grid per step count: `random`, global low-entropy, `block` for each `--block-sizes`, and `low_entropy_last` (block 4). Arguments: `--target-cell Z X` (default 6 8), `--initial-box Z0 X0 Z1 X1` (default 7 7 9 9), `--steps` (default 16), `--block-sizes` (default 4), `--schedule` (quartercos), `--last-eps`, `--l2-cap`, `--seed`, `--specs <json>` (replaces the default grid), `--native` (adds the 1024-step random reference), `--device`.
Outputs: per sampler a folder `<out_dir>/<id>/` (id = first 12 hex digits of the SHA256 of the sampler spec) with `blocks.npy` (global ids) and `commit_step.npy` (forward index that revealed each voxel, 0 = never; written when the sampler records it); at the top level `rows.json` (one row per sampler: spec, nfe, seconds, array metrics including seam errors, SHA256 of the blocks) and `identity.json` (case, window, seed, model identity, device).

### E3b: per-band cross-entropy on revealed states (`streamvoxel.eval.state_probe`)

Teacher-forced probe: for revealed fractions 0.1, 0.3, 0.5, 0.7, 0.9 and several mask families, the state holds the ground truth at revealed positions and MASK elsewhere; one forward; statistics over masked voxels (NLL, accuracy, mean entropy, AUROC of entropy for predicting errors, ECE), overall, for air and solid voxels, and for 4 groups of 64 layers. Mask families: `iid`, `strata`, `entropy_<name>` (from a recorded `commit_step.npy`), `above_surface_first`, `band_topdown`, `band_bottomup`, `air_last`, `air_last_soft` (the last two use ground truth and are upper bounds). Masks are seeded independently of the model, so every model sees identical states.

```bash
scripts/eval_state_probe.sh <case> <mapping> <vocabulary> <base> <control> <out_dir> --picks lel=<grid_out>/<id>/commit_step.npy
```

Arguments: `--target-cell`, `--initial-box` (must match the grid run), `--picks name=path ...`, `--l2-cap`, `--seed`, `--time {actual,zero}`, `--device`. Output: `state_probe_summary.json`. Run once per checkpoint pair and compare the summaries.

### E4: batching schedules, stale edges and seams (`streamvoxel.eval.batching_schedule`)

Compares how windows are admitted and batched while the world grows outward from the initial 2x2-cell box. Every window needs `--steps` forwards. Arms:

- `serial`: one window at a time (dynamic, batch 1).
- `naive_fixed`: fixed batches of `--batch` windows (the next batch starts only when all have finished), nearest ready cells, no separation rule.
- `safe_fixed`: fixed batches with the dependency-safe admission of `stream.scheduler.pick_window` (a candidate may not touch an in-flight window).
- `safe_continuous`: dynamic slots (a freed slot is refilled at once) with the same admission: the streaming backend's policy (`plan_inserts` mode `dynamic`).

Schedule model (`simulate`, pure CPU, uses the real `plan_inserts` and `pick_window`). Per arm it reports `windows`, `forwards`, `mean_batch`, `batch_hist` (active windows per forward), `empty_slot_rate` (idle slot-forwards over slot-forwards), `windows_per_forward`, and stale edges: pairs of in-flight windows that share an edge or a corner (`metrics.stale_edges`), as `stale_edges_total` (summed over forwards), `forwards_with_stale`, `stale_edges_max`. Admission is nearest-first to `--player` (voxels).

Quality (needs a GPU): the schedule is replayed with the product sampler. A window is conditioned on the cells committed when it started (windows in flight are absent from its condition) and is scored when it ends with `metrics.array_metrics` against the neighbours committed by then. `scores.json` holds per arm `quality` (mean accuracy, mean `seam_occupancy_error` and `seam_height_jump_mae` over windows with a committed neighbour), `voxel` (`voxel_scores`, one pooled group) and per-window `records`. Windows are sampled one after another with the batched condition semantics; batched forward time is not measured here (use `docs/benchmark.md` for throughput).

```bash
scripts/eval_batching.sh <case> <mapping> <vocabulary> <base> <control> <out_dir> --batch 4 --windows 32
PYTHONPATH=. python -m streamvoxel.eval.batching_schedule --schedule-only --output <new dir> --batch 4 --windows 32   # CPU
```

Arguments: `--arms` (default all four), `--batch` (4), `--windows` (32), `--steps` (16), `--player Z X` (128 128), `--diagonal` (safe arms exclude only edge neighbours), `--last-eps`, `--l2-cap`, `--seed`, `--schedule-only`, `--device`. Outputs: `schedule.json` (windows, batch sizes and stale counts per forward, per arm), `blocks_<arm>.npy` (GPU run), `scores.json`.

## Not included yet

- E1 (unconditional FID against external baselines): coming soon.

## Status of results

Quality numbers are not reported in this repository yet (TBD).

## Tests

`PYTHONPATH=. python -m pytest -q tests/eval tests/viewer` (CPU).
