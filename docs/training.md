# Training

Training has three stages. Stages 1 and 2 train the unconditional MD4 DiT with `python -m streamvoxel.train.train_base`; stage 3 trains the neighbour-conditioned ControlNet with `python -m streamvoxel.train.train_controlnet`. Both entries accept `--config <yaml>`: the yaml only sets argparse defaults, command-line flags win, unknown keys are an error, and required flags supplied by the yaml count as satisfied.

Run the wrappers from the repository root:

```bash
scripts/train_base.sh <config.yaml> <run_dir> <num_gpus> [extra flags]
scripts/train_controlnet.sh <config.yaml> <run_dir> <num_gpus> [extra flags]
```

Both use `torch.distributed.run --standalone`, refuse an existing `<run_dir>`, and pass extra flags through. To stop a run cleanly, create `<run_dir>/stop_request.json` (checked every step; a checkpoint is written first) or send SIGTERM (`train_base` also handles SIGINT, `train_controlnet` also SIGUSR1).

## Stage 1: time-conditioned base

`configs/base_stage1.yaml`: 25-layer DiT, hidden 768, 8 heads, patch size 4, time-conditioned (`time_conditioned: true`), i.i.d. random MD4 masks, global batch 512, 46000 steps in the original run.

## Stage 2: half-mask fine-tune, time-free

`configs/base_stage2_stratified.yaml`. Starts from the stage-1 weights (`init_weights`), writes the time embedding at t = 0.5 into a constant (`freeze_time_at: 0.5`) and freezes the time MLP, so the model no longer takes a timestep (`timefree_arch: true`).

Half-mask (`half_mask: true`): half of the data-parallel ranks draw i.i.d. random MD4 masks, the other half draw height-strata masks. With one GPU the single rank alternates the role every optimizer step; otherwise an even number of ranks is required (3 GPUs is not supported) and `global_batch` must be divisible by `microbatch * num_gpus`.

Strata masks (`streamvoxel/objectives/md4.py`, `strata_mask`): the 256 layers are split into bands of `strata_band` layers (default 8). Each band draws its own noise level t in [0, 1]. A band that is entirely air in the target is masked with probability sin(pi t / 2); any other band is masked with probability t. Voxels are independent inside a band. This makes the contexts seen in training resemble the spatially structured contexts of sampling (whole height regions already revealed), instead of uniformly scattered ones. The loss weights follow the corresponding mask probability.

## Stage 3: ControlNet

`configs/controlnet.yaml`. A ControlNet branch (`control_depth` 13, `control_hidden_dim` 384) is trained on the frozen stage-2 base (`--base`). Conditions are the L1 neighbours (up to 8 tiles around the target), L2 far tiles and biomes (`biome_patch`, `biome_halo`). A training sample is a group of `side` x `side` known tiles with ring targets around them. `half_mask: true` applies the same random/strata split. Validation and best-checkpoint selection weight the random and strata buckets equally.

## Configuration keys

Stage 1 and 2 (flags of `train_base`):

| Key | Meaning |
| --- | --- |
| `data_dir` | Immutable tile-index directory (see data_format.md). `source_dir` overrides the source corpus location. |
| `tile_cache_dir`, `read_backend` | `raw_tiles`, `world`, `compact_cache` or `zstd_cache`; the cache backends need `tile_cache_dir`. |
| `batch_prefetch`, `loader_prefetch`, `workers` | Loader pipeline depth and worker processes. `batch_prefetch` must be 0 on a single GPU. |
| `depth`, `hidden_dim`, `heads`, `patch_size`, `mlp_ratio` | DiT size. `patch_size` is 2 or 4. |
| `global_batch`, `microbatch` | Accumulation is `global_batch / (microbatch * gpus)`. |
| `lr`, `ema_decay`, `ema_update_every`, `seed` | Optimizer and EMA. |
| `steps` | Total steps (0 in stage 2: run until stopped). `max_wall_hours` stops after a wall-clock budget. |
| `save_every`, `eval_every`, `first_validation_step`, `eval_batch_size`, `eval_worlds`, `eval_limit`, `eval_include_test`, `sample_steps` | Checkpoint and validation schedule. |
| `time_conditioned` / `timefree_arch` | Choose the architecture (stage 1 / stage 2). |
| `init_weights` | Load weights only (fresh optimizer and EMA). |
| `freeze_time_at` | Convert the time MLP at this t into a constant and freeze it. |
| `half_mask`, `strata_band` | Half-mask recipe and band height. |
| `resume`, `resume_reshard`, `override_resume_lr` | Resume from a checkpoint (also onto a different GPU count) and replace the stored learning rate. |

Stage 3 (flags of `train_controlnet`):

| Key | Meaning |
| --- | --- |
| `dataset`, `base` | ControlNet dataset directory and the stage-2 base checkpoint (both required). |
| `side` | Side length of the known region of a training group (8 in the config). |
| `global_batch`, `microbatch`, `cell_chunk`, `prefetch_depth` | Batching and loader depth. |
| `lr`, `warmup_steps`, `lr_schedule` (`constant` or `cosine`), `cosine_start_step`, `cosine_steps`, `min_lr` | Learning-rate schedule. |
| `steps` | 0 trains until `stop_request.json`. |
| `eval_every`, `save_every`, `sample_steps`, `eval_ordinary_contexts`, `eval_full_contexts` | Validation and checkpoint schedule. |
| `attention_backend` | `concat` or `shared`. |
| `compile_blocks`, `compile_loss`, `persistent_autocast`, `async_reduce`, `target_checkpoint` | Throughput options. |
| `biome_patch`, `biome_halo` | Biome conditioning geometry. |
| `control_depth`, `control_hidden_dim` | ControlNet size. |
| `half_mask` | Half random / half strata masks. |
| `resume`, `allow_world_size_change`, `allow_lr_change`, `allow_schedule_change`, `allow_runtime_change` | Resume and the explicit permissions to change settings when resuming. |
| `skip_dataset_validation`, `serial_model_load` | Skip full-file re-verification at start; load the model rank by rank to bound CPU memory. |

Run-time learning-rate control (`train_base` only): `lr_control.json` in the run directory is read by rank 0 and applied each step (`streamvoxel/train/lr_control.py`).

## Reproducing the original schedule of stage 2

The original stage 2 ran in segments: 1 GPU for initialisation and freezing (lr 1e-4), then 4 GPUs at lr 1e-4 up to step 10532, then a resume at lr 1e-5 up to step 11000 (the selected checkpoint). The last segment is:

```bash
scripts/train_base.sh configs/base_stage2_stratified.yaml <new_run_dir> 4 \
    --resume <previous_run_dir>/<checkpoint>.pt --resume-reshard --override-resume-lr --lr 1e-5
```

## Known gaps

- Stage 1 hyperparameters are only partly recorded. `base_stage1.yaml` uses the values of the first segment (lr 1e-4, EMA 0.99); the later segments (steps 38000 to 46000) used lr 1e-5, EMA 0.9 and microbatch 32. The per-segment lr/EMA of steps 0 to 38000 and the resume commands between segments are not recorded.
- The run config of the exported ControlNet checkpoint is missing (only the checkpoint remains). `controlnet.yaml` is taken from the run it was resumed from; the final `--steps` and `--save-every` values are unknown (the original runs used `steps: 0` and were stopped via `stop_request.json`).
- `configs/controlnet.yaml` lists `sample_steps: 1024` and `microbatch: 8` as recorded; the effective accumulation depends on your GPU count.
- Dataset paths in all configs are placeholders.
