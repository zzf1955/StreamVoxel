# Benchmark

`scripts/bench_ops.py` measures and checks the streaming fast path (`streamvoxel/stream/fast_ops.py`) against the reference slot forward. It needs a GPU and a prepared run directory.

```bash
CUDA_VISIBLE_DEVICES=<gpu> PYTHONPATH=. python scripts/bench_ops.py <mode> --run <run_dir> [--out <json>]
```

Prepare the run directory first (it must hold `config.json` and `panel/`):

```bash
mkdir -p <run_dir> && cp configs/stream_2x4090.json <run_dir>/config.json   # after filling the <...> paths
PYTHONPATH=. python -m streamvoxel.stream.session prepare --run <run_dir>
```

## Options

`--run` (required), `--base`/`--control` (default: the checkpoints in the run's `config.json`), `--world` (default 161), `--box Z0 X0 Z1 X1` (default 64 64 192 192; the initial known box the conditions are cut from), `--out <json>`, `--configs a,b,...`, `--config <name>`, `--cap` (64), `--windows` (8), `--rounds` (10), `--verbose`.

Fast-path configurations (`--config`/`--configs`): `kv_sdpa`, `kv_flex`, `kv_sdpa_gather`, `kv_flex_gather`, `kv_flex_gather_compile`, `kv_flex_gather_compile_emulate`. The last one is the shipped setting (packed K/V, FlexAttention, gather patchify, compiled blocks with emulated BF16 casts).

## Modes

| Mode | What it measures |
| --- | --- |
| `profile` | `SlotForward` (or `FastSlotForward` with `--config`) graph time per batch bucket 1/2/4/8, per-module CUDA-event split and top kernels at buckets 4 and 8. |
| `speed` | Interleaved A/B graph timing of `SlotForward` against the `--configs` variants (min/median over `--rounds`). This is the single-forward comparison in the README. |
| `forward` | Numerics and timing of every fast configuration: gather vs one-hot patchify, logits against the unpadded exact reference (argmax agreement, `|dlogits|`), bucket 4 vs bucket 8, local K/V cache bitwise check. |
| `sample` | End-to-end sampling of `--windows` windows with the reference and fast slots (plain, batched, CUDA-graph select, Triton entropy): wall time per forward and per consume, token agreement against the reference, repeat and other-seed controls, accuracy against the ground truth. |
| `gate` | The worker's numerical gate (padded CUDA graph vs unpadded forward) on the run's initial frontier. |
| `fp8` | Measurement only: FP8 (E4M3) backbone GEMMs against the BF16 fast path; `--configs mlp` or `all` picks the scope. |

Conditions are real: ground-truth cells of one panel world (read only) and targets with 2 to 8 L1 tiles.

## Reproducing the overview numbers

The numbers in the README (single forward about 281 -> 44 ms; time per window about 14.3 -> 2.11 s) are internal measurements on 2 x RTX 4090 D, headless.

Single forward (reference `SlotForward` against the shipped fast path):

```bash
CUDA_VISIBLE_DEVICES=<gpu> PYTHONPATH=. python scripts/bench_ops.py speed --run <run_dir> \
    --configs kv_flex_gather_compile_emulate --rounds 10 --out speed.json
```

End-to-end sampling and agreement with the reference:

```bash
CUDA_VISIBLE_DEVICES=<gpu> PYTHONPATH=. python scripts/bench_ops.py sample --run <run_dir> \
    --cap 64 --windows 8 --config kv_flex_gather_compile_emulate --out sample.json
```

Live throughput on two GPUs (time per window): run the session with `scripts/run_stream.sh configs/stream_2x4090.json <run_dir>` and read `<run_dir>/timing_r0.json`, `<run_dir>/timing_r1.json` (per-rank forward and GPU time) and `<run_dir>/results/window_times.jsonl`. Set `headless_windows` and `defaults.paused: false` in the config to measure without a viewer.

Results depend on the GPU model, driver, PyTorch version and on other jobs sharing the GPU; check `nvidia-smi` before timing.
