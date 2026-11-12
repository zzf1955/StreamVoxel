#!/usr/bin/env bash
# Single-node multi-GPU base training (stage 1 or 2, chosen by the config).
# usage (from the repo root): scripts/train_base.sh <config.yaml> <run_dir> <num_gpus> [extra train_base flags...]
# Fill the <...> placeholders in the config first, or override them: ... --data-dir X --tile-cache-dir Y
set -euo pipefail
(( $# >= 3 )) || { echo "usage: $0 <config.yaml> <run_dir> <num_gpus> [flags...]" >&2; exit 2; }
CONFIG="$1"; RUN="$2"; GPUS="$3"; shift 3
[[ -f "$CONFIG" ]] || { echo "missing config: $CONFIG" >&2; exit 1; }
[[ ! -e "$RUN" ]] || { echo "refusing to reuse existing run dir: $RUN" >&2; exit 1; }
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec python -u -m torch.distributed.run --standalone --nproc_per_node="$GPUS" \
  -m streamvoxel.train.train_base --config "$CONFIG" --run-dir "$RUN" "$@"
