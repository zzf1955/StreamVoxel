#!/usr/bin/env bash
# E3b: per-height-band CE on the reveal states a sampler actually produced.
# Usage: scripts/eval_state_probe.sh <case> <mapping.json> <vocabulary.json> <base.pt> <control.pt> <out_dir> [extra args]
# Pass the sampler_grid outputs with --picks name=<out>/<id>/commit_step.npy (same window and --target-cell as the grid run).
set -euo pipefail
[ $# -ge 6 ] || { echo "usage: eval_state_probe.sh <case> <mapping> <vocabulary> <base> <control> <out_dir> [args]" >&2; exit 1; }
case=$1 mapping=$2 vocabulary=$3 base=$4 control=$5 out=$6
shift 6
export PYTHONPATH=.
python -m streamvoxel.eval.state_probe --case "$case" --mapping "$mapping" --vocabulary "$vocabulary" \
    --base "$base" --control "$control" --output "$out" "$@"
