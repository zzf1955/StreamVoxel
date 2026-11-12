#!/usr/bin/env bash
# E3a/E3c/E3d: sampler grid on one window. Usage: scripts/eval_sampler_grid.sh <case> <mapping.json> <vocabulary.json> <base.pt> <control.pt> <out_dir> [extra args]
# Run from the repository root on a GPU machine. <out_dir> must not exist yet. Extra args go to streamvoxel.eval.sampler_grid (e.g. --steps 8 16 --block-sizes 1 4).
set -euo pipefail
[ $# -ge 6 ] || { echo "usage: eval_sampler_grid.sh <case> <mapping> <vocabulary> <base> <control> <out_dir> [args]" >&2; exit 1; }
case=$1 mapping=$2 vocabulary=$3 base=$4 control=$5 out=$6
shift 6
export PYTHONPATH=.
python -m streamvoxel.eval.sampler_grid --case "$case" --mapping "$mapping" --vocabulary "$vocabulary" \
    --base "$base" --control "$control" --output "$out" "$@"
