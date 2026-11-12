#!/usr/bin/env bash
# E2: stream 32x32 out to 256x256. Usage: scripts/eval_stream_rollout.sh <case> <mapping.json> <vocabulary.json> <base.pt> <control.pt> <out_dir> [extra args]
# Run from the repository root on a GPU machine. Top-down LPIPS needs --client-jar <minecraft client jar> or STREAMVOXEL_MC_JAR.
set -euo pipefail
[ $# -ge 6 ] || { echo "usage: eval_stream_rollout.sh <case> <mapping> <vocabulary> <base> <control> <out_dir> [args]" >&2; exit 1; }
case=$1 mapping=$2 vocabulary=$3 base=$4 control=$5 out=$6
shift 6
export PYTHONPATH=.
python -m streamvoxel.eval.stream_rollout --case "$case" --mapping "$mapping" --vocabulary "$vocabulary" \
    --base "$base" --control "$control" --output "$out" "$@"
