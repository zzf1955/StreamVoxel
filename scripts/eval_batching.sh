#!/usr/bin/env bash
# E4: batching schedules, stale dependency edges, seam error. Usage: scripts/eval_batching.sh <case> <mapping.json> <vocabulary.json> <base.pt> <control.pt> <out_dir> [extra args]
# Run from the repository root on a GPU machine. <out_dir> must not exist yet. Extra args go to streamvoxel.eval.batching_schedule (e.g. --batch 4 --windows 32 --arms serial safe_continuous).
# Schedule statistics only, no GPU: PYTHONPATH=. python -m streamvoxel.eval.batching_schedule --schedule-only --output <out_dir> --batch 4 --windows 32
set -euo pipefail
[ $# -ge 6 ] || { echo "usage: eval_batching.sh <case> <mapping> <vocabulary> <base> <control> <out_dir> [args]" >&2; exit 1; }
case=$1 mapping=$2 vocabulary=$3 base=$4 control=$5 out=$6
shift 6
export PYTHONPATH=.
python -m streamvoxel.eval.batching_schedule --case "$case" --mapping "$mapping" --vocabulary "$vocabulary" \
    --base "$base" --control "$control" --output "$out" "$@"
