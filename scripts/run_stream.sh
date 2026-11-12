#!/usr/bin/env bash
# prepare -> gate -> run for the streaming backend. Usage: scripts/run_stream.sh <config.json> <run_dir>
# Run from the repository root on the GPU machine. <run_dir> must not exist yet (prepare creates its panel/).
set -euo pipefail
config=${1:?usage: run_stream.sh <config.json> <run_dir>}
run=${2:?usage: run_stream.sh <config.json> <run_dir>}
[ -e "$run/panel" ] && { echo "$run/panel exists; use a new run_dir" >&2; exit 1; }
mkdir -p "$run"
cp "$config" "$run/config.json"
export PYTHONPATH=.
python -m streamvoxel.stream.session prepare --run "$run"
python -m streamvoxel.stream.session gate --run "$run"
python -m streamvoxel.stream.session fresh --run "$run"
python -m streamvoxel.stream.session run --run "$run" 2>&1 | tee -a "$run/session.log"
