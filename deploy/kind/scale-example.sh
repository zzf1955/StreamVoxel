#!/usr/bin/env bash
# EXAMPLE ONLY (not an autoscaler): one scaling decision for one world from the coordinator's metrics.json.
#   wanted = ceil(sv_requested_batch / SV_CAPACITY), clamped to [SV_MIN_WORKERS, SV_MAX_WORKERS]
# Prints the decision; with --apply it also runs `kubectl scale` on the worker Deployment. A real controller would add
# hysteresis (only act after the condition held for a while), skip scaling up while the frontier is the bottleneck,
# and pick which worker to drain first (see docs/streaming.md, "Kubernetes").
# Env: SV_KUBECONFIG, SV_WORLD (default w1), SV_CAPACITY (default 8), SV_MIN_WORKERS (1), SV_MAX_WORKERS (2).
set -euo pipefail
: "${SV_KUBECONFIG:?}"
K="kubectl --kubeconfig $SV_KUBECONFIG"
WORLD="${SV_WORLD:-w1}"
POD="$($K get pod -l app=sv-coordinator,world="$WORLD" -o jsonpath='{.items[0].metadata.name}')"
METRICS="$($K exec "$POD" -- sh -c 'cat "$SV_RUN/metrics.json"')"
CURRENT="$($K get deployment "sv-worker-$WORLD" -o jsonpath='{.spec.replicas}')"
WANTED="$(printf '%s' "$METRICS" | python3 -c '
import json, math, sys
m = json.load(sys.stdin)
cap, lo, hi = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
print(min(hi, max(lo, math.ceil(m["sv_requested_batch"] / cap))))
' "${SV_CAPACITY:-8}" "${SV_MIN_WORKERS:-1}" "${SV_MAX_WORKERS:-2}")"
printf '%s' "$METRICS" | python3 -c '
import json, sys
m = json.load(sys.stdin)
print("requested_batch", m["sv_requested_batch"], "slots_total", m["sv_slots_total"], "ready", m["sv_workers_ready"])'
echo "replicas: current $CURRENT, wanted $WANTED"
if [ "${1:-}" = "--apply" ] && [ "$WANTED" != "$CURRENT" ]; then
  $K scale deployment "sv-worker-$WORLD" --replicas="$WANTED"
fi
