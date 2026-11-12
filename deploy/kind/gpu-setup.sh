#!/usr/bin/env bash
# Prepare the GPU kind node (run after kind create with the gpu node image, see gpu-node.Dockerfile).
# Needs host Docker with the nvidia runtime, and nvidia-ctk / nvidia-cdi-hook on the host PATH.
# Required: SV_GPU_UUIDS = comma-separated UUIDs of the allowed GPUs (each must be among the NVIDIA_VISIBLE_DEVICES the node
#           image was built with; building the node image for more cards than are used lets a later
#           `docker restart <node>` + this script switch cards without recreating the cluster). SV_GPU_UUID (one UUID) still works.
# Optional: SV_NODE (default sv-dev-control-plane).
# The device plugin then reports one nvidia.com/gpu per allowed card, and each Pod that asks for one GPU gets exactly one
# of them (through CDI; the other cards' device nodes are not in the Pod's device allow-list).
# Idempotent. After `docker restart <node>` the node /dev is recreated (all /dev/nvidiaN come back), so re-run this script;
# the CDI spec, svgpu runtime handler and binaries in the node's writable layer survive, the /dev cleanup does not.
# On a running cluster, delete the device plugin Pod afterwards (it may have started before this script and still list
# the old cards): kubectl -n kube-system delete pod -l name=nvidia-device-plugin-ds. The containerd restart at the end
# can exceed systemd's timeout on a loaded host; it completes anyway (check `systemctl is-active containerd` in the node).
set -euo pipefail
UUIDS="${SV_GPU_UUIDS:-${SV_GPU_UUID:-}}"
[ -n "$UUIDS" ] || { echo "set SV_GPU_UUIDS to the allowed GPU UUIDs (comma-separated)" >&2; exit 1; }
NODE="${SV_NODE:-sv-dev-control-plane}"
HERE="$(cd "$(dirname "$0")" && pwd)"

# /dev/nvidiaN: N is the driver minor number, not the index; look it up from each UUID.
MINORS=""
for uuid in ${UUIDS//,/ }; do
  minor="$(docker exec "$NODE" nvidia-smi -q -i "$uuid" | awk -F: '/Minor Number/{gsub(/ /,"",$2); print $2; exit}')"
  case "$minor" in ''|*[!0-9]*) echo "cannot determine minor number of $uuid (got '$minor')" >&2; exit 1;; esac
  echo "allowed GPU $uuid -> /dev/nvidia$minor"
  MINORS="$MINORS $minor"
done

# kind nodes are privileged and see every /dev/nvidia<N>; drop all but the allowed cards (node tmpfs only).
docker exec "$NODE" sh -c 'keep=" '"$MINORS"' "; for d in /dev/nvidia[0-9]*; do n="${d#/dev/nvidia}"; case "$keep" in *" $n "*) ;; *) rm -f "$d";; esac; done'
# Privileged node: NVML sees all cards until their nodes are removed; now exactly the allowed UUIDs must remain.
VISIBLE="$(docker exec "$NODE" nvidia-smi --query-gpu=uuid --format=csv,noheader | tr -d ' \r' | sort | paste -sd, -)"
WANTED="$(echo "${UUIDS//,/ }" | tr ' ' '\n' | sed '/^$/d' | sort | paste -sd, -)"
[ "$VISIBLE" = "$WANTED" ] || { echo "node nvidia-smi sees '$VISIBLE', expected exactly '$WANTED'" >&2; exit 1; }
docker cp "$(command -v nvidia-ctk)" "$NODE":/usr/local/bin/nvidia-ctk
docker cp "$(command -v nvidia-cdi-hook)" "$NODE":/usr/bin/nvidia-cdi-hook
docker exec "$NODE" sh -c 'chmod +x /usr/bin/nvidia-cdi-hook; mkdir -p /etc/cdi; nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml'
docker cp "$HERE/gpu-runtime.py" "$NODE":/usr/local/bin/gpu-runtime.py
# shellcheck disable=SC2086
docker exec "$NODE" python3 /usr/local/bin/gpu-runtime.py $MINORS
docker exec "$NODE" systemctl restart containerd || echo "containerd restart reported a failure; check it in the node" >&2
echo "done (on a running cluster: restart the device plugin Pod)"
