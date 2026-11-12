#!/usr/bin/env bash
# Create the GPU kind cluster from scratch and load the images it needs.
# Required env: SV_GPU_UUIDS (comma-separated; SV_GPU_UUID for one card), SV_DATA_DIR (read-only data), SV_OUT_DIR
#               (writable output), SV_KUBECONFIG, SV_KIND_IMAGE (node image built from gpu-node.Dockerfile for those GPUs).
# Optional: SV_LOAD_IMAGES (space-separated extra local images to load, e.g. the streaming image).
set -euo pipefail
: "${SV_DATA_DIR:?}" "${SV_OUT_DIR:?}" "${SV_KUBECONFIG:?}" "${SV_KIND_IMAGE:?}"
export SV_GPU_UUIDS="${SV_GPU_UUIDS:-${SV_GPU_UUID:?set SV_GPU_UUIDS}}"
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$SV_OUT_DIR"
RENDERED="$(mktemp)"; trap 'rm -f "$RENDERED"' EXIT
envsubst '${SV_DATA_DIR} ${SV_OUT_DIR}' < "$HERE/kind-config.gpu.yaml" > "$RENDERED"
SV_KIND_CONFIG="$RENDERED" "$HERE/up.sh"
"$HERE/gpu-setup.sh"
for img in nvcr.io/nvidia/k8s-device-plugin:v0.17.0 nvidia/cuda:12.4.1-base-ubuntu22.04 busybox:latest ${SV_LOAD_IMAGES:-}; do
  "$HERE/load-image.sh" "$img"
done
K="kubectl --kubeconfig $SV_KUBECONFIG"
$K apply -f "$HERE/../k8s/runtimeclass-svgpu.yaml" -f "$HERE/../k8s/nvidia-device-plugin.yaml"
