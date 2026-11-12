#!/usr/bin/env bash
set -euo pipefail
NAME="${SV_KIND_NAME:-sv-dev}"
KUBECONFIG_PATH="${SV_KUBECONFIG:-./kubeconfig}"
HERE="$(cd "$(dirname "$0")" && pwd)"
kind create cluster --name "$NAME" --image "${SV_KIND_IMAGE:-kindest/node:r27-pinned}" \
  --config "${SV_KIND_CONFIG:-$HERE/kind-config.yaml}" --kubeconfig "$KUBECONFIG_PATH" --wait 180s
kubectl --kubeconfig "$KUBECONFIG_PATH" get nodes -o wide
