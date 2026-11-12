#!/usr/bin/env bash
set -euo pipefail
KUBECONFIG_PATH="${SV_KUBECONFIG:-./kubeconfig}"
HERE="$(cd "$(dirname "$0")" && pwd)"
K="kubectl --kubeconfig $KUBECONFIG_PATH"
# Registries may be unreachable: import a locally present image into the node first.
if docker image inspect busybox:latest >/dev/null 2>&1; then
  kind load docker-image busybox:latest --name "${SV_KIND_NAME:-sv-dev}"
fi
trap '$K delete -f "$HERE/../k8s/smoke-cpu.yaml" --ignore-not-found' EXIT
$K apply -f "$HERE/../k8s/smoke-cpu.yaml"
$K wait --for=condition=Ready pod/sv-smoke-cpu --timeout=120s
$K logs sv-smoke-cpu
