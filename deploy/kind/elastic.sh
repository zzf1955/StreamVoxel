#!/usr/bin/env bash
# One elastic world on the cluster: coordinator Deployment + Service, worker Deployment, authkey Secret, config ConfigMap.
# Usage: deploy/kind/elastic.sh {secret|config|coordinator|workers|all|down}
#   secret      : Secret sv-authkey-<world>; a random 32-byte hex key unless SV_AUTHKEY_SOURCE names a key file (then replaced)
#   config      : ConfigMap sv-config-<world> from SV_CONFIG (a finished run config) or rendered from
#                 deploy/k8s/stream_elastic.template.json with SV_PANEL_DIR SV_MAPPING SV_VOCAB SV_BASE_CKPT SV_CONTROL_CKPT
#   coordinator : the coordinator (needs SV_RUN_NAME, a run folder name that does not exist yet under /mnt/sv-out/runs)
#   workers     : the worker Deployment with SV_WORKERS replicas (default 1); may be applied before the coordinator
#   all         : secret (if missing), config, coordinator, workers
#   down        : delete the world's Deployments, Service, ConfigMap and Secret (the run folder on the volume stays)
# Env: SV_KUBECONFIG, SV_WORLD (default w1), SV_IMAGE, SV_DATA_DIR, SV_UID/SV_GID (default: the caller's ids),
#      SV_CAPACITY (slots per GPU, default 8), SV_WORKER_NAME (default sv-worker-<world>), SV_AUTH_SECRET (default sv-authkey-<world>).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
K8S="$HERE/../k8s"
: "${SV_KUBECONFIG:?}"
K="kubectl --kubeconfig $SV_KUBECONFIG"
export SV_WORLD="${SV_WORLD:-w1}"
export SV_UID="${SV_UID:-$(id -u)}" SV_GID="${SV_GID:-$(id -g)}"
export SV_CAPACITY="${SV_CAPACITY:-8}" SV_WORKERS="${SV_WORKERS:-1}"
export SV_WORKER_NAME="${SV_WORKER_NAME:-sv-worker-$SV_WORLD}" SV_AUTH_SECRET="${SV_AUTH_SECRET:-sv-authkey-$SV_WORLD}"

secret() {
  if [ -n "${SV_AUTHKEY_SOURCE:-}" ]; then
    $K create secret generic "$SV_AUTH_SECRET" --from-file=authkey="$SV_AUTHKEY_SOURCE" --dry-run=client -o yaml | $K apply -f -
  elif ! $K get secret "$SV_AUTH_SECRET" >/dev/null 2>&1; then
    local key; key="$(mktemp)"
    head -c 32 /dev/urandom | od -An -tx1 | tr -d ' \n' > "$key"
    $K create secret generic "$SV_AUTH_SECRET" --from-file=authkey="$key" || { rm -f "$key"; return 1; }
    rm -f "$key"
  fi
  $K label secret "$SV_AUTH_SECRET" world="$SV_WORLD" --overwrite
}

config() {
  local src="${SV_CONFIG:-}" tmp=""
  if [ -z "$src" ]; then
    : "${SV_PANEL_DIR:?}" "${SV_MAPPING:?}" "${SV_VOCAB:?}" "${SV_BASE_CKPT:?}" "${SV_CONTROL_CKPT:?}"
    tmp="$(mktemp)"; src="$tmp"
    envsubst '${SV_PANEL_DIR} ${SV_MAPPING} ${SV_VOCAB} ${SV_BASE_CKPT} ${SV_CONTROL_CKPT}' < "$K8S/stream_elastic.template.json" > "$src"
  fi
  $K create configmap "sv-config-$SV_WORLD" --from-file=config.json="$src" --dry-run=client -o yaml | $K apply -f -
  [ -z "$tmp" ] || rm -f "$tmp"
  $K label configmap "sv-config-$SV_WORLD" world="$SV_WORLD" --overwrite
}

coordinator() {
  : "${SV_IMAGE:?}" "${SV_DATA_DIR:?}" "${SV_RUN_NAME:?}"
  envsubst '${SV_WORLD} ${SV_RUN_NAME} ${SV_IMAGE} ${SV_DATA_DIR} ${SV_UID} ${SV_GID}' < "$K8S/elastic-coordinator.yaml" | $K apply -f -
}

workers() {
  : "${SV_IMAGE:?}" "${SV_DATA_DIR:?}"
  envsubst '${SV_WORLD} ${SV_WORKER_NAME} ${SV_AUTH_SECRET} ${SV_IMAGE} ${SV_DATA_DIR} ${SV_UID} ${SV_GID} ${SV_CAPACITY} ${SV_WORKERS}' \
    < "$K8S/elastic-worker.yaml" | $K apply -f -
}

case "${1:-}" in
  secret) secret ;;
  config) config ;;
  coordinator) coordinator ;;
  workers) workers ;;
  all) secret; config; coordinator; workers ;;
  down)
    $K delete deployment -l world="$SV_WORLD" --ignore-not-found
    $K delete service,configmap,secret -l world="$SV_WORLD" --ignore-not-found ;;
  *) echo "usage: $0 {secret|config|coordinator|workers|all|down}" >&2; exit 2 ;;
esac
