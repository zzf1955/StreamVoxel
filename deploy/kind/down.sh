#!/usr/bin/env bash
set -euo pipefail
NAME="${SV_KIND_NAME:-sv-dev}"
KUBECONFIG_PATH="${SV_KUBECONFIG:-./kubeconfig}"
kind delete cluster --name "$NAME" --kubeconfig "$KUBECONFIG_PATH"
