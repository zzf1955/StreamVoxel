#!/usr/bin/env bash
# Load a local Docker image into the kind node without a temporary tar file (`kind load docker-image` writes the whole
# image to $TMPDIR first, which can fill a small system disk): docker save is streamed into the node's containerd.
# Skips the copy when the node already has an image with the same ID. Usage: load-image.sh <image> [<image> ...]
# Optional: SV_KIND_NAME (default sv-dev).
set -euo pipefail
NODE="${SV_KIND_NAME:-sv-dev}-control-plane"
for img in "$@"; do
  id="$(docker image inspect --format '{{.Id}}' "$img")"
  if docker exec "$NODE" crictl images -q --no-trunc | grep -qx "$id"; then
    echo "$img ($id) already on $NODE"
    continue
  fi
  echo "loading $img ($id) into $NODE"
  docker save "$img" | docker exec -i "$NODE" ctr --namespace=k8s.io images import --all-platforms --digests -
done
