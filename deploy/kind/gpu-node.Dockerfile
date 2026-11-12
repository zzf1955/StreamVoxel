# Usage: docker build -f gpu-node.Dockerfile --build-arg GPU=<gpu-uuid>[,<gpu-uuid>...] -t sv-node:gpu .
# GPU is required (the UUIDs of the allowed cards, comma-separated); there is no default.
ARG BASE=kindest/node:r27-pinned
FROM ${BASE}
ARG GPU
RUN test -n "${GPU}" || { echo "build-arg GPU (uuid list) is required" >&2; exit 1; }
ENV NVIDIA_VISIBLE_DEVICES=${GPU}
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
