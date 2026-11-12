# Training/inference image. Build context = repository root. No weights or data are baked in.
# BASE can be overridden (e.g. a locally cached CUDA base image) when the registry is not reachable.
# gcc + Python headers are needed because triton / FlexAttention compile kernels at run time.
ARG BASE=nvidia/cuda:12.8.0-runtime-ubuntu22.04
FROM ${BASE}
ENV DEBIAN_FRONTEND=noninteractive
# Ubuntu 22.04 ships Python 3.10 (torch 2.11 supports it); avoids the deadsnakes PPA, whose key server may be unreachable.
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-venv python3-dev build-essential curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
ARG PIP_INDEX_URL=https://pypi.org/simple
ENV PIP_INDEX_URL=${PIP_INDEX_URL}
RUN pip install --no-cache-dir torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128 --extra-index-url ${PIP_INDEX_URL}
WORKDIR /workspace
COPY . /workspace
RUN pip install --no-cache-dir ".[triton]"
ENV PYTHONPATH=/workspace
# ENTRYPOINT intentionally empty: the Pod/Job spec supplies the command.
