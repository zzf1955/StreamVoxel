# Deployment (kind, single node)

Requires `docker`, `kind`, `kubectl`, `envsubst`. Variables: `SV_KIND_NAME` (default `sv-dev`),
`SV_KUBECONFIG` (default `./kubeconfig`; never written to the user's default kubeconfig),
optional `SV_KIND_IMAGE`, `SV_KIND_CONFIG`.

    deploy/kind/up.sh       # kind create cluster (CPU config, API server on 127.0.0.1)
    deploy/kind/smoke.sh    # apply deploy/k8s/smoke-cpu.yaml, wait, print logs, delete
    deploy/kind/down.sh     # kind delete cluster (only the named cluster)

## GPU cluster

    docker build -f deploy/kind/gpu-node.Dockerfile --build-arg GPU=<uuid>[,<uuid>...] -t sv-node:gpu .
    deploy/kind/gpu-up.sh        # cluster + gpu-setup + image load (env: SV_GPU_UUIDS SV_DATA_DIR SV_OUT_DIR SV_KUBECONFIG SV_KIND_IMAGE [SV_LOAD_IMAGES])
    deploy/kind/gpu-setup.sh     # re-run after a node restart (env: SV_GPU_UUIDS), then delete the device plugin Pod
    deploy/kind/load-image.sh    # docker image -> node containerd, streamed (no temporary tar on the local disk)

`SV_GPU_UUIDS` lists the cards the cluster may use (comma-separated UUIDs; `SV_GPU_UUID` for one card still works).
`gpu-setup.sh` keeps only those cards' device nodes in the (privileged) kind node, generates the CDI spec and the
`svgpu` runtime handler for the device plugin. The device plugin then reports one `nvidia.com/gpu` per card, and a Pod
that asks for `nvidia.com/gpu: 1` gets exactly one of them through CDI: inside the Pod `nvidia-smi` shows one card,
as index 0, and the other cards' device nodes cannot be opened. Workload Pods must not use `runtimeClassName: svgpu`
(that handler adds every allowed card; it exists only for the device plugin Pod).

Single-GPU streaming run (see comments in each file):

    deploy/docker/Dockerfile                # build with --build-arg BASE=<local cuda base> if the registry is unreachable
    deploy/k8s/stream_1gpu.template.json    # render with envsubst into <SV_OUT_DIR>/<run>.json
    deploy/k8s/stream-headless.yaml         # Job (envsubst), headless run on 1 GPU

## Elastic streaming on Kubernetes

One world = one coordinator Deployment (replicas 1, CPU only) + its Service + one worker Deployment (one GPU per Pod)
+ an authkey Secret + a config ConfigMap. `deploy/kind/elastic.sh` creates and removes them:

    export SV_KUBECONFIG=... SV_WORLD=w1 SV_IMAGE=sv-stream:<tag> SV_DATA_DIR=<read-only data dir>
    SV_CONFIG=<run config.json> deploy/kind/elastic.sh config     # or SV_PANEL_DIR/SV_MAPPING/SV_VOCAB/SV_BASE_CKPT/SV_CONTROL_CKPT + template
    deploy/kind/elastic.sh secret                                  # random key (or SV_AUTHKEY_SOURCE=<key file>)
    SV_RUN_NAME=<new run folder name> deploy/kind/elastic.sh coordinator
    SV_WORKERS=1 SV_CAPACITY=8 deploy/kind/elastic.sh workers      # any order relative to the coordinator
    kubectl --kubeconfig $SV_KUBECONFIG scale deployment sv-worker-w1 --replicas=2
    deploy/kind/elastic.sh down                                    # the run folder on the output volume stays

| Object | Name | Notes |
| --- | --- | --- |
| Secret | `sv-authkey-<world>` | key `authkey` (>= 16 bytes), mounted as a file; `SV_AUTHKEY_FILE` points at it in both Pods |
| ConfigMap | `sv-config-<world>` | key `config.json`, the run config with `"listen": ["0.0.0.0", 7788]` (`k8s/stream_elastic.template.json`) |
| Service | `sv-coord-<world>` | ClusterIP, port 7788; workers connect to `sv-coord-<world>:7788` |
| Deployment | `sv-coord-<world>` | replicas 1, `Recreate`, no GPU; `prepare`, `fresh`, `run` into `/mnt/sv-out/runs/<SV_RUN_NAME>` (`k8s/elastic-coordinator.yaml`) |
| Deployment | `sv-worker-<world>` | `nvidia.com/gpu: 1` per Pod, `SV_CAPACITY` slots, grace period 120 s (`k8s/elastic-worker.yaml`) |

Design choices:

- **GPUs.** Each worker Pod gets its own physical card from the device plugin (the cluster is set up with several
  allowed cards, `SV_GPU_UUIDS`), so adding a Pod adds real GPU capacity. Time-slicing one card between Pods (device
  plugin sharing) would only show that joining and leaving work, not a speed-up; it is not used here.
- **Coordinator.** A Deployment rather than a Job so that the Service always selects it by label and so that it is
  managed like the workers; `Recreate` makes sure there are never two coordinators of one world. It needs no GPU (the
  numerical gate runs on every joining worker instead). Ready = the session loop runs (its `metrics.json` was written
  in the last 15 s). Until then the Service has no endpoint, so a worker's connect is refused and retried. When the
  session has ended the container stays up, not ready, so that the run folder can still be read (`kubectl exec`, the
  viewer transport); a restart in the middle of a session is not supported (the run folder exists, the new container
  exits with an error). The run folder lives on a hostPath volume here; an emptyDir or a PVC works the same way.
- **Workers.** Configured by environment (`SV_CONNECT`, `SV_CAPACITY`, `SV_CONNECT_TIMEOUT`, `SV_AUTHKEY_FILE`,
  `SV_NVSMI_INDEX`, `SV_READY_FILE`, `SV_WORKER_RUN`; see `python -m streamvoxel.stream.worker_main --help`). The
  command is the Python process itself (exec form), so SIGTERM reaches it. Readiness = the file `/scratch/ready`,
  which the worker creates once the model is built and the gate passed, and removes when it asks to leave. A
  readiness probe does not route traffic here (the worker connects out); it shows when a new Pod is serving.
  No liveness probe: loading and compiling take about a minute on a quiet host and over ten minutes on a busy one
  (measured), and a stuck worker is detected by the coordinator
  (heartbeat) anyway. Worker scratch (slot logs, traces) is an emptyDir and goes away with the Pod. The torch.compile
  and Triton kernel caches (many small files written while the worker builds) are on a memory-backed emptyDir (4 GiB
  limit, counted as Pod memory), so a slow or busy node disk does not stretch the build; each new Pod compiles again.
- **Leaving.** SIGTERM (scale down, rollout, `kubectl delete pod`) starts a drain; no `preStop` hook is needed.
  The coordinator waits at most `drain_timeout_s` (default 60 s) for the windows the worker owns, so
  `terminationGracePeriodSeconds` (120 s) must be longer; otherwise the Pod is killed and its windows are re-done
  elsewhere. If the draining worker is the last ready one, the coordinator keeps it until another worker is ready
  (scaling to 0 does not stop the world; delete the world instead). A force delete (`--grace-period=0 --force`) leaves
  the worker about 2 s before SIGKILL, so it is seen as lost and its windows are reclaimed.
- **Capacity.** Each worker announces its slot count (`SV_CAPACITY`) in its hello; the coordinator sums the ready
  workers. Use one worker Deployment per GPU type (with a `nodeSelector`) and set its `SV_CAPACITY` from this table:

  | GPU | `SV_CAPACITY` | Source |
  | --- | --- | --- |
  | RTX 4090 (24 GB) | 8 | measured (8 slots per card in all runs) |
  | A800 (80 GB) | 16 (placeholder) | not measured; only 8 slots per card were ever run |

- **Code and weights.** The image carries the code; coordinator and workers must run the same image tag, because
  the coordinator refuses a worker whose code digest (hash of every Python file of the package) differs from its
  own. Checkpoints and panel data are not in the image: the host data directory is mounted read-only into the kind
  node (`kind-config.gpu.yaml` extraMounts) and from there into the Pods at the same absolute path, so the paths in
  the run config resolve in every Pod. The worker checks the checkpoint hashes against the panel after loading;
  setting `SV_BASE`/`SV_CONTROL` makes it hash them before joining too (a few seconds per checkpoint).
- **Scaling.** No HorizontalPodAutoscaler. `deploy/kind/scale-example.sh` is an example of one scaling decision from
  the coordinator's `metrics.json` (`sv_requested_batch` against `SV_CAPACITY`), printed, and applied with `--apply`.
- **More worlds.** Each world is its own set of the objects above with another `SV_WORLD` (and `SV_RUN_NAME`), and its
  own Secret, so a worker can only join the world whose key it has. Not exercised.
