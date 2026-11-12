# Streaming backend

`streamvoxel/stream/` generates 16x16 full-height windows one after another around a moving player. A window is conditioned on the committed windows around it (L1: up to 8 neighbours; L2: the nearest `l2_cap` committed windows) and on biomes, then sampled with the low-entropy-last sampler. Many windows are in flight at the same time.

## Architecture

| Module | Role |
| --- | --- |
| `session.py` | Coordinator process. Owns scheduler state, the committed world, player control and the wire (the packets the viewer reads). Spawns one worker per GPU. Entry: `python -m streamvoxel.stream.session {prepare,gate,fresh,run} --run <run_dir>`. |
| `worker.py` | One process per GPU (so two GPUs never share a GIL). Holds a replica of the committed world, builds conditions for the windows the coordinator picks, runs the slot forward and the samplers, returns finished windows and frame rows. Messages carry an epoch; older epochs are dropped. |
| `scheduler.py` | Pure scheduling logic: `separated`/`pick_window` (admission), `plan_inserts` (which idle slots start), `rank_share` (split of the global batch over ranks), `make_item` (condition of one window), biome crop. |
| `engine.py` | `SlotForward`: fixed-capacity slot forward. Conditions are copied into padded key buffers; one CUDA graph per bucket size (1, 2, 4, 8) runs only the leading slots. `open_model` loads base and control checkpoints. |
| `fast_ops.py` | `FastSlotForward` and fast selection. Same model and weights as `SlotForward`; see below. |
| `slot.py` | `LowEntropyLastSlot`: per-slot sampler trajectory (one marking forward, then block-quota steps; the deferred low-entropy voxels are committed at the end without a forward). `ForwardBudget` counts forwards. |
| `condition.py` | Request/condition layout and `Mapping` (global 981 ids <-> compact 304 ids). |
| `model_loading.py` | Checkpoint loading helpers. |
| `reserve.py`, `telemetry.py` | GPU memory reservation and per-rank GPU telemetry. |

## Admission rule

A candidate window must touch a committed window and must not be too close to a window that is still being sampled. With `diagonal: false` (default) the Chebyshev distance to every in-flight window must be at least 2, so two in-flight windows never share an edge or a corner and no window ever reads a neighbour that is not committed yet. With `diagonal: true` only edge neighbours are excluded (Manhattan distance >= 2). Among admissible cells the one nearest to the player is chosen (`pick_window`). Committed snapshots are the only condition source.

Slot refill (`plan_inserts`): in `dynamic` mode any idle slot is refilled while fewer than `batch` windows are active; in `fixed` mode a new group starts only when every active window has finished.

## Fast path

`fast_ops: true` selects `FastSlotForward` and `FastLowEntropyLastSlot`:

- Local (L1 + biome) cross-attention K/V is computed once per window; prefix and local keys are packed into one buffer, valid keys first.
- FlexAttention over the packed keys with a per-slot block mask, so each slot pays only for its own keys and no key-length buckets are needed.
- Patchify by gather (`embedding_bag`) instead of a one-hot 3D convolution.
- Transformer blocks compiled with `torch.compile`, with emulated precision casts to match eager BF16.
- Entropy and block-quota selection run inside the CUDA graph without host synchronisation (Triton entropy kernel when `triton` is installed); one host wait per forward.

The fast path reproduces the reference sampler bit for bit for the selection and within a documented logit tolerance for the forward; `scripts/bench_ops.py gate` and `forward` check this (see benchmark.md).

## Configuration

`configs/stream_2x4090.json` is copied to `<run_dir>/config.json`. Required keys: `world`, `parent_panel`, `mapping`, `vocabulary`, `base`, `control`, `initial_box_zx`, `visible_gpus`. Replace every `<...>` value.

| Key | Meaning | Config value |
| --- | --- | --- |
| `world` | World id inside the panel | 161 |
| `parent_panel` | Panel directory with `manifest.json`, `COMPLETE` and the case folders | placeholder |
| `mapping`, `vocabulary` | Compact mapping json and vocabulary json | placeholder |
| `base`, `control` | Checkpoints | placeholder |
| `initial_box_zx` | Initially known box `[z0, x0, z1, x1]` in 16-voxel cells | `[112, 112, 160, 160]` |
| `visible_gpus` | Physical GPU indices (as nvidia-smi numbers them), one worker each; worker r runs with `CUDA_VISIBLE_DEVICES=visible_gpus[r]` | `[0, 1]` |
| `slots_per_gpu` | Slots per GPU | 8 |
| `l1_cap`, `l2_cap` | Max L1 neighbours and L2 far windows per condition | 8, 256 |
| `block_size` | Block size of the sampler | 4 |
| `defaults` | Initial player-adjustable settings: `steps`, `batch`, `mode` (`dynamic` or `fixed`), `paused` | 128, 8, dynamic, true |
| `last_eps` | Low-entropy-last threshold | 1e-4 |
| `max_windows`, `max_gpu_seconds`, `idle_timeout_seconds` | Stop conditions of `run` | 100000, 21600, 21600 |
| `headless_windows` | If > 0, generate that many windows without a viewer | 0 |
| `max_display_lag_windows` | Max windows the sampler may run ahead of the viewer | 32 |
| `reserve_memory`, `leave_free_gb` | Hold GPU memory and give it back down to this free margin | true, 1 |
| `diagonal` | Admission variant above | false |
| `frame_interval_s` | Interval of progressive frames | 0.3 |
| `fast_ops` | Use the fast path | true |
| `progressive_frames` | Stream partial frames while a window is sampling | true |

`air_bank` is an optional extra key (read by the session, not set in the shipped config).

## Running

`scripts/run_stream.sh <config.json> <run_dir>` (from the repository root; `<run_dir>` must not exist) performs, in order:

1. `python -m streamvoxel.stream.session prepare --run <run_dir>`: verifies the parent panel manifest against `COMPLETE`, checks that the base checkpoint's SHA256 equals the identity stored in the ControlNet checkpoint, links the panel files into `<run_dir>/panel`, and writes `frozen_source_manifest.json` (hashes of the source files used).
2. `... session gate --run <run_dir>`: loads the first GPU and checks the padded CUDA-graph forward against the unpadded reference on the run's initial frontier; writes `numerical_gate.json`.
3. `... session fresh --run <run_dir>`: removes artifacts of earlier sessions (wire, results, logs, gate output); config and panel stay.
4. `... session run --run <run_dir>`: serves until `max_windows`, `max_gpu_seconds` or `idle_timeout_seconds`. Log: `<run_dir>/session.log`; per-GPU timing: `<run_dir>/timing_r0.json`, `timing_r1.json`.

## Connecting the viewer

The session writes packets to `<run_dir>/wire`. On the viewer machine, `viewer/bridge.py` starts `python -m viewer.transport --run <run_dir> --after <ack>` on the GPU machine over SSH. The transport sends everything written since the last batch as one compressed line and waits for the viewer's acknowledgement before the next batch; player position and settings travel back on the same channel. See viewer.md.

## Elastic mode: workers join over the network

By default `session run` starts one local process per entry of `visible_gpus`. In elastic mode the session instead listens on a port and GPU workers on other machines join and leave while it runs. Local mode is unchanged.

### Starting the coordinator

Add `listen` to the run config (it replaces `visible_gpus`, which is then not needed), put a shared secret in a file and start the session as usual:

    "listen": ["0.0.0.0", 7788],
    "elastic": {"drain_timeout_s": 60, "min_workers": 1}

    SV_AUTHKEY_FILE=/path/to/authkey python -m streamvoxel.stream.session run --run <run_dir>

The key file holds at least 16 bytes (`SV_AUTHKEY` with the key itself also works). The session computes a digest of its own code, and it starts generation (sends `go`) only when `min_workers` workers are ready. The `elastic` block takes the fields of `ElasticSettings` in `session.py`: `lost_after_s` (15, silence after which a worker is lost), `building_lost_after_s` (60, same while the model loads), `drain_timeout_s` (60), `no_worker_timeout_s` (120, no worker at all ends the session), `hb_interval_s`, `snapshot_chunk`, `max_pending_bytes`, `min_workers`, `max_slots_per_worker` and `max_slots_total` (ceiling of the batch a player can request, 64). `session gate` ignores `listen` and runs the numerical gate on the first local GPU.

### Starting a worker

    SV_AUTHKEY_FILE=/path/to/authkey python -m streamvoxel.stream.worker_main --connect <host>:7788 --capacity 8

- `--capacity`: number of slots on this GPU; the coordinator may lower it to `max_slots_per_worker`.
- `--connect-timeout`: how long to keep retrying while the coordinator is not listening yet (default 120 s). A wrong key or a rejection is never retried.
- `--nvsmi-index`: physical GPU for telemetry (default: first entry of `CUDA_VISIBLE_DEVICES`, else 0).
- `--base`, `--control`: local paths of the two checkpoints. Without them the paths from the coordinator's config are used, so they must exist on the worker machine.
- Scratch files (slot logs, traces) go to `SV_WORKER_RUN` or a temporary folder.

The coordinator refuses a worker whose code differs from its own (`code_digest`, a hash of every Python file of the package) or, when `--base`/`--control` are given, whose checkpoint hashes differ from the panel. A worker that loaded other weights than the panel expects stops with an error after loading. Refusals and failed joins are written to `<run_dir>/worker_events.jsonl`. Three failed joins stop the session only when no worker is ready.

### Leaving: drain and SIGTERM

SIGTERM makes a worker ask to leave. The coordinator stops giving it new windows, waits until the windows it already owns are finished (at most `drain_timeout_s`), then tells it to stop. A worker that disappears without draining is declared lost after `lost_after_s`; its unfinished windows are given to other workers, and windows it had already finished are published from the coordinator's copy. If the leaving worker is the last ready one, the request is kept and granted as soon as another worker is ready, so the world keeps growing meanwhile. Under Kubernetes the pod's `terminationGracePeriodSeconds` must be longer than `drain_timeout_s` (see `deploy/README.md`).

### What is not reproducible

Which worker generates which window depends on timing, so two elastic runs with the same seed do not produce the same world, and the order of commits can differ. A window that is re-done after a loss starts again from the beginning. Local mode with fixed settings stays deterministic. Elastic mode has been exercised on the CPU with fake workers only; real GPU workers, real network latency and the snapshot transfer speed are not verified.

## Kubernetes

`deploy/` holds a kind setup and manifests for one elastic world; `deploy/README.md` has the commands and the reasons
behind each choice. In short:

- One world is a coordinator Deployment (one replica, CPU only) behind a ClusterIP Service `sv-coord-<world>` on port
  7788, a worker Deployment `sv-worker-<world>` with one GPU per Pod, a Secret with the authkey (mounted as the file
  `SV_AUTHKEY_FILE`) and a ConfigMap with the run config. Workers reach the coordinator by the Service name. Scaling
  the world is `kubectl scale deployment sv-worker-<world> --replicas=N`.
- Workers take their settings from the environment as well as from flags (a flag wins): `SV_CONNECT` (host:port),
  `SV_CAPACITY`, `SV_CONNECT_TIMEOUT`, `SV_NVSMI_INDEX`, `SV_BASE`, `SV_CONTROL`, `SV_READY_FILE` (exists while the
  worker is ready; used by the readiness probe), `SV_WORKER_RUN`. The worker sets `CUDA_DEVICE_ORDER=PCI_BUS_ID`
  unless it is already set, so CUDA and nvidia-smi number the cards the same way.
- Start order does not matter: a worker retries the connection (refused, or a name that does not resolve yet) for
  `SV_CONNECT_TIMEOUT` seconds; a wrong key or a refusal is not retried.
- Worker exit codes: 0 finished (drained, stopped by the coordinator, or SIGTERM before it joined), 1 error while
  serving, 2 bad settings, 3 wrong authkey, 4 refused by the coordinator (code digest, weights, capacity), 5 coordinator
  not reachable in time. A Deployment restarts the container in every case; delete the worker Deployment when the
  world ends.
- Scale down and `kubectl delete pod` send SIGTERM, which drains the worker (above). The Pod's grace period (120 s)
  is longer than `drain_timeout_s` (60 s). A force delete gives about 2 s, so the worker is lost and its windows are
  re-done by the others; the Deployment then starts a replacement Pod, which joins as a new worker.
- Capacity per GPU type comes from `SV_CAPACITY` of that type's worker Deployment (8 slots measured on RTX 4090;
  the A800 value is a placeholder).
- The coordinator's `metrics.json` (in the run folder) carries `sv_requested_batch`, `sv_slots_total` and the worker
  counts; `deploy/kind/scale-example.sh` shows a single scaling decision from it. It is an example, not an autoscaler.
- Several worlds (several players): one coordinator Deployment, Service, Secret, ConfigMap and worker Deployment per
  world, named by the world. Worlds share nothing; a worker belongs to one world. Not exercised.
- Checked on a single-node kind cluster with two RTX 4090 cards (one per worker Pod), headless, 3000 windows. The
  following held: a worker started before the coordinator joins after retrying; scaling to two workers adds a worker
  that runs the gate and shares the batch; a force-deleted worker is detected as lost, its unfinished windows are
  re-done and the Deployment's replacement Pod joins; scaling back drains a worker with nothing reclaimed; a worker
  with a wrong key exits with code 3 while the session continues; every window is committed exactly once. Not
  exercised there: the drain timeout, several worlds, workers on other machines.
