# Viewer

The viewer lets a player fly through the world while it is generated. A vanilla Minecraft Java server runs locally in spectator mode; a bridge process applies the windows committed by the GPU session to that server, and a normal Java client connects to it. No sampling happens on the viewer machine.

```text
GPU machine:  stream session  ->  <run_dir>/wire  ->  python -m viewer.transport   (stdout)
                                                              | ssh
viewer machine:  viewer.bridge: receiver thread -> recording.jsonl + play buffer -> RCON
                                -> local server (StreamVoxelServerHook)  <-  Java client
```

Transport and playback are separate. The transport sends the session's wire packets in order, unmerged, as bounded compressed lines (at most 256 packets or about 384 KB of packet JSON each, zlib + base64), with no ack gating; TCP back-pressure paces it. After a dropped link the bridge reconnects by itself (backing off to at most 15 s) with `--after <last packet received>`, and the packets already buffered keep playing. The receiver thread verifies sequence numbers and sha256 of every packet, appends the raw line to `<viewer_root>/recording.jsonl` and queues the packets in a `viewer.playback.PlayBuffer`. Each packet carries the session's emit time; playback starts once `--delay` seconds (default 3) of material are buffered, advances at `--speed` times real time, releases every packet the playhead has passed, stops and refills when the buffer runs dry, and skips idle gaps longer than 1 s. If the display falls behind, the frame rows of packets released in one tick are merged per cell (`streamvoxel.common.live_rows.coalesce`), so density drops instead of smoothness. The ack in `delivery_state.json` and in the control line means "played on the server"; the session uses it to bound how far the display may lag. On restart the bridge replays received-but-unplayed packets from `recording.jsonl`.

## External requirements (not distributed)

- Minecraft Java Edition server jar for game version 26.2 (data version 4903), passed as `--server`. The client must be the same version, and the server runs with `online-mode=true`, so the player needs a legitimate account.
- JDK 25, passed as `--java`. The `javac` next to the `java` binary is used at start-up to compile `viewer/StreamVoxelServerHook.java`.
- For exporting the initial world only: `node` and the `prismarine-nbt` package (directory passed as `--nbt`), and a void template world (`--template`).
- SSH access from the viewer machine to the GPU machine (key-based, non-interactive). The GPU machine needs this repository (the bridge runs `python -m viewer.transport` there).
- Python >= 3.10 on the viewer machine with this repository on `PYTHONPATH` (or installed); `viewer.start`, `viewer.bridge` and `viewer.world_export` import `streamvoxel.common`, and `world_export` also needs numpy.

The server listens on 127.0.0.1 only (default server port 25582, RCON port 25583, overridable by `server_port` and `rcon_port` in the viewer config). `server_xmx` sets the server heap (default 2G).

## Steps

1. On the GPU machine, prepare a streaming run (see streaming.md): `python -m streamvoxel.stream.session prepare --run <run_dir>`. This creates `<run_dir>/panel`.
2. Export the initial world from the panel (viewer machine or anywhere with node):

   ```bash
   python -m viewer.world_export --panel <run_dir>/panel --output <viewer_root> \
       --template <void_template_world> --node <node> --nbt <prismarine_nbt_dir>
   ```

   `<viewer_root>` must not exist. It receives `blocks.npy`, `world/` (with the `streamvoxel_live` data pack), `initial_world.zip`, `export.json` and `COMPLETE`.
3. Write a viewer config json (see below) with `local_root` = `<viewer_root>`.
4. Start the viewer stack:

   ```bash
   python -m viewer.start --config <viewer_config.json> --java <jdk25>/bin/java --server <server.jar>
   ```

   It verifies the export hashes, unpacks the initial world, writes `server.properties` and `eula.txt` into `<viewer_root>`, starts the server through the hook, and starts the bridge (`python -m viewer.bridge --config <viewer_config.json>`). It prints the address `127.0.0.1:<port>`. Every start resets `<viewer_root>` to the initial world.
5. Start the streaming session on the GPU machine (`scripts/run_stream.sh configs/stream_2x4090.json <run_dir>`; the session must use the same `initial_box_zx` as the export, otherwise the bridge stops with an error).
6. Join the printed address with the Java 26.2 client. Session settings (steps, batch, mode, pause, speed) are controlled from the game (`/trigger streamvoxel_menu` reopens the menu; `[回起点]` or `/trigger streamvoxel_home` teleports to the spawn point in `export.json`, which a world reset does automatically); `defaults` in the config sets the initial values.

## Viewer config keys

Read by `viewer/start.py` and `viewer/bridge.py`: `local_root`, `ssh` (SSH destination), `remote_source` (path of this repository on the GPU machine), `remote_python`, `remote_run` (the `<run_dir>`), `fallback_player_zx`, `air_bank_setup` (list of `{x, y, z, state}` entries for the cave/void air reference blocks written by the export), `session`, `defaults` (`steps`, `batch`, `mode`, `paused`, `speed`), optional `server_port`, `rcon_port`, `server_xmx`.

## Offline replay

`recording.jsonl` is the compressed recording of the whole generation process (every packet, at its original cadence). To fetch all packets a run has written so far in one go and replay them without a GPU machine:

```bash
python -m viewer.record --config <viewer_config.json> --out <recording>.jsonl
python -m viewer.start --config <viewer_config.json> --java <jdk25>/bin/java --server <server.jar> --replay <recording>.jsonl [--speed 2]
```

`viewer.record` runs `python -m viewer.transport --once` over SSH (a partial download stays as `<recording>.part`). Replay plays into a freshly reset initial world and does not connect to the GPU machine. On every start an existing non-empty `recording.jsonl` is renamed to `recording_<timestamp>.jsonl` rather than deleted.

## Server notes

- Triggers: a `/trigger` objective only accepts a command while it is enabled for the player, and a click disables it. At start the bridge writes a per-tick function `streamvoxel_live:tick` into the data pack (`minecraft:tick` tag, then `reload`); the server re-enables every `streamvoxel_*` objective each tick and parks the clicked value in `#<key> streamvoxel_cfg`, so the bridge polls one flag (`#any`) every 0.1 s and takes the values only when it is set. Command feedback gamerules (`send_command_feedback`, `command_block_output`, `log_admin_commands`) are turned off.
- Template cache: the vanilla server caches every placed structure template forever and each frame row has a new id, which exhausts the heap. `StreamVoxelServerHook` removes the `streamvoxel_live:` templates from the server's template cache every 0.5 s, and the bridge deletes the `.nbt` file after placing.
- Stopping the server waits 30 s and then sends SIGKILL (a full heap can hang the shutdown hook).
- The transport's control receiver ignores bad lines and failed writes instead of exiting, and the whole process exits when stdin closes (the viewer is gone), so no orphan transport is left behind.

## Profiling

The bridge writes `bridge_profile.jsonl` under `<viewer_root>` (including `buffer_s`, `group:packets`, `lag:emit_to_recv`, `lag:emit_to_shown`). Summarise it with `python -m viewer.profile <bridge_profile.jsonl>`.

## Tests

`PYTHONPATH=. python -m pytest -q tests/viewer` (CPU): macro splitting, player position fallback, menu and telemetry length, trigger round trip, home button, transport telemetry, play buffer timing (`test_playback.py`), packet order and hash, settings clamping, world-export crop.
