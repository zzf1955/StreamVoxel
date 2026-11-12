"""Install and start the local Java 26.2 viewer used by the low-entropy-last bridge."""
import argparse
import json
import os
from pathlib import Path
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import time
import zipfile

from streamvoxel.common.io import atomic_json, digest


def available(port):
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', port))


STATE = ('world', 'server.pid', 'bridge.pid', 'connection.json', 'delivery.jsonl', 'delivery_state.json', 'active_cells.json',
         'reset.json', 'settings.json', 'logs', 'bridge_profile.jsonl', 'bridge.log', 'server.log', 'bridge_error_*.json')


def stop(root, name, marker):
    """Terminate the process a receipt names, but only if its command line still carries `marker`."""
    receipt = root / name
    if not receipt.exists():
        return
    pid = int(receipt.read_text())
    try:
        command = subprocess.run(['ps', '-p', str(pid), '-o', 'command='], capture_output=True, text=True).stdout
        if marker in command:
            os.kill(pid, signal.SIGTERM)
            for _ in range(60):
                time.sleep(.5)
                if marker not in subprocess.run(['ps', '-p', str(pid), '-o', 'command='], capture_output=True, text=True).stdout:
                    break
            else:
                # A server out of heap can hang in its shutdown hook; it holds nothing worth saving.
                os.kill(pid, signal.SIGKILL)
                time.sleep(1)
                if marker in subprocess.run(['ps', '-p', str(pid), '-o', 'command='], capture_output=True, text=True).stdout:
                    raise TimeoutError(f'{name} process {pid} did not exit')
    except ProcessLookupError:
        pass


def wipe(root):
    """Every start begins from the initial world: stop the previous server and bridge, delete their state."""
    stop(root, 'bridge.pid', 'viewer.bridge')
    stop(root, 'server.pid', 'server.jar')
    # The recording is the only copy of a generation process that is gone from the GPU box: set it aside, never delete it.
    recording = root / 'recording.jsonl'
    if recording.exists() and recording.stat().st_size:
        recording.rename(root / time.strftime('recording_%Y%m%d_%H%M%S.jsonl', time.localtime(recording.stat().st_mtime)))
    for pattern in STATE:
        for path in root.glob(pattern):
            shutil.rmtree(path) if path.is_dir() else path.unlink()


def start(config, java, server, replay=None, speed=1.):
    cfg = json.loads(config.read_text())
    root = Path(cfg['local_root'])
    export = json.loads((root / 'export.json').read_text())
    if digest(root / 'export.json') != (root / 'COMPLETE').read_text().strip():
        raise RuntimeError('export.json does not match COMPLETE')
    if digest(root / 'initial_world.zip') != export['archive_sha256']:
        raise RuntimeError('initial world archive hash mismatch')
    wipe(root)
    world = root / 'world'
    if not world.exists():
        world.mkdir()
        with zipfile.ZipFile(root / 'initial_world.zip') as archive:
            for name in archive.namelist():
                target = world / name
                if not target.resolve().is_relative_to(world.resolve()):
                    raise ValueError('Unsafe archive member')
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(name))
        for name, hashed in export['world_files'].items():
            if digest(world / name) != hashed:
                raise RuntimeError(f'World file hash mismatch: {name}')
    port = cfg.get('server_port', 25582)
    rcon_port = cfg.get('rcon_port', 25583)
    available(port)
    available(rcon_port)
    password = secrets.token_hex(24)
    atomic_json(root / 'connection.json', dict(server_port=port, rcon_port=rcon_port, rcon_password=password))
    (root / 'connection.json').chmod(0o600)
    (root / 'eula.txt').write_text('eula=true\n')
    (root / 'server.properties').write_text(
        f'level-name=world\nserver-ip=127.0.0.1\nserver-port={port}\nonline-mode=true\nenable-rcon=true\n'
        f'rcon.port={rcon_port}\nrcon.password={password}\nbroadcast-rcon-to-ops=false\nview-distance=8\n'
        f'simulation-distance=4\nmax-players=1\ngamemode=spectator\nforce-gamemode=true\ndifficulty=peaceful\n'
        f'spawn-protection=0\nmax-tick-time=120000\npause-when-empty-seconds=0\nmotd=StreamVoxel low-entropy-last live\n')
    (root / 'server.properties').chmod(0o600)
    # The vanilla bundler started through StreamVoxelServerHook: the bridge's fly speed and the live template cache (see that file).
    hook = root / 'server_hook'
    subprocess.run([str(Path(java).with_name('javac')), '-d', str(hook), str(Path(__file__).with_name('StreamVoxelServerHook.java'))], check=True)
    log = (root / 'server.log').open('ab')
    proc = subprocess.Popen([str(java), '-Xms512M', f'-Xmx{cfg.get("server_xmx", "2G")}', '-XX:ActiveProcessorCount=4',
                             '--add-opens', 'java.base/java.lang=ALL-UNNAMED', f'-Dstreamvoxel.flySpeedFile={root / "fly_speed.txt"}',
                             '-cp', f'{hook}{os.pathsep}{server}', 'StreamVoxelServerHook', '--nogui'], cwd=root, stdin=subprocess.DEVNULL, stdout=log,
                            stderr=subprocess.STDOUT, start_new_session=True)
    log.close()
    (root / 'server.pid').write_text(str(proc.pid) + '\n')
    for _ in range(240):
        if proc.poll() is not None:
            raise RuntimeError(f'Native server exited {proc.returncode}; inspect server.log')
        if 'RCON running on' in (root / 'server.log').read_text(errors='replace'):
            break
        time.sleep(.5)
    else:
        raise TimeoutError('Native server startup; inspect server.log')
    bridge_log = (root / 'bridge.log').open('ab')
    bridge = subprocess.Popen([sys.executable, '-u', '-m', 'viewer.bridge', '--config', str(config)]
                              + (['--replay', str(replay), '--speed', str(speed)] if replay else []),
                              stdout=bridge_log, stderr=subprocess.STDOUT, start_new_session=True)
    bridge_log.close()
    (root / 'bridge.pid').write_text(str(bridge.pid) + '\n')
    print(json.dumps(dict(address=f'127.0.0.1:{port}', game_version='26.2', server_pid=proc.pid, bridge_pid=bridge.pid)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'java', 'server'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--replay', type=Path, help='play this recording into the fresh initial world instead of connecting to the GPU box')
    parser.add_argument('--speed', type=float, default=1.)
    args = parser.parse_args()
    start(args.config.resolve(), args.java.resolve(), args.server.resolve(), args.replay and args.replay.resolve(), args.speed)


if __name__ == '__main__':
    main()
