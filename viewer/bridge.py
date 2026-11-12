"""Vanilla Java viewer for the low-entropy-last live session. Stdlib only; no sampling on the Mac."""
from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path
import re
import shlex
import socket
import struct
import subprocess
import threading
import time
import zlib

from streamvoxel.common.io import atomic_json
from streamvoxel.common.live_rows import coalesce, expand_row, special_table, states_digest
from streamvoxel.common.live_wire import LIMITS, clamp_settings, unbatch, verify_packet
from streamvoxel.common.preview_states import load_registry
from viewer.playback import PlayBuffer
from viewer.profile import Profile

PRESETS = (8, 16, 32, 64, 128)
BATCHES = (1, 2, 4, 8, 16)
SPEEDS = (100, 70, 50, 30, 15)
EXTRA = ('menu', 'reset', 'world', 'run', 'batch', 'mode', 'move', 'speed', 'home')
KEYS = ('steps',) + EXTRA
DEFAULTS = dict(steps=32, batch=8, mode='dynamic', paused=True, speed=100)
PROF = None   # set by `bridge`; stage timings go to bridge_profile.jsonl (see viewer.profile)


def span(key):
    return PROF.span(key) if PROF else contextlib.nullcontext()


class Rcon:
    def __init__(self, port, password, tag='rcon'):
        self.tag = tag
        self.socket = socket.create_connection(('127.0.0.1', port), timeout=30)
        self.ident = 0
        self.request(3, password)

    def read(self, count):
        data = b''
        while len(data) < count:
            chunk = self.socket.recv(count - len(data))
            if not chunk:
                raise EOFError('RCON disconnected')
            data += chunk
        return data

    def request(self, kind, text):
        self.ident += 1
        body = struct.pack('<ii', self.ident, kind) + text.encode() + b'\0\0'
        self.socket.sendall(struct.pack('<i', len(body)) + body)
        length = struct.unpack('<i', self.read(4))[0]
        if not 10 <= length <= 4 * 1024 * 1024:
            raise ValueError('Invalid RCON size')
        response = self.read(length)
        ident, _typ = struct.unpack('<ii', response[:8])
        if ident != self.ident:
            raise RuntimeError('RCON authentication/sequence mismatch')
        return response[8:-2].decode()

    def command(self, text):
        if len(text.encode()) > 1400:
            raise ValueError('RCON command exceeds safe input bound')
        with span(f'{self.tag}:' + ' '.join(text.split()[:2])):
            result = self.request(2, text)
        for marker in ('Unknown or incomplete command', 'Unknown function', 'Failed to place', 'No elements matching',
                       'That position is not loaded', 'There is no template with ID'):
            if marker in result:
                raise RuntimeError(f'{text}: {result}')
        return result

    def close(self):
        self.socket.close()


def player_position(rcon, fallback):
    text = rcon.command('data get entity @a[limit=1] Pos')
    match = re.search(r'\[(-?[\d.eE+]+)d?,\s*(-?[\d.eE+]+)d?,\s*(-?[\d.eE+]+)d?\]', text)
    if not match:
        return dict(player_present=False, player_zx=fallback)
    x, y, z = map(float, match.groups())
    return dict(player_present=True, player_zx=[z, x], player_y=y)


def apply_speed(rcon, root, percent):
    """Walking and flying speed as a percent of vanilla (0.1 and 0.05). Flight goes through the server's fly speed hook."""
    rcon.command(f'attribute @a minecraft:movement_speed base set {0.1 * percent / 100:.4f}')
    tmp = root / 'fly_speed.tmp'
    tmp.write_text(f'{0.05 * percent / 100:.5f}\n')
    tmp.replace(root / 'fly_speed.txt')


def menu_rows(settings):
    button = lambda text, command, color, on=False: dict(
        text=text, color='green' if on else color, bold=on, click_event=dict(action='run_command', command=command))
    paused = settings.get('paused', False)
    rows = [[dict(text=''), dict(text='StreamVoxel low-entropy-last', color='gold', bold=True),
             dict(text='  /trigger streamvoxel_menu 重开', color='gray')]]
    rows.append([dict(text=''),
                 button('[开始]', '/trigger streamvoxel_run set 2', 'aqua', not paused), dict(text=' '),
                 button('[暂停]', '/trigger streamvoxel_run set 1', 'aqua', paused),
                 dict(text='  暂停不再开新地块，正在生成的会跑完', color='dark_gray')])
    row = [dict(text=''), dict(text='Cap: ', color='gray')]
    for value in PRESETS:
        row.append(button(f'[{value}]', f'/trigger streamvoxel_steps set {value}', 'aqua', settings.get('steps') == value))
        row.append(dict(text=' '))
    rows.append(row)
    row = [dict(text=''), dict(text='Batch: ', color='gray')]
    for value in BATCHES:
        row.append(button(f'[{value}]', f'/trigger streamvoxel_batch set {value}', 'aqua', settings.get('batch') == value))
        row.append(dict(text=' '))
    row.append(dict(text='(两卡均分，16 就是每卡 8)', color='dark_gray'))
    rows.append(row)
    row = [dict(text=''), dict(text='移动速度: ', color='gray')]
    for value in SPEEDS:
        row.append(button(f'[{value}%]', f'/trigger streamvoxel_speed set {value}', 'aqua', settings.get('speed', 100) == value))
        row.append(dict(text=' '))
    row.append(dict(text='(行走和飞行一起，或 /trigger streamvoxel_speed set <1-100>)', color='dark_gray'))
    rows.append(row)
    dynamic = settings.get('mode') != 'fixed'
    rows.append([dict(text=''), dict(text='批次: ', color='gray'),
                 button('[动态补位]', '/trigger streamvoxel_mode set 1', 'aqua', dynamic), dict(text=' '),
                 button('[固定批次]', '/trigger streamvoxel_mode set 2', 'aqua', not dynamic)])
    flying = settings.get('move', 'fly') != 'walk'
    rows.append([dict(text=''), dict(text='移动: ', color='gray'),
                 button('[飞行]', '/trigger streamvoxel_move set 1', 'aqua', flying), dict(text=' '),
                 button('[行走]', '/trigger streamvoxel_move set 2', 'aqua', not flying)])
    rows.append([dict(text=''), button('[回起点]', '/trigger streamvoxel_home set 1', 'green'),
                 dict(text='  传送到初始出生点（reset 后也会自动回去）', color='dark_gray')])
    rows.append([dict(text=''), dict(text='World: ', color='gray'), button('[上一个]', '/trigger streamvoxel_world set 1', 'aqua'),
                 dict(text=' '), button('[下一个]', '/trigger streamvoxel_world set 2', 'aqua'),
                 dict(text=' (或 /trigger streamvoxel_world set <id>) '), button('[清空]', '/trigger streamvoxel_reset set 1', 'red')])
    return rows


def telemetry_commands(packet):
    commands = ['bossbar set streamvoxel:cfg visible false']
    for gpu in packet['gpus']:
        label = gpu.get('index', gpu.get('rank'))
        limit = max(float(gpu.get('power_limit_w') or 1), 1)
        value = int(round(min(100, max(0, float(gpu['power_w']) / limit * 100))))
        bar = f'streamvoxel:gpu{gpu["rank"]}'
        text = f'GPU{label} {float(gpu["power_w"]):.0f}W'
        commands += [f'bossbar set {bar} name ' + json.dumps(dict(text=text)),
                     f'bossbar set {bar} value {value}', f'bossbar set {bar} players @a',
                     f'bossbar set {bar} visible true']
    return commands


def setup_ui(rcon):
    for rule in ('send_command_feedback', 'command_block_output', 'log_admin_commands'):
        rcon.command(f'gamerule {rule} false')   # no "Triggered [...]" echo lingering in the chat corner
    for key in KEYS:
        rcon.command(f'scoreboard objectives add streamvoxel_{key} trigger')
    rcon.command('scoreboard objectives add streamvoxel_cfg dummy')
    for bar in ('gpu0', 'gpu1', 'cfg'):
        rcon.command(f'bossbar add streamvoxel:{bar} ' + json.dumps(dict(text=bar)))
        rcon.command(f'bossbar set streamvoxel:{bar} max 100')
    rcon.command('bossbar set streamvoxel:gpu0 color green')
    rcon.command('bossbar set streamvoxel:gpu1 color blue')
    rcon.command('bossbar set streamvoxel:cfg color yellow')
    rcon.command('bossbar set streamvoxel:cfg visible false')
    for key in KEYS + ('any',):
        rcon.command(f'scoreboard players set #{key} streamvoxel_cfg 0')


def install_tick(rcon, world):
    """Server-side trigger handling, run every game tick by the datapack.

    A trigger objective only accepts `/trigger` while it is enabled for the player, and a click disables it. Re-enabling
    from the bridge's poll loop left a window (poll period, RCON queueing behind placements) in which menu clicks were
    refused. Here the server re-enables every objective each tick and parks the clicked value in `#<key> streamvoxel_cfg`;
    the bridge then needs one query (`#any`) per poll instead of one per objective.
    """
    data = world / 'datapacks/streamvoxel_live/data'
    (data / 'streamvoxel_live/function').mkdir(parents=True, exist_ok=True)
    (data / 'minecraft/tags/function').mkdir(parents=True, exist_ok=True)
    lines = []
    for key in KEYS:
        hit = f'@a[scores={{streamvoxel_{key}=1..}}]'
        lines += [f'execute as {hit} run scoreboard players operation #{key} streamvoxel_cfg = @s streamvoxel_{key}',
                  f'execute if entity {hit} run scoreboard players set #any streamvoxel_cfg 1',
                  f'scoreboard players set {hit} streamvoxel_{key} 0',
                  f'scoreboard players enable @a streamvoxel_{key}']
    (data / 'streamvoxel_live/function/tick.mcfunction').write_text('\n'.join(lines) + '\n')
    (data / 'minecraft/tags/function/tick.json').write_text(json.dumps(dict(values=['streamvoxel_live:tick'])))
    rcon.command('reload')
    for _ in range(100):
        # The reload finishes on the server thread; the function exists once it is back.
        if 'Unknown function' not in rcon.request(2, 'function streamvoxel_live:tick'):
            return
        time.sleep(.1)
    raise RuntimeError('streamvoxel_live:tick did not load')


def take_triggers(rcon):
    """{key: value} for every trigger clicked since the last call."""
    if 'passed' not in rcon.command('execute if score #any streamvoxel_cfg matches 1..'):
        return {}
    rcon.command('scoreboard players set #any streamvoxel_cfg 0')
    got = {}
    for key in KEYS:
        match = re.search(r'has (-?\d+)', rcon.command(f'scoreboard players get #{key} streamvoxel_cfg'))
        if match and int(match.group(1)) > 0:
            got[key] = int(match.group(1))
            rcon.command(f'scoreboard players set #{key} streamvoxel_cfg 0')
    return got


def home_position(cfg):
    x, y, z = json.loads((Path(cfg['local_root']) / 'export.json').read_text())['spawn_xyz']
    return x + .5, y, z + .5


def go_home(rcon, cfg):
    x, y, z = home_position(cfg)
    rcon.command(f'execute in minecraft:overworld run tp @a {x} {y} {z}')


def run_chunks(rcon, runs):
    chunk = []
    for run in runs:
        trial = chunk + [run]
        command = 'data modify storage streamvoxel_live:frame runs set value ' + json.dumps(trial, separators=(',', ':'))
        if len(command.encode()) > 1200 and chunk:
            rcon.command('data modify storage streamvoxel_live:frame runs set value ' + json.dumps(chunk, separators=(',', ':')))
            rcon.command('function streamvoxel_live:loop')
            chunk = [run]
        else:
            chunk = trial
    if chunk:
        rcon.command('data modify storage streamvoxel_live:frame runs set value ' + json.dumps(chunk, separators=(',', ':')))
        rcon.command('function streamvoxel_live:loop')


def bank_cells(cfg):
    return [(item['z'] // 16, item['x'] // 16) for item in cfg.get('air_bank_setup', [])]


def force_add(rcon, cells, cfg, deadline=30):
    """Add force-load tickets without dropping tickets held for other live windows."""
    wanted = list(cells) + bank_cells(cfg)
    for z, x in wanted:
        rcon.command(f'forceload add {x * 16} {z * 16}')
    limit = time.monotonic() + deadline
    for z, x in wanted:
        while 'time is' not in rcon.command(f'execute if loaded {x * 16} 0 {z * 16} run time query gametime'):
            if time.monotonic() > limit:
                raise TimeoutError('Native target chunks not loaded')
            time.sleep(.05)


def setup_bank(rcon, cfg):
    setup = cfg.get('air_bank_setup', [])
    if not setup:
        return
    rcon.command('forceload remove all')
    force_add(rcon, [], cfg)
    for block in sorted(setup, key=lambda item: 'air' in item['state']):
        for _ in range(20):
            rcon.command(f'setblock {block["x"]} {block["y"]} {block["z"]} {block["state"]}')
            if 'passed' in rcon.command(f'execute if block {block["x"]} {block["y"]} {block["z"]} {block["state"]}'):
                break
            time.sleep(.25)
        else:
            raise RuntimeError(f'Air bank block not placed: {block}')


def clear_cell(rcon, z, x):
    for y0, y1 in ((-64, 63), (64, 191)):
        rcon.command(f'fill {x * 16} {y0} {z * 16} {x * 16 + 15} {y1} {z * 16 + 15} air')


class Codec:
    """What the viewer needs to expand packed rows: the block-state registry and the session's air bank."""

    def __init__(self, cfg):
        self.states = load_registry()['idToState']
        self.special = special_table(self.states)
        # Known before any packet, so a reconnect that resumes after the session packet can still expand rows.
        self.bank = {str(self.states.index(entry['state'])): [entry['x'], entry['y'], entry['z']]
                     for entry in cfg['air_bank_setup'] if entry['state'] in ('minecraft:cave_air', 'minecraft:void_air')}
        self.initial = set()

    def session(self, packet):
        if packet.get('states_sha256') != states_digest(self.states):
            raise RuntimeError('Session block-state registry differs from the viewer registry')
        bank = {str(k): list(v) for k, v in packet['air_bank'].items()}
        if bank != self.bank:
            raise RuntimeError(f'Session air bank {bank} != viewer air bank {self.bank}')
        self.initial = {tuple(cell) for cell in packet['initial_cells']}


def apply_rows(rcon, world, session, seq, rows, codec):
    folder = world / 'generated/streamvoxel_live/structure'
    folder.mkdir(parents=True, exist_ok=True)
    for row in rows:
        z, x = row['target']
        name = f'{session}/p{seq:08d}r{row["row"]}'
        with span('expand'):
            template, runs = expand_row(row, codec.states, codec.special, codec.bank)
        if template:
            path = folder / (name + '.nbt')
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix('.tmp')
            with span('write'):
                tmp.write_bytes(template)
                tmp.replace(path)
            rcon.command(f'place template streamvoxel_live:{name} {x * 16} -64 {z * 16} none none 1 0 strict')
            path.unlink()
        run_chunks(rcon, runs)


def apply_packet(rcon, world, packet, session, cfg, codec, hud=''):
    """`hud` is appended to the action bar after a frame; None skips the action bar (not the last frame of a tick)."""
    kind = packet['kind']
    if kind == 'session':
        exported = json.loads((Path(cfg['local_root']) / 'export.json').read_text())['initial_box_zx']
        if list(packet['initial_box_zx']) != list(exported):
            raise RuntimeError(f'Session initial box {packet["initial_box_zx"]} != exported world box {exported}: '
                               'known cells outside the session box would be regenerated')
        codec.session(packet)
        return
    if kind == 'window_start':
        z, x = packet['target']
        force_add(rcon, [(z, x)], cfg)
        if (z, x) not in codec.initial:
            clear_cell(rcon, z, x)
        return
    if kind == 'reset':
        rcon.command('tellraw @a ' + json.dumps(dict(text=f'Clearing {len(packet["clear"])} windows, loading World {packet["world"]}...', color='gold')))
        rcon.command('forceload remove all')
        cells = [tuple(cell) for cell in packet['clear']]
        for start in range(0, len(cells), 24):
            group = cells[start:start + 24]
            force_add(rcon, group, cfg)
            for z, x in group:
                clear_cell(rcon, z, x)
        frames = packet['frames']
        force_add(rcon, [tuple(row['target']) for row in frames], cfg)
        apply_rows(rcon, world, session, packet['seq'], frames, codec)
        rcon.command('save-all flush')
        go_home(rcon, cfg)
        rcon.command('tellraw @a ' + json.dumps(dict(text=f'World {packet["world"]} ready，已传送回起点（/trigger streamvoxel_home 随时回来，/trigger streamvoxel_menu 重开菜单）', color='green')))
        return
    if kind == 'frame':
        apply_rows(rcon, world, session, packet['seq'], packet['frames'], codec)
        if hud is not None:
            rcon.command('title @a actionbar ' + json.dumps(dict(
                text=f'windows {packet.get("committed", 0)} | in flight {packet.get("inflight", 0)}/{packet.get("slots", 0)} | cap {packet.get("cap", "?")}{hud}',
                color='aqua')))
    elif kind == 'window_commit':
        rcon.command('save-all flush')
        z, x = packet['target']
        if (z, x) not in codec.initial:
            rcon.command(f'forceload remove {x * 16} {z * 16}')
    elif kind == 'complete':
        rcon.command('tellraw @a ' + json.dumps(dict(text=f'Live session ended: {packet["new_windows"]} windows', color='green')))
        rcon.command('save-all flush')
    elif kind == 'failed':
        raise RuntimeError(packet)


def load_spool(path, played):
    """Packets of the local recording after `played`, plus the last sequence number it holds.

    The recording is the transport lines exactly as received; a torn last line (crash mid-write) is cut off.
    """
    packets, last, good = [], played, 0
    if not path.exists():
        return packets, last
    data = path.read_bytes()
    for line in data.splitlines(keepends=True):
        try:
            message = json.loads(line)
            batch = unbatch(message) if message.get('kind') == 'batch' else [message]
        except (ValueError, zlib.error):
            break
        if not line.endswith(b'\n'):
            break
        good += len(line)
        for packet in batch:
            if packet['seq'] > last:
                packets.append(verify_packet(packet, last + 1))
                last = packet['seq']
    if good < len(data):
        with path.open('r+b') as handle:
            handle.truncate(good)
    return packets, last


def bridge(cfg, replay=None, speed=1., delay=3.):
    """Live: ssh transport -> receiver thread -> local recording + play buffer -> steady playback on the Java server.

    `replay` is a recording (transport lines) played from its first packet into a freshly started world.
    """
    global PROF
    root = Path(cfg['local_root'])
    PROF = Profile(root / 'bridge_profile.jsonl')
    world = root / 'world'
    journal = root / 'delivery.jsonl'
    ack_path = root / 'delivery_state.json'
    spool_path = root / 'recording.jsonl'
    ack = json.loads(ack_path.read_text())['ack'] if ack_path.exists() else -1   # last packet played on the server
    buffer = PlayBuffer(delay=delay, speed=speed)
    if replay:
        replayed, received = load_spool(Path(replay), -1)
        for packet in replayed:
            buffer.push(packet, time.monotonic())
        ack = -1
    else:
        pending, received = load_spool(spool_path, ack)
        for packet in pending:
            buffer.push(packet, time.monotonic())
    connection = json.loads((root / 'connection.json').read_text())
    fallback = cfg['fallback_player_zx']
    rcon = Rcon(connection['rcon_port'], connection['rcon_password'])
    if 'streamvoxel_live' not in rcon.command('datapack list enabled'):
        rcon.command('datapack enable "file/streamvoxel_live"')
    setup_bank(rcon, cfg)
    setup_ui(rcon)
    install_tick(rcon, world)
    settings_path = root / 'settings.json'
    reset_path = root / 'reset.json'
    reset = json.loads(reset_path.read_text()) if reset_path.exists() else dict(seq=0, world_req=None)
    settings = clamp_settings(json.loads(settings_path.read_text()) if settings_path.exists() else {}, dict(DEFAULTS, **cfg.get('defaults', {})))
    ssh = None   # the live transport process; replaced by the receiver thread on every reconnect
    stopped = threading.Event()
    errors = []
    telemetry = []   # newest telemetry line from the receiver, taken by the player
    state = dict(ack=ack, ready=True, reset_seq=reset['seq'], world_req=reset['world_req'], **settings, **player_position(rcon, fallback))
    lock = threading.Lock()
    codec = Codec(cfg)
    active_path = root / 'active_cells.json'

    def feedback():
        nonlocal reset
        poll = None
        try:
            poll = Rcon(connection['rcon_port'], connection['rcon_password'], tag='poll')
            current = dict(settings)
            greeted = False
            position = player_position(poll, fallback)
            tick = 0
            while not stopped.is_set():
                if tick % 3 == 0:
                    position = player_position(poll, fallback)
                got = take_triggers(poll)
                if got:
                    changed = False
                    value = got.get('steps')
                    if value is not None:
                        current['steps'] = value
                        changed = True
                    value = got.get('batch')
                    if value is not None:
                        current['batch'] = value
                        changed = True
                    value = got.get('speed')
                    if value is not None:
                        current['speed'] = value
                        apply_speed(poll, root, min(max(value, 1), 100))
                        changed = True
                    value = got.get('mode')
                    if value is not None:
                        current['mode'] = 'dynamic' if value == 1 else 'fixed'
                        changed = True
                    value = got.get('run')
                    if value is not None:
                        # 1 = pause, 2 = run. Zero is an illegal trigger value on this client.
                        current['paused'] = value != 2
                        changed = True
                    value = got.get('move')
                    if value is not None:
                        current['move'] = 'walk' if value == 2 else 'fly'
                        if current['move'] == 'walk':
                            poll.command('gamemode adventure @a')
                        else:
                            poll.command('gamemode spectator @a')
                        changed = True
                    if changed:
                        move = current.get('move', 'fly')
                        current = clamp_settings(current, current)
                        current['move'] = move
                        atomic_json(settings_path, current)
                        poll.command('tellraw @a ' + json.dumps(dict(
                            text=f'cap {current["steps"]} | batch {current["batch"]} | 速度 {current["speed"]}% | {current["mode"]} | '
                                 f'{"暂停" if current["paused"] else "运行"} | {"行走" if current["move"] == "walk" else "飞行"}',
                            color='gold')))
                    for key in ('reset', 'world'):
                        triggered = got.get(key)
                        if triggered is not None:
                            reset = dict(seq=reset['seq'] + 1, world_req=triggered if key == 'world' else None)
                            atomic_json(reset_path, reset)
                            poll.command('tellraw @a ' + json.dumps(dict(text='Reset requested; in-flight windows are dropped', color='gold')))
                    if got.get('home') is not None:
                        go_home(poll, cfg)
                menu = got.get('menu') is not None
                if menu or (position['player_present'] and not greeted):
                    if position['player_present']:
                        apply_speed(poll, root, current.get('speed', 100))
                    for row in menu_rows(current):
                        poll.command('tellraw @a ' + json.dumps(row))
                    greeted = greeted or position['player_present']
                tick += 1
                with lock:
                    state.update(position, reset_seq=reset['seq'], world_req=reset['world_req'], **current)
                    message = dict(kind='control', **state)
                if (tick - 1) % 3 and not got:
                    stopped.wait(.1)
                    continue
                link = ssh
                if link:
                    try:
                        link.stdin.write(json.dumps(message) + '\n')
                        link.stdin.flush()
                    except (OSError, ValueError):
                        pass   # link is being re-established; the next control line carries the same state
                stopped.wait(.1)
        except BaseException as error:
            errors.append(str(error))
            stopped.set()
        finally:
            if poll:
                poll.close()

    thread = threading.Thread(target=feedback, daemon=True)
    thread.start()
    buffer_lock = threading.Lock()
    ended = threading.Event()

    def receive():
        """Read transport lines as they arrive: verify, record verbatim, queue. Reconnects by itself, so a dropped
        link only drains the play buffer instead of stopping playback."""
        nonlocal received, ssh
        attempt = 0
        spool = spool_path.open('a')
        while not (stopped.is_set() or ended.is_set()):
            ssh = subprocess.Popen(
                ['ssh', '-o', 'Compression=yes', '-o', 'ConnectTimeout=20', '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=3',
                 cfg['ssh'], f'cd {shlex.quote(cfg["remote_source"])} && {shlex.quote(cfg["remote_python"])} -u -m viewer.transport '
                              f'--run {shlex.quote(cfg["remote_run"])} --after {received}'],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
            began = time.monotonic()
            try:
                idle = time.perf_counter()
                for line in ssh.stdout:
                    PROF.add('wait:ssh_line', time.perf_counter() - idle)
                    with span('decode'):
                        message = json.loads(line)
                        if message.get('kind') == 'telemetry':
                            telemetry[:] = [message]
                            idle = time.perf_counter()
                            continue
                        packets = unbatch(message) if message.get('kind') == 'batch' else [message]
                        for offset, packet in enumerate(packets):
                            verify_packet(packet, received + 1 + offset)
                    spool.write(line)
                    spool.flush()
                    arrived, wall = time.monotonic(), time.time_ns()
                    PROF.add('batch:packets', len(packets) / 1e3)
                    PROF.add('batch:kbytes', len(line) / 1e6)
                    if packets[0].get('emitted_unix_ns'):
                        PROF.add('lag:emit_to_recv', (wall - packets[0]['emitted_unix_ns']) / 1e9)
                    with buffer_lock:
                        for packet in packets:
                            buffer.push(packet, arrived)
                        received = packets[-1]['seq']
                    if packets[-1]['kind'] in ('complete', 'failed'):
                        ended.set()
                        return
                    idle = time.perf_counter()
            except (ValueError, zlib.error) as error:
                errors.append(f'{type(error).__name__}: {error}')   # corrupt data is not a link drop: stop loudly
                return
            finally:
                if ssh.poll() is None:
                    ssh.terminate()
            status = ssh.wait()
            atomic_json(root / f'bridge_error_{attempt % 10:02d}.json',
                        dict(type='TransportClosed', message=f'status={status}', attempt=attempt + 1, unix=time.time()))
            attempt = 0 if time.monotonic() - began > 60 else attempt + 1
            stopped.wait(min(15, 2 ** min(attempt, 4)))

    if not replay:
        threading.Thread(target=receive, daemon=True).start()
    active = json.loads(active_path.read_text()) if active_path.exists() else []
    shown, shown_message = 0., None
    try:
        while True:
            if errors:
                raise RuntimeError(errors)
            now = time.monotonic()
            with buffer_lock:
                due = buffer.due(now)
                ahead = buffer.buffered()
            if telemetry and telemetry[0] is not shown_message and now - shown >= .5:
                shown_message = telemetry[0]
                with span('telemetry'):
                    for text in telemetry_commands(shown_message):
                        rcon.command(text)
                shown = time.monotonic()
            if not due:
                PROF.tick()
                time.sleep(.005)
                continue
            PROF.add('buffer_s', ahead)
            PROF.add('group:packets', len(due) / 1e3)
            with span('coalesce'):
                group = coalesce(due) if len(due) > 1 else due
            last_frame = max((i for i, packet in enumerate(group) if packet['kind'] == 'frame'), default=-1)
            hud = f' | buffer {ahead:.1f}s' + (f' | x{speed:g}' if speed != 1 else '')
            for index, packet in enumerate(group):
                began = time.monotonic()
                with span('apply:' + packet['kind']):
                    apply_packet(rcon, world, packet, cfg['session'], cfg, codec, hud if index == last_frame else None)
                took = time.monotonic() - began
                if took >= PROF.slow:
                    PROF.event('slow', seq=packet['seq'], packet=packet['kind'], seconds=round(took, 3),
                               rows=len(packet.get('frames') or ()), voxels=sum(len(row.get('sparse', '')) for row in packet.get('frames') or ()))
                bookkeeping = time.perf_counter()
                if packet['kind'] == 'window_start':
                    active.append(packet['target'])
                    atomic_json(active_path, active)
                elif packet['kind'] == 'window_commit':
                    active = [cell for cell in active if cell != packet['target']]
                    atomic_json(active_path, active)
                elif packet['kind'] in ('reset', 'complete'):
                    active = []
                    atomic_json(active_path, active)
                ack = packet['seq']
                shown_ns = time.time_ns()
                with journal.open('a') as handle:
                    handle.write(json.dumps(dict(seq=ack, kind=packet['kind'], sha256=packet['sha256'], emitted_unix_ns=packet.get('emitted_unix_ns'),
                                                 displayed_unix_ns=shown_ns, apply_seconds=took, group=len(due), buffer_s=round(ahead, 3))) + '\n')
                atomic_json(ack_path, dict(ack=ack, packet_sha256=packet['sha256'], state=packet['kind']))
                with lock:
                    state['ack'] = ack
                PROF.add('book', time.perf_counter() - bookkeeping)
                if packet.get('emitted_unix_ns'):
                    PROF.add('lag:emit_to_shown', (shown_ns - packet['emitted_unix_ns']) / 1e9)
                if packet['kind'] == 'complete' and not replay:
                    return
            PROF.tick()
    finally:
        stopped.set()
        thread.join(timeout=3)
        rcon.close()
        if ssh and ssh.poll() is None:
            ssh.terminate()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--replay', type=Path, help='play a recording (recording.jsonl or `record` output) from its first packet')
    parser.add_argument('--speed', type=float, default=1., help='replay speed')
    parser.add_argument('--delay', type=float, default=3., help='seconds of material buffered before playback starts')
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    root = Path(cfg['local_root'])
    # The link to the GPU box is a SOCKS-proxied SSH that drops for tens of seconds at a time. Delivery resumes from
    # the local ack, so never give up: back off to at most 15 s between attempts and keep one error file per attempt
    # (cycling) so the cause of the last drops stays inspectable. A clean `complete` packet returns normally.
    attempt = 0
    while True:
        began = time.monotonic()
        try:
            bridge(cfg, args.replay, args.speed, args.delay)
            return
        except (OSError, EOFError, RuntimeError, TimeoutError) as error:
            atomic_json(root / f'bridge_error_{attempt % 10:02d}.json',
                        dict(type=type(error).__name__, message=str(error), attempt=attempt + 1, unix=time.time()))
            # A run that lasted a while was a working connection that dropped: restart the backoff.
            attempt = 0 if time.monotonic() - began > 60 else attempt + 1
            time.sleep(min(15, 2 ** min(attempt, 4)))


if __name__ == '__main__':
    main()
