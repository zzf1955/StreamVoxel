import json
import numpy as np
import pytest
from viewer.bridge import menu_rows, player_position, run_chunks, take_triggers, telemetry_commands
from viewer.transport import telemetry_line


class FakeRcon:
    def __init__(self, pending=None):
        self.commands = []
        self.pending = dict(pending or {})

    def command(self, text):
        self.commands.append(text)
        if text.startswith('data get'):
            return 'Player has the following entity data: [12.5d, 83.0d, -91.25d]'
        if text.startswith('execute if score #any'):
            return 'Test passed' if self.pending else 'Test failed'
        for key in ('steps', 'batch', 'menu', 'home'):
            if text.startswith(f'scoreboard players get #{key} '):
                return f'#{key} has {self.pending.pop(key, 0)} [streamvoxel_cfg]'
        return 'ok'


def test_shared_macro_commands_are_bounded_and_lossless():
    rcon = FakeRcon()
    runs = [dict(x=i, ex=i, y=70, z=12, state='minecraft:water[level=0]') for i in range(300)]
    run_chunks(rcon, runs)
    sets = [c for c in rcon.commands if 'set value ' in c]
    assert all(len(c.encode()) <= 1200 for c in sets)
    restored = []
    for c in sets:
        restored += json.loads(c.split('set value ', 1)[1])
    assert restored == runs
    assert rcon.commands.count('function streamvoxel_live:loop') == len(sets)
    assert player_position(rcon, [0, 0]) == dict(player_present=True, player_zx=[-91.25, 12.5], player_y=83.)


def test_player_position_falls_back_without_a_player():
    class Empty:
        def command(self, text):
            return 'No entity was found'
    assert player_position(Empty(), [3, 4]) == dict(player_present=False, player_zx=[3, 4])


def test_menu_fits_rcon_and_marks_current_choice():
    rows = menu_rows(dict(steps=32, batch=8, speed=100))
    assert all(len(json.dumps(r).encode()) + len('tellraw @a ') <= 1400 for r in rows)
    buttons = [c for r in rows for c in r if 'click_event' in c]
    commands = {c['click_event']['command'] for c in buttons}
    assert {'/trigger streamvoxel_steps set 32', '/trigger streamvoxel_batch set 16', '/trigger streamvoxel_speed set 15'} <= commands
    assert {c['text'] for c in buttons if c.get('bold')} >= {'[32]', '[8]', '[100%]'}


def test_telemetry_commands_fit_rcon():
    gpu = lambda r: dict(rank=r, power_w=312.2, power_limit_w=400.)
    commands = telemetry_commands(dict(gpus=[gpu(0), gpu(1)]))
    assert any('GPU1 312W' in c for c in commands) and 'bossbar set streamvoxel:gpu1 value 78' in commands
    assert all(len(c.encode()) <= 1400 for c in commands)


def test_trigger_roundtrip_collects_clicked_values_and_clears_them():
    rcon = FakeRcon(dict(steps=16, home=1))
    assert take_triggers(rcon) == dict(steps=16, home=1)
    assert 'scoreboard players set #steps streamvoxel_cfg 0' in rcon.commands
    assert 'scoreboard players set #any streamvoxel_cfg 0' in rcon.commands
    assert take_triggers(FakeRcon()) == {}


def test_menu_has_a_home_button():
    buttons = [c for r in menu_rows(dict()) for c in r if 'click_event' in c]
    assert '/trigger streamvoxel_home set 1' in {c['click_event']['command'] for c in buttons}


def test_transport_telemetry_line_only_when_new_and_carries_applied(tmp_path):
    assert telemetry_line(tmp_path, None) == (None, None)
    (tmp_path / 'telemetry_r0.json').write_text(json.dumps(dict(rank=0, unix_ns=5, util=1.)))
    (tmp_path / 'applied_settings.json').write_text(json.dumps(dict(steps=32, batch=4, gpus=1)))
    line, stamp = telemetry_line(tmp_path, None)
    assert line['kind'] == 'telemetry' and line['applied']['steps'] == 32 and stamp == [5]
    assert telemetry_line(tmp_path, stamp) == (None, stamp)


def test_packet_order_hash_and_base64(tmp_path):
    from streamvoxel.common.live_wire import Publisher, decoded, encoded, verify_packet
    publisher = Publisher(tmp_path)
    publisher.publish('frame', payload=encoded(b'\x00\xffabc'))
    packet = json.loads((tmp_path / 'wire/00000000.json').read_text())
    assert decoded(verify_packet(packet, 0)['payload']) == b'\x00\xffabc'
    with pytest.raises(ValueError):
        verify_packet(packet, 1)
    packet['kind'] = 'forged'
    with pytest.raises(ValueError):
        verify_packet(packet, 0)


def test_settings_are_clamped():
    from streamvoxel.common.live_wire import clamp_settings
    defaults = dict(steps=32, batch=4)
    assert clamp_settings({}, defaults) == defaults
    assert clamp_settings(dict(steps=0, batch=9999), defaults) == dict(steps=1, batch=64)
    assert clamp_settings(dict(steps='x', batch=None), defaults) == defaults


def test_export_crops_ground_truth_to_the_initial_box(tmp_path, monkeypatch):
    import viewer.world_export as export
    states = ['minecraft:air', 'minecraft:stone', 'minecraft:dirt']
    monkeypatch.setattr(export, 'load_registry', lambda: dict(idToState=states + ['minecraft:cave_air', 'minecraft:void_air']))
    monkeypatch.setattr(export, 'preview_lookup', lambda space, vocab: np.arange(3, dtype=np.uint16))
    panel = tmp_path / 'panel'
    (panel / 'case00').mkdir(parents=True)
    case = dict(world='w0', initial_box_zx=[0, 0, 2, 2])
    (panel / 'manifest.json').write_text(json.dumps(dict(cases=[case])))
    (panel / 'vocabulary.json').write_text('{}')
    np.save(panel / 'case00/gt_global.npy', np.ones((256, 2, 2), np.uint16))
    seen = {}

    def fake_run(cmd, env, check):
        seen['cmd'] = cmd
        (tmp_path / 'out' / 'world').mkdir()
        (tmp_path / 'out' / 'world' / 'level.dat').write_bytes(b'x')

    monkeypatch.setattr(export.subprocess, 'run', fake_run)
    export.build(panel, tmp_path / 'out', tmp_path / 'tpl', 'node', tmp_path / 'nbt')
    out = tmp_path / 'out'
    blocks = np.load(out / 'blocks.npy')
    assert blocks.shape == (256, 256, 544) and blocks[0, 0, 0] == 1 and blocks[64, 0, 516] == 1
    assert (out / 'world/datapacks/streamvoxel_live/data/streamvoxel_live/function/loop.mcfunction').exists()
    report = json.loads((out / 'export.json').read_text())
    assert report['spawn_xyz'] == [1, 140, 1] and set(report['air_bank']) == {'3', '4'}
    assert (out / 'COMPLETE').read_text().strip() == export.digest(out / 'export.json')
