"""Compact frame rows for the low-entropy-last wire, and their expansion into native placements on the viewer.

A row carries only the visible (non-air) voxels a frame committed in one cell: sorted positions as uint16
deltas followed by uint16 global block ids, xz-compressed (preset 2: 7% smaller than zlib
at ~2 ms a row; preset 6 saves 17% but costs ~13 ms; the positions of a random unmask order are most of the bytes) and base64-encoded. Air needs no placement
because the viewer clears a cell when its window starts. The viewer rebuilds the structure template and
the strict commands for property-sensitive states from the same block-state registry (`states_digest`).
"""
from __future__ import annotations

import gzip
import hashlib
import json
import struct
import lzma

import numpy as np

from streamvoxel.common.live_wire import decoded, encoded, packet_sha


DATA_VERSION = 4903
# One NBT compound per block: pos (list of 3 ints), state (int), end tag. 36 bytes, big-endian.
_BLOCK = np.dtype([('pos_tag', 'S7'), ('count', '>i4'), ('x', '>i4'), ('y', '>i4'), ('z', '>i4'),
                   ('state_tag', 'S8'), ('state', '>i4'), ('end', 'u1')])
_AIRS = ('minecraft:cave_air', 'minecraft:void_air')


def states_digest(states):
    return hashlib.sha256(json.dumps(list(states), ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def special_table(states):
    """True for states that never enter a template: property-sensitive ones and the special airs."""
    return np.asarray(['[' in state or state in _AIRS for state in states], bool)


def _sparse_sha(positions, raw):
    return hashlib.sha256(positions.astype('<u2').tobytes() + raw.astype('<u2').tobytes()).hexdigest()


def pack_row(target, positions, raw):
    """`positions` are flat YZX indices in one [256,16,16] cell, `raw` their global ids."""
    positions = np.asarray(positions, np.int64)
    raw = np.asarray(raw, np.uint16)
    keep = raw != 0
    order = np.argsort(positions[keep], kind='stable')
    pos = positions[keep][order].astype(np.uint16)
    val = raw[keep][order]
    body = np.diff(pos, prepend=np.uint16(0)).astype('<u2').tobytes() + val.astype('<u2').tobytes()
    return dict(target=[int(v) for v in target], committed=int(len(raw)), visible=int(len(pos)),
                sparse=encoded(lzma.compress(body, preset=2)), sparse_sha256=_sparse_sha(pos, val))


def unpack_row(row):
    body = lzma.decompress(decoded(row['sparse']))
    count = int(row['visible'])
    if len(body) != 4 * count:
        raise ValueError('Sparse row length mismatch')
    pos = np.cumsum(np.frombuffer(body[:2 * count], '<u2'), dtype=np.int64).astype(np.uint16)
    val = np.frombuffer(body[2 * count:], '<u2').astype(np.uint16)
    if _sparse_sha(pos, val) != row['sparse_sha256']:
        raise ValueError('Sparse row hash mismatch')
    return pos.astype(np.int32), val


def merge_rows(rows):
    """One row for several rows of the same cell and window, in order; their voxel sets are disjoint."""
    if len(rows) == 1:
        return rows[0]
    parts = [unpack_row(row) for row in rows]
    merged = pack_row(rows[-1]['target'], np.concatenate([p for p, _ in parts]), np.concatenate([v for _, v in parts]))
    merged['committed'] = sum(int(row['committed']) for row in rows)
    return dict(rows[-1], **merged)


def coalesce(packets):
    """Consecutive wire packets with each cell's frame rows merged into its last frame packet.

    A row may only move later, past nothing that touches its cell: rows of one cell belong to one window,
    which starts with `window_start` and ends after its last frame, and `reset`/`session` packets are never
    crossed. Merged frame packets keep their sequence numbers and are re-hashed.
    """
    era, window, last, rows = 0, {}, {}, {}
    keys = []
    for index, packet in enumerate(packets):
        if packet['kind'] in ('reset', 'session'):
            era += 1
        elif packet['kind'] == 'window_start':
            window[tuple(packet['target'])] = window.get(tuple(packet['target']), 0) + 1
        elif packet['kind'] == 'frame':
            mine = []
            for row in packet['frames']:
                key = (era, window.get(tuple(row['target']), 0), tuple(row['target']))
                rows.setdefault(key, []).append(row)
                last[key] = index
                mine.append(key)
            keys.append(mine)
            continue
        keys.append(None)
    out = []
    for index, packet in enumerate(packets):
        if packet['kind'] != 'frame' or all(last[key] == index and len(rows[key]) == 1 for key in keys[index]):
            out.append(packet)
            continue
        if packet_sha(packet) != packet['sha256']:
            raise ValueError(f'Frame packet {packet["seq"]} hash mismatch before coalescing')
        packet = dict(packet, frames=[merge_rows(rows[key]) for key in keys[index] if last[key] == index], coalesced=True)
        packet.pop('sha256')
        packet['sha256'] = packet_sha(packet)
        out.append(packet)
    return out


def _string(text):
    value = text.encode('utf8')
    return struct.pack('>H', len(value)) + value


def _tag(kind, name, payload):
    return bytes([kind]) + _string(name) + payload


def structure_nbt(positions, raw, states):
    """Uncompressed structure NBT, byte-identical to `frontier_structure_replay.structure_bytes` before gzip."""
    positions = np.asarray(positions, np.int64)
    unique, ranks = np.unique(np.asarray(raw), return_inverse=True)
    palette = []
    for idx in unique:
        name, sep, props = states[int(idx)].partition('[')
        entry = _tag(8, 'Name', _string(name))
        if sep:
            entry += _tag(10, 'Properties', b''.join(_tag(8, k, _string(v)) for k, v in (p.split('=') for p in props[:-1].split(','))) + b'\0')
        palette.append(entry + b'\0')
    blocks = np.zeros(len(positions), _BLOCK)
    blocks['pos_tag'] = b'\x09\x00\x03pos\x03'
    blocks['count'] = 3
    blocks['x'], blocks['y'], blocks['z'] = positions % 16, positions // 256, (positions // 16) % 16
    blocks['state_tag'] = b'\x03\x00\x05state'
    blocks['state'] = ranks.reshape(-1)
    root = (_tag(3, 'DataVersion', struct.pack('>i', DATA_VERSION)) +
            _tag(9, 'size', b'\x03' + struct.pack('>i3i', 3, 16, 256, 16)) +
            _tag(9, 'palette', b'\x0a' + struct.pack('>i', len(palette)) + b''.join(palette)) +
            _tag(9, 'blocks', b'\x0a' + struct.pack('>i', len(blocks)) + blocks.tobytes()) +
            _tag(9, 'entities', b'\x0a' + struct.pack('>i', 0)) + b'\0')
    return b'\x0a\x00\x00' + root


def special_runs(positions, raw, states, target, bank):
    """Strict commands for property-sensitive states, x-runs of one state within a row; same as `preview_delta`."""
    runs = []
    z, x = target
    i = 0
    while i < len(positions):
        j = i + 1
        while (j < len(positions) and positions[j] == positions[j - 1] + 1 and positions[j] // 16 == positions[i] // 16
               and raw[j] == raw[i]):
            j += 1
        pos = int(positions[i])
        state = states[int(raw[i])]
        entry = dict(x=x * 16 + pos % 16, ex=x * 16 + int(positions[j - 1]) % 16, y=pos // 256 - 64, z=z * 16 + (pos // 16) % 16,
                     state=state)
        if state in _AIRS:
            # One source voxel per special air state, cloned only to fresh cells.
            sx, sy, sz = bank[str(int(raw[i]))]
            for xx in range(entry['x'], entry['ex'] + 1):
                runs.append(dict(x=xx, y=entry['y'], z=entry['z'], sx=sx, sy=sy, sz=sz))
        else:
            runs.append(entry)
        i = j
    return runs


def expand_row(row, states, special, bank):
    """(gzip template bytes or None, strict runs) for a packed row. `special` is `special_table(states)`."""
    positions, raw = unpack_row(row)
    mask = special[raw]
    simple = ~mask
    template = gzip.compress(structure_nbt(positions[simple], raw[simple], states), compresslevel=1, mtime=0) if simple.any() else None
    return template, special_runs(positions[mask], raw[mask], states, row['target'], bank)
