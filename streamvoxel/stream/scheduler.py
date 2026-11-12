"""Window admission for dynamic low-entropy-last batches.

A candidate must touch a committed window and not share an edge with any window that is still sampling.
By default that means Chebyshev distance >= 2. With `diagonal`, only the four edge neighbours are excluded
(Manhattan distance >= 2), so live windows may touch at a corner and a checkerboard of A/B windows emerges as
the frontier advances. Committed snapshots are the only condition source.
"""
from __future__ import annotations

import hashlib

import numpy as np

from streamvoxel.common.colored_frontier import ready_cells
from streamvoxel.stream.condition import Item

QUARTS = 64  # biome quarts per world side; labels repeat with this period, array has a one-quart halo


def tiled_biome_crop(biomes, z, x):
    """6x6 quart crop (one-quart halo) for cell (z, x), wrapping around the world's own labels.

    Identical to `biomes[:, 4z:4z+6, 4x:4x+6]` for in-world cells away from the edge."""
    zi = (np.arange(z * 4 - 1, z * 4 + 5) % QUARTS) + 1
    xi = (np.arange(x * 4 - 1, x * 4 + 5) % QUARTS) + 1
    return biomes[:, zi][:, :, xi]


def seed(world, z, x):
    return int.from_bytes(hashlib.sha256(f'player-frontier-v1:{world}:{z}:{x}'.encode()).digest()[:8],'little')%(2**63-1)


def chebyshev(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def separated(cell, others, gap=2, diagonal=False):
    if diagonal:
        return all(abs(cell[0] - other[0]) + abs(cell[1] - other[1]) >= 2 for other in others)
    return all(chebyshev(cell, other) >= gap for other in others)


def rank_share(batch, ranks, rank):
    """This rank's share of a global batch. Earlier ranks receive the remainder."""
    batch = max(0, int(batch))
    ranks = max(1, int(ranks))
    base, extra = divmod(batch, ranks)
    return base + (1 if int(rank) < extra else 0)


def plan_inserts(idle_indexes, active_count, batch, mode):
    """Which idle slots may start a window on this forward.

    `dynamic` refills any free slot while fewer than `batch` windows are active.
    `fixed` waits until every active window has finished, then starts up to `batch` together.
    """
    batch = max(0, int(batch))
    if mode == 'fixed':
        if active_count:
            return []
        return list(idle_indexes)[:batch]
    room = max(0, batch - int(active_count))
    return list(idle_indexes)[:room]


def pick_window(committed, inflight, player_zx, *, cell_size=16, diagonal=False):
    """Nearest ready cell that is not in flight and does not touch an in-flight cell (see `separated`). None when the frontier is blocked."""
    pz, px = player_zx
    best = None
    blocked = set(inflight)
    for z, x in ready_cells(committed):
        cell = (z, x)
        if cell in blocked or not separated(cell, blocked, diagonal=diagonal):
            continue
        distance = ((z + .5) * cell_size - pz) ** 2 + ((x + .5) * cell_size - px) ** 2
        key = (distance, z, x)
        if best is None or key < best[0]:
            best = (key, cell)
    return None if best is None else best[1]


def _relative_xz(coords, z, x):
    return np.asarray([[(b - x) * 16, (a - z) * 16] for a, b in coords], np.int64)


def biome_volume(biomes, z, x, spec):
    """Halo crop in the model's biome shape, centered in the 12x12 tensor `build_group` reads."""
    if spec.biome_halo == 4:
        crop = tiled_biome_crop(biomes, z, x)
    else:
        pad = spec.biome_halo // 4
        zi = (np.arange(z * 4 - pad, z * 4 + 4 + pad) % 64) + 1
        xi = (np.arange(x * 4 - pad, x * 4 + 4 + pad) % 64) + 1
        crop = biomes[:, zi][:, :, xi]
    if crop.shape != spec.biome_shape:
        raise ValueError(f'Biome crop {crop.shape} != {spec.biome_shape}')
    padded = np.zeros((spec.height // 4, 12, 12), np.int64)
    lo = (12 - crop.shape[1]) // 2
    padded[:, lo:lo + crop.shape[1], lo:lo + crop.shape[2]] = crop.astype(np.int64)
    return padded


def make_item(cells, target, biomes, spec, world, cap, l2_cap):
    """One `Item` in the target-relative frame. `cells` is the committed snapshot; in-flight tiles are absent."""
    z, x = target
    if target in cells:
        raise ValueError('Target already committed')
    near = sorted((a, b) for a in range(z - 1, z + 2) for b in range(x - 1, x + 2)
                  if (a, b) != target and (a, b) in cells)
    if not near:
        raise ValueError('Target has no committed neighbor')
    if len(near) > 8:
        raise ValueError('More than 8 L1 tiles')
    far = sorted(sorted(cells, key=lambda c: ((c[0] - z) ** 2 + (c[1] - x) ** 2, c))[:l2_cap])
    height, tile = spec.height, spec.tile
    l1 = np.stack([np.asarray(cells[c], np.int16) for c in near])
    l2 = np.stack([np.asarray(cells[c], np.int16) for c in far])
    if l1.shape != (len(near), height, tile, tile) or l2.shape != (len(far), height, tile, tile):
        raise ValueError('Condition tile has the wrong shape')
    return Item(
        id=f'{z}_{x}', seed=seed(world, z, x), steps=int(cap), temperature=1.0,
        l1_blocks=l1, l1_xz=_relative_xz(near, z, x), l2_blocks=l2, l2_xz=_relative_xz(far, z, x),
        biomes=biome_volume(biomes, z, x, spec), known=None)


def member_shares(batch, members):
    """Split a global batch over workers in proportion to weight (integer largest remainder).

    `members` is `[(wid, capacity, weight)]` in ascending wid. Returns `{wid: share}`. With equal capacities and weights
    this equals `rank_share` value for value (the remainder goes to the smaller wid); shares never exceed capacity
    while `batch <= sum(capacity)`.
    """
    batch = max(0, int(batch))
    if not members:
        return {}
    total = sum(weight for _, _, weight in members)
    numerator = {wid: batch * weight for wid, _, weight in members}
    share = {wid: numerator[wid] // total for wid, _, _ in members}
    rest = batch - sum(share.values())
    for wid, _, _ in sorted(members, key=lambda m: (-(numerator[m[0]] % total), m[0]))[:rest]:
        share[wid] += 1
    over = 0
    for wid, capacity, _ in members:
        if share[wid] > capacity:
            over += share[wid] - capacity
            share[wid] = capacity
    for wid, capacity, _ in members:
        take = min(over, capacity - share[wid])
        share[wid] += take
        over -= take
    return share
