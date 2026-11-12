"""Shared plumbing of the evaluation entries: case loading and the conditioned forward of one window.

A case folder holds `gt_global.npy` [256,Z,X] uint16 (global ids, YZX, Z and X multiples of 16) and `biomes.npy` [64,Z/4+2,X/4+2]
(quart labels with a one-quart halo). `--mapping` is the compact mapping json, `--vocabulary` the vocabulary json with `idToCleanName`.
"""
import json
from pathlib import Path

import numpy as np
import torch

from streamvoxel.stream.condition import Mapping
from streamvoxel.stream.engine import encode_item, open_model
from streamvoxel.stream.scheduler import make_item
from streamvoxel.train.runtime import autocast

AIR_NAMES = ('minecraft:air', 'minecraft:cave_air', 'minecraft:void_air')


def load_case(case, mapping, vocabulary):
    case = Path(case)
    gt = np.load(case / 'gt_global.npy')
    biomes = np.load(case / 'biomes.npy')
    if gt.ndim != 3 or gt.shape[1] % 16 or gt.shape[2] % 16:
        raise ValueError(f'gt_global must be [Y,16*Z,16*X], got {gt.shape}')
    mapping = Mapping.from_json(mapping)
    names = json.loads(Path(vocabulary).read_text())['idToCleanName']
    air_global = [i for i, name in enumerate(names) if name in AIR_NAMES]
    air_compact = sorted({int(mapping.to_compact[i]) for i in air_global if mapping.to_compact[i] >= 0})
    air_token = int(mapping.to_compact[names.index('minecraft:air')])
    return gt, biomes, mapping, air_global, air_compact, air_token


def initial_canvas(gt, mapping, box):
    """Compact canvas with only the initial cells `box = (z0, x0, z1, x1)` (in 16-voxel cells) filled; -1 elsewhere."""
    z0, x0, z1, x1 = box
    canvas = np.full(gt.shape, -1, np.int16)
    canvas[:, z0 * 16:z1 * 16, x0 * 16:x1 * 16] = mapping.compact(gt[:, z0 * 16:z1 * 16, x0 * 16:x1 * 16], 'global', 1 << 30)
    known = np.zeros((gt.shape[1] // 16, gt.shape[2] // 16), bool)
    known[z0:z1, x0:x1] = True
    return canvas, known


def cells_of(canvas, known):
    return {(int(z), int(x)): canvas[:, z * 16:(z + 1) * 16, x * 16:(x + 1) * 16] for z, x in zip(*np.where(known))}


def window_forward(model, item, spec, device):
    """`forward(tokens[1,Y,Z,X], t) -> logits[1,C,Y,Z,X]` with the item's condition encoded once (exact, unpadded)."""
    prefix, local, positions = encode_item(model, item, spec, device)
    stacked = [(key.unsqueeze(0), value.unsqueeze(0)) for key, value in prefix]
    origin = torch.tensor([[0., spec.y_origin, 0.]], device=device)

    def forward(tokens, t):
        with autocast(device):
            return model(tokens, timestep=torch.full((1,), t, device=device), origin_xyz=origin, group_origin_xyz=origin[0],
                         prefix_kv=stacked, local=local, local_positions=positions)
    return forward


def window_item(canvas, known, target, biomes, spec, seed, steps, l2_cap):
    item = make_item(cells_of(canvas, known), target, biomes, spec, 0, steps, l2_cap)
    item.seed = int(seed)
    return item


def open_eval_model(base, control, device):
    return open_model(base, control, torch.device(device))
