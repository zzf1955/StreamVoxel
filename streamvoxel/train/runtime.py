"""Training-runtime helpers: autocast, exact RNG capture/restore, and the resumable data cursor.

Moved without behavior change from the pre-refactor random-mask trainer/dataset modules so
checkpoints (cursor state, per-rank RNG state) resume identically.
"""
from __future__ import annotations

from contextlib import nullcontext
import random

import numpy as np
import torch


def autocast(device):
    return torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else nullcontext()


def rng_state(device):
    return {'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state(device) if device.type == 'cuda' else None,
            'numpy': np.random.get_state(), 'python': random.getstate()}


def restore_rng(value, device):
    torch.set_rng_state(value['torch'])
    if device.type == 'cuda':
        torch.cuda.set_rng_state(value['cuda'], device)
    np.random.set_state(value['numpy'])
    random.setstate(value['python'])


class ShuffledCursor:
    """Exact deterministic stream, including across an epoch boundary and restart."""

    def __init__(self, size, seed, state=None):
        self.size = int(size)
        self.seed = int(seed)
        self.epoch = 0
        self.offset = 0
        self.order = None
        if self.size <= 0:
            raise ValueError('Empty training set')
        if state:
            if state['size'] != self.size or state['seed'] != self.seed:
                raise ValueError('Data cursor identity mismatch')
            self.epoch = state['epoch']
            self.offset = state['offset']

    def take(self, count):
        result = []
        while len(result) < count:
            if self.order is None:
                self.order = torch.randperm(
                    self.size, generator=torch.Generator().manual_seed(self.seed + self.epoch)).tolist()
            n = min(count - len(result), self.size - self.offset)
            result.extend(self.order[self.offset:self.offset + n])
            self.offset += n
            if self.offset == self.size:
                self.epoch += 1
                self.offset = 0
                self.order = None
        return result

    def state_dict(self):
        return {'size': self.size, 'seed': self.seed, 'epoch': self.epoch, 'offset': self.offset}
