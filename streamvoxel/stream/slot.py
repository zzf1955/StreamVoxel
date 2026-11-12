"""Per-slot low-entropy-last trajectory. The forward count is 1 marking pass plus the solid prefix of a cap-length quartercos schedule."""
from __future__ import annotations

import torch

from streamvoxel.sampling.block_quota import block_quota_select, block_view
from streamvoxel.sampling.entropy import entropy_scores
from streamvoxel.sampling.low_entropy_last import low_entropy_last_quotas, low_entropy_last_update
from streamvoxel.sampling.schedule import actual_time


class ForwardBudget:
    """Counts forwards the way `LowEntropyLastSlot.consume` does: one marking forward, then one forward per solid quota.

    `arm(total)` sets the whole-window forward count, including the marking forward. A total of 20 finishes
    at the end of the 20th forward and is idle before the 21st.
    """

    def __init__(self):
        self.phase = 'idle'
        self.solid = 0
        self.left = 0
        self.nfe = 0

    def arm(self, total):
        total = int(total)
        if total < 1:
            raise ValueError('A window needs at least the marking forward')
        self.phase = 'marking'
        self.solid = total - 1
        self.left = 0
        self.nfe = 0

    def forward(self):
        """Advance one forward. Returns True when this forward finishes the window."""
        if self.phase == 'idle':
            return False
        self.nfe += 1
        if self.phase == 'marking':
            self.phase = 'solid'
            self.left = self.solid
        else:
            self.left -= 1
        if self.left == 0:
            self.phase = 'idle'
            return True
        return False


class LowEntropyLastSlot:
    """One canvas window inside a batched forward.

    RNG matches `LowEntropyLastRule`: the block order is `seed + 9173`, and the only draws on the sample generator
    are the categorical samples of solid steps. The marking forward commits nothing. `finish` writes the deferred last
    voxels to `air_token` and does not take another forward.
    """

    def __init__(self, shape, classes, air_ids, air_token, last_eps, block_size, device):
        self.shape = tuple(shape)
        self.total = int(torch.tensor(self.shape).prod())
        self.classes = int(classes)
        self.air_ids = tuple(air_ids)
        self.air_token = int(air_token)
        self.last_eps = float(last_eps)
        self.block_size = int(block_size)
        self.device = torch.device(device)
        self.phase = 'idle'
        self.target = None
        self.seed = None
        self.cap = None
        self.last = None
        self.quotas = None
        self.solid_calls = 0
        self.actual = 0
        self.nfe = 0
        self.order = None
        self.generator = None
        self.last_count = 0
        self.step_map = torch.zeros(self.shape, dtype=torch.int16, device='cpu')
        self.values = torch.full(self.shape, -1, dtype=torch.int16, device='cpu')

    @property
    def idle(self):
        return self.phase == 'idle'

    def arm(self, seed, cap, target):
        self.phase = 'marking'
        self.seed = int(seed)
        self.cap = int(cap)
        self.target = tuple(target)
        self.nfe = 0
        self.actual = 0
        self.solid_calls = 0
        self.quotas = None
        self.last_count = 0
        self.last = torch.zeros(self.shape, dtype=torch.bool, device=self.device)
        blocks = self.total // self.block_size ** 3
        order_rng = torch.Generator(device=self.device).manual_seed(self.seed + 9173)
        self.order = torch.randperm(blocks, device=self.device, generator=order_rng)
        self.generator = torch.Generator(device=self.device).manual_seed(self.seed)
        self.step_map = torch.zeros(self.shape, dtype=torch.int16)
        self.values = torch.full(self.shape, -1, dtype=torch.int16)
        self.air = torch.as_tensor(list(self.air_ids), device=self.device)

    def drop(self):
        self.phase = 'idle'
        self.target = None
        self.quotas = None

    def time(self, tokens=None):
        """Masked fraction mapped to model time. Solid steps commit exactly their quota, so the count needs no GPU read."""
        if self.phase == 'idle':
            return 1.0
        committed = sum(self.quotas[:self.solid_calls]) if self.quotas else 0
        return actual_time((self.total - committed) / self.total)

    def consume(self, logits, tokens):
        """One forward for this slot. Returns `(finish_or_none, commit_or_none)`.

        `tokens` is the slot's `[Y,Z,X]` buffer and is updated in place. `logits` is `[C,Y,Z,X]`.
        """
        if self.phase == 'idle':
            return None, None
        if self.phase == 'marking':
            self.nfe += 1
            air_p = logits.float().softmax(0)[self.air].sum(0)
            mask = torch.ones(self.shape, dtype=torch.bool, device=tokens.device)
            solid = torch.zeros(self.shape, dtype=torch.bool, device=tokens.device)
            self.last = low_entropy_last_update(self.last, mask, air_p, solid, self.last_eps, False, True)
            self.last_count = int(self.last.sum())
            self.quotas = low_entropy_last_quotas(self.total, self.cap, self.last_count, 'quartercos')
            self.phase = 'solid'
            self.solid_calls = 0
            if not self.quotas:
                return self._finish(tokens), None
            return None, None
        if self.solid_calls >= len(self.quotas):
            return self._finish(tokens), None
        k = self.quotas[self.solid_calls]
        selectable = (tokens == self.classes) & ~self.last
        non_last = ~self.last
        # Every solid step commits exactly its quota of non-last voxels (asserted below), so the counts are host-side sums.
        committed = sum(self.quotas[:self.solid_calls])
        frac = (committed + k) / (self.total - self.last_count)
        n_total = block_view(non_last.contiguous(), self.block_size).sum(1)
        chosen = block_quota_select(entropy_scores(logits), selectable, k, self.block_size, self.order, frac, n_total)
        if bool((chosen.sum() != k) | (chosen & ~selectable).any()):
            raise AssertionError('low-entropy-last solid step did not meet its quota')
        probabilities = logits.permute(1, 2, 3, 0)[chosen].float().softmax(-1)
        values = torch.multinomial(probabilities, 1, generator=self.generator).squeeze(-1)
        tokens[chosen] = values
        values_cpu = values.detach().to(dtype=torch.int16, device='cpu')
        self.solid_calls += 1
        self.actual += 1
        self.nfe += 1
        flat = chosen.reshape(-1).cpu()
        positions = torch.nonzero(flat, as_tuple=False).flatten()
        self.step_map.reshape(-1)[positions] = self.nfe
        self.values.reshape(-1)[positions] = values_cpu
        commit = dict(positions=positions.numpy().astype('int32'),
                      values=values_cpu.numpy(), step=int(self.nfe))
        if self.solid_calls >= len(self.quotas):
            return self._finish(tokens), commit
        return None, commit

    def _finish(self, tokens):
        tokens[self.last] = self.air_token
        last_cpu = self.last.detach().cpu()
        self.step_map[last_cpu] = self.nfe
        self.values[last_cpu] = self.air_token
        record = dict(target=self.target, seed=self.seed, cap=self.cap, nfe=int(self.nfe), actual=int(self.actual),
                      last_count=int(self.last_count), last_frac=float(self.last_count) / self.total,
                      step_map=self.step_map.clone(), values=self.values.clone())
        self.phase = 'idle'
        self.target = None
        return record
