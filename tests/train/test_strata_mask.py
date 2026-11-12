"""Height-banded masks, the half-mask batch split, and test Worlds added to validation."""
import math

import numpy as np
import torch

from streamvoxel.train.validation_base import select_eval_worlds
from streamvoxel.objectives.md4 import linear_mask_weight, mask_probability, mask_weight, strata_mask
from streamvoxel.train.train_base import RankBatches, half_mask_plan, train_step, training_steps


def test_zero_step_target_keeps_counting():
    steps = training_steps(2500, 0)
    assert [next(steps) for _ in range(3)] == [2501, 2502, 2503]
    assert list(training_steps(2, 4)) == [3, 4]


def test_half_mask_ranks_cover_512_samples_once():
    counts, roles = half_mask_plan(512, 8, 4)
    assert counts == [16, 16, 16, 16]
    assert roles == ["random", "random", "strata", "strata"]
    gathered = []
    for rank, count in enumerate(counts):
        batches = iter(RankBatches(4096, 3, rank, 4, 8, 512))
        local = [next(batches) for _ in range(count)]
        assert len(local) == count
        gathered.extend(index for batch in local for index in batch)
    assert len(gathered) == 512 and len(set(gathered)) == 512


def test_pure_air_bands_use_sine_and_other_bands_use_a_linear_rate():
    tokens = torch.zeros((2, 16, 2, 2), dtype=torch.long)
    tokens[0, 0, 0, 0] = 4
    tokens[1, 8:] = 4
    times = torch.full((2, 2), 0.5)
    uniform = torch.full(tokens.shape, 0.6)
    _corrupted, mask, timestep, weight = strata_mask(tokens, 8, [0], band=8, times=times, uniform=uniform)
    sine = mask_probability(times)
    assert torch.allclose(timestep, torch.full((2,), 0.5))
    assert torch.equal(mask[0, :8], torch.zeros_like(mask[0, :8]))
    assert torch.equal(mask[0, 8:], torch.ones_like(mask[0, 8:]))
    assert torch.equal(mask[1, :8], torch.ones_like(mask[1, :8]))
    assert torch.equal(mask[1, 8:], torch.zeros_like(mask[1, 8:]))
    assert 0.6 < float(sine[0, 0])
    assert torch.allclose(weight[0, :8], torch.full_like(weight[0, :8], float(linear_mask_weight(times[0, 0]))))
    assert torch.allclose(weight[0, 8:], torch.full_like(weight[0, 8:], float(mask_weight(times[0, 1]))))
    assert torch.allclose(weight[1, :8], torch.full_like(weight[1, :8], math.pi / 2))
    assert torch.allclose(weight[1, 8:], torch.full_like(weight[1, 8:], 2))


def test_strata_train_step_reports_its_loss_separately_from_the_other_roles():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.zeros(4))

        def forward(self, tokens, **_kwargs):
            return self.bias[None, :, None, None, None].expand(len(tokens), 4, *tokens.shape[1:])

    class Loader:
        def __iter__(self):
            while True:
                yield dict(target=torch.zeros((2, 16, 4, 4), dtype=torch.long),
                           origin_xyz=torch.zeros(2, 3))

    model = Model()
    result = train_step(model, torch.optim.SGD(model.parameters(), lr=0.), iter(Loader()),
                        type("Env", (), {"device": torch.device("cpu"), "enabled": False, "world_size": 1})(),
                        accumulation=1, num_classes=4, grad_clip=1., mask_role="strata", air_ids=[0], strata_band=8)
    assert result["train_strata_samples"] == 2
    assert result["train_random_samples"] == 0
    assert result["train_strata_masked_voxels"] == result["masked_voxels"] > 0
    assert abs(result["train_strata_md4_loss"] - result["train_md4_loss"]) < 1e-6


def test_evaluation_keeps_validation_worlds_and_appends_test_worlds():
    class Data:
        def __init__(self, worlds):
            self.records = np.array([[wi, 0, 0] for wi in worlds for _ in range(2)], dtype=np.int32)

    validation, test = Data([3, 8]), Data([3, 11, 12, 15, 16, 19, 20])
    chosen = select_eval_worlds(validation, 8, test=test)
    assert chosen == [3, 8, 11, 12, 15, 16, 19, 20]
    assert validation.records[:, 0].tolist() == [3, 3, 8, 8, 11, 11, 12, 12, 15, 15, 16, 16, 19, 19, 20, 20]
