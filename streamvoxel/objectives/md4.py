"""Dream-Cubed MD4 target corruption and weighted CE."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def mask_probability(time: torch.Tensor) -> torch.Tensor:
    return torch.sin(time * (math.pi / 2))


def mask_weight(time: torch.Tensor) -> torch.Tensor:
    """MD4 weight p'(t)/p(t) for p(t)=sin(π t / 2)."""
    return (math.pi / 2) / torch.tan(time * (math.pi / 2))


def linear_mask_probability(time: torch.Tensor) -> torch.Tensor:
    return time


def linear_mask_weight(time: torch.Tensor) -> torch.Tensor:
    """MD4 weight p'(t)/p(t) for the linear schedule p(t)=t."""
    return 1 / time


def band_all_air(tokens, air_ids, *, band):
    """[B,Y,Z,X] -> [B,Y/band] bool, true when a height band contains only air."""
    if tokens.ndim != 4 or tokens.shape[1] % band:
        raise ValueError("Expected [B,Y,Z,X] whose height is divisible by the band")
    b, y, z, x = tokens.shape
    bands = y // band
    return torch.isin(tokens.reshape(b, bands, -1), torch.as_tensor(air_ids, device=tokens.device)).all(-1)


def strata_mask(tokens, num_classes, air_ids, *, generator=None, band=8, times=None, uniform=None):
    """Mask each height band at its own noise level. Pure-air bands use sin(π t / 2); other bands use p=t."""
    if tokens.ndim != 4 or tokens.dtype not in (torch.int32, torch.int64):
        raise ValueError("Expected integer [B,Y,Z,X] target")
    if band < 1 or tokens.shape[1] % band:
        raise ValueError("Height must be divisible by the strata band")
    if ((tokens < 0) | (tokens >= num_classes)).any():
        raise ValueError("Targets must be real categories")
    air = torch.as_tensor(list(air_ids), dtype=torch.long)
    if air.numel() == 0 or (air < 0).any() or (air >= num_classes).any() or air.unique().numel() != air.numel():
        raise ValueError("Invalid air compact ids")
    b, y, z, x = tokens.shape
    bands = y // band
    if times is None:
        times = torch.rand((b, bands), device=tokens.device, generator=generator)
    if times.shape != (b, bands) or not torch.isfinite(times).all() or ((times < 0) | (times > 1)).any():
        raise ValueError("Expected one noise level in [0,1] per height band")
    times = times.clamp(1e-4, 1 - 1e-4)
    all_air = band_all_air(tokens, air, band=band)
    probability = torch.where(all_air, mask_probability(times), linear_mask_probability(times))
    weight = torch.where(all_air, mask_weight(times), linear_mask_weight(times))
    probability = probability[:, :, None, None, None].expand(b, bands, band, z, x).reshape(tokens.shape).contiguous()
    weight = weight[:, :, None, None, None].expand(b, bands, band, z, x).reshape(tokens.shape).contiguous()
    if uniform is None:
        uniform = torch.rand(tokens.shape, device=tokens.device, generator=generator)
    if uniform.shape != tokens.shape:
        raise ValueError("Uniform draw must match the target")
    mask = uniform < probability
    for i in torch.where(~mask.flatten(1).any(1))[0].tolist():
        position = torch.randint(tokens[i].numel(), (1,), device=tokens.device, generator=generator)
        mask[i].flatten()[position] = True
    corrupted = tokens.clone()
    corrupted[mask] = num_classes
    return corrupted, mask, times.mean(1), weight


def md4_mask(tokens, num_classes, *, generator=None, times=None):
    if tokens.ndim != 4 or tokens.dtype not in (torch.int32, torch.int64):
        raise ValueError("Expected integer [B,Y,Z,X] target")
    if ((tokens < 0) | (tokens >= num_classes)).any():
        raise ValueError("Targets must be real categories")
    b = len(tokens)
    if times is None:
        times = torch.rand((b,), device=tokens.device, generator=generator)
    if times.shape != (b,) or not torch.isfinite(times).all() or ((times < 0) | (times > 1)).any():
        raise ValueError("Expected times in [0,1]")
    # Avoid the singular cotangent weight at exactly zero.
    times = times.clamp(1e-4, 1-1e-4)
    probability = mask_probability(times)
    uniform = torch.rand(tokens.shape, device=tokens.device, generator=generator)
    mask = uniform < probability[:, None, None, None]
    for i in torch.where(~mask.flatten(1).any(1))[0].tolist():
        position = torch.randint(tokens[i].numel(), (1,), device=tokens.device, generator=generator)
        mask[i].flatten()[position] = True
    corrupted = tokens.clone()
    corrupted[mask] = num_classes
    weight = mask_weight(times)
    return corrupted, mask, times, weight


def weighted_masked_ce_sums(logits, targets, mask, weight, *, by_sample=False):
    if (logits.ndim != 5 or logits.shape[0] != len(targets)
            or logits.shape[2:] != targets.shape[1:] or mask.shape != targets.shape
            or mask.dtype != torch.bool or weight.shape not in ((len(targets),), targets.shape)
            or not mask.any()):
        raise ValueError("Invalid aligned logits, targets, mask, or MD4 weights")
    selected = logits.permute(0, 2, 3, 4, 1)[mask].float()
    labels = targets[mask].long()
    if ((labels < 0) | (labels >= logits.shape[1])).any():
        raise ValueError("Targets outside real vocabulary")
    per_voxel = F.cross_entropy(selected, labels, reduction="none")
    expanded_weight = (weight[:, None, None, None].expand_as(mask) if weight.shape == (len(targets),) else weight)[mask].float()
    weighted = per_voxel * expanded_weight
    correct = selected.detach().argmax(-1) == labels
    result = (weighted.sum(), correct.sum(), mask.sum())
    if not by_sample:
        return result
    counts = mask.flatten(1).sum(1)
    owners = torch.repeat_interleave(torch.arange(len(targets), device=mask.device), counts)
    losses = weighted.detach().new_zeros((len(targets),)).scatter_add_(0, owners, weighted.detach())
    successes = counts.new_zeros(counts.shape).scatter_add_(0, owners, correct.long())
    return (*result, losses, successes, counts)
