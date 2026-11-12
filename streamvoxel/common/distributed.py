"""Small torchrun helpers shared by the active remote trainers."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistributedEnv:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def init_distributed(device_arg: str = "auto") -> DistributedEnv:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("torchrun training requires CUDA")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        dist.init_process_group(backend="nccl", device_id=device)
    elif device_arg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_arg)
    return DistributedEnv(rank=rank, local_rank=local_rank, world_size=world_size, device=device)


def barrier(env: DistributedEnv) -> None:
    if env.enabled:
        dist.barrier()


def cleanup(env: DistributedEnv) -> None:
    if env.enabled and dist.is_initialized():
        dist.destroy_process_group()


def all_reduce_sum(value: torch.Tensor, env: DistributedEnv) -> torch.Tensor:
    if env.enabled:
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value
