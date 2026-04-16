from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


@dataclass
class DistributedContext:
    is_distributed: bool
    rank: int
    world_size: int
    local_rank: int
    device: torch.device

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


def setup_distributed() -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_distributed = world_size > 1

    if torch.cuda.is_available():
        if is_distributed:
            torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank if is_distributed else 0)
    else:
        device = torch.device("cpu")

    if is_distributed and not dist.is_initialized():
        backend = "nccl" if device.type == "cuda" else "gloo"
        dist.init_process_group(backend=backend)

    return DistributedContext(
        is_distributed=is_distributed,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        device=device,
    )


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def wrap_model_for_distributed(
    model: torch.nn.Module,
    distributed_context: DistributedContext,
) -> torch.nn.Module:
    if not distributed_context.is_distributed:
        return model

    if distributed_context.device.type == "cuda":
        return DDP(
            model,
            device_ids=[distributed_context.local_rank],
            output_device=distributed_context.local_rank,
            find_unused_parameters=False,
        )

    return DDP(model, find_unused_parameters=False)


def all_reduce_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor)
    return tensor

