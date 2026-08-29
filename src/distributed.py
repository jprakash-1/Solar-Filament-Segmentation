"""Minimal DDP helpers for multi-GPU training via `torchrun`.

Deliberately not used unless launched with torchrun -- a plain `python -m src.train`
still runs single-process exactly as before (detected via the RANK env var torchrun
sets; absent otherwise).
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


def is_distributed() -> bool:
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def setup_distributed() -> tuple[int, int, int]:
    """Call once at the start of main() when is_distributed() is True.
    Returns (local_rank, rank, world_size).
    """
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    # Pass device_id explicitly -- without it, NCCL logs "Guessing device ID... This
    # can cause a hang if rank to GPU mapping is heterogeneous" and (observed on
    # Kaggle T4 x2) can leave one GPU idle at 0% while the other does all the work,
    # instead of splitting compute across both. torch.cuda.set_device() alone isn't
    # enough for newer NCCL to pick this up eagerly.
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size, device_id=torch.device(f"cuda:{local_rank}"))
    return local_rank, rank, world_size


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()
