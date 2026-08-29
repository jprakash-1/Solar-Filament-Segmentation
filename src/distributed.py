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
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    return local_rank, rank, world_size


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()
