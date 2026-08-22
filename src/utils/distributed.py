"""Multi-process helpers shared by train.py, infer.py, and evaluate.py.

Only train.py needs true DDP (gradient sync via nn.parallel.DistributedDataParallel) --
infer.py/evaluate.py just need each rank to know its shard of the work and, for
evaluate.py, a way to gather results back to rank 0 at the end.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


def is_distributed() -> bool:
    return "LOCAL_RANK" in os.environ and torch.cuda.is_available()


def setup_distributed() -> tuple[int, int, int]:
    """Initialize the process group (torchrun sets these env vars). Returns (local_rank, rank, world_size)."""
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return local_rank, dist.get_rank(), dist.get_world_size()
