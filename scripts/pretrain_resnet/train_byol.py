#!/usr/bin/env python3
"""BYOL domain-SSL pretraining for a 1-channel ResNet50 on the GONG H-alpha corpus.
See RESNET_PRETRAIN_PLAN.md for the full design this implements (sections 5-9).

Multi-GPU: launch with torchrun to train with DistributedDataParallel across all
visible GPUs (e.g. Kaggle's T4 x2). A plain `python train_byol.py` still runs
single-process on one GPU/CPU/MPS, unchanged -- distributed mode is only entered
when torchrun's RANK/WORLD_SIZE env vars are present (distributed.is_distributed()),
same gate `jp-mvp1:src/train.py` uses.

DDP/BYOL interaction notes (read before changing this file):
  - Only the *online* sub-network (encoder/projector/predictor) needs gradients;
    target stays requires_grad=False (set in model.BYOL.__init__), so DDP's
    gradient-sync naturally skips it -- no find_unused_parameters needed, every
    online param gets a gradient every forward (both views go through the online
    path).
  - SyncBatchNorm is applied before DDP-wrapping so BatchNorm statistics are
    computed across both GPUs, not per-GPU independently -- matters more here than
    in jp-mvp1 since per-GPU batch may be modest at 224px/ResNet50.
  - update_target() (the EMA step) is called on the *unwrapped* module
    (`model.module` under DDP) after every optimizer step, identically by every
    rank (no rank-dependent branching) -- this can't reproduce the exact
    asymmetric-forward()-call NCCL hang `jp-mvp1:src/train.py`'s docstring
    documents (that bug came from validation running forward() on rank 0 only
    through the DDP-wrapped model). Health-check evaluation below applies the same
    "always go through model.module, never the DDP wrapper, for rank-0-only work"
    rule.

Usage:
    python scripts/pretrain_resnet/train_byol.py --max-images 200 --epochs 1
    torchrun --nproc_per_node=2 scripts/pretrain_resnet/train_byol.py --epochs 150
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from dataset import HalphaBYOLDataset, date_grouped_split, discover_images
from distributed import cleanup_distributed, is_distributed, setup_distributed
from health_checks import embedding_std, nearest_neighbors
from model import BYOL, byol_loss

logger = logging.getLogger("pretrain_resnet.train_byol")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images-dir", type=Path, default=Path("data/processed/gong_pretrain"))
    p.add_argument("--manifest", type=Path, default=Path("data/processed/gong_pretrain/manifest.csv"))
    p.add_argument("--crop-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=128, help="per-GPU batch size")
    p.add_argument("--epochs", type=int, default=150, help="total epochs planned across ALL sessions, not per-session")
    p.add_argument("--warmup-epochs", type=int, default=10)
    p.add_argument("--base-lr", type=float, default=None, help="defaults to 0.2 * (per-GPU batch * world_size) / 256")
    p.add_argument("--tau-base", type=float, default=0.996)
    p.add_argument("--val-fraction", type=float, default=0.04)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None, help="cuda / mps / cpu -- auto-detected if omitted (ignored under torchrun, which sets the GPU per rank)")
    p.add_argument("--num-workers", type=int, default=0, help="DataLoader worker processes; keep 0 for local CPU/MPS debugging, raise (e.g. 4) on GPU to stop data loading from bottlenecking GPU utilization")
    p.add_argument("--checkpoint-out", type=Path, default=Path("outputs/checkpoints/resnet50_byol.pt"))
    p.add_argument("--resume", type=Path, default=None, help="checkpoint to resume from, e.g. the mounted halpha-byol-ckpt dataset's latest.pt")
    p.add_argument("--log-csv", type=Path, default=Path("outputs/logs/byol_train_log.csv"))
    p.add_argument("--session-budget-seconds", type=float, default=8 * 3600, help="self-stop with margin before Kaggle's session cap")
    p.add_argument("--health-check-every", type=int, default=5, help="epochs between held-out val loss + embedding-std + nearest-neighbor checks")
    p.add_argument("--nn-grid-out", type=Path, default=Path("outputs/logs/byol_nn_grid.png"))
    p.add_argument("--max-images", type=int, default=None, help="debug/smoke-test cap on corpus size")
    return p.parse_args()


def resolve_device(local_rank: int | None, requested: str | None) -> torch.device:
    if local_rank is not None:
        return torch.device(f"cuda:{local_rank}")
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def cosine_lr(epoch: int, total_epochs: int, warmup_epochs: int, base_lr: float) -> float:
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    progress = min((epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs), 1.0)
    return 0.5 * base_lr * (1 + math.cos(math.pi * progress))


def tau_schedule(epoch: int, total_epochs: int, tau_base: float) -> float:
    """Cosine-annealed 0.996 -> 1.0 over training -- the target network tracks the
    online network more loosely early on, converging to near-frozen by the end.
    BYOL's actual recipe (not an optional refinement). Purely a function of
    epoch/total_epochs, so it survives checkpoint-resume automatically as long as
    --epochs is kept consistent across sessions."""
    progress = min(epoch / max(1, total_epochs - 1), 1.0)
    return 1.0 - (1.0 - tau_base) * (1 + math.cos(math.pi * progress)) / 2


def save_checkpoint(path: Path, model_module: torch.nn.Module, optimizer, scaler, epoch: int, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Stringify args (argparse gives us Path objects for several fields) so the
    # checkpoint round-trips through torch.load's default weights_only=True
    # (torch >= 2.6) without needing an allowlisted-globals workaround.
    args_dict = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    torch.save({
        "model": model_module.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "args": args_dict,
    }, path)


@torch.no_grad()
def run_validation(model_module: torch.nn.Module, val_loader: DataLoader, device: torch.device) -> float:
    model_module.eval()
    total_loss, n_batches = 0.0, 0
    for view1, view2 in val_loader:
        view1, view2 = view1.to(device), view2.to(device)
        o1, o2, t1, t2 = model_module(view1, view2)
        total_loss += byol_loss(o1, o2, t1, t2).item()
        n_batches += 1
    model_module.train()
    return total_loss / max(1, n_batches)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    args = parse_args()
    session_start = time.time()

    distributed = is_distributed()
    if distributed:
        local_rank, rank, world_size = setup_distributed()
    else:
        local_rank, rank, world_size = None, 0, 1
    device = resolve_device(local_rank, args.device)
    is_main = rank == 0

    base_lr = args.base_lr if args.base_lr is not None else 0.2 * (args.batch_size * world_size) / 256
    if is_main:
        logger.info(f"world_size={world_size} device={device} per_gpu_batch={args.batch_size} base_lr={base_lr:.4f}")

    model = BYOL().to(device)
    if distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[local_rank])
    model_module = model.module if distributed else model

    optimizer = torch.optim.SGD(
        [p for p in model_module.parameters() if p.requires_grad],
        lr=base_lr, momentum=0.9, weight_decay=1e-6,
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    start_epoch = 0
    if args.resume and args.resume.exists():
        ckpt = torch.load(args.resume, map_location=device)
        model_module.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        if is_main:
            logger.info(f"resumed from {args.resume} at epoch {start_epoch}")

    records = discover_images(args.images_dir, args.manifest)
    if args.max_images:
        records = records[: args.max_images]
    train_records, val_records = date_grouped_split(records, args.val_fraction, args.seed)

    train_dataset = HalphaBYOLDataset(train_records, crop_size=args.crop_size)
    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed) if distributed else None
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=sampler, shuffle=(sampler is None),
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"), drop_last=True,
    )

    val_loader = None
    if is_main and val_records:
        val_dataset = HalphaBYOLDataset(val_records, crop_size=args.crop_size)
        val_loader = DataLoader(val_dataset, batch_size=min(args.batch_size, len(val_dataset)), shuffle=False, num_workers=0, drop_last=False)

    stop_early = torch.zeros(1, device=device) if distributed else None
    # all-reduced so every rank agrees on when to stop -- checking wall-clock
    # independently per rank could let one rank exit while another keeps
    # iterating, hanging DDP's collective ops.

    for epoch in range(start_epoch, args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        lr = cosine_lr(epoch, args.epochs, args.warmup_epochs, base_lr)
        tau = tau_schedule(epoch, args.epochs, args.tau_base)
        for g in optimizer.param_groups:
            g["lr"] = lr

        model.train()
        epoch_loss, n_batches = 0.0, 0
        loader_iter = tqdm(train_loader, desc=f"epoch {epoch}", disable=not is_main)
        for view1, view2 in loader_iter:
            view1, view2 = view1.to(device, non_blocking=True), view2.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, enabled=use_amp):
                o1, o2, t1, t2 = model(view1, view2)
                loss = byol_loss(o1, o2, t1, t2)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            model_module.update_target(tau)  # unwrapped module, called identically by every rank
            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(1, n_batches)
        if is_main:
            val_loss, emb_std = float("nan"), float("nan")
            do_health_check = val_loader is not None and (epoch % args.health_check_every == 0 or epoch == args.epochs - 1)
            if do_health_check:
                val_loss = run_validation(model_module, val_loader, device)
                emb_std = embedding_std(model_module.online_encoder, val_loader, device)
                nearest_neighbors(model_module.online_encoder, val_loader, device, grid_out=str(args.nn_grid_out))
                logger.info(f"epoch {epoch} | train_loss {avg_loss:.4f} | val_loss {val_loss:.4f} | emb_std {emb_std:.4f} | tau {tau:.5f} | lr {lr:.2e}")
            else:
                logger.info(f"epoch {epoch} | train_loss {avg_loss:.4f} | tau {tau:.5f} | lr {lr:.2e}")

            args.log_csv.parent.mkdir(parents=True, exist_ok=True)
            write_header = not args.log_csv.exists()
            with open(args.log_csv, "a", newline="") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(["epoch", "train_loss", "val_loss", "embedding_std", "tau", "lr"])
                writer.writerow([epoch, avg_loss, val_loss, emb_std, tau, lr])

            save_checkpoint(args.checkpoint_out, model_module, optimizer, scaler, epoch, args)

            if time.time() - session_start > args.session_budget_seconds:
                if stop_early is not None:
                    stop_early += 1
                else:
                    logger.info(f"session budget reached at epoch {epoch}, stopping cleanly")
                    break

        if distributed:
            dist.broadcast(stop_early, src=0)
            if stop_early.item() > 0:
                if is_main:
                    logger.info(f"session budget reached at epoch {epoch}, stopping cleanly")
                break

    if distributed:
        cleanup_distributed()


if __name__ == "__main__":
    main()
