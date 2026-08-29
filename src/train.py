#!/usr/bin/env python3
"""MVP1 training loop -- tiny U-Net, few epochs, img_size=256, pipeline validation
over accuracy. Explicitly not in scope: augmentation beyond a flip, LR scheduling,
multi-fold CV, TTA, mixed-precision tuning -- all post-MVP1 iteration.

Note on the Dice number logged per epoch: this uses src/metrics.dice_score directly
(mean over the val batch) rather than torchmetrics.segmentation.DiceScore -- a quick
check found DiceScore's default aggregation behavior non-obvious for the empty-mask
edge case, and src/metrics.py is the same implementation used for final local
Dice/PQ evaluation (scripts/baseline_classical.py --split val), so per-epoch and
final numbers stay consistent with each other by construction.

Multi-GPU: launch with torchrun instead of plain `python` to train with
DistributedDataParallel across all visible GPUs (e.g. Kaggle's T4 x2). A plain
`python -m src.train` still runs single-process on one GPU/CPU/MPS, unchanged --
distributed mode is only entered when torchrun's RANK/WORLD_SIZE env vars are
present (src/distributed.is_distributed()). Per-epoch train_loss/train_dice printed
in distributed mode are rank 0's own shard only (not all-reduced across ranks) --
a deliberate simplification for MVP1; directionally informative, not exact.
Validation runs on rank 0 only, against the full val set, so val_dice/checkpoint
selection stays exact regardless of world size.

Caveat: this DDP path has not been run against real multi-GPU hardware (only
logic-reviewed + a local CPU/gloo dry-run, which hung inconclusively in a sandboxed
environment -- see README.md's "Known gotchas"). Smoke-test with --epochs 1 first.

Usage:
    python -m src.train
    python -m src.train --epochs 10 --img-size 256
    torchrun --nproc_per_node=2 -m src.train --epochs 10 --batch-size 16
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from src.dataset import FilamentDataset, group_split
from src.distributed import cleanup_distributed, is_distributed, setup_distributed
from src.metrics import dice_score
from src.model import build_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--data-json",
        type=Path,
        default=Path("data/raw/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"),
    )
    p.add_argument(
        "--images-dir",
        type=Path,
        default=Path("data/raw/MAGFiLO_1.0_Kaggle_2026/train/train_images"),
    )
    p.add_argument("--img-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None, help="cuda / mps / cpu -- auto-detected if omitted")
    p.add_argument("--checkpoint-out", type=Path, default=Path("outputs/checkpoints/mvp1_unet.pt"))
    p.add_argument("--log-csv", type=Path, default=Path("outputs/logs/train_log.csv"), help="per-epoch train/val loss+dice, overwritten each run")
    return p.parse_args()


def dice_loss_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    inter = (probs * targets).sum(dim=(1, 2, 3))
    denom = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


def combined_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, targets) + dice_loss_from_logits(logits, targets)


def resolve_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_epoch(model: nn.Module, loader: DataLoader, device: torch.device, optimizer=None) -> tuple[float, float]:
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss, total_dice, n_batches = 0.0, 0.0, 0
    context = torch.enable_grad() if train_mode else torch.no_grad()
    with context:
        for img, mask, _image_id, _orig_size in loader:
            img, mask = img.to(device), mask.to(device)
            logits = model(img)
            loss = combined_loss(logits, mask)

            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            preds = (torch.sigmoid(logits) > 0.5).float()
            batch_dice = float(torch.mean(torch.tensor([dice_score(preds[i, 0].cpu().numpy() > 0, mask[i, 0].cpu().numpy() > 0) for i in range(preds.shape[0])])))
            total_loss += loss.item()
            total_dice += batch_dice
            n_batches += 1
    return total_loss / n_batches, total_dice / n_batches


def main() -> None:
    args = parse_args()

    distributed = is_distributed()
    if distributed:
        local_rank, rank, world_size = setup_distributed()
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank, world_size = 0, 1
        device = resolve_device(args.device)
    is_main = rank == 0

    if is_main:
        print(f"Using device: {device}" + (f"  (distributed, world_size={world_size})" if distributed else ""))

    full_ds = FilamentDataset(args.data_json, args.images_dir, image_ids=[], img_size=args.img_size)
    train_ids, val_ids = group_split(full_ds.coco, val_fraction=args.val_fraction, seed=args.seed)
    if is_main:
        print(f"train: {len(train_ids)} images, val: {len(val_ids)} images (grouped by file_name)")

    train_ds = FilamentDataset(args.data_json, args.images_dir, train_ids, img_size=args.img_size)
    val_ds = FilamentDataset(args.data_json, args.images_dir, val_ids, img_size=args.img_size)

    if distributed:
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler, num_workers=0)
    else:
        train_sampler = None
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)  # rank 0 only, full val set

    if distributed and not is_main:
        dist.barrier()  # let rank 0 download+cache the pretrained encoder weights first, avoiding a concurrent-download race
    model = build_model().to(device)
    if distributed and is_main:
        dist.barrier()
    if distributed:
        model = DDP(model, device_ids=[local_rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_dice = -1.0
    if is_main:
        args.checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
        args.log_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.log_csv, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "train_dice", "val_loss", "val_dice"])

    epoch_range = range(1, args.epochs + 1)
    for epoch in (tqdm(epoch_range, desc="epochs") if is_main else epoch_range):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_loss, train_dice = run_epoch(model, train_loader, device, optimizer)

        if distributed:
            dist.barrier()  # keep other ranks from racing ahead into next epoch while rank 0 validates/checkpoints

        if is_main:
            val_loss, val_dice = run_epoch(model, val_loader, device, optimizer=None)
            print(f"epoch {epoch:>2}: train_loss={train_loss:.4f} train_dice={train_dice:.4f}  val_loss={val_loss:.4f} val_dice={val_dice:.4f}")

            with open(args.log_csv, "a", newline="") as f:
                csv.writer(f).writerow([epoch, train_loss, train_dice, val_loss, val_dice])

            if val_dice > best_val_dice:
                best_val_dice = val_dice
                state_dict = model.module.state_dict() if distributed else model.state_dict()
                torch.save(
                    {
                        "model_state_dict": state_dict,
                        "img_size": args.img_size,
                        "val_dice": val_dice,
                        "epoch": epoch,
                        "val_fraction": args.val_fraction,
                        "seed": args.seed,
                    },
                    args.checkpoint_out,
                )
                print(f"  -> saved new best checkpoint (val_dice={val_dice:.4f}) to {args.checkpoint_out}")

        if distributed:
            dist.barrier()  # don't let other ranks start the next epoch until rank 0's validation/checkpoint is done

    if is_main:
        print(f"Done. Best val_dice={best_val_dice:.4f}, checkpoint at {args.checkpoint_out}")

    if distributed:
        cleanup_distributed()


if __name__ == "__main__":
    main()
