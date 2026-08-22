#!/usr/bin/env python3
"""Train U-Net (ConvNeXt encoder) for binary filament segmentation.

Usage:
    python scripts/train.py
    python scripts/train.py --epochs 40 --batch-size 8
    python scripts/train.py --debug-limit 20 --epochs 1   # fast smoke test

Multi-GPU (DistributedDataParallel, e.g. Kaggle's T4x2):
    torchrun --nproc_per_node=2 scripts/train.py --config configs/config_kaggle.yaml
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.coco_utils import index_annotations_by_image, index_images_by_id, load_coco  # noqa: E402
from src.data.dataset import FilamentSegDataset  # noqa: E402
from src.data.transforms import build_train_transforms, build_val_transforms, describe_transforms  # noqa: E402
from src.models.unet_convnext import build_model  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.device import get_device  # noqa: E402
from src.utils.distributed import is_distributed, setup_distributed  # noqa: E402
from src.utils.losses import DiceBCELoss  # noqa: E402
from src.utils.metrics import EpochDiceScore, compute_batch_iou_dice  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None, help="total batch size, split evenly across GPUs under torchrun")
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--debug-limit", type=int, default=None, help="cap train images (val capped to 1/5 of this) for a fast smoke test")
    p.add_argument("--resume", action="store_true", help="resume from checkpoint_dir/best.pt if it exists")
    return p.parse_args()


def split_image_ids(ann_index: dict, val_fraction: float, seed: int) -> tuple[list[str], list[str]]:
    ids = sorted(ann_index.keys())
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_val = max(1, int(len(ids) * val_fraction))
    return ids[n_val:], ids[:n_val]


def compute_run_name(image_size: int, batch_size: int, gradient_checkpointing: bool) -> str:
    """Auto-named per-run output folder: {timestamp}_img{size}_bs{batch}[_gc], so different
    configs (e.g. this session's 768px vs 1536px+checkpointing experiments) don't clobber each
    other's checkpoints/logs under the shared checkpoint_dir/log_dir roots."""
    tag = f"img{image_size}_bs{batch_size}"
    if gradient_checkpointing:
        tag += "_gc"
    return f"{datetime.now():%Y%m%d_%H%M%S}_{tag}"


def find_latest_run(root: Path) -> str | None:
    """Most recently modified subdirectory of `root` that has a best.pt, or None.

    Deterministic filesystem read, not clock-based -- safe to call independently on every DDP
    rank for --resume without needing to broadcast the result (unlike compute_run_name, which
    uses datetime.now() and is only ever used by rank 0 for anything).
    """
    if not root.exists():
        return None
    candidates = [d for d in root.iterdir() if d.is_dir() and (d / "best.pt").exists()]
    if not candidates:
        return None
    return max(candidates, key=lambda d: d.stat().st_mtime).name


def reduce_mean(value_sum: float, count: int, device: torch.device, distributed: bool) -> float:
    """Global (sum / count) across all ranks -- each rank only sums over its own data shard."""
    if not distributed:
        return value_sum / max(1, count)
    stats = torch.tensor([value_sum, float(count)], device=device)
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    return (stats[0] / stats[1]).item()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    distributed = is_distributed()
    if distributed:
        local_rank, rank, world_size = setup_distributed()
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank, world_size = 0, 1
        device = get_device(cfg["train"]["device"])

    is_main = rank == 0
    set_seed(cfg["seed"] + rank)  # decorrelate augmentation RNG streams across ranks

    if is_main:
        print(f"Using device: {device}" + (f" (DDP, world_size={world_size})" if distributed else ""))

    model_cfg = dict(cfg["model"])
    if device.type == "mps" and model_cfg.get("decoder_use_norm", "batchnorm") == "batchnorm":
        print("MPS backend: torch.batch_norm backward is broken on this torch build -> disabling decoder_use_norm")
        model_cfg["decoder_use_norm"] = False

    coco = load_coco(Path(cfg["data"]["train_annotations"]))
    ann_index = index_annotations_by_image(coco)
    img_index = index_images_by_id(coco)

    train_ids, val_ids = split_image_ids(ann_index, cfg["data"]["val_fraction"], cfg["seed"])
    if args.debug_limit:
        train_ids = train_ids[: args.debug_limit]
        val_ids = val_ids[: max(1, args.debug_limit // 5)]
    if is_main:
        print(f"train images: {len(train_ids)}  val images: {len(val_ids)}")

    image_size = cfg["data"]["image_size"]
    images_dir = Path(cfg["data"]["train_images_dir"])

    train_transform = build_train_transforms(image_size)
    val_transform = build_val_transforms(image_size)
    if is_main:
        print(f"Train augmentations: {describe_transforms(train_transform)}")
        print(f"Val transforms:      {describe_transforms(val_transform)}")

    train_ds = FilamentSegDataset(train_ids, img_index, ann_index, images_dir, train_transform)
    val_ds = FilamentSegDataset(val_ids, img_index, ann_index, images_dir, val_transform)

    global_batch_size = args.batch_size or cfg["train"]["batch_size"]
    per_device_batch_size = max(1, global_batch_size // world_size) if distributed else global_batch_size
    num_workers = args.num_workers if args.num_workers is not None else cfg["data"]["num_workers"]

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, seed=cfg["seed"]) if distributed else None
    train_loader = DataLoader(
        train_ds, batch_size=per_device_batch_size, shuffle=(train_sampler is None), sampler=train_sampler,
        num_workers=num_workers, pin_memory=(device.type == "cuda"), drop_last=len(train_ds) > per_device_batch_size,
    )
    # Validation runs on rank 0 only (val set is small) -- avoids cross-rank metric reduction.
    val_loader = DataLoader(
        # per_device_batch_size, not global_batch_size: validation runs on rank 0 alone (see the
        # comment above), so it must fit on a single GPU -- global_batch_size is sized for the
        # *sum* across all ranks and would OOM there once resolution leaves little headroom.
        val_ds, batch_size=per_device_batch_size, shuffle=False, num_workers=num_workers, pin_memory=(device.type == "cuda"),
    )
    if is_main:
        print(
            f"Batches per epoch: {len(train_loader)} train, {len(val_loader)} val "
            f"(batch_size={per_device_batch_size}/rank, global={global_batch_size}, num_workers={num_workers})"
        )

    model = build_model(**model_cfg).to(device)
    if distributed:
        model = DDP(model, device_ids=[local_rank])
    # Validation calls this directly instead of model(...): DDP's forward() broadcasts buffers
    # (BatchNorm running stats) from rank 0 to all ranks as a collective on every call, but only
    # rank 0 runs validation -- going through DDP there deadlocks waiting for the other ranks.
    eval_model = model.module if distributed else model
    if is_main:
        n_params = sum(p.numel() for p in eval_model.parameters())
        print(f"Model: {model_cfg['encoder_name']} encoder, decoder_use_norm={model_cfg['decoder_use_norm']}, {n_params / 1e6:.1f}M params")

    criterion = DiceBCELoss(**cfg["loss"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])

    epochs = args.epochs or cfg["train"]["epochs"]
    warmup_epochs = min(cfg["train"]["warmup_epochs"], max(epochs - 1, 0))  # leave >=1 epoch for cosine decay
    if warmup_epochs > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if is_main:
        print(f"LR schedule: warmup_epochs={warmup_epochs}, base_lr={cfg['train']['lr']}, epochs={epochs}")

    amp_enabled = cfg["train"]["amp"] and device.type in ("cuda", "mps")
    amp_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    scaler = torch.amp.GradScaler(device="cuda", enabled=(device.type == "cuda" and amp_enabled))
    if is_main:
        print(f"AMP: {'enabled (' + str(amp_dtype) + ')' if amp_enabled else 'disabled'}")

    checkpoint_root = Path(cfg["train"]["checkpoint_dir"])
    log_root = Path(cfg["train"]["log_dir"])
    if args.resume:
        run_name = find_latest_run(checkpoint_root)
        if run_name is None:
            run_name = compute_run_name(image_size, global_batch_size, model_cfg.get("gradient_checkpointing", False))
            if is_main:
                print(f"--resume passed but no prior run found under {checkpoint_root} -- starting a new run instead")
    else:
        run_name = compute_run_name(image_size, global_batch_size, model_cfg.get("gradient_checkpointing", False))
    if is_main:
        print(f"Run: {run_name}")
    checkpoint_dir = checkpoint_root / run_name
    log_dir = log_root / run_name
    threshold = cfg["train"]["val_threshold"]
    early_stopping_patience = cfg["train"]["early_stopping_patience"]
    best_dice = -1.0
    epochs_without_improvement = 0
    start_epoch = 1
    effective_cfg = {**cfg, "model": model_cfg}

    if args.resume:
        resume_path = checkpoint_dir / "best.pt"
        if resume_path.exists():
            ckpt = torch.load(resume_path, map_location=device, weights_only=False)
            eval_model.load_state_dict(ckpt["model"])
            if "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
                scaler.load_state_dict(ckpt["scaler"])
            elif is_main:
                print(f"  {resume_path} is an old-format checkpoint (weights only) -- optimizer restarts fresh")
            best_dice = ckpt.get("best_dice", -1.0)
            epochs_without_improvement = ckpt.get("epochs_without_improvement", 0)
            start_epoch = ckpt["epoch"] + 1
            # Rebuilding scheduler.state_dict() directly would restore the *previous* run's
            # T_max/milestones (shaped for whatever --epochs it used), which silently breaks the
            # LR curve if this run's --epochs differs. Replaying .step() re-derives the correct
            # position against *this* run's schedule instead.
            for _ in range(start_epoch - 1):
                scheduler.step()
            if is_main:
                print(f"Resumed from {resume_path} (epoch {ckpt['epoch']}, best_dice={best_dice:.4f})")
        elif is_main:
            print(f"--resume passed but {resume_path} doesn't exist -- starting fresh")

    log_file = None
    writer = None
    if is_main:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "train_log.csv"
        resuming_log = start_epoch > 1 and log_path.exists()
        log_file = open(log_path, "a" if resuming_log else "w", newline="")
        writer = csv.writer(log_file)
        if not resuming_log:
            writer.writerow(["epoch", "train_loss", "val_loss", "val_iou", "val_dice", "val_dice_torchmetrics", "lr", "seconds"])

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        train_loss_sum, train_count = 0.0, 0
        train_iter = tqdm(train_loader, desc=f"epoch {epoch}/{epochs} [train]") if is_main else train_loader
        for images, masks, _ in train_iter:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            if amp_enabled:
                with torch.autocast(device_type=device.type, dtype=amp_dtype):
                    logits = model(images)
                    loss = criterion(logits, masks)
                if device.type == "cuda":
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            else:
                logits = model(images)
                loss = criterion(logits, masks)
                loss.backward()
                optimizer.step()
            train_loss_sum += loss.item() * images.size(0)
            train_count += images.size(0)
        train_loss = reduce_mean(train_loss_sum, train_count, device, distributed)
        scheduler.step()

        if distributed:
            dist.barrier()  # other ranks wait here while rank 0 validates + checkpoints below

        if is_main:
            eval_model.eval()
            val_loss_sum, iou_sum, dice_sum, n_batches = 0.0, 0.0, 0.0, 0
            epoch_dice_metric = EpochDiceScore(device, threshold=threshold)
            with torch.no_grad():
                for images, masks, _ in tqdm(val_loader, desc=f"epoch {epoch}/{epochs} [val]"):
                    images, masks = images.to(device), masks.to(device)
                    logits = eval_model(images)
                    loss = criterion(logits, masks)
                    val_loss_sum += loss.item() * images.size(0)
                    iou, dice = compute_batch_iou_dice(logits, masks, threshold=threshold)
                    iou_sum += iou
                    dice_sum += dice
                    n_batches += 1
                    epoch_dice_metric.update(logits, masks)
            val_loss = val_loss_sum / len(val_ds)
            val_iou = iou_sum / max(1, n_batches)
            val_dice = dice_sum / max(1, n_batches)
            val_dice_tm = epoch_dice_metric.compute()

            is_new_best = val_dice_tm > best_dice
            if is_new_best:
                best_dice = val_dice_tm
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            lr_now = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t0
            print(
                f"epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"val_iou={val_iou:.4f} val_dice={val_dice:.4f} val_dice_tm={val_dice_tm:.4f} "
                f"lr={lr_now:.2e} no_improve={epochs_without_improvement}/{early_stopping_patience} ({elapsed:.1f}s)"
            )
            writer.writerow([epoch, train_loss, val_loss, val_iou, val_dice, val_dice_tm, lr_now, elapsed])
            log_file.flush()

            checkpoint_payload = {
                "model": eval_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "config": effective_cfg,
                "epoch": epoch,
                "best_dice": best_dice,
                "epochs_without_improvement": epochs_without_improvement,
            }
            torch.save(checkpoint_payload, checkpoint_dir / "last.pt")
            print(f"  saved {checkpoint_dir / 'last.pt'}")
            if is_new_best:
                torch.save({**checkpoint_payload, "val_dice": best_dice}, checkpoint_dir / "best.pt")
                print(f"  new best (val_dice={best_dice:.4f}) -> saved {checkpoint_dir / 'best.pt'}")

            should_stop = epochs_without_improvement >= early_stopping_patience
            if should_stop:
                print(f"  early stopping: val_dice_tm hasn't improved in {early_stopping_patience} epochs")
        else:
            should_stop = False

        if distributed:
            dist.barrier()  # hold other ranks until rank 0 finishes validation/checkpointing before next epoch
            stop_tensor = torch.tensor([int(should_stop)], device=device)
            dist.broadcast(stop_tensor, src=0)  # all ranks must break together, or later collectives mismatch
            should_stop = bool(stop_tensor.item())

        if should_stop:
            break

    if is_main:
        log_file.close()
        print("Training complete.")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
