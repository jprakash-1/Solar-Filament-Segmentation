#!/usr/bin/env python3
"""Train U-Net (ConvNeXt encoder) for binary filament segmentation.

Usage:
    python scripts/train.py
    python scripts/train.py --epochs 40 --batch-size 8
    python scripts/train.py --debug-limit 20 --epochs 1   # fast smoke test
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.coco_utils import index_annotations_by_image, index_images_by_id, load_coco  # noqa: E402
from src.data.dataset import FilamentSegDataset  # noqa: E402
from src.data.transforms import build_train_transforms, build_val_transforms  # noqa: E402
from src.models.unet_convnext import build_model  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.device import get_device  # noqa: E402
from src.utils.losses import DiceBCELoss  # noqa: E402
from src.utils.metrics import EpochDiceScore, compute_batch_iou_dice  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--debug-limit", type=int, default=None, help="cap train images (val capped to 1/5 of this) for a fast smoke test")
    return p.parse_args()


def split_image_ids(ann_index: dict, val_fraction: float, seed: int) -> tuple[list[str], list[str]]:
    ids = sorted(ann_index.keys())
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_val = max(1, int(len(ids) * val_fraction))
    return ids[n_val:], ids[:n_val]


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    device = get_device(cfg["train"]["device"])
    print(f"Using device: {device}")

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
    print(f"train images: {len(train_ids)}  val images: {len(val_ids)}")

    image_size = cfg["data"]["image_size"]
    images_dir = Path(cfg["data"]["train_images_dir"])

    train_ds = FilamentSegDataset(train_ids, img_index, ann_index, images_dir, build_train_transforms(image_size))
    val_ds = FilamentSegDataset(val_ids, img_index, ann_index, images_dir, build_val_transforms(image_size))

    batch_size = args.batch_size or cfg["train"]["batch_size"]
    num_workers = args.num_workers if args.num_workers is not None else cfg["data"]["num_workers"]

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=(device.type == "cuda"), drop_last=len(train_ds) > batch_size,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(device.type == "cuda"),
    )

    model = build_model(**model_cfg).to(device)
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs via nn.DataParallel")
        model = nn.DataParallel(model)

    criterion = DiceBCELoss(**cfg["loss"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])

    epochs = args.epochs or cfg["train"]["epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    amp_enabled = cfg["train"]["amp"] and device.type in ("cuda", "mps")
    amp_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    scaler = torch.amp.GradScaler(device="cuda", enabled=(device.type == "cuda" and amp_enabled))

    checkpoint_dir = Path(cfg["train"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(cfg["train"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)

    threshold = cfg["train"]["val_threshold"]
    best_dice = -1.0
    effective_cfg = {**cfg, "model": model_cfg}

    with open(log_dir / "train_log.csv", "w", newline="") as log_file:
        writer = csv.writer(log_file)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_iou", "val_dice", "val_dice_torchmetrics", "lr", "seconds"])

        for epoch in range(1, epochs + 1):
            t0 = time.time()

            model.train()
            train_loss_sum = 0.0
            for images, masks, _ in tqdm(train_loader, desc=f"epoch {epoch}/{epochs} [train]"):
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
            train_loss = train_loss_sum / len(train_ds)
            scheduler.step()

            model.eval()
            val_loss_sum, iou_sum, dice_sum, n_batches = 0.0, 0.0, 0.0, 0
            epoch_dice_metric = EpochDiceScore(device, threshold=threshold)
            with torch.no_grad():
                for images, masks, _ in tqdm(val_loader, desc=f"epoch {epoch}/{epochs} [val]"):
                    images, masks = images.to(device), masks.to(device)
                    logits = model(images)
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

            lr_now = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t0
            print(
                f"epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"val_iou={val_iou:.4f} val_dice={val_dice:.4f} val_dice_tm={val_dice_tm:.4f} ({elapsed:.1f}s)"
            )
            writer.writerow([epoch, train_loss, val_loss, val_iou, val_dice, val_dice_tm, lr_now, elapsed])
            log_file.flush()

            model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save({"model": model_state, "config": effective_cfg, "epoch": epoch}, checkpoint_dir / "last.pt")
            if val_dice_tm > best_dice:
                best_dice = val_dice_tm
                torch.save(
                    {"model": model_state, "config": effective_cfg, "epoch": epoch, "val_dice": best_dice},
                    checkpoint_dir / "best.pt",
                )
                print(f"  new best (val_dice={best_dice:.4f}) -> saved {checkpoint_dir / 'best.pt'}")

    print("Training complete.")


if __name__ == "__main__":
    main()
