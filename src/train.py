#!/usr/bin/env python3
"""Stage 4 supervised fine-tuning loop -- extends MVP1's tiny-U-Net training script
to (a) load a domain-pretrained ResNet50 encoder (the GONG Halpha BYOL checkpoint
from the jp-pretraining-data-prep branch's scripts/pretrain_resnet/) via
`--encoder-checkpoint`, and (b) apply the fine-tuning-specific safeguards
PRETRAIN_PLAN.md sections 5.4-5.6 specify once real score, not interface
correctness, is the goal: a recall-biased Tversky+BCE loss, layer-wise LR decay
across the ResNet stages, an encoder freeze warmup, joint image+mask
augmentation, and model selection on validation Panoptic Quality (not just Dice).

Every new flag defaults to something that still runs `jp-mvp1`'s original
resnet18/ImageNet/BCE+Dice recipe when `--encoder-checkpoint` is omitted --
this script is a superset, not a fork, of that behavior.

Resolution: `--img-size` defaults to 2048 (native, no downsampling) per explicit
project direction -- this differs from both the BYOL pretraining crop (224px) and
the one real MVP1 baseline run (1024px, val PQ=0.3071, see mvp1-solar.ipynb on
jp-mvp1). RESNET_PRETRAIN_PLAN.md section 7 flags a pretrain/fine-tune resolution
mismatch as a BatchNorm-calibration risk, but treats it as something that
resolves itself over a few epochs of training at the new resolution, not as
disqualifying -- accepted here as a known, documented tradeoff, not an oversight.
Training a ResNet50 U-Net at full 2048x2048 is far heavier than either of those
two references, hence two additions neither had: mixed precision (`--amp`) and a
deliberately conservative default `--batch-size` that MUST be re-tuned empirically
on real hardware (see configs/finetune_resnet50_kaggle.yaml) -- do not trust the
default blindly, per this repo's own kaggle.md Step 4 corollary ("resolution and
model capacity both cost VRAM... re-tune batch size empirically every time you
change resolution").

Multi-GPU / DDP behavior, NCCL gotchas, and the rank-0-only validation discipline
are all unchanged from `jp-mvp1`'s `src/train.py` -- see that file's docstring and
README.md's "Known gotchas" for the full writeup; not re-derived here.

Usage:
    python -m src.train --epochs 1 --encoder-name resnet18   # MVP1-equivalent smoke test
    python -m src.train --encoder-name resnet50 --encoder-checkpoint path/to/resnet50_byol_encoder.pt
    python -m src.train --linear-probe-only --encoder-name resnet50 --encoder-checkpoint <ckpt>  # forgetting tripwire baseline
    torchrun --nproc_per_node=2 -m src.train --encoder-name resnet50 --encoder-checkpoint <ckpt> --batch-size 2
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
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
from src.metrics import aggregate_pq, dice_score, panoptic_quality
from src.model import build_model
from src.postprocess import mask_to_instances

# ResNet stages in encoder-depth order (stem first, deepest/closest-to-decoder
# last) -- used for both the freeze-warmup toggle and the layer-wise LR groups.
RESNET_STAGE_ATTRS = ["layer1", "layer2", "layer3", "layer4"]


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
    p.add_argument("--img-size", type=int, default=2048, help="native resolution by default -- no downsampling (see module docstring)")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=8, help="per-process under DDP; re-tune empirically at img-size=2048, 8 is almost certainly too large on a 16GB GPU")
    p.add_argument("--num-workers", type=int, default=0, help="DataLoader worker processes; keep 0 for local CPU/MPS debugging, raise (e.g. 4) on GPU to stop data loading from bottlenecking GPU utilization")
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None, help="cuda / mps / cpu -- auto-detected if omitted")
    p.add_argument("--checkpoint-out", type=Path, default=Path("outputs/checkpoints/finetune_resnet50.pt"))
    p.add_argument("--log-csv", type=Path, default=Path("outputs/logs/finetune_log.csv"), help="per-epoch train/val loss+dice+PQ, overwritten each run")

    # Encoder / architecture
    p.add_argument("--encoder-name", default="resnet18", help="smp.Unet encoder_name; use resnet50 with --encoder-checkpoint for the domain-pretrained path")
    p.add_argument("--encoder-checkpoint", type=Path, default=None, help="export_encoder.py output (see scripts/verify_encoder_checkpoint.py to sanity-check first); omit to fall back to plain ImageNet weights (MVP1 behavior)")
    p.add_argument("--encoder-weights", default="imagenet", help="ignored when --encoder-checkpoint is set")

    # Loss
    p.add_argument("--loss", choices=["bce_dice", "tversky_bce"], default="tversky_bce", help="bce_dice matches jp-mvp1's original loss; tversky_bce is PRETRAIN_PLAN.md section 5.4's recall-biased fine-tuning default")

    # Optimizer / fine-tuning schedule -- only take effect when --encoder-checkpoint
    # is set; otherwise this behaves exactly like jp-mvp1's plain Adam(lr) run.
    p.add_argument("--lr", type=float, default=1e-3, help="used directly when no --encoder-checkpoint; ignored (superseded by --head-lr) otherwise")
    p.add_argument("--head-lr", type=float, default=1e-3, help="decoder+head LR when fine-tuning a checkpointed encoder")
    p.add_argument("--encoder-lr-decay", type=float, default=0.75, help="per-stage LR decay factor going from decoder back to the stem (PRETRAIN_PLAN.md section 5.5)")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--freeze-epochs", type=int, default=3, help="epochs (1-indexed, inclusive) during which the encoder is frozen before unfreezing")
    p.add_argument("--linear-probe-only", action="store_true", help="freeze the encoder for the entire run (forgetting-tripwire baseline, PRETRAIN_PLAN.md section 5.6) -- overrides --freeze-epochs")

    # Mixed precision
    p.add_argument("--amp", choices=["auto", "on", "off"], default="auto", help="auto enables AMP only on CUDA; MPS/CPU autocast support is inconsistent enough not to default it on")

    # Validation-time postprocessing (mirrors src/postprocess.py's defaults so
    # training-time val PQ and the offline src/infer.py check stay comparable)
    p.add_argument("--prob-thresh", type=float, default=0.5)
    p.add_argument("--min-area-px", type=int, default=15)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def dice_loss_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    inter = (probs * targets).sum(dim=(1, 2, 3))
    denom = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


def bce_dice_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, targets) + dice_loss_from_logits(logits, targets)


def tversky_loss_from_logits(logits: torch.Tensor, targets: torch.Tensor, alpha: float = 0.3, beta: float = 0.7, eps: float = 1e-6) -> torch.Tensor:
    """alpha < beta biases toward recall -- thin filament structures are the ones
    a model under-predicts first (PRETRAIN_PLAN.md section 5.4)."""
    probs = torch.sigmoid(logits)
    tp = (probs * targets).sum(dim=(1, 2, 3))
    fp = (probs * (1 - targets)).sum(dim=(1, 2, 3))
    fn = ((1 - probs) * targets).sum(dim=(1, 2, 3))
    tversky = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    return 1.0 - tversky.mean()


def tversky_bce_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return 0.5 * F.binary_cross_entropy_with_logits(logits, targets) + tversky_loss_from_logits(logits, targets)


LOSSES = {"bce_dice": bce_dice_loss, "tversky_bce": tversky_bce_loss}


# ---------------------------------------------------------------------------
# Augmentation -- joint image+mask, label-preserving only (PRETRAIN_PLAN.md
# section 5.6: no canonical "up" on the Sun, so rotation/flip are always valid;
# any geometric warp must be applied identically to image and mask or a few
# pixels of misalignment corrupts a large fraction of a thin positive region)
# ---------------------------------------------------------------------------

def build_train_transform() -> A.Compose:
    return A.Compose([
        A.RandomRotate90(p=0.75),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomGamma(gamma_limit=(80, 120), p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
        A.GaussNoise(std_range=(0.02, 0.08), p=0.3),
        A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(0.02, 0.08), hole_width_range=(0.02, 0.08), p=0.3),
    ])


# ---------------------------------------------------------------------------
# Layer-wise LR parameter groups + freeze warmup
# ---------------------------------------------------------------------------

def build_param_groups(model: nn.Module, head_lr: float, encoder_lr_decay: float) -> list[dict]:
    """Stem+layer1..layer4 as 5 depth-ordered groups (decay applied per stage
    going back from the decoder), everything else (decoder + segmentation head)
    at head_lr, per PRETRAIN_PLAN.md section 5.5's layerwise_lr() formula."""
    encoder = model.encoder
    stem_params = list(encoder.conv1.parameters()) + list(encoder.bn1.parameters())
    stage_param_lists = [stem_params] + [list(getattr(encoder, name).parameters()) for name in RESNET_STAGE_ATTRS]
    num_stages = len(stage_param_lists)

    encoder_param_ids = {id(p) for p in encoder.parameters()}
    head_params = [p for p in model.parameters() if id(p) not in encoder_param_ids]

    groups = []
    for depth, params in enumerate(stage_param_lists):
        lr = head_lr * (encoder_lr_decay ** (num_stages - 1 - depth))
        groups.append({"params": params, "lr": lr})
    groups.append({"params": head_params, "lr": head_lr})
    return groups


def set_encoder_requires_grad(model: nn.Module, requires_grad: bool) -> None:
    for p in model.encoder.parameters():
        p.requires_grad = requires_grad


def resolve_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Train / validate
# ---------------------------------------------------------------------------

def run_train_epoch(model, loader, device, optimizer, loss_fn, scaler, use_amp) -> tuple[float, float]:
    model.train()
    total_loss, total_dice, n_batches = 0.0, 0.0, 0
    for img, mask, _image_ids, _orig_size in loader:
        img, mask = img.to(device), mask.to(device)
        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(img)
            loss = loss_fn(logits, mask)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        preds = (torch.sigmoid(logits) > 0.5).float()
        batch_dice = float(torch.mean(torch.tensor([dice_score(preds[i, 0].detach().cpu().numpy() > 0, mask[i, 0].cpu().numpy() > 0) for i in range(preds.shape[0])])))
        total_loss += loss.item()
        total_dice += batch_dice
        n_batches += 1
    return total_loss / n_batches, total_dice / n_batches


@torch.no_grad()
def validate(model, loader, device, dataset, loss_fn, prob_thresh, min_area_px, use_amp) -> tuple[float, float, dict]:
    """One pass over the val set computing loss/Dice *and* Panoptic Quality
    together (PRETRAIN_PLAN.md section 5.6: model selection on PQ, not just
    Dice/loss, since those don't guarantee good instance separation). Reuses
    each batch's own predictions for both -- at img_size=2048 (native) the
    postprocessing resize below is a no-op; at any other resolution this PQ is
    an approximation at training resolution, not the authoritative native-2048
    number -- rerun `python -m src.infer --split val` for that.
    """
    model.eval()
    total_loss, total_dice, n_batches = 0.0, 0.0, 0
    per_image_results = []
    for img, mask, image_ids, _orig_size in loader:
        img, mask = img.to(device), mask.to(device)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(img)
            loss = loss_fn(logits, mask)
        probs = torch.sigmoid(logits).float()
        preds = (probs > 0.5).float()
        batch_dice = float(torch.mean(torch.tensor([dice_score(preds[i, 0].cpu().numpy() > 0, mask[i, 0].cpu().numpy() > 0) for i in range(preds.shape[0])])))
        total_loss += loss.item()
        total_dice += batch_dice
        n_batches += 1

        probs_np = probs.cpu().numpy()
        for i, image_id in enumerate(image_ids):
            binary = (probs_np[i, 0] > prob_thresh).astype(np.uint8)
            pred_instances = mask_to_instances(binary, min_area_px=min_area_px)
            gt_native = dataset.get_instance_masks(image_id)
            h, w = binary.shape
            gt_instances = [cv2.resize(g, (w, h), interpolation=cv2.INTER_NEAREST) for g in gt_native]
            per_image_results.append(panoptic_quality(gt_instances, pred_instances))

    agg = aggregate_pq(per_image_results)
    return total_loss / n_batches, total_dice / n_batches, agg


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

    use_amp = args.amp == "on" or (args.amp == "auto" and device.type == "cuda")
    if args.amp == "on" and device.type != "cuda":
        use_amp = False  # autocast on non-CUDA is unreliable enough not to honor an explicit "on" either
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp) if device.type == "cuda" else None

    fine_tuning = args.encoder_checkpoint is not None
    freeze_epochs = args.epochs if args.linear_probe_only else args.freeze_epochs

    if is_main:
        print(f"Using device: {device}" + (f"  (distributed, world_size={world_size})" if distributed else ""))
        print(f"encoder={args.encoder_name} checkpoint={args.encoder_checkpoint} fine_tuning={fine_tuning} amp={use_amp} loss={args.loss}")
        if args.linear_probe_only:
            print("--linear-probe-only set: encoder stays frozen for the entire run (forgetting-tripwire baseline)")

    full_ds = FilamentDataset(args.data_json, args.images_dir, image_ids=[], img_size=args.img_size)
    train_ids, val_ids = group_split(full_ds.coco, val_fraction=args.val_fraction, seed=args.seed)
    if is_main:
        print(f"train: {len(train_ids)} images, val: {len(val_ids)} images (grouped by file_name)")

    train_transform = build_train_transform()
    train_ds = FilamentDataset(args.data_json, args.images_dir, train_ids, img_size=args.img_size, transform=train_transform)
    val_ds = FilamentDataset(args.data_json, args.images_dir, val_ids, img_size=args.img_size)

    pin_memory = device.type == "cuda"
    loader_kwargs = {"num_workers": args.num_workers, "pin_memory": pin_memory}
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    if distributed:
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler, **loader_kwargs)
    else:
        train_sampler = None
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)  # rank 0 only, full val set

    if distributed and not is_main:
        dist.barrier()  # let rank 0 download+cache the pretrained encoder weights first, avoiding a concurrent-download race
    model = build_model(
        encoder_name=args.encoder_name,
        encoder_weights=args.encoder_weights,
        encoder_checkpoint=args.encoder_checkpoint,
    ).to(device)
    if distributed and is_main:
        dist.barrier()

    loss_fn = LOSSES[args.loss]

    if fine_tuning:
        plain_model = model  # not yet DDP-wrapped
        param_groups = build_param_groups(plain_model, head_lr=args.head_lr, encoder_lr_decay=args.encoder_lr_decay)
        if is_main:
            print(f"layer-wise LR groups (head_lr={args.head_lr}, decay={args.encoder_lr_decay}): "
                  + ", ".join(f"{g['lr']:.2e}" for g in param_groups))
        optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if distributed:
        # find_unused_parameters=True is required whenever a freeze/unfreeze
        # schedule can run (fine_tuning=True): DDP is constructed here, while
        # every parameter still has requires_grad=True (the freeze toggle only
        # happens later, per-epoch, inside the loop below). Once
        # set_encoder_requires_grad(..., False) runs for the frozen epochs, the
        # encoder's parameters stop producing gradients entirely -- but DDP's
        # default (find_unused_parameters=False) assumes every parameter it saw
        # at construction time will receive one on every backward call, and
        # raises ("Expected to have finished reduction... Parameter indices
        # which did not receive grad...") the moment that assumption breaks.
        # Confirmed directly: this crashed on real Kaggle T4 x2 hardware on the
        # very first training step of epoch 1 (freeze_epochs=3 by default), with
        # every encoder parameter listed as unused. find_unused_parameters=True
        # makes DDP tolerate a param not getting a gradient in a given
        # iteration (a small per-iteration traversal cost) instead of asserting
        # the set of trainable parameters never changes -- exactly this branch's
        # use case, so only pay for it here, not on a plain (non-fine-tuning) run.
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=fine_tuning)

    best_val_pq = -1.0
    if is_main:
        args.checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
        args.log_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.log_csv, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "train_dice", "val_loss", "val_dice", "val_pq_mean", "val_pq_pooled", "encoder_frozen"])

    epoch_range = range(1, args.epochs + 1)
    for epoch in (tqdm(epoch_range, desc="epochs") if is_main else epoch_range):
        plain_model = model.module if distributed else model
        encoder_frozen = fine_tuning and epoch <= freeze_epochs
        if fine_tuning:
            set_encoder_requires_grad(plain_model, requires_grad=not encoder_frozen)

        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_loss, train_dice = run_train_epoch(model, train_loader, device, optimizer, loss_fn, scaler, use_amp)

        if distributed:
            dist.barrier()  # keep other ranks from racing ahead into next epoch while rank 0 validates/checkpoints

        if is_main:
            # Validate through the unwrapped module -- DDP broadcasts BatchNorm
            # buffers on every forward() by default, and this loop only runs on
            # rank 0, so calling the DDP wrapper here would issue collectives
            # rank 1 never joins (the exact NCCL watchdog hang jp-mvp1's
            # src/train.py already hit and documented; see README.md).
            val_loss, val_dice, val_pq = validate(plain_model, val_loader, device, val_ds, loss_fn, args.prob_thresh, args.min_area_px, use_amp)
            print(
                f"epoch {epoch:>2}: train_loss={train_loss:.4f} train_dice={train_dice:.4f}  "
                f"val_loss={val_loss:.4f} val_dice={val_dice:.4f} val_pq={val_pq['mean_per_image_pq']:.4f} "
                f"(pooled={val_pq['pooled_pq']:.4f}){'  [encoder frozen]' if encoder_frozen else ''}"
            )

            with open(args.log_csv, "a", newline="") as f:
                csv.writer(f).writerow([epoch, train_loss, train_dice, val_loss, val_dice, val_pq["mean_per_image_pq"], val_pq["pooled_pq"], encoder_frozen])

            if val_pq["mean_per_image_pq"] > best_val_pq:
                best_val_pq = val_pq["mean_per_image_pq"]
                torch.save(
                    {
                        "model_state_dict": plain_model.state_dict(),
                        "img_size": args.img_size,
                        "encoder_name": args.encoder_name,
                        "val_dice": val_dice,
                        "val_pq": best_val_pq,
                        "epoch": epoch,
                        "val_fraction": args.val_fraction,
                        "seed": args.seed,
                        "loss": args.loss,
                    },
                    args.checkpoint_out,
                )
                print(f"  -> saved new best checkpoint (val_pq={best_val_pq:.4f}) to {args.checkpoint_out}")

        if distributed:
            dist.barrier()  # don't let other ranks start the next epoch until rank 0's validation/checkpoint is done

    if is_main:
        print(f"Done. Best val_pq={best_val_pq:.4f}, checkpoint at {args.checkpoint_out}")

    if distributed:
        cleanup_distributed()


if __name__ == "__main__":
    main()
