#!/usr/bin/env python3
"""Export a clean, standalone ResNet50 encoder checkpoint from a BYOL training
checkpoint. train_byol.py's checkpoint carries the full BYOL module (online +
target encoder/projector/predictor, optimizer, scaler) -- Stage 4 segmentation
fine-tuning only wants the trained online_encoder's weights, not that
scaffolding. See RESNET_PRETRAIN_PLAN.md section 5 for why the encoder alone is
the reusable artifact.

Usage:
    python scripts/pretrain_resnet/export_encoder.py \
        --checkpoint /kaggle/working/checkpoints/best.pt \
        --out /kaggle/working/resnet50_byol_encoder.pt
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from model import BYOL

logger = logging.getLogger("pretrain_resnet.export_encoder")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True, help="train_byol.py checkpoint, e.g. best.pt or latest.pt")
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def export_encoder(checkpoint_path: Path, out_path: Path) -> None:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    # pretrained=False: load_state_dict below overwrites every weight anyway, so
    # skip the ImageNet download entirely rather than fetching it just to discard it.
    model = BYOL(pretrained=False)
    model.load_state_dict(ckpt["model"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "encoder_state_dict": model.online_encoder.state_dict(),
        "source_checkpoint": str(checkpoint_path),
        "source_epoch": ckpt.get("epoch"),
    }, out_path)
    logger.info(f"exported online_encoder from {checkpoint_path} (epoch {ckpt.get('epoch')}) -> {out_path}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    args = parse_args()
    export_encoder(args.checkpoint, args.out)


if __name__ == "__main__":
    main()
