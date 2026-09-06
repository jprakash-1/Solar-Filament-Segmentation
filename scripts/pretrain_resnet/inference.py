#!/usr/bin/env python3
"""Standalone inference with an exported BYOL encoder checkpoint (see
export_encoder.py): embed a fresh random sample of Halpha frames and render a
nearest-neighbor grid. This is a separate check from the nn_grid.png
train_byol.py already produces mid-training -- it exercises the *exported*,
scaffolding-free checkpoint (no BYOL projector/target network in memory) the
way Stage 4 fine-tuning would actually load it, and uses a deterministic
disk-centered crop rather than training's randomly-augmented views, since
inference wants one reproducible embedding per image.

Usage:
    python scripts/pretrain_resnet/inference.py \
        --encoder-checkpoint /kaggle/working/resnet50_byol_encoder.pt \
        --images-dir /kaggle/input/halpha-preprocessed \
        --manifest /kaggle/input/halpha-preprocessed/manifest.csv \
        --num-images 64 --grid-out /kaggle/working/inference_nn_grid.png
"""

from __future__ import annotations

import argparse
import csv
import logging
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from dataset import ImageRecord, discover_images
from health_checks import _save_neighbor_grid
from model import resnet50_1ch

logger = logging.getLogger("pretrain_resnet.inference")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--encoder-checkpoint", type=Path, required=True, help="output of export_encoder.py")
    p.add_argument("--images-dir", type=Path, required=True)
    p.add_argument("--manifest", type=Path, default=None)
    p.add_argument("--crop-size", type=int, default=224)
    p.add_argument("--num-images", type=int, default=64, help="random sample of images to embed")
    p.add_argument("--k", type=int, default=5, help="neighbors shown per query")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default=None, help="cuda / mps / cpu -- auto-detected if omitted")
    p.add_argument("--grid-out", type=Path, default=Path("outputs/logs/inference_nn_grid.png"))
    p.add_argument("--embeddings-out", type=Path, default=None,
                    help="optional .npy path to also save the [n, embed_dim] embedding matrix")
    return p.parse_args()


class CenterCropDataset(Dataset):
    """Deterministic disk-centered crop, no augmentation -- unlike dataset.py's
    disk_bounded_crop (randomized, built for BYOL's two-view training pretext),
    inference wants exactly one reproducible view per image."""

    def __init__(self, records: list[ImageRecord], crop_size: int):
        self.records = records
        self.crop_size = crop_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> torch.Tensor:
        rec = self.records[idx]
        img = np.array(Image.open(rec.path), dtype=np.float32) / 255.0
        h, w = img.shape[:2]
        half = self.crop_size // 2
        x0 = int(min(max(rec.cx - half, 0), w - self.crop_size))
        y0 = int(min(max(rec.cy - half, 0), h - self.crop_size))
        crop = img[y0 : y0 + self.crop_size, x0 : x0 + self.crop_size]
        return torch.from_numpy(crop[None, :, :].copy())


def load_encoder(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(checkpoint_path, map_location=device)
    encoder = resnet50_1ch(pretrained=False)
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    encoder.to(device).eval()
    return encoder


@torch.no_grad()
def embed_all(encoder: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    embeddings, images = [], []
    for batch in loader:
        batch = batch.to(device)
        emb = F.normalize(encoder(batch), dim=-1)
        embeddings.append(emb.cpu())
        images.append(batch.cpu())
    return torch.cat(embeddings, dim=0), torch.cat(images, dim=0)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    args = parse_args()
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    )

    records = discover_images(args.images_dir, args.manifest)
    n_sample = min(args.num_images, len(records))
    sample = random.Random(args.seed).sample(records, n_sample)

    encoder = load_encoder(args.encoder_checkpoint, device)
    loader = DataLoader(CenterCropDataset(sample, args.crop_size), batch_size=args.batch_size, shuffle=False, num_workers=0)

    embeddings, images = embed_all(encoder, loader, device)

    if args.embeddings_out is not None:
        args.embeddings_out.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.embeddings_out, embeddings.numpy())
        # Sibling CSV, row-aligned with the .npy -- lets a notebook color a 2D
        # projection of the embeddings by capture date without re-embedding.
        meta_out = args.embeddings_out.with_suffix(".csv")
        with open(meta_out, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["path", "date"])
            for rec in sample:
                writer.writerow([str(rec.path), rec.date])
        logger.info(f"saved [{embeddings.shape[0]}, {embeddings.shape[1]}] embeddings -> {args.embeddings_out} (metadata -> {meta_out})")

    k = min(args.k, n_sample - 1)
    sims = embeddings @ embeddings.T
    sims.fill_diagonal_(-1.0)
    topk = sims.topk(k, dim=1).indices

    args.grid_out.parent.mkdir(parents=True, exist_ok=True)
    _save_neighbor_grid(images, topk, str(args.grid_out))
    logger.info(f"embedded {n_sample} images, saved neighbor grid -> {args.grid_out}")


if __name__ == "__main__":
    main()
