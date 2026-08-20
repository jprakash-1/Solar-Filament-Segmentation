#!/usr/bin/env python3
"""Run inference on the test set and write a competition submission CSV.

One row per predicted filament instance: filament_id (test filename stem +
index, e.g. 20111114063134Lh_1) and segmentation_rle (COCO RLE counts,
mask size fixed at 2048x2048 per the competition's Submission File section).

Usage:
    python scripts/infer.py --checkpoint outputs/checkpoints/best.pt
    python scripts/infer.py --checkpoint outputs/checkpoints/best.pt --limit 5 --visualize 5
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.data.coco_utils import load_image  # noqa: E402
from src.data.dataset import FilamentTestDataset  # noqa: E402
from src.data.transforms import build_val_transforms  # noqa: E402
from src.models.unet_convnext import build_model  # noqa: E402
from src.utils.device import get_device  # noqa: E402
from src.utils.postprocess import mask_to_instances, rle_encode  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

INSTANCE_COLORS = [
    (31, 119, 180), (214, 39, 40), (44, 160, 44), (148, 103, 189),
    (255, 127, 14), (23, 190, 207), (227, 119, 194), (188, 189, 34),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--test-images-dir", type=Path, default=None, help="override checkpoint config's test images dir")
    p.add_argument("--output", type=Path, default=None, help="override output CSV path")
    p.add_argument("--limit", type=int, default=None, help="only run on the first N test images (debug)")
    p.add_argument("--visualize", type=int, default=10, help="save prediction overlay PNGs for the first N images (0 disables)")
    p.add_argument("--prob-threshold", type=float, default=None, help="override postprocess.prob_threshold")
    return p.parse_args()


def draw_instances(image: np.ndarray, instances: list[np.ndarray]) -> np.ndarray:
    overlay = image.copy()
    for i, inst in enumerate(instances):
        color = INSTANCE_COLORS[i % len(INSTANCE_COLORS)]
        contours, _ = cv2.findContours(inst.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, thickness=3)
    return overlay


def save_visualization(image: np.ndarray, instances: list[np.ndarray], title: str, out_path: Path) -> None:
    overlay = draw_instances(image, instances)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image)
    axes[0].set_title("Input")
    axes[1].imshow(overlay)
    axes[1].set_title(f"Predicted instances ({len(instances)})")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    set_seed(cfg["seed"])

    device = get_device(cfg["train"]["device"])
    print(f"Using device: {device}")

    model = build_model(**cfg["model"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    test_images_dir = args.test_images_dir or Path(cfg["data"]["test_images_dir"])
    image_size = cfg["data"]["image_size"]
    dataset = FilamentTestDataset(test_images_dir, transform=build_val_transforms(image_size))
    if args.limit:
        dataset.files = dataset.files[: args.limit]
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=cfg["data"]["num_workers"])
    print(f"Running inference on {len(dataset)} test images")

    pp = dict(cfg["postprocess"])
    if args.prob_threshold is not None:
        pp["prob_threshold"] = args.prob_threshold

    submission_dir = Path(cfg["inference"]["submission_dir"])
    submission_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output or submission_dir / f"submission_{datetime.now():%Y%m%d_%H%M%S}.csv"

    viz_n = args.visualize
    viz_dir = Path("outputs/eda/predictions")
    if viz_n > 0:
        viz_dir.mkdir(parents=True, exist_ok=True)

    n_filaments = 0
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filament_id", "segmentation_rle"])

        for idx, (image, stem, orig_h, orig_w) in enumerate(tqdm(loader, desc="infer")):
            image = image.to(device)
            orig_h, orig_w = int(orig_h.item()), int(orig_w.item())
            stem = stem[0]

            with torch.no_grad():
                logits = model(image)
                probs = torch.sigmoid(logits)
                probs = F.interpolate(probs, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
            prob_map = probs.squeeze().cpu().numpy()
            binary_pred = prob_map > pp["prob_threshold"]

            instances = mask_to_instances(
                binary_pred, min_area=pp["min_instance_area"], use_watershed=pp["use_watershed"],
                watershed_min_distance=pp["watershed_min_distance"],
            )
            for k, inst_mask in enumerate(instances, start=1):
                writer.writerow([f"{stem}_{k}", rle_encode(inst_mask)])
            n_filaments += len(instances)

            if idx < viz_n:
                display_image = load_image(test_images_dir, dataset.files[idx])
                save_visualization(display_image, instances, title=f"{stem} ({len(instances)} filaments)", out_path=viz_dir / f"{stem}_pred.png")

    print(f"Wrote {n_filaments} filament rows for {len(dataset)} images to {output_path}")
    if viz_n > 0:
        print(f"Saved visualizations to {viz_dir}")


if __name__ == "__main__":
    main()
