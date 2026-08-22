#!/usr/bin/env python3
"""Run inference on the test set and write a competition submission CSV.

One row per predicted filament instance: filament_id (test filename stem +
index, e.g. 20111114063134Lh_1) and segmentation_rle (COCO RLE counts,
mask size fixed at 2048x2048 per the competition's Submission File section).

Usage:
    python scripts/infer.py --checkpoint outputs/checkpoints/best.pt
    python scripts/infer.py --checkpoint outputs/checkpoints/best.pt --limit 5 --visualize 5

Multi-GPU (each rank predicts its own shard of the test set independently, then rank 0
merges the results -- no gradient sync needed, unlike training):
    torchrun --nproc_per_node=2 scripts/infer.py --checkpoint outputs/checkpoints/best.pt
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
import torch.distributed as dist
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
from src.utils.distributed import is_distributed, setup_distributed  # noqa: E402
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
    p.add_argument("--watershed-min-distance", type=int, default=None, help="override postprocess.watershed_min_distance")
    p.add_argument("--min-instance-area", type=int, default=None, help="override postprocess.min_instance_area")
    p.add_argument("--no-watershed", action="store_true", help="disable watershed, use plain connected components instead")
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

    distributed = is_distributed()
    if distributed:
        local_rank, rank, world_size = setup_distributed()
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank, world_size = 0, 1
        device = None  # resolved below from the checkpoint's config
    is_main = rank == 0

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    set_seed(cfg["seed"])

    if device is None:
        device = get_device(cfg["train"]["device"])
    if is_main:
        print(f"Using device: {device}" + (f" (multi-GPU, world_size={world_size})" if distributed else ""))

    model = build_model(**cfg["model"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    test_images_dir = args.test_images_dir or Path(cfg["data"]["test_images_dir"])
    image_size = cfg["data"]["image_size"]
    dataset = FilamentTestDataset(test_images_dir, transform=build_val_transforms(image_size))
    if args.limit:
        dataset.files = dataset.files[: args.limit]
    total_images = len(dataset.files)
    if is_main:
        print(f"Running inference on {total_images} test images")
    if distributed:
        dataset.files = dataset.files[rank::world_size]  # each rank's predictions are independent rows -- order doesn't matter

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=cfg["data"]["num_workers"])

    pp = dict(cfg["postprocess"])
    if args.prob_threshold is not None:
        pp["prob_threshold"] = args.prob_threshold
    if args.watershed_min_distance is not None:
        pp["watershed_min_distance"] = args.watershed_min_distance
    if args.min_instance_area is not None:
        pp["min_instance_area"] = args.min_instance_area
    if args.no_watershed:
        pp["use_watershed"] = False
    if is_main and (args.prob_threshold, args.watershed_min_distance, args.min_instance_area, args.no_watershed) != (None, None, None, False):
        print(f"postprocess overrides: {pp}")

    submission_dir = Path(cfg["inference"]["submission_dir"])
    submission_dir.mkdir(parents=True, exist_ok=True)  # every rank writes its own file here, not just rank 0
    output_path = args.output or submission_dir / f"submission_{datetime.now():%Y%m%d_%H%M%S}.csv"
    write_path = submission_dir / f".partial_rank{rank}.csv" if distributed else output_path

    viz_n = args.visualize
    viz_dir = Path("outputs/eda/predictions")
    if is_main and viz_n > 0:
        viz_dir.mkdir(parents=True, exist_ok=True)

    if is_main:
        iterator = tqdm(loader, desc="infer")
    else:
        print(f"[rank {rank}] processing {len(dataset)} images...")
        iterator = loader

    n_filaments = 0
    with open(write_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filament_id", "segmentation_rle"])

        for idx, (image, stem, orig_h, orig_w) in enumerate(iterator):
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

            if is_main and idx < viz_n:
                display_image = load_image(test_images_dir, dataset.files[idx])
                save_visualization(display_image, instances, title=f"{stem} ({len(instances)} filaments)", out_path=viz_dir / f"{stem}_pred.png")

    if not is_main:
        print(f"[rank {rank}] wrote {n_filaments} rows for {len(dataset)} images")

    if distributed:
        dist.barrier()  # wait for every rank's partial file before merging
        if is_main:
            total_rows = 0
            with open(output_path, "w", newline="") as out_f:
                writer = csv.writer(out_f)
                writer.writerow(["filament_id", "segmentation_rle"])
                for r in range(world_size):
                    partial_path = submission_dir / f".partial_rank{r}.csv"
                    with open(partial_path) as in_f:
                        reader = csv.reader(in_f)
                        next(reader)  # that shard's own header
                        for row in reader:
                            writer.writerow(row)
                            total_rows += 1
                    partial_path.unlink()
            print(f"Wrote {total_rows} filament rows for {total_images} images to {output_path}")
        dist.destroy_process_group()
    else:
        print(f"Wrote {n_filaments} filament rows for {total_images} images to {output_path}")

    if is_main and viz_n > 0:
        print(f"Saved visualizations to {viz_dir}")


if __name__ == "__main__":
    main()
