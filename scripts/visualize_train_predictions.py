#!/usr/bin/env python3
"""Qualitative check: raw image | ground-truth instances | predicted instances,
side by side, for a sample of TRAINING-set images (not val, not test).

Distinct from scripts/evaluate.py (held-out val-set metrics) and scripts/infer.py
(ground-truth-less test-set predictions): this looks at data the model actually trained
on, as a sanity check that predictions are in the right ballpark at all.

Usage:
    python scripts/visualize_train_predictions.py --checkpoint outputs/checkpoints/best.pt
    python scripts/visualize_train_predictions.py --checkpoint outputs/checkpoints/best.pt --n 5 --watershed-min-distance 140
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.coco_utils import index_annotations_by_image, index_images_by_id, load_coco, load_image  # noqa: E402
from src.data.masks import rasterize_instance_masks  # noqa: E402
from src.data.transforms import build_val_transforms  # noqa: E402
from src.models.unet_convnext import build_model  # noqa: E402
from src.utils.device import get_device  # noqa: E402
from src.utils.postprocess import mask_to_instances  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

INSTANCE_COLORS = [
    (31, 119, 180), (214, 39, 40), (44, 160, 44), (148, 103, 189),
    (255, 127, 14), (23, 190, 207), (227, 119, 194), (188, 189, 34),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--n", type=int, default=6, help="number of training images to visualize")
    p.add_argument("--output-dir", type=Path, default=Path("outputs/train_samples_predictions"))
    p.add_argument("--prob-threshold", type=float, default=None, help="override postprocess.prob_threshold")
    p.add_argument("--watershed-min-distance", type=int, default=None, help="override postprocess.watershed_min_distance")
    p.add_argument("--min-instance-area", type=int, default=None, help="override postprocess.min_instance_area")
    p.add_argument("--no-watershed", action="store_true", help="disable watershed, use plain connected components instead")
    return p.parse_args()


def split_image_ids(ann_index: dict, val_fraction: float, seed: int) -> tuple[list[str], list[str]]:
    ids = sorted(ann_index.keys())
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_val = max(1, int(len(ids) * val_fraction))
    return ids[n_val:], ids[:n_val]


def predict_probability_map(model, image: np.ndarray, image_size: int, device: torch.device) -> np.ndarray:
    orig_h, orig_w = image.shape[:2]
    transformed = build_val_transforms(image_size)(image=image)["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(transformed)
        probs = torch.sigmoid(logits)
        probs = F.interpolate(probs, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
    return probs.squeeze().cpu().numpy()


def draw_instances(image: np.ndarray, instances: list[np.ndarray]) -> np.ndarray:
    overlay = image.copy()
    for i, inst in enumerate(instances):
        color = INSTANCE_COLORS[i % len(INSTANCE_COLORS)]
        contours, _ = cv2.findContours(inst.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, thickness=3)
    return overlay


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

    coco = load_coco(Path(cfg["data"]["train_annotations"]))
    ann_index = index_annotations_by_image(coco)
    img_index = index_images_by_id(coco)
    train_ids, _ = split_image_ids(ann_index, cfg["data"]["val_fraction"], cfg["seed"])

    n = min(args.n, len(train_ids))
    sample_ids = random.Random(cfg["seed"]).sample(train_ids, n)
    print(f"Visualizing {n} of {len(train_ids)} training images (checkpoint epoch {ckpt.get('epoch')})")

    images_dir = Path(cfg["data"]["train_images_dir"])
    image_size = cfg["data"]["image_size"]
    pp = dict(cfg["postprocess"])
    if args.prob_threshold is not None:
        pp["prob_threshold"] = args.prob_threshold
    if args.watershed_min_distance is not None:
        pp["watershed_min_distance"] = args.watershed_min_distance
    if args.min_instance_area is not None:
        pp["min_instance_area"] = args.min_instance_area
    if args.no_watershed:
        pp["use_watershed"] = False
    print(f"postprocess: {pp}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for image_id in sample_ids:
        img_meta = img_index[image_id]
        anns = ann_index[image_id]
        image = load_image(images_dir, img_meta["file_name"])
        h, w = img_meta["height"], img_meta["width"]

        gt_instances = rasterize_instance_masks(anns, h, w)

        prob_map = predict_probability_map(model, image, image_size, device)
        binary_pred = prob_map > pp["prob_threshold"]
        pred_instances = mask_to_instances(
            binary_pred, min_area=pp["min_instance_area"], use_watershed=pp["use_watershed"],
            watershed_min_distance=pp["watershed_min_distance"],
        )

        gt_overlay = draw_instances(image, gt_instances)
        pred_overlay = draw_instances(image, pred_instances)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(image)
        axes[0].set_title("Input (no annotation)")
        axes[1].imshow(gt_overlay)
        axes[1].set_title(f"Ground truth ({len(gt_instances)} filaments)")
        axes[2].imshow(pred_overlay)
        axes[2].set_title(f"Predicted ({len(pred_instances)} filaments)")
        for ax in axes:
            ax.axis("off")
        fig.suptitle(image_id, fontsize=10)
        fig.tight_layout()

        out_path = args.output_dir / f"{image_id}_train_compare.png"
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        print(f"  saved {out_path}")

    print(f"Done. Saved {n} comparison plots to {args.output_dir}")


if __name__ == "__main__":
    main()
