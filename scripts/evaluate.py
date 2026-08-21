#!/usr/bin/env python3
"""Instance-level evaluation (Panoptic Quality + Dice) on the held-out val split.

Runs the full inference + post-processing pipeline (the same one used for
submission) against ground truth, to sanity-check against the competition's
actual leaderboard metric before submitting. This is slow (connected
components + instance matching per image) so it's a separate, on-demand
script rather than part of every training epoch.

Usage:
    python scripts/evaluate.py --checkpoint outputs/checkpoints/best.pt
    python scripts/evaluate.py --checkpoint outputs/checkpoints/last.pt --debug-limit 20
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.coco_utils import index_annotations_by_image, index_images_by_id, load_coco, load_image  # noqa: E402
from src.data.masks import rasterize_instance_masks, rasterize_semantic_mask  # noqa: E402
from src.data.transforms import build_val_transforms  # noqa: E402
from src.models.unet_convnext import build_model  # noqa: E402
from src.utils.average_precision import mean_average_precision  # noqa: E402
from src.utils.device import get_device  # noqa: E402
from src.utils.panoptic import panoptic_quality  # noqa: E402
from src.utils.postprocess import mask_to_instances  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--debug-limit", type=int, default=None, help="cap number of val images evaluated")
    p.add_argument("--iou-thresh", type=float, default=None, help="override PQ matching IoU threshold")
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
    _, val_ids = split_image_ids(ann_index, cfg["data"]["val_fraction"], cfg["seed"])
    if args.debug_limit:
        val_ids = val_ids[: args.debug_limit]
    print(f"Evaluating on {len(val_ids)} val images (checkpoint epoch {ckpt.get('epoch')})")

    images_dir = Path(cfg["data"]["train_images_dir"])
    image_size = cfg["data"]["image_size"]
    pp = cfg["postprocess"]
    iou_thresh = args.iou_thresh if args.iou_thresh is not None else cfg["evaluate"]["iou_thresh"]

    pq_scores, dice_scores, iou_scores = [], [], []
    all_gt_instances, all_pred_instances, all_pred_scores = [], [], []
    for image_id in tqdm(val_ids, desc="evaluate"):
        img_meta = img_index[image_id]
        anns = ann_index[image_id]
        image = load_image(images_dir, img_meta["file_name"])
        h, w = img_meta["height"], img_meta["width"]

        prob_map = predict_probability_map(model, image, image_size, device)
        binary_pred = prob_map > pp["prob_threshold"]

        gt_semantic = rasterize_semantic_mask(anns, h, w).astype(bool)
        gt_instances = rasterize_instance_masks(anns, h, w)
        pred_instances = mask_to_instances(
            binary_pred, min_area=pp["min_instance_area"], use_watershed=pp["use_watershed"],
            watershed_min_distance=pp["watershed_min_distance"],
        )

        inter = np.logical_and(binary_pred, gt_semantic).sum()
        union = np.logical_or(binary_pred, gt_semantic).sum()
        dice = 2 * inter / (binary_pred.sum() + gt_semantic.sum()) if (binary_pred.sum() + gt_semantic.sum()) > 0 else 1.0
        iou = inter / union if union > 0 else 1.0
        pq = panoptic_quality(gt_instances, pred_instances, iou_thresh=iou_thresh)

        dice_scores.append(dice)
        iou_scores.append(iou)
        pq_scores.append(pq)

        pred_scores = [float(prob_map[inst].mean()) for inst in pred_instances]
        all_gt_instances.append(gt_instances)
        all_pred_instances.append(pred_instances)
        all_pred_scores.append(pred_scores)

    mAP, ap_per_thresh = mean_average_precision(all_gt_instances, all_pred_instances, all_pred_scores)

    print()
    print(f"mean Dice (semantic):  {np.mean(dice_scores):.4f}")
    print(f"mean IoU  (semantic):  {np.mean(iou_scores):.4f}")
    print(f"mean Panoptic Quality: {np.mean(pq_scores):.4f}  (IoU thresh={iou_thresh})")
    print(f"mAP@[.5:.95]:          {mAP:.4f}")
    print(f"  AP@0.50:             {ap_per_thresh[0.5]:.4f}")
    print(f"  AP@0.75:             {ap_per_thresh[0.75]:.4f}")


if __name__ == "__main__":
    main()
