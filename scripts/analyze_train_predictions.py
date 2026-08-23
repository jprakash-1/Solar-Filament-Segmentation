#!/usr/bin/env python3
"""Per-image analysis of predictions across the TRAINING set (not val, not test).

Computes semantic Dice/IoU and instance-level Panoptic Quality per training image (the
same numbers scripts/evaluate.py computes internally but only reports as an aggregate
mean over the val split), saves them to a CSV for real analysis, prints summary
statistics, and auto-generates input/ground-truth/predicted comparison plots for the
worst- (and best-) performing images -- targeted debugging instead of random
spot-checks like scripts/visualize_train_predictions.py does.

No mAP here: mAP needs a single global precision-recall curve across the whole dataset
(see src/utils/average_precision.py) and doesn't decompose into a per-image number, so
it isn't useful for outlier-finding and would just slow this down.

Usage:
    python scripts/analyze_train_predictions.py --checkpoint outputs/checkpoints/best.pt
    python scripts/analyze_train_predictions.py --checkpoint outputs/checkpoints/best.pt --watershed-min-distance 140 --worst-n 20

Multi-GPU (each rank computes metrics for its own shard, gathered to rank 0 for the
CSV/summary/worst-best visualization -- same pattern as evaluate.py's mAP gather):
    torchrun --nproc_per_node=2 scripts/analyze_train_predictions.py --checkpoint outputs/checkpoints/best.pt
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from tqdm import tqdm  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.coco_utils import index_annotations_by_image, index_images_by_id, load_coco, load_image  # noqa: E402
from src.data.masks import rasterize_instance_masks, rasterize_semantic_mask  # noqa: E402
from src.data.transforms import build_val_transforms  # noqa: E402
from src.models.unet_convnext import build_model  # noqa: E402
from src.utils.device import get_device  # noqa: E402
from src.utils.distributed import is_distributed, setup_distributed  # noqa: E402
from src.utils.panoptic import panoptic_quality  # noqa: E402
from src.utils.postprocess import mask_to_instances  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

INSTANCE_COLORS = [
    (31, 119, 180), (214, 39, 40), (44, 160, 44), (148, 103, 189),
    (255, 127, 14), (23, 190, 207), (227, 119, 194), (188, 189, 34),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--debug-limit", type=int, default=None, help="cap number of training images analyzed")
    p.add_argument("--iou-thresh", type=float, default=None, help="override PQ matching IoU threshold")
    p.add_argument("--prob-threshold", type=float, default=None, help="override postprocess.prob_threshold")
    p.add_argument("--watershed-min-distance", type=int, default=None, help="override postprocess.watershed_min_distance")
    p.add_argument("--min-instance-area", type=int, default=None, help="override postprocess.min_instance_area")
    p.add_argument("--no-watershed", action="store_true", help="disable watershed, use plain connected components instead")
    p.add_argument("--sort-by", choices=["pq", "dice", "iou"], default="pq", help="metric used to rank worst/best images")
    p.add_argument("--worst-n", type=int, default=15, help="number of worst-ranked images to visualize (0 disables)")
    p.add_argument("--best-n", type=int, default=5, help="number of best-ranked images to visualize, for contrast (0 disables)")
    p.add_argument("--output-dir", type=Path, default=Path("outputs/train_analysis"))
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


def save_comparison(image_id: str, images_dir: Path, img_meta: dict, gt_instances, pred_instances, out_path: Path) -> None:
    image = load_image(images_dir, img_meta["file_name"])
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
        print(f"Using device: {device}" + (f"  ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else "") + (f" (multi-GPU, world_size={world_size})" if distributed else ""))
        if device.type == "cpu":
            print(
                f"WARNING: no GPU detected -- this will be slow. "
                f"torch.cuda.is_available()={torch.cuda.is_available()}, "
                f"cfg['train']['device']={cfg['train']['device']!r} (from the checkpoint's saved config)"
            )

    model = build_model(**cfg["model"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    coco = load_coco(Path(cfg["data"]["train_annotations"]))
    ann_index = index_annotations_by_image(coco)
    img_index = index_images_by_id(coco)
    train_ids, _ = split_image_ids(ann_index, cfg["data"]["val_fraction"], cfg["seed"])
    if args.debug_limit:
        train_ids = train_ids[: args.debug_limit]
    if is_main:
        print(f"Analyzing {len(train_ids)} training images (checkpoint epoch {ckpt.get('epoch')})")
    shard_ids = train_ids[rank::world_size] if distributed else train_ids

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
    if is_main:
        print(f"postprocess: {pp}")
    iou_thresh = args.iou_thresh if args.iou_thresh is not None else cfg["evaluate"]["iou_thresh"]

    if is_main:
        iterator = tqdm(shard_ids, desc="analyze")
    else:
        print(f"[rank {rank}] analyzing {len(shard_ids)} images...")
        iterator = shard_ids

    records = []
    for image_id in iterator:
        img_meta = img_index[image_id]
        anns = ann_index[image_id]
        image = load_image(images_dir, img_meta["file_name"])
        h, w = img_meta["height"], img_meta["width"]

        gt_semantic = rasterize_semantic_mask(anns, h, w).astype(bool)
        gt_instances = rasterize_instance_masks(anns, h, w)

        prob_map = predict_probability_map(model, image, image_size, device)
        binary_pred = prob_map > pp["prob_threshold"]
        pred_instances = mask_to_instances(
            binary_pred, min_area=pp["min_instance_area"], use_watershed=pp["use_watershed"],
            watershed_min_distance=pp["watershed_min_distance"],
        )

        inter = np.logical_and(binary_pred, gt_semantic).sum()
        union = np.logical_or(binary_pred, gt_semantic).sum()
        dice = 2 * inter / (binary_pred.sum() + gt_semantic.sum()) if (binary_pred.sum() + gt_semantic.sum()) > 0 else 1.0
        iou = inter / union if union > 0 else 1.0
        pq = panoptic_quality(gt_instances, pred_instances, iou_thresh=iou_thresh)

        records.append({
            "image_id": image_id, "dice": float(dice), "iou": float(iou), "pq": float(pq),
            "n_gt": len(gt_instances), "n_pred": len(pred_instances),
        })

    if not is_main:
        print(f"[rank {rank}] done")

    if distributed:
        gathered = [None] * world_size if is_main else None
        dist.gather_object(records, gathered, dst=0)
        if is_main:
            records = [r for shard in gathered for r in shard]

    if is_main:
        args.output_dir.mkdir(parents=True, exist_ok=True)

        csv_path = args.output_dir / "per_image_metrics.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["image_id", "dice", "iou", "pq", "n_gt", "n_pred"])
            writer.writeheader()
            writer.writerows(records)
        print(f"Wrote {len(records)} rows to {csv_path}")

        dice_vals = np.array([r["dice"] for r in records])
        iou_vals = np.array([r["iou"] for r in records])
        pq_vals = np.array([r["pq"] for r in records])
        print()
        for name, vals in [("dice", dice_vals), ("iou", iou_vals), ("pq", pq_vals)]:
            print(f"{name}: mean={vals.mean():.4f} median={np.median(vals):.4f} std={vals.std():.4f} min={vals.min():.4f} max={vals.max():.4f}")
        print(f"images with pq == 0 (complete failures): {int((pq_vals == 0).sum())} / {len(records)}")

        fig, axes = plt.subplots(1, 4, figsize=(20, 4))
        axes[0].hist(pq_vals, bins=30)
        axes[0].set_title("Panoptic Quality distribution")
        axes[1].hist(dice_vals, bins=30)
        axes[1].set_title("Dice distribution")
        axes[2].hist(iou_vals, bins=30)
        axes[2].set_title("IoU distribution")
        n_gt_vals = [r["n_gt"] for r in records]
        n_pred_vals = [r["n_pred"] for r in records]
        max_n = max(max(n_gt_vals, default=0), max(n_pred_vals, default=0)) + 1
        axes[3].scatter(n_gt_vals, n_pred_vals, alpha=0.3, s=10)
        axes[3].plot([0, max_n], [0, max_n], "r--", linewidth=1)
        axes[3].set_xlabel("n_gt")
        axes[3].set_ylabel("n_pred")
        axes[3].set_title("Predicted vs GT instance count")
        fig.tight_layout()
        summary_path = args.output_dir / "summary.png"
        fig.savefig(summary_path, dpi=120)
        plt.close(fig)
        print(f"Saved {summary_path}")

        records_sorted = sorted(records, key=lambda r: r[args.sort_by])

        def visualize(subset: list[dict], subdir: str) -> None:
            if not subset:
                return
            out_dir = args.output_dir / subdir
            out_dir.mkdir(parents=True, exist_ok=True)
            for r in subset:
                image_id = r["image_id"]
                img_meta = img_index[image_id]
                anns = ann_index[image_id]
                h, w = img_meta["height"], img_meta["width"]
                gt_instances = rasterize_instance_masks(anns, h, w)
                image = load_image(images_dir, img_meta["file_name"])
                prob_map = predict_probability_map(model, image, image_size, device)
                binary_pred = prob_map > pp["prob_threshold"]
                pred_instances = mask_to_instances(
                    binary_pred, min_area=pp["min_instance_area"], use_watershed=pp["use_watershed"],
                    watershed_min_distance=pp["watershed_min_distance"],
                )
                out_path = out_dir / f"{args.sort_by}={r[args.sort_by]:.3f}_{image_id}.png"
                save_comparison(image_id, images_dir, img_meta, gt_instances, pred_instances, out_path)
            print(f"Saved {len(subset)} comparison plots to {out_dir}")

        if args.worst_n > 0:
            visualize(records_sorted[: args.worst_n], "worst")
        if args.best_n > 0:
            visualize(records_sorted[-args.best_n :][::-1], "best")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
