#!/usr/bin/env python3
"""Where predictions are actually going wrong, across the TRAINING set.

Dice/IoU/PQ (scripts/evaluate.py, scripts/analyze_train_predictions.py) are single
numbers that conflate precision and recall and say nothing about *why* a match failed
or what an unmatched instance looks like. This script splits pixel-level
precision/recall apart (which direction is the model biased?), looks at how tight
correctly-matched instances actually are (does a boundary-aware loss term make sense?),
and characterizes false positives/negatives by size and shape (is it a postprocessing
min-area problem, or a genuine model sensitivity/shape problem?) -- meant as a concrete
input to the loss-function/architecture decision, not just another metric.

Usage:
    python scripts/error_analysis.py --checkpoint outputs/checkpoints/best.pt
    python scripts/error_analysis.py --checkpoint outputs/checkpoints/best.pt --watershed-min-distance 140

Multi-GPU (each rank analyzes its own shard, gathered to rank 0 for the report -- same
pattern as evaluate.py's mAP gather):
    torchrun --nproc_per_node=2 scripts/error_analysis.py --checkpoint outputs/checkpoints/best.pt
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from skimage.measure import regionprops

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
from src.utils.panoptic import panoptic_quality_detailed  # noqa: E402
from src.utils.postprocess import mask_to_instances  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--debug-limit", type=int, default=None, help="cap number of training images analyzed")
    p.add_argument("--iou-thresh", type=float, default=None, help="override PQ matching IoU threshold")
    p.add_argument("--prob-threshold", type=float, default=None, help="override postprocess.prob_threshold")
    p.add_argument("--watershed-min-distance", type=int, default=None, help="override postprocess.watershed_min_distance")
    p.add_argument("--min-instance-area", type=int, default=None, help="override postprocess.min_instance_area")
    p.add_argument("--no-watershed", action="store_true", help="disable watershed, use plain connected components instead")
    p.add_argument("--output-dir", type=Path, default=Path("outputs/error_analysis"))
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


def instance_shape(mask: np.ndarray) -> tuple[float, float]:
    """(area, eccentricity) via skimage regionprops -- eccentricity near 0 is blob-like
    (circle), near 1 is elongated (line-like, what a real filament should look like)."""
    props = regionprops(mask.astype(np.uint8))
    if not props:
        return 0.0, 0.0
    return float(props[0].area), float(props[0].eccentricity)


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
        iterator = tqdm(shard_ids, desc="error_analysis")
    else:
        print(f"[rank {rank}] analyzing {len(shard_ids)} images...")
        iterator = shard_ids

    pixel_records = []  # per-image precision/recall
    matched_ious: list[float] = []
    fn_records = []  # {"area": ..., "eccentricity": ...} per missed GT instance
    fp_records = []  # same, per false-positive predicted instance
    gt_areas = []  # every GT instance's area, for the "are misses concentrated in small filaments" comparison
    tp_areas = []  # every TP predicted instance's area, for the "are FPs smaller than real detections" comparison

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
        precision = inter / binary_pred.sum() if binary_pred.sum() > 0 else 1.0
        recall = inter / gt_semantic.sum() if gt_semantic.sum() > 0 else 1.0
        pixel_records.append({"image_id": image_id, "precision": float(precision), "recall": float(recall)})

        detail = panoptic_quality_detailed(gt_instances, pred_instances, iou_thresh=iou_thresh)
        matched_ious.extend(detail["matched_ious"])

        matched_pred_indices = set(range(len(pred_instances))) - set(detail["fp_indices"])
        for i in range(len(gt_instances)):
            area, ecc = instance_shape(gt_instances[i])
            gt_areas.append(area)
            if i in detail["fn_indices"]:
                fn_records.append({"area": area, "eccentricity": ecc})
        for j in range(len(pred_instances)):
            area, ecc = instance_shape(pred_instances[j])
            if j in detail["fp_indices"]:
                fp_records.append({"area": area, "eccentricity": ecc})
            elif j in matched_pred_indices:
                tp_areas.append(area)

    if not is_main:
        print(f"[rank {rank}] done")

    if distributed:
        local_results = {
            "pixel": pixel_records, "matched_ious": matched_ious,
            "fn": fn_records, "fp": fp_records, "gt_areas": gt_areas, "tp_areas": tp_areas,
        }
        gathered = [None] * world_size if is_main else None
        dist.gather_object(local_results, gathered, dst=0)
        if is_main:
            pixel_records = [r for shard in gathered for r in shard["pixel"]]
            matched_ious = [x for shard in gathered for x in shard["matched_ious"]]
            fn_records = [r for shard in gathered for r in shard["fn"]]
            fp_records = [r for shard in gathered for r in shard["fp"]]
            gt_areas = [x for shard in gathered for x in shard["gt_areas"]]
            tp_areas = [x for shard in gathered for x in shard["tp_areas"]]

    if is_main:
        args.output_dir.mkdir(parents=True, exist_ok=True)

        csv_path = args.output_dir / "pixel_precision_recall.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["image_id", "precision", "recall"])
            writer.writeheader()
            writer.writerows(pixel_records)
        print(f"Wrote {len(pixel_records)} rows to {csv_path}")

        precisions = np.array([r["precision"] for r in pixel_records])
        recalls = np.array([r["recall"] for r in pixel_records])
        matched_ious_arr = np.array(matched_ious) if matched_ious else np.array([0.0])
        fn_areas = np.array([r["area"] for r in fn_records]) if fn_records else np.array([0.0])
        fp_areas = np.array([r["area"] for r in fp_records]) if fp_records else np.array([0.0])
        fn_ecc = np.array([r["eccentricity"] for r in fn_records]) if fn_records else np.array([0.0])
        fp_ecc = np.array([r["eccentricity"] for r in fp_records]) if fp_records else np.array([0.0])
        gt_areas_arr = np.array(gt_areas) if gt_areas else np.array([0.0])
        tp_areas_arr = np.array(tp_areas) if tp_areas else np.array([0.0])

        fig, axes = plt.subplots(1, 1, figsize=(6, 4))
        axes.hist(matched_ious_arr, bins=30)
        axes.set_title(f"Matched-instance IoU distribution (n={len(matched_ious)})")
        axes.set_xlabel("IoU")
        fig.tight_layout()
        iou_hist_path = args.output_dir / "matched_iou_hist.png"
        fig.savefig(iou_hist_path, dpi=120)
        plt.close(fig)
        print(f"Saved {iou_hist_path}")

        # Area distributions are heavily right-skewed (many small instances, occasional huge
        # spurious blobs) -- a handful of outliers otherwise swamp a linear-scale histogram and
        # hide the informative bulk of the distribution near zero. Log-scale bins fix that.
        area_bins = np.logspace(0, np.log10(max(fn_areas.max(), fp_areas.max(), 10)), 30)

        fig, axes = plt.subplots(1, 4, figsize=(20, 4))
        axes[0].hist(np.maximum(fn_areas, 1), bins=area_bins)
        axes[0].set_xscale("log")
        axes[0].set_title(f"FN (missed) instance area (n={len(fn_records)})")
        axes[1].hist(np.maximum(fp_areas, 1), bins=area_bins)
        axes[1].set_xscale("log")
        axes[1].set_title(f"FP (spurious) instance area (n={len(fp_records)})")
        axes[2].hist(fn_ecc, bins=30, range=(0, 1))
        axes[2].set_title("FN eccentricity (0=blob, 1=elongated)")
        axes[3].hist(fp_ecc, bins=30, range=(0, 1))
        axes[3].set_title("FP eccentricity (0=blob, 1=elongated)")
        fig.tight_layout()
        fn_fp_path = args.output_dir / "fn_fp_characteristics.png"
        fig.savefig(fn_fp_path, dpi=120)
        plt.close(fig)
        print(f"Saved {fn_fp_path}")

        print()
        print("=== Error analysis summary ===")
        print(f"pixel-level precision: mean={precisions.mean():.4f} median={np.median(precisions):.4f}")
        print(f"pixel-level recall:    mean={recalls.mean():.4f} median={np.median(recalls):.4f}")
        bias = precisions.mean() - recalls.mean()
        if abs(bias) < 0.03:
            print(f"  -> precision and recall are close (diff={bias:+.4f}) -- no strong over/under-prediction bias")
        elif bias > 0:
            print(f"  -> precision > recall (diff={bias:+.4f}): model UNDER-predicts (misses real foreground more than it invents false foreground)")
        else:
            print(f"  -> recall > precision (diff={bias:+.4f}): model OVER-predicts (invents false foreground more than it misses real foreground)")
        print()
        print(f"matched-instance IoU: mean={matched_ious_arr.mean():.4f} median={np.median(matched_ious_arr):.4f} (n={len(matched_ious)})")
        print(f"  -> {'tight matches, boundary precision is not the main problem' if np.median(matched_ious_arr) > 0.75 else 'loose matches even when correctly detected -- boundary/shape loss term likely to help'}")
        print()
        print(f"FN (missed) instance area:   mean={fn_areas.mean():.1f} median={np.median(fn_areas):.1f}  vs overall GT area: mean={gt_areas_arr.mean():.1f} median={np.median(gt_areas_arr):.1f}")
        print(f"  -> {'misses are concentrated in smaller/fainter filaments' if np.median(fn_areas) < np.median(gt_areas_arr) else 'misses are not obviously smaller than average -- not purely a small-object sensitivity problem'}")
        print(f"FP (spurious) instance area: mean={fp_areas.mean():.1f} median={np.median(fp_areas):.1f}  vs correctly-matched area: mean={tp_areas_arr.mean():.1f} median={np.median(tp_areas_arr):.1f}")
        print(f"  -> {'false positives are smaller/noisier than real detections -- min_instance_area postprocessing tuning has more room' if np.median(fp_areas) < np.median(tp_areas_arr) else 'false positives are not obviously smaller than real detections -- a postprocessing area filter alone will not fix these'}")
        print(f"FN eccentricity: mean={fn_ecc.mean():.3f}  FP eccentricity: mean={fp_ecc.mean():.3f} (near 1 = elongated/filament-like, near 0 = blob-like)")
        print(f"  -> {'false positives are blob-shaped, not filament-shaped -- a shape-based postprocessing filter is worth trying' if fp_ecc.mean() < 0.7 else 'false positives are themselves elongated/filament-like -- likely a genuine model confusion, not filterable by shape alone'}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
