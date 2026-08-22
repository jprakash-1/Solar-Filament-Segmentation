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
    python scripts/evaluate.py --checkpoint outputs/checkpoints/best.pt --watershed-min-distance 45

Sweep mode: predicts every image once, then tries many watershed_min_distance values
cheaply against the cached predictions (skips redundant GPU forward passes), prints a
PQ table, and auto-computes full mAP for the best value found:
    python scripts/evaluate.py --checkpoint outputs/checkpoints/best.pt --sweep-watershed-min-distance
    python scripts/evaluate.py --checkpoint outputs/checkpoints/best.pt --sweep-watershed-min-distance 10,30,50,70

Multi-GPU (each rank predicts + postprocesses its own shard of the val set
independently, then all ranks' results are gathered to rank 0 to compute mAP over the
whole set -- no gradient sync needed, unlike training):
    torchrun --nproc_per_node=2 scripts/evaluate.py --checkpoint outputs/checkpoints/best.pt
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.coco_utils import index_annotations_by_image, index_images_by_id, load_coco, load_image  # noqa: E402
from src.data.masks import rasterize_instance_masks, rasterize_semantic_mask  # noqa: E402
from src.data.transforms import build_val_transforms  # noqa: E402
from src.models.unet_convnext import build_model  # noqa: E402
from src.utils.average_precision import mean_average_precision  # noqa: E402
from src.utils.device import get_device  # noqa: E402
from src.utils.distributed import is_distributed, setup_distributed  # noqa: E402
from src.utils.panoptic import panoptic_quality  # noqa: E402
from src.utils.postprocess import mask_to_instances  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

DEFAULT_SWEEP_GRID = "10,20,30,40,50,60,75,90,110,130"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--debug-limit", type=int, default=None, help="cap number of val images evaluated")
    p.add_argument("--iou-thresh", type=float, default=None, help="override PQ matching IoU threshold")
    p.add_argument("--prob-threshold", type=float, default=None, help="override postprocess.prob_threshold")
    p.add_argument("--watershed-min-distance", type=int, default=None, help="override postprocess.watershed_min_distance")
    p.add_argument("--min-instance-area", type=int, default=None, help="override postprocess.min_instance_area")
    p.add_argument("--no-watershed", action="store_true", help="disable watershed, use plain connected components instead")
    p.add_argument(
        "--sweep-watershed-min-distance", type=str, nargs="?", const=DEFAULT_SWEEP_GRID, default=None,
        help="sweep mode: comma-separated watershed_min_distance values to try against cached predictions "
             "(predicts once, tries every value cheaply on CPU). Prints a PQ table, then auto-computes full "
             "mAP for the best one. Pass with no value for a default grid.",
    )
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


def compute_instances_and_scores(
    binary_pred: np.ndarray, prob_map: np.ndarray, pp: dict
) -> tuple[list[np.ndarray], list[float]]:
    pred_instances = mask_to_instances(
        binary_pred, min_area=pp["min_instance_area"], use_watershed=pp["use_watershed"],
        watershed_min_distance=pp["watershed_min_distance"],
    )
    pred_scores = [float(prob_map[inst].mean()) for inst in pred_instances]
    return pred_instances, pred_scores


def main() -> None:
    args = parse_args()
    sweeping = args.sweep_watershed_min_distance is not None
    sweep_values = [int(v) for v in args.sweep_watershed_min_distance.split(",")] if sweeping else None

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
        print(f"Using device: {device}" + (f"  ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
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
    _, val_ids = split_image_ids(ann_index, cfg["data"]["val_fraction"], cfg["seed"])
    if args.debug_limit:
        val_ids = val_ids[: args.debug_limit]
    if is_main:
        print(f"Evaluating on {len(val_ids)} val images (checkpoint epoch {ckpt.get('epoch')})")
    shard_ids = val_ids[rank::world_size] if distributed else val_ids

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
    if is_main and (args.prob_threshold, args.watershed_min_distance, args.min_instance_area, args.no_watershed) != (None, None, None, False):
        print(f"postprocess overrides: {pp}")
    if is_main and sweeping:
        print(f"Sweep mode: will try watershed_min_distance in {sweep_values} against cached predictions")
    iou_thresh = args.iou_thresh if args.iou_thresh is not None else cfg["evaluate"]["iou_thresh"]

    if is_main:
        iterator = tqdm(shard_ids, desc="evaluate")
    else:
        print(f"[rank {rank}] evaluating {len(shard_ids)} images...")
        iterator = shard_ids

    dice_scores, iou_scores, all_gt_instances = [], [], []
    if sweeping:
        all_binary_preds, all_prob_maps = [], []
    else:
        pq_scores, all_pred_instances, all_pred_scores = [], [], []

    for image_id in iterator:
        img_meta = img_index[image_id]
        anns = ann_index[image_id]
        image = load_image(images_dir, img_meta["file_name"])
        h, w = img_meta["height"], img_meta["width"]

        prob_map = predict_probability_map(model, image, image_size, device)
        binary_pred = prob_map > pp["prob_threshold"]

        gt_semantic = rasterize_semantic_mask(anns, h, w).astype(bool)
        gt_instances = rasterize_instance_masks(anns, h, w)

        inter = np.logical_and(binary_pred, gt_semantic).sum()
        union = np.logical_or(binary_pred, gt_semantic).sum()
        dice = 2 * inter / (binary_pred.sum() + gt_semantic.sum()) if (binary_pred.sum() + gt_semantic.sum()) > 0 else 1.0
        iou = inter / union if union > 0 else 1.0
        dice_scores.append(dice)
        iou_scores.append(iou)
        all_gt_instances.append(gt_instances)

        if sweeping:
            # Deferred: pred_instances/scores depend on watershed_min_distance, which varies per
            # sweep point below -- cache the (cheap-to-store) inputs instead of recomputing per image.
            all_binary_preds.append(binary_pred)
            all_prob_maps.append(prob_map)
        else:
            pred_instances, pred_scores = compute_instances_and_scores(binary_pred, prob_map, pp)
            pq_scores.append(panoptic_quality(gt_instances, pred_instances, iou_thresh=iou_thresh))
            all_pred_instances.append(pred_instances)
            all_pred_scores.append(pred_scores)

    if not is_main:
        print(f"[rank {rank}] done")

    if distributed:
        # mAP (and, in sweep mode, the grid search itself) needs every image's data in one place --
        # gather_object is the collective for arbitrary picklable Python objects (here, lists of numpy
        # mask/probability arrays), unlike the tensor-only collectives (all_reduce/broadcast) train.py
        # uses. This call itself is a synchronization point, so no separate barrier is needed.
        local_results = {"dice": dice_scores, "iou": iou_scores, "gt": all_gt_instances}
        if sweeping:
            local_results["binary_pred"] = all_binary_preds
            local_results["prob_map"] = all_prob_maps
        else:
            local_results["pq"] = pq_scores
            local_results["pred"] = all_pred_instances
            local_results["scores"] = all_pred_scores
        gathered = [None] * world_size if is_main else None
        dist.gather_object(local_results, gathered, dst=0)
        if is_main:
            dice_scores = [x for r in gathered for x in r["dice"]]
            iou_scores = [x for r in gathered for x in r["iou"]]
            all_gt_instances = [x for r in gathered for x in r["gt"]]
            if sweeping:
                all_binary_preds = [x for r in gathered for x in r["binary_pred"]]
                all_prob_maps = [x for r in gathered for x in r["prob_map"]]
            else:
                pq_scores = [x for r in gathered for x in r["pq"]]
                all_pred_instances = [x for r in gathered for x in r["pred"]]
                all_pred_scores = [x for r in gathered for x in r["scores"]]

    best_wmd = None
    if is_main and sweeping:
        n_gt_total = sum(len(g) for g in all_gt_instances)
        results_table = []
        for wmd in tqdm(sweep_values, desc="sweep"):
            sweep_pp = {**pp, "watershed_min_distance": wmd}
            pq_list, n_pred_total = [], 0
            for binary_pred, prob_map, gt_instances in zip(all_binary_preds, all_prob_maps, all_gt_instances):
                pred_instances, _ = compute_instances_and_scores(binary_pred, prob_map, sweep_pp)
                pq_list.append(panoptic_quality(gt_instances, pred_instances, iou_thresh=iou_thresh))
                n_pred_total += len(pred_instances)
            results_table.append((wmd, float(np.mean(pq_list)), n_pred_total))

        results_table.sort(key=lambda r: r[1], reverse=True)
        print(f"\n{'watershed_min_distance':>24} {'mean PQ':>10} {'pred/gt instances':>20}")
        for wmd, mean_pq, n_pred in results_table:
            print(f"{wmd:>24} {mean_pq:>10.4f} {f'{n_pred}/{n_gt_total}':>20}")

        best_wmd = results_table[0][0]
        print(f"\nBest: watershed_min_distance={best_wmd} (mean PQ={results_table[0][1]:.4f}) -- computing full mAP for this config below.\n")
        pp = {**pp, "watershed_min_distance": best_wmd}

        pq_scores, all_pred_instances, all_pred_scores = [], [], []
        for binary_pred, prob_map, gt_instances in zip(all_binary_preds, all_prob_maps, all_gt_instances):
            pred_instances, pred_scores = compute_instances_and_scores(binary_pred, prob_map, pp)
            pq_scores.append(panoptic_quality(gt_instances, pred_instances, iou_thresh=iou_thresh))
            all_pred_instances.append(pred_instances)
            all_pred_scores.append(pred_scores)

    if is_main:
        n_gt_total = sum(len(g) for g in all_gt_instances)
        n_pred_total = sum(len(p) for p in all_pred_instances)
        print(
            f"Computing mAP@[.5:.95]: {n_pred_total} predicted instances vs {n_gt_total} ground-truth "
            f"instances across {len(val_ids)} images, over 10 IoU thresholds -- this is the slow part "
            f"(full-resolution mask matching), progress bar below."
        )
        t_map = time.time()
        mAP, ap_per_thresh = mean_average_precision(all_gt_instances, all_pred_instances, all_pred_scores, show_progress=True)
        print(f"mAP computation took {time.time() - t_map:.1f}s")

        print()
        print(f"mean Dice (semantic):  {np.mean(dice_scores):.4f}")
        print(f"mean IoU  (semantic):  {np.mean(iou_scores):.4f}")
        print(f"mean Panoptic Quality: {np.mean(pq_scores):.4f}  (IoU thresh={iou_thresh})")
        print(f"mAP@[.5:.95]:          {mAP:.4f}")
        print(f"  AP@0.50:             {ap_per_thresh[0.5]:.4f}")
        print(f"  AP@0.75:             {ap_per_thresh[0.75]:.4f}")
        if sweeping:
            print(f"(winning watershed_min_distance={best_wmd} -- pass --watershed-min-distance {best_wmd} next time to reproduce this without re-sweeping)")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
