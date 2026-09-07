#!/usr/bin/env python3
"""Fuse duplicate-annotator versions of the same underlying image into one
canonical instance set per `file_name`, per the routing rule agreed in
`jp-analysis:memory.md`'s "Training-strategy discussion" entry (not yet built
there -- this is that build).

Why: 296 of 707 unique training images were independently labeled by 2-3
annotators (`outputs/dup/findings.md`, mean inter-annotator mask IoU 0.471,
2 groups at IoU=0.000). Training on each annotator version as its own
whole-image sample means the model gets flatly contradictory full-image
supervision for the same input pixels depending on which version a batch
draws. This script resolves that once, offline, rather than at every
training step.

Routing rule (per group of >=2 annotator versions for one `file_name`):
- **Near-zero mean-pairwise-IoU groups** (`--near-zero-threshold`, default
  0.1): annotators marked essentially disjoint filaments, not competing
  opinions about one shape -- take the **union** of every version's
  instances, unmatched.
- **Everything else**: **per-instance-matched fusion**, not a flat
  semantic-mask union (a flat union risks welding two nearby distinct
  filaments into one blob, which directly hurts PQ's instance-separation
  scoring -- see `src/metrics.panoptic_quality`). The first version is the
  matching reference; each other version's instances are greedily
  IoU-matched against it (`--match-iou-threshold`, default 0.3 -- looser
  than PQ's own 0.5 since this is finding "the same real filament," not
  scoring a prediction). Matched clusters are fused via **strict per-pixel
  majority vote** (`vote > 0.5`, i.e. more than half the versions in that
  cluster must agree a pixel is foreground -- this deliberately degrades to
  *intersection* for a 2-annotator cluster rather than union, since
  `memory.md` explicitly warns a union would "systematically inflate/thicken
  every mask beyond what either annotator actually drew"; a genuine 3-way
  majority for a 3-annotator cluster). Unmatched instances (caught by only
  one version, no correspondence in the others) are kept as their own
  independent instance, not dropped -- no evidence they're spurious rather
  than a real miss by the other annotator(s).

Fused masks are re-vectorized back to a single polygon via
`cv2.findContours` (largest contour only, matching `MVP1_PLAN.md`'s own
documented single-polygon-per-instance assumption the rest of the pipeline
already relies on) so the output stays plug-compatible with every existing
consumer (`src/dataset.FilamentDataset`, `src/crop_dataset.FilamentCropDataset`,
`pycocotools`) without inventing a parallel raster format.

Usage:
    python scripts/fuse_duplicate_annotations.py \
        --data-json data/raw/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json \
        --out data/processed/MAGFiLO_fused_train.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from pycocotools.coco import COCO
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.metrics import _iou_matrix  # noqa: E402 -- reuse the bbox-restricted IoU computation as-is; matching loop below is small enough to keep local rather than touching src/metrics.py (on the live training critical path)

logger = logging.getLogger("fuse_duplicate_annotations")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--data-json",
        type=Path,
        default=Path("data/raw/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"),
    )
    p.add_argument("--out", type=Path, default=Path("data/processed/MAGFiLO_fused_train.json"))
    p.add_argument("--near-zero-threshold", type=float, default=0.1, help="mean pairwise semantic-mask IoU below this routes a duplicate group to union instead of matched fusion")
    p.add_argument("--match-iou-threshold", type=float, default=0.3, help="IoU threshold for matching one annotator version's instance to another's -- looser than PQ's own 0.5 since this is finding correspondence, not scoring a prediction")
    return p.parse_args()


def greedy_match(gt_masks: list[np.ndarray], pred_masks: list[np.ndarray], iou_thresh: float) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    """Greedy unique IoU matching -- same pattern src/metrics.panoptic_quality uses
    internally, kept local here (not imported) so this script never touches
    src/metrics.py, which sits on the live training critical path.
    Returns (matched_pairs=[(gt_idx, pred_idx, iou), ...], unmatched_gt_indices, unmatched_pred_indices).
    """
    n_gt, n_pred = len(gt_masks), len(pred_masks)
    if n_gt == 0 or n_pred == 0:
        return [], list(range(n_gt)), list(range(n_pred))
    iou_matrix = _iou_matrix(gt_masks, pred_masks)
    candidates = sorted(
        ((iou_matrix[i, j], i, j) for i in range(n_gt) for j in range(n_pred) if iou_matrix[i, j] > iou_thresh),
        key=lambda t: -t[0],
    )
    matched_gt, matched_pred, pairs = set(), set(), []
    for iou, i, j in candidates:
        if i in matched_gt or j in matched_pred:
            continue
        matched_gt.add(i)
        matched_pred.add(j)
        pairs.append((i, j, iou))
    unmatched_gt = [i for i in range(n_gt) if i not in matched_gt]
    unmatched_pred = [j for j in range(n_pred) if j not in matched_pred]
    return pairs, unmatched_gt, unmatched_pred


def instance_masks_for(coco: COCO, image_id) -> tuple[list[dict], list[np.ndarray]]:
    ann_ids = coco.getAnnIds(imgIds=[image_id])
    anns = coco.loadAnns(ann_ids)
    return anns, [coco.annToMask(ann) for ann in anns]


def semantic_mask_from(instance_masks: list[np.ndarray], h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    for m in instance_masks:
        mask = np.maximum(mask, m)
    return mask


def mean_pairwise_iou(semantic_masks: list[np.ndarray]) -> float:
    ious = []
    for i in range(len(semantic_masks)):
        for j in range(i + 1, len(semantic_masks)):
            inter = np.logical_and(semantic_masks[i], semantic_masks[j]).sum()
            union = np.logical_or(semantic_masks[i], semantic_masks[j]).sum()
            ious.append(float(inter) / float(union) if union > 0 else 1.0)
    return float(np.mean(ious)) if ious else 1.0


def mask_to_instance_dict(mask: np.ndarray) -> dict | None:
    """Re-vectorize a binary mask to a single COCO-style polygon instance dict.
    Returns None for a degenerate/empty mask -- caller drops it with a warning
    rather than crashing on a real edge case in real data."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)  # single-polygon-per-instance, matching MVP1_PLAN.md's documented assumption the rest of the pipeline relies on
    if len(largest) < 3:
        return None
    poly = largest.reshape(-1, 2).astype(np.float64)
    area = float(cv2.contourArea(largest))
    if area <= 0:
        return None
    x, y, bw, bh = cv2.boundingRect(largest)
    flat = poly.flatten().tolist()
    if flat[:2] != flat[-2:]:
        flat = flat + flat[:2]  # close the polygon (first == last), matching the documented annotation format
    return {"segmentation": [flat], "area": area, "bbox": [float(x), float(y), float(bw), float(bh)]}


def ann_to_instance_dict(ann: dict) -> dict:
    return {"segmentation": ann["segmentation"], "area": ann["area"], "bbox": ann["bbox"]}


def fuse_matched_group(coco: COCO, group_image_ids: list, h: int, w: int, match_iou_thresh: float) -> list[dict]:
    per_version_masks: list[list[np.ndarray]] = []
    for iid in group_image_ids:
        _, masks = instance_masks_for(coco, iid)
        per_version_masks.append(masks)

    ref_masks = per_version_masks[0]
    clusters: list[list[np.ndarray]] = [[m] for m in ref_masks]
    leftover: list[np.ndarray] = []

    for v in range(1, len(per_version_masks)):
        pairs, _unmatched_ref, unmatched_v = greedy_match(ref_masks, per_version_masks[v], match_iou_thresh)
        for ref_i, v_j, _iou in pairs:
            clusters[ref_i].append(per_version_masks[v][v_j])
        for v_j in unmatched_v:
            leftover.append(per_version_masks[v][v_j])

    fused: list[dict] = []
    for cluster in clusters + [[m] for m in leftover]:
        if len(cluster) == 1:
            fused_mask = cluster[0]
        else:
            vote = np.stack(cluster, axis=0).astype(np.float32).mean(axis=0)
            fused_mask = (vote > 0.5).astype(np.uint8)  # strict majority -- degrades to intersection for a 2-way cluster, deliberately not union
        inst = mask_to_instance_dict(fused_mask)
        if inst is not None:
            fused.append(inst)
        else:
            logger.warning(f"dropped a degenerate/empty fused instance in group {group_image_ids}")
    return fused


def process_group(coco: COCO, group_image_ids: list, h: int, w: int, near_zero_threshold: float, match_iou_thresh: float) -> tuple[list[dict], str]:
    if len(group_image_ids) == 1:
        anns = coco.loadAnns(coco.getAnnIds(imgIds=group_image_ids))
        return [ann_to_instance_dict(a) for a in anns], "single"

    semantic_masks = []
    for iid in group_image_ids:
        _, masks = instance_masks_for(coco, iid)
        semantic_masks.append(semantic_mask_from(masks, h, w))
    mean_iou = mean_pairwise_iou(semantic_masks)

    if mean_iou < near_zero_threshold:
        fused = []
        for iid in group_image_ids:
            anns = coco.loadAnns(coco.getAnnIds(imgIds=[iid]))
            fused.extend(ann_to_instance_dict(a) for a in anns)
        return fused, "union"

    return fuse_matched_group(coco, group_image_ids, h, w, match_iou_thresh), "matched-fusion"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    args = parse_args()

    coco = COCO(str(args.data_json))
    by_file: dict[str, list] = defaultdict(list)
    for image_id, info in coco.imgs.items():
        by_file[info["file_name"]].append(image_id)

    fused_images, fused_annotations = [], []
    route_counts = {"single": 0, "union": 0, "matched-fusion": 0}
    ann_counter = 0
    for file_name, group_ids in tqdm(by_file.items(), desc="fusing"):
        info = coco.imgs[group_ids[0]]
        h, w = info["height"], info["width"]
        instances, route = process_group(coco, group_ids, h, w, args.near_zero_threshold, args.match_iou_threshold)
        route_counts[route] += 1

        fused_images.append({"id": file_name, "file_name": file_name, "height": h, "width": w})
        for inst in instances:
            ann_counter += 1
            fused_annotations.append({
                "id": f"fused-{ann_counter}",
                "image_id": file_name,
                "category_id": 1,
                "segmentation": inst["segmentation"],
                "area": inst["area"],
                "bbox": inst["bbox"],
                "iscrowd": 0,
            })

    out_data = {
        "info": {}, "licenses": [],
        "categories": [{"id": 1, "name": "filament"}],
        "images": fused_images,
        "annotations": fused_annotations,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out_data))
    logger.info(f"routes: {route_counts} -- {len(fused_images)} fused images, {len(fused_annotations)} fused instances -> {args.out}")


if __name__ == "__main__":
    main()
