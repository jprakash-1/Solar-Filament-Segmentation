"""COCO-style mask Average Precision, averaged over IoU thresholds 0.50-0.95.

Single-class (the competition's filament instances are category-agnostic), so this is
what COCO calls "AP" -- averaging over classes as well would make it "mAP", but with one
class the two are the same number. Used only for local sanity-checking
(scripts/evaluate.py), same as panoptic_quality in src/utils/panoptic.py.
"""

from __future__ import annotations

import numpy as np


def compute_ap_at_iou(
    gt_by_image: list[list[np.ndarray]],
    pred_by_image: list[list[np.ndarray]],
    scores_by_image: list[list[float]],
    iou_thresh: float,
) -> float:
    total_gt = sum(len(g) for g in gt_by_image)
    total_pred = sum(len(p) for p in pred_by_image)
    if total_gt == 0:
        return 1.0 if total_pred == 0 else 0.0
    if total_pred == 0:
        return 0.0

    flat_preds = []  # (image_idx, mask, score)
    for image_idx, (preds, scores) in enumerate(zip(pred_by_image, scores_by_image)):
        for mask, score in zip(preds, scores):
            flat_preds.append((image_idx, mask, score))
    flat_preds.sort(key=lambda p: p[2], reverse=True)

    matched_gt = [set() for _ in gt_by_image]
    tps = np.zeros(len(flat_preds))
    fps = np.zeros(len(flat_preds))

    for i, (image_idx, pred_mask, _score) in enumerate(flat_preds):
        gts = gt_by_image[image_idx]
        best_iou, best_j = 0.0, -1
        for j, g in enumerate(gts):
            if j in matched_gt[image_idx]:
                continue
            inter = np.logical_and(pred_mask, g).sum()
            if inter == 0:
                continue
            union = np.logical_or(pred_mask, g).sum()
            iou = inter / union
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= iou_thresh:
            tps[i] = 1
            matched_gt[image_idx].add(best_j)
        else:
            fps[i] = 1

    cum_tp = np.cumsum(tps)
    cum_fp = np.cumsum(fps)
    recalls = cum_tp / total_gt
    precisions = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)

    # COCO's 101-point interpolation: precision at recall level r is the max precision
    # observed at any recall >= r.
    recall_levels = np.linspace(0.0, 1.0, 101)
    interpolated = np.zeros_like(recall_levels)
    for k, r in enumerate(recall_levels):
        above = precisions[recalls >= r]
        interpolated[k] = above.max() if above.size > 0 else 0.0
    return float(interpolated.mean())


def mean_average_precision(
    gt_by_image: list[list[np.ndarray]],
    pred_by_image: list[list[np.ndarray]],
    scores_by_image: list[list[float]],
    iou_thresholds: np.ndarray | None = None,
) -> tuple[float, dict[float, float]]:
    if iou_thresholds is None:
        iou_thresholds = np.arange(0.5, 1.0, 0.05)
    per_thresh = {
        round(float(t), 2): compute_ap_at_iou(gt_by_image, pred_by_image, scores_by_image, round(float(t), 2))
        for t in iou_thresholds
    }
    return float(np.mean(list(per_thresh.values()))), per_thresh
