"""COCO-style mask Average Precision, averaged over IoU thresholds 0.50-0.95.

Single-class (the competition's filament instances are category-agnostic), so this is
what COCO calls "AP" -- averaging over classes as well would make it "mAP", but with one
class the two are the same number. Used only for local sanity-checking
(scripts/evaluate.py), same as panoptic_quality in src/utils/panoptic.py.
"""

from __future__ import annotations

import numpy as np
from tqdm import tqdm


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """(rmin, rmax, cmin, cmax), or None if the mask is empty."""
    rows = np.any(mask, axis=1)
    if not rows.any():
        return None
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return int(rmin), int(rmax), int(cmin), int(cmax)


def _boxes_overlap(a: tuple[int, int, int, int] | None, b: tuple[int, int, int, int] | None) -> bool:
    if a is None or b is None:
        return False
    a_rmin, a_rmax, a_cmin, a_cmax = a
    b_rmin, b_rmax, b_cmin, b_cmax = b
    return a_rmin <= b_rmax and b_rmin <= a_rmax and a_cmin <= b_cmax and b_cmin <= a_cmax


def compute_ap_at_iou(
    gt_by_image: list[list[np.ndarray]],
    pred_by_image: list[list[np.ndarray]],
    scores_by_image: list[list[float]],
    iou_thresh: float,
    show_progress: bool = False,
) -> float:
    total_gt = sum(len(g) for g in gt_by_image)
    total_pred = sum(len(p) for p in pred_by_image)
    if total_gt == 0:
        return 1.0 if total_pred == 0 else 0.0
    if total_pred == 0:
        return 0.0

    gt_bboxes = [[_bbox(g) for g in gts] for gts in gt_by_image]

    flat_preds = []  # (image_idx, mask, bbox, score)
    for image_idx, (preds, scores) in enumerate(zip(pred_by_image, scores_by_image)):
        for mask, score in zip(preds, scores):
            flat_preds.append((image_idx, mask, _bbox(mask), score))
    flat_preds.sort(key=lambda p: p[3], reverse=True)

    matched_gt = [set() for _ in gt_by_image]
    tps = np.zeros(len(flat_preds))
    fps = np.zeros(len(flat_preds))

    iterator = tqdm(flat_preds, desc=f"AP@{iou_thresh}", leave=False) if show_progress else flat_preds
    for i, (image_idx, pred_mask, pred_bbox, _score) in enumerate(iterator):
        gts = gt_by_image[image_idx]
        bboxes = gt_bboxes[image_idx]
        best_iou, best_j = 0.0, -1
        for j, g in enumerate(gts):
            if j in matched_gt[image_idx] or not _boxes_overlap(pred_bbox, bboxes[j]):
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
    show_progress: bool = False,
) -> tuple[float, dict[float, float]]:
    if iou_thresholds is None:
        iou_thresholds = np.arange(0.5, 1.0, 0.05)
    thresholds = [round(float(t), 2) for t in iou_thresholds]
    iterator = tqdm(thresholds, desc="mAP thresholds") if show_progress else thresholds
    per_thresh = {
        t: compute_ap_at_iou(gt_by_image, pred_by_image, scores_by_image, t, show_progress=show_progress)
        for t in iterator
    }
    return float(np.mean(list(per_thresh.values()))), per_thresh
