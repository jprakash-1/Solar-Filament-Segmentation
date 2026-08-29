"""Local Dice + Panoptic Quality, mirroring the competition's own metric family so
local numbers are a meaningful stand-in for leaderboard PQ before spending a
submission slot. Convention for the empty-prediction/empty-GT edge case (both empty
-> perfect score of 1.0, excluded from nothing) is a reasonable default but has NOT
been verified against the organizers' self-evaluation notebook -- do that before
trusting absolute local PQ numbers, not just relative ones across experiments.
"""

from __future__ import annotations

import numpy as np


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    return int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1


def _iou_matrix(gt_masks: list[np.ndarray], pred_masks: list[np.ndarray]) -> np.ndarray:
    """Bounding-box-restricted IoU computation -- a full O(n_gt * n_pred) pairwise
    logical_and/or over full-resolution (2048x2048) masks is fine for a handful of
    instances but becomes pathological if a noisy model/baseline produces hundreds
    or thousands of predicted instances (observed with an early, under-tuned version
    of scripts/baseline_classical.py: 3267 predicted instances on one image made
    this loop take hours). Skip the full-array op entirely for non-overlapping
    bounding boxes (the common case once instance counts get large), and restrict it
    to the small bbox-intersection region otherwise.
    """
    n_gt, n_pred = len(gt_masks), len(pred_masks)
    gt_areas = [int(m.sum()) for m in gt_masks]
    pred_areas = [int(m.sum()) for m in pred_masks]
    gt_boxes = [_bbox(m) for m in gt_masks]
    pred_boxes = [_bbox(m) for m in pred_masks]

    iou_matrix = np.zeros((n_gt, n_pred))
    for i in range(n_gt):
        gb = gt_boxes[i]
        if gb is None or gt_areas[i] == 0:
            continue
        gy0, gy1, gx0, gx1 = gb
        for j in range(n_pred):
            pb = pred_boxes[j]
            if pb is None or pred_areas[j] == 0:
                continue
            py0, py1, px0, px1 = pb
            y0, y1 = max(gy0, py0), min(gy1, py1)
            x0, x1 = max(gx0, px0), min(gx1, px1)
            if y0 >= y1 or x0 >= x1:
                continue  # bounding boxes don't overlap -> IoU is exactly 0
            inter = np.logical_and(gt_masks[i][y0:y1, x0:x1], pred_masks[j][y0:y1, x0:x1]).sum()
            if inter == 0:
                continue
            union = gt_areas[i] + pred_areas[j] - inter
            iou_matrix[i, j] = inter / union if union > 0 else 0.0
    return iou_matrix


def dice_score(pred_semantic: np.ndarray, gt_semantic: np.ndarray) -> float:
    """Semantic (pixel-level) Dice between two binary masks."""
    inter = np.logical_and(pred_semantic, gt_semantic).sum()
    denom = pred_semantic.sum() + gt_semantic.sum()
    if denom == 0:
        return 1.0
    return float(2.0 * inter) / float(denom)


def panoptic_quality(gt_masks: list[np.ndarray], pred_masks: list[np.ndarray], iou_thresh: float = 0.5) -> dict:
    """Per-image PQ = SQ x RQ, matched by IoU > iou_thresh (unique matching guaranteed
    by the >0.5 threshold). Returns a dict with pq/sq/rq/tp/fp/fn so callers can
    aggregate however they need (mean over images, or pooled TP/FP/FN then one PQ).
    """
    n_gt, n_pred = len(gt_masks), len(pred_masks)
    if n_gt == 0 and n_pred == 0:
        return {"pq": 1.0, "sq": 1.0, "rq": 1.0, "tp": 0, "fp": 0, "fn": 0, "sum_tp_iou": 0.0}
    if n_gt == 0 or n_pred == 0:
        # everything is an FP or FN -- RQ collapses to 0, PQ is 0 regardless of SQ
        return {"pq": 0.0, "sq": 0.0, "rq": 0.0, "tp": 0, "fp": n_pred, "fn": n_gt, "sum_tp_iou": 0.0}

    iou_matrix = _iou_matrix(gt_masks, pred_masks)

    candidates = [(iou_matrix[i, j], i, j) for i in range(n_gt) for j in range(n_pred) if iou_matrix[i, j] > iou_thresh]
    candidates.sort(key=lambda t: -t[0])

    matched_gt, matched_pred, tp_ious = set(), set(), []
    for iou, i, j in candidates:
        if i in matched_gt or j in matched_pred:
            continue
        matched_gt.add(i)
        matched_pred.add(j)
        tp_ious.append(iou)

    tp = len(tp_ious)
    fp = n_pred - len(matched_pred)
    fn = n_gt - len(matched_gt)
    denom = tp + 0.5 * fp + 0.5 * fn
    sq = float(np.mean(tp_ious)) if tp_ious else 0.0
    rq = tp / denom if denom > 0 else 0.0
    pq = (sum(tp_ious) / denom) if denom > 0 else 0.0

    return {"pq": pq, "sq": sq, "rq": rq, "tp": tp, "fp": fp, "fn": fn, "sum_tp_iou": float(sum(tp_ious))}


def aggregate_pq(per_image_results: list[dict]) -> dict:
    """Two aggregation conventions, report both: mean-of-per-image-PQ (what most
    public notebooks report) and pooled-TP/FP/FN-then-one-PQ (matches the formal
    per-dataset PQ definition more closely). Compare against the self-eval notebook
    to see which one they use before treating either as authoritative.
    """
    mean_pq = float(np.mean([r["pq"] for r in per_image_results])) if per_image_results else 0.0
    total_tp = sum(r["tp"] for r in per_image_results)
    total_fp = sum(r["fp"] for r in per_image_results)
    total_fn = sum(r["fn"] for r in per_image_results)
    denom = total_tp + 0.5 * total_fp + 0.5 * total_fn
    total_tp_iou = sum(r["sum_tp_iou"] for r in per_image_results)
    pooled_pq = (total_tp_iou / denom) if denom > 0 else 0.0
    return {
        "mean_per_image_pq": mean_pq,
        "pooled_pq": pooled_pq,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
    }
