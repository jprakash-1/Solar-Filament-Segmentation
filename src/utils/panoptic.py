"""Panoptic Quality metric, matching the Overview page's exact formula
(itself the standard PQ definition from Kirillov et al., CVPR 2019):

    PQ = sum(IoU over TP matches) / (|TP| + 0.5*|FP| + 0.5*|FN|)

A ground-truth/predicted instance pair is a true positive if their IoU > 0.5
(a threshold this high guarantees at most one valid match per instance, so
matching is unique). Used only for local sanity-checking (scripts/evaluate.py)
against the actual leaderboard metric.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def panoptic_quality_detailed(gt_masks: list[np.ndarray], pred_masks: list[np.ndarray], iou_thresh: float = 0.5) -> dict:
    """Same matching as panoptic_quality(), but also returns the per-match IoUs and
    which instances were left unmatched -- the detail scripts/error_analysis.py needs
    that the plain scalar PQ throws away."""
    if not gt_masks and not pred_masks:
        return {"pq": 1.0, "matched_ious": [], "fn_indices": [], "fp_indices": []}
    if not gt_masks:
        return {"pq": 0.0, "matched_ious": [], "fn_indices": [], "fp_indices": list(range(len(pred_masks)))}
    if not pred_masks:
        return {"pq": 0.0, "matched_ious": [], "fn_indices": list(range(len(gt_masks))), "fp_indices": []}

    n_gt, n_pred = len(gt_masks), len(pred_masks)
    iou_matrix = np.zeros((n_gt, n_pred))
    for i, g in enumerate(gt_masks):
        for j, p in enumerate(pred_masks):
            inter = np.logical_and(g, p).sum()
            if inter == 0:
                continue
            union = np.logical_or(g, p).sum()
            iou_matrix[i, j] = inter / union

    gt_idx, pred_idx = linear_sum_assignment(-iou_matrix)

    matched_ious = []
    matched_gt, matched_pred = set(), set()
    for i, j in zip(gt_idx, pred_idx):
        if iou_matrix[i, j] > iou_thresh:
            matched_ious.append(float(iou_matrix[i, j]))
            matched_gt.add(i)
            matched_pred.add(j)

    fn_indices = [i for i in range(n_gt) if i not in matched_gt]
    fp_indices = [j for j in range(n_pred) if j not in matched_pred]

    tp = len(matched_ious)
    fp = len(fp_indices)
    fn = len(fn_indices)
    denom = tp + 0.5 * fp + 0.5 * fn
    pq = sum(matched_ious) / denom if denom > 0 else 1.0

    return {"pq": pq, "matched_ious": matched_ious, "fn_indices": fn_indices, "fp_indices": fp_indices}


def panoptic_quality(gt_masks: list[np.ndarray], pred_masks: list[np.ndarray], iou_thresh: float = 0.5) -> float:
    return panoptic_quality_detailed(gt_masks, pred_masks, iou_thresh)["pq"]
