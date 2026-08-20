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


def panoptic_quality(gt_masks: list[np.ndarray], pred_masks: list[np.ndarray], iou_thresh: float = 0.5) -> float:
    if not gt_masks and not pred_masks:
        return 1.0
    if not gt_masks or not pred_masks:
        return 0.0

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

    tp_ious = []
    matched_gt, matched_pred = set(), set()
    for i, j in zip(gt_idx, pred_idx):
        if iou_matrix[i, j] > iou_thresh:
            tp_ious.append(iou_matrix[i, j])
            matched_gt.add(i)
            matched_pred.add(j)

    tp = len(tp_ious)
    fp = n_pred - len(matched_pred)
    fn = n_gt - len(matched_gt)
    denom = tp + 0.5 * fp + 0.5 * fn
    return sum(tp_ious) / denom if denom > 0 else 1.0
