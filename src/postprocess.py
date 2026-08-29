"""Binary mask -> per-instance mask extraction.

Discipline that matters for PQ: upsample the *probability* map to the native
2048x2048 resolution first, threshold second, connected-component-label third.
Thresholding at low resolution and upsampling the binary mask instead introduces
blocky artifacts and can merge nearby filaments that shouldn't be merged -- a direct
hit to PQ's RQ term (over-merging).
"""

from __future__ import annotations

import cv2
import numpy as np


def mask_to_instances(binary_mask: np.ndarray, min_area_px: int = 15) -> list[np.ndarray]:
    """binary_mask: (H, W) uint8/bool array at the target (native) resolution.
    Returns a list of (H, W) uint8 binary masks, one per connected component,
    dropping components smaller than min_area_px (speckle-noise filter).
    """
    n_labels, labels = cv2.connectedComponentsWithStats(binary_mask.astype(np.uint8), connectivity=8)[:2]
    instances = []
    for lbl in range(1, n_labels):  # 0 = background
        inst = (labels == lbl).astype(np.uint8)
        if inst.sum() >= min_area_px:
            instances.append(inst)
    return instances


def prob_map_to_instances(prob_map_low_res: np.ndarray, target_size: tuple[int, int], prob_thresh: float = 0.5, min_area_px: int = 15) -> list[np.ndarray]:
    """prob_map_low_res: (h, w) float probability map at model resolution.
    target_size: (H, W) to upsample to before thresholding (native image size).
    """
    h, w = target_size
    prob_full = cv2.resize(prob_map_low_res, (w, h), interpolation=cv2.INTER_LINEAR)
    binary_full = (prob_full > prob_thresh).astype(np.uint8)
    return mask_to_instances(binary_full, min_area_px=min_area_px)
