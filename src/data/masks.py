"""Rasterize MAGFiLO COCO polygon annotations into pixel masks.

The competition scores category-agnostic filament instances (see the Overview
page: "this competition focuses exclusively on the segmentation masks of
solar filaments"), so the training target merges all 4 annotation categories
into a single foreground class. Per-instance masks are kept separately for
local Panoptic-Quality evaluation against ground truth.
"""

from __future__ import annotations

import cv2
import numpy as np

from src.data.coco_utils import polygons_from_segmentation


def rasterize_semantic_mask(anns: list[dict], height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    for ann in anns:
        for poly in polygons_from_segmentation(ann["segmentation"]):
            cv2.fillPoly(mask, [poly.round().astype(np.int32)], color=1)
    return mask


def rasterize_instance_masks(anns: list[dict], height: int, width: int) -> list[np.ndarray]:
    instances = []
    for ann in anns:
        mask = np.zeros((height, width), dtype=np.uint8)
        for poly in polygons_from_segmentation(ann["segmentation"]):
            cv2.fillPoly(mask, [poly.round().astype(np.int32)], color=1)
        instances.append(mask.astype(bool))
    return instances
