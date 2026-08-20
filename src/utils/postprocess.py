"""Post-processing: binary mask -> filament instances -> COCO RLE.

The model predicts one semantic foreground probability map, but the
competition scores individual filament instances (Panoptic Quality) and
explicitly penalizes over-merging. Distance-transform watershed splits
touching/adjacent filaments that a plain connected-components pass would
merge into a single instance.
"""

from __future__ import annotations

import numpy as np
from pycocotools import mask as mask_utils
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed


def mask_to_instances(
    binary_mask: np.ndarray,
    min_area: int = 20,
    use_watershed: bool = True,
    watershed_min_distance: int = 8,
) -> list[np.ndarray]:
    binary_mask = binary_mask.astype(bool)
    if not binary_mask.any():
        return []

    if use_watershed:
        distance = ndi.distance_transform_edt(binary_mask)
        coords = peak_local_max(distance, min_distance=watershed_min_distance, labels=binary_mask)
        peak_mask = np.zeros_like(distance, dtype=bool)
        peak_mask[tuple(coords.T)] = True
        markers, _ = ndi.label(peak_mask)
        labels = watershed(-distance, markers, mask=binary_mask)
    else:
        labels, _ = ndi.label(binary_mask)

    instances = []
    for label_id in range(1, labels.max() + 1):
        inst_mask = labels == label_id
        if inst_mask.sum() >= min_area:
            instances.append(inst_mask)
    return instances


def rle_encode(mask: np.ndarray) -> str:
    fortran_mask = np.asfortranarray(mask.astype(np.uint8))
    rle = mask_utils.encode(fortran_mask)
    counts = rle["counts"]
    return counts.decode("utf-8") if isinstance(counts, bytes) else counts


def rle_decode(counts: str, height: int, width: int) -> np.ndarray:
    rle = {"counts": counts.encode("utf-8"), "size": [height, width]}
    return mask_utils.decode(rle).astype(bool)
