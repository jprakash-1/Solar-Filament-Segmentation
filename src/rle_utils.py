"""Binary mask <-> COCO RLE helpers, matching the submission format's exact contract:
RLE *counts* only (not the full {size, counts} dict), size fixed and implicit at
2048x2048 for every image. Always encode at native 2048x2048 -- never encode a
resized mask and resize the RLE after the fact.
"""

from __future__ import annotations

import numpy as np
from pycocotools import mask as maskUtils

SUBMISSION_SIZE = (2048, 2048)


def mask_to_rle_counts(binary_mask: np.ndarray) -> str:
    """binary_mask: (H, W) array of 0/1 (or bool). Returns the RLE counts string only."""
    fortran_mask = np.asfortranarray(binary_mask.astype(np.uint8))
    rle = maskUtils.encode(fortran_mask)
    return rle["counts"].decode("utf-8")


def rle_counts_to_mask(counts: str, size: tuple[int, int] = SUBMISSION_SIZE) -> np.ndarray:
    """Inverse of mask_to_rle_counts -- round-trip check / GT decoding."""
    rle = {"size": list(size), "counts": counts.encode("utf-8")}
    return maskUtils.decode(rle)
