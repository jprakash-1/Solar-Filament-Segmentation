"""COCO-style annotation parsing and the training Dataset.

Gotcha this file exists specifically to handle correctly: `images[].id` is a string
that encodes both an annotator/batch id and the underlying filename (e.g.
"010401-20160920230134Lh"), and the same physical JPEG can appear multiple times
under different `image_id`s -- one per independent annotator pass. Splitting by
`image_id` would let the same pixels leak across train/val under a different
annotator's polygons. Group key for splitting is `file_name`, never `image_id`.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset


def group_split(coco: COCO, val_fraction: float = 0.15, seed: int = 0) -> tuple[list[str], list[str]]:
    """Group-aware train/val split over image_ids, grouped by file_name so that
    duplicate-annotator versions of the same underlying image never land on
    opposite sides of the split.
    """
    image_ids = sorted(coco.imgs.keys())
    file_names = [coco.imgs[iid]["file_name"] for iid in image_ids]
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    train_idx, val_idx = next(splitter.split(image_ids, groups=file_names))
    train_ids = [image_ids[i] for i in train_idx]
    val_ids = [image_ids[i] for i in val_idx]
    return train_ids, val_ids


class FilamentDataset(Dataset):
    """MVP1 target: one binary semantic mask per image (union of all instance
    polygons). Instance separation happens at postprocessing time (src/postprocess.py),
    not in the loss -- the fastest path to a working submission, and a legitimate
    longer-term approach too (several public notebooks use exactly this).

    Images with zero annotations are kept as valid negatives (empty mask), not
    filtered out -- the model needs to learn what "no filament" looks like, and the
    test set will contain filament-free frames.
    """

    def __init__(self, coco_json: str | Path, img_dir: str | Path, image_ids: list[str], img_size: int = 256):
        self.coco = COCO(str(coco_json))
        self.img_dir = Path(img_dir)
        self.ids = image_ids
        self.img_size = img_size

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        image_id = self.ids[idx]
        info = self.coco.imgs[image_id]
        img = cv2.imread(str(self.img_dir / info["file_name"]), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(self.img_dir / info["file_name"])
        h, w = img.shape
        assert (h, w) == (info["height"], info["width"]), f"size mismatch for {info['file_name']}: read {(h, w)}, json says {(info['height'], info['width'])}"

        ann_ids = self.coco.getAnnIds(imgIds=[image_id])
        anns = self.coco.loadAnns(ann_ids)
        semantic_mask = np.zeros((h, w), dtype=np.uint8)
        for ann in anns:
            m = self.coco.annToMask(ann)
            semantic_mask = np.maximum(semantic_mask, m)

        img_rs = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        mask_rs = cv2.resize(semantic_mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        img_t = torch.from_numpy(img_rs).float().unsqueeze(0) / 255.0
        mask_t = torch.from_numpy(mask_rs).float().unsqueeze(0)
        return img_t, mask_t, image_id, (h, w)

    def get_instance_masks(self, image_id: str) -> list[np.ndarray]:
        """Native-resolution per-instance GT masks for local PQ evaluation."""
        ann_ids = self.coco.getAnnIds(imgIds=[image_id])
        anns = self.coco.loadAnns(ann_ids)
        return [self.coco.annToMask(ann) for ann in anns]
