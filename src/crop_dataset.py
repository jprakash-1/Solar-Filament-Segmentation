"""Crop-based training Dataset -- Stage 1 of the two-stage curriculum agreed in
`jp-analysis:memory.md`'s "Training-strategy discussion" entry. Samples
foreground-biased crops instead of feeding the whole 2048x2048 image per training
step, directly targeting the ~204:1 background:foreground pixel imbalance
(`outputs/class_imbalance`) that whole-image training wastes most of its gradient
signal on.

Intended to run against `scripts/fuse_duplicate_annotations.py`'s fused COCO json
(one canonical instance set per underlying image, duplicate-annotator disagreement
already resolved) rather than the raw MAGFiLO json directly -- nothing here
requires that specifically, but crop-centering on a random instance from a
duplicate-annotator image that hasn't been fused would still hit the same
whole-image-disagreement problem this Stage was built to route around.

Val/inference stay on `FilamentDataset` (whole-image, unchanged) -- this dataset
is train-only. Kept as a separate module rather than a mode inside
`FilamentDataset` so the existing whole-image path (currently training
successfully on Kaggle) is never touched by this change.
"""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from torch.utils.data import Dataset


def _rasterize_crop_mask(anns: list[dict], x0: int, y0: int, crop_size: int) -> np.ndarray:
    """Fill each annotation's polygon(s) directly onto a crop_size x crop_size
    canvas, translated by (-x0, -y0) -- avoids allocating a full-frame (2048x2048)
    mask per instance per crop the way FilamentDataset's coco.annToMask does,
    which matters here since this runs crops_per_image times more often per
    underlying image. cv2.fillPoly clips out-of-canvas coordinates correctly
    (verified directly, not assumed), so a polygon straddling the crop boundary
    -- the common case for an instance-centered crop -- rasterizes correctly with
    no extra bounds-checking needed.
    """
    mask = np.zeros((crop_size, crop_size), dtype=np.uint8)
    for ann in anns:
        for poly in ann["segmentation"]:
            pts = np.array(poly, dtype=np.float64).reshape(-1, 2)
            pts[:, 0] -= x0
            pts[:, 1] -= y0
            cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 1)
    return mask


class FilamentCropDataset(Dataset):
    """Per `memory.md`'s Stage 1 spec: crops centered on a randomly chosen
    instance (with jitter, not dead-center every time), padded generously enough
    that nearby instances falling in the same window are NOT masked out -- the
    crop's target is whatever fused instances overlap the window, not just the
    one that was centered on. Mixed with a minority of pure uniform-random crops
    (`bg_crop_fraction`) for background context, matching `analysis.md` section
    6's "keeping some pure-background crops to avoid inflating false positives."

    Crops are *sampled*, not one-per-image -- `__len__` is
    `len(image_ids) * crops_per_image`, cycling through source images that many
    times per epoch, standard practice for patch-based training.
    """

    def __init__(
        self,
        coco_json: str | Path,
        img_dir: str | Path,
        image_ids: list[str],
        crop_size: int = 512,
        crops_per_image: int = 4,
        bg_crop_fraction: float = 0.2,
        transform=None,
        seed: int = 0,
    ):
        self.coco = COCO(str(coco_json))
        self.img_dir = Path(img_dir)
        self.ids = image_ids
        self.crop_size = crop_size
        self.crops_per_image = crops_per_image
        self.bg_crop_fraction = bg_crop_fraction
        self.transform = transform
        self._base_seed = seed
        self._epoch = 0

    def __len__(self) -> int:
        return len(self.ids) * self.crops_per_image

    def set_epoch(self, epoch: int) -> None:
        """Call once per epoch (mirrors DistributedSampler.set_epoch, already used
        for train_sampler in src/train.py) so crop sampling varies epoch to epoch
        while every (epoch, idx) pair still deterministically reproduces the same
        crop. Deliberately NOT using persistent mutable RNG state (e.g. a stored
        random.Random instance advanced across calls) -- DataLoader workers are
        forked copies of this dataset object, so shared mutable state would start
        every worker from an identical copy and correlate their random streams
        (a well-known multiprocessing-DataLoader pitfall). Deriving the seed fresh
        from (base_seed, epoch, idx) inside __getitem__ instead sidesteps that
        entirely -- no state to share incorrectly."""
        self._epoch = epoch

    def _rng_for(self, idx: int) -> random.Random:
        seed_val = (self._base_seed * 1_000_003 + self._epoch * 9_973 + idx) % (2**31)
        return random.Random(seed_val)

    def _sample_crop_origin(self, rng: random.Random, anns: list[dict], h: int, w: int) -> tuple[int, int]:
        max_x0 = max(0, w - self.crop_size)
        max_y0 = max(0, h - self.crop_size)
        half = self.crop_size / 2

        use_background = (not anns) or (rng.random() < self.bg_crop_fraction)
        if use_background:
            return rng.randint(0, max_x0), rng.randint(0, max_y0)

        chosen = rng.choice(anns)
        bx, by, bw, bh = chosen["bbox"]
        center_x, center_y = bx + bw / 2, by + bh / 2
        jitter = self.crop_size * 0.2  # so the instance isn't dead-center every time
        center_x += rng.uniform(-jitter, jitter)
        center_y += rng.uniform(-jitter, jitter)
        x0 = int(min(max(center_x - half, 0), max_x0))
        y0 = int(min(max(center_y - half, 0), max_y0))
        return x0, y0

    def __getitem__(self, idx: int):
        image_id = self.ids[idx % len(self.ids)]
        info = self.coco.imgs[image_id]
        img = cv2.imread(str(self.img_dir / info["file_name"]), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(self.img_dir / info["file_name"])
        h, w = img.shape
        assert (h, w) == (info["height"], info["width"]), f"size mismatch for {info['file_name']}: read {(h, w)}, json says {(info['height'], info['width'])}"

        ann_ids = self.coco.getAnnIds(imgIds=[image_id])
        anns = self.coco.loadAnns(ann_ids)

        crop = self.crop_size
        if h < crop or w < crop:
            # Smaller than the crop (shouldn't happen for native 2048x2048 MAGFiLO
            # frames, but don't silently misbehave on an unexpected input size):
            # pad the source image up to crop_size first, then crop covers it whole.
            pad_h, pad_w = max(0, crop - h), max(0, crop - w)
            img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
            h, w = img.shape
        rng = self._rng_for(idx)
        x0, y0 = self._sample_crop_origin(rng, anns, h, w)

        img_crop = img[y0:y0 + crop, x0:x0 + crop]
        mask_crop = _rasterize_crop_mask(anns, x0, y0, crop)

        if self.transform is not None:
            augmented = self.transform(image=img_crop, mask=mask_crop)
            img_crop, mask_crop = augmented["image"], augmented["mask"]

        img_t = torch.from_numpy(img_crop).float().unsqueeze(0) / 255.0
        mask_t = torch.from_numpy(mask_crop).float().unsqueeze(0)
        return img_t, mask_t, image_id, (h, w)
