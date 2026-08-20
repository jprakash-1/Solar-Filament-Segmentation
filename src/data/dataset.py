"""PyTorch Dataset classes for the filament segmentation model."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from src.data.coco_utils import load_image
from src.data.masks import rasterize_semantic_mask


class FilamentSegDataset(Dataset):
    """Train/val dataset: image + rasterized binary foreground mask."""

    def __init__(
        self,
        image_ids: list[str],
        img_index: dict[str, dict],
        ann_index: dict[str, list[dict]],
        images_dir: Path,
        transform=None,
    ) -> None:
        self.image_ids = image_ids
        self.img_index = img_index
        self.ann_index = ann_index
        self.images_dir = Path(images_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx]
        img_meta = self.img_index[image_id]
        anns = self.ann_index.get(image_id, [])

        image = load_image(self.images_dir, img_meta["file_name"])
        mask = rasterize_semantic_mask(anns, img_meta["height"], img_meta["width"])

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]

        mask = mask.float().unsqueeze(0) if isinstance(mask, torch.Tensor) else torch.from_numpy(mask).float().unsqueeze(0)
        return image, mask, image_id


class FilamentTestDataset(Dataset):
    """Inference dataset: image only, iterating raw files in a directory."""

    def __init__(self, images_dir: Path, transform=None) -> None:
        self.images_dir = Path(images_dir)
        self.files = sorted(p.name for p in self.images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        self.transform = transform

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        file_name = self.files[idx]
        image = load_image(self.images_dir, file_name)
        orig_h, orig_w = image.shape[:2]

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        stem = Path(file_name).stem
        return image, stem, orig_h, orig_w
