"""Dataset for BYOL pretraining on the GONG H-alpha corpus.

Reuses two already-verified pieces rather than re-deriving them:
  - the manifest-coverage-gap workaround from `scripts/pretrain_data/thin_manifest.py`
    (directory listing is the source of truth for which files exist; manifest.csv is
    consulted opportunistically for cx/cy/r and falls back to GONG's standardized
    geometry otherwise) -- see RESNET_PRETRAIN_PLAN.md section 3.
  - the disk-bounded rejection-sampling crop from PRETRAIN_PLAN.md section 4.1 (a crop
    centered off-disk would let the pretext task shortcut on "is this patch
    background").

See RESNET_PRETRAIN_PLAN.md sections 3, 4, 8 for the full design this implements.
"""

from __future__ import annotations

import csv
import logging
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from torch.utils.data import Dataset

logger = logging.getLogger("pretrain_resnet.dataset")

FNAME_RE = re.compile(r"^(\d{8})(\d{6})([A-Za-z])h\.jpe?g$")

# GONG's own processing pipeline re-registers every frame to this standardized
# geometry (verified in scripts/pretrain_data/preprocess_gong.py) -- used as a
# fallback for files missing from --manifest, same as thin_manifest.py.
DEFAULT_CX, DEFAULT_CY, DEFAULT_R = 1024.0, 1024.0, 900.0


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    date: str  # YYYYMMDD, used for the held-out split (section 8)
    cx: float
    cy: float
    r: float


def parse_filename(name: str) -> tuple[str, str, str] | None:
    """Returns (date, time, site) from a `<YYYYMMDDHHMMSS><site>h.jpeg` filename, or None."""
    m = FNAME_RE.match(name)
    if not m:
        return None
    return m.group(1), m.group(2), m.group(3)


def load_metadata(manifest_csv: Path | None) -> dict[str, dict]:
    """Maps JPEG path (as written in manifest.csv) -> {cx, cy, r} row dict."""
    if manifest_csv is None or not manifest_csv.exists():
        return {}
    with open(manifest_csv) as f:
        return {row["path"]: row for row in csv.DictReader(f)}


def discover_images(images_dir: Path, manifest_csv: Path | None = None) -> list[ImageRecord]:
    """Directory listing is the source of truth for which files exist -- manifest.csv
    only covers a subset of the on-disk corpus (see module docstring). Files that
    don't parse as the expected `<date><time><site>h.jpeg` shape are skipped."""
    metadata = load_metadata(manifest_csv)
    records: list[ImageRecord] = []
    n_unparsed = 0
    n_fallback = 0
    for path in sorted(images_dir.glob("*.jpeg")):
        parsed = parse_filename(path.name)
        if parsed is None:
            n_unparsed += 1
            continue
        date, _time, _site = parsed
        row = metadata.get(str(path))
        if row is None:
            n_fallback += 1
            cx, cy, r = DEFAULT_CX, DEFAULT_CY, DEFAULT_R
        else:
            cx, cy, r = float(row["cx"]), float(row["cy"]), float(row["r"])
        records.append(ImageRecord(path=path, date=date, cx=cx, cy=cy, r=r))
    if n_unparsed:
        logger.warning(f"skipped {n_unparsed} files that didn't match the expected filename pattern")
    logger.info(f"discovered {len(records)} images ({n_fallback} using fallback geometry, manifest covered {len(records) - n_fallback})")
    return records


def date_grouped_split(records: list[ImageRecord], val_fraction: float = 0.04, seed: int = 0) -> tuple[list[ImageRecord], list[ImageRecord]]:
    """Held out ~3-5% of the corpus as a fixed validation set, grouped by *date* --
    not by row -- so a held-out day's frames aren't near-duplicates of a training
    day's (same-site consecutive-hour SSIM 0.91 per PRETRAIN_PLAN.md section 3; that
    redundancy would leak straight across a row-level split). See section 8."""
    dates = sorted({r.date for r in records})
    rng = random.Random(seed)
    rng.shuffle(dates)
    n_val_dates = max(1, round(len(dates) * val_fraction))
    val_dates = set(dates[:n_val_dates])
    train = [r for r in records if r.date not in val_dates]
    val = [r for r in records if r.date in val_dates]
    logger.info(f"date-grouped split: {len(dates)} dates -> {len(dates) - n_val_dates} train / {n_val_dates} val dates ({len(train)} / {len(val)} images)")
    return train, val


def disk_bounded_crop(img: np.ndarray, cx: float, cy: float, r: float, crop_size: int) -> np.ndarray:
    """Sample the crop's *center* uniformly within the solar disk (not an unbounded
    crop over the full frame) -- an unbounded crop can land mostly off-disk (pure
    background), which would let the pretext task shortcut on "is this patch
    background" rather than learning filament-relevant texture. The crop can still
    extend past the limb near its edges -- just not be centered on pure background.
    Ported from PRETRAIN_PLAN.md section 4.1 (same crop the ViT-MAE dataset uses)."""
    half = crop_size / 2
    h, w = img.shape[:2]
    for _ in range(10):  # a handful of rejection-sample attempts is plenty
        angle = random.uniform(0, 2 * np.pi)
        radius = r * (random.random() ** 0.5)  # uniform over disk *area*, not radius
        cx_s = cx + radius * np.cos(angle)
        cy_s = cy + radius * np.sin(angle)
        x0, y0 = int(cx_s - half), int(cy_s - half)
        if 0 <= x0 and x0 + crop_size <= w and 0 <= y0 and y0 + crop_size <= h:
            return img[y0 : y0 + crop_size, x0 : x0 + crop_size]
    # fallback: disk-centered crop, in-bounds given the standard r=900, crop<=2*900
    cx_i, cy_i = int(cx), int(cy)
    half_i = int(half)
    y0 = min(max(cy_i - half_i, 0), h - crop_size)
    x0 = min(max(cx_i - half_i, 0), w - crop_size)
    return img[y0 : y0 + crop_size, x0 : x0 + crop_size]


def make_view(img: np.ndarray, cx: float, cy: float, r: float, crop_size: int, blur_p: float) -> np.ndarray:
    """One BYOL augmented view. Grayscale removes the need for color jitter -- gamma
    + brightness/contrast is the direct analog -- everything else has a direct
    analog to BYOL's original recipe. See RESNET_PRETRAIN_PLAN.md section 4.
    `blur_p` implements BYOL's *asymmetric* blur (1.0 for view 1, 0.1 for view 2 --
    call site's responsibility, verified detail from the paper, not arbitrary)."""
    crop = disk_bounded_crop(img, cx, cy, r, crop_size)
    # rotation: any angle valid, no canonical "up" on the Sun
    k = random.randint(0, 3)
    crop = np.rot90(crop, k).copy()
    if random.random() < 0.5:
        crop = np.fliplr(crop).copy()
    if random.random() < 0.8:
        gamma = random.uniform(0.7, 1.3)
        crop = np.clip(crop, 0, 1) ** gamma
    if random.random() < 0.5:
        contrast = random.uniform(0.8, 1.2)
        crop = np.clip((crop - 0.5) * contrast + 0.5, 0, 1)
    if random.random() < blur_p:
        crop = gaussian_filter(crop, sigma=random.uniform(0.1, 1.0))
    return crop.astype(np.float32)


class HalphaBYOLDataset(Dataset):
    """Returns (view1, view2) float32 arrays [crop_size, crop_size] per image --
    caller (or a collate step) adds the channel dim. Full native corpus, not the
    thinned manifest -- BYOL's redundancy tolerance means the near-duplicate
    same-site-consecutive-hour frames aren't harmful here, see section 1/3."""

    def __init__(self, records: list[ImageRecord], crop_size: int = 224):
        self.records = records
        self.crop_size = crop_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        rec = self.records[idx]
        img = np.array(Image.open(rec.path), dtype=np.float32) / 255.0
        view1 = make_view(img, rec.cx, rec.cy, rec.r, self.crop_size, blur_p=1.0)
        view2 = make_view(img, rec.cx, rec.cy, rec.r, self.crop_size, blur_p=0.1)
        return view1[None, :, :], view2[None, :, :]
