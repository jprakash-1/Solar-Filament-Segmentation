#!/usr/bin/env python3
"""MVP1 Option A -- classical CV baseline, zero training required.

Purpose (per MVP1 plan section 3.4): validate the *entire*
postprocess -> RLE -> submission.csv -> upload path today, decoupled from any
model-training bugs. This is meant to be the literal first submission, not a good
one.

Method: filaments are dark, elongated regions relative to their local surroundings
on the (roughly circular, bright) solar disk. Flatten large-scale brightness
variation with a heavily-blurred local-background estimate, threshold the residual
(Otsu, restricted to disk pixels) to find locally-darker-than-neighborhood regions,
clean up with morphological opening, then split into instances via connected
components (src/postprocess.py) -- the same postprocessing discipline the U-Net path
will use.

Usage:
    python scripts/baseline_classical.py
    python scripts/baseline_classical.py --split val --data-json <train_json>  # local PQ check
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import FilamentDataset, group_split  # noqa: E402
from src.metrics import aggregate_pq, panoptic_quality  # noqa: E402
from src.postprocess import mask_to_instances  # noqa: E402
from src.submission import build_submission, validate_submission  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--test-images-dir",
        type=Path,
        default=Path("data/raw/MAGFiLO_1.0_Kaggle_2026/test/test_images"),
    )
    p.add_argument(
        "--data-json",
        type=Path,
        default=Path("data/raw/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"),
        help="train COCO json, only used with --split val for local PQ evaluation",
    )
    p.add_argument(
        "--train-images-dir",
        type=Path,
        default=Path("data/raw/MAGFiLO_1.0_Kaggle_2026/train/train_images"),
    )
    p.add_argument("--split", choices=["test", "val"], default="test", help="'test' produces the real submission; 'val' evaluates local PQ/Dice on the held-out train split")
    p.add_argument("--val-fraction", type=float, default=0.15, help="must match src/train.py's --val-fraction to compare PQ against the U-Net on the same held-out images")
    p.add_argument("--seed", type=int, default=0, help="must match src/train.py's --seed to compare PQ against the U-Net on the same held-out images")
    p.add_argument("--blur-sigma", type=float, default=61.0, help="Gaussian sigma for the local-background estimate")
    p.add_argument("--min-area-px", type=int, default=15)
    p.add_argument("--out", type=Path, default=Path("outputs/submissions/baseline_classical.csv"))
    return p.parse_args()


def disk_mask(gray: np.ndarray, threshold: int = 10) -> np.ndarray:
    mask = (gray > threshold).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n_labels <= 1:
        return mask.astype(bool)
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return labels == largest


def detect_filament_instances(gray: np.ndarray, blur_sigma: float, min_area_px: int, fg_percentile: float = 99.5) -> list[np.ndarray]:
    """fg_percentile: threshold the residual at this percentile of disk-interior
    values (default 99.5, i.e. keep the top 0.5%) rather than Otsu. Otsu was tried
    first and picked a threshold low enough to let granulation texture bridge into
    one enormous connected blob (300k+ px, covering a large fraction of the disk) --
    a percentile grounded in the known ~0.4-0.5% foreground fraction from
    outputs/class_imbalance/findings.md gives a far more sane candidate count
    (dozens, not tens of thousands, of components) at comparable recall.
    """
    mask = disk_mask(gray)
    background = cv2.GaussianBlur(gray, (0, 0), blur_sigma)
    residual = background.astype(np.int16) - gray.astype(np.int16)  # positive where darker than local background
    residual_clipped = np.clip(residual, 0, 255).astype(np.uint8)
    residual_clipped[~mask] = 0  # never detect outside the disk

    disk_vals = residual_clipped[mask]
    if disk_vals.max() == 0:
        return []
    thresh_val = np.percentile(disk_vals, fg_percentile)
    binary = (residual_clipped > thresh_val).astype(np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    return mask_to_instances(binary, min_area_px=min_area_px)


def run_on_test(args: argparse.Namespace) -> None:
    image_paths = sorted(args.test_images_dir.glob("*.jpeg"))
    print(f"Running classical baseline over {len(image_paths)} test images...")

    image_stem_to_instances = {}
    n_instances_total = 0
    for path in tqdm(image_paths, desc="baseline inference"):
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        instances = detect_filament_instances(gray, args.blur_sigma, args.min_area_px)
        image_stem_to_instances[path.stem] = instances
        n_instances_total += len(instances)

    print(f"Total predicted instances: {n_instances_total} ({n_instances_total / len(image_paths):.1f}/image)")
    df = build_submission(image_stem_to_instances, args.out)
    print(f"Wrote {len(df)} rows to {args.out}")

    expected_stems = [p.stem for p in image_paths]
    result = validate_submission(args.out, expected_image_stems=expected_stems)
    for w in result["warnings"]:
        print(f"  WARNING: {w}")
    if result["errors"]:
        print(f"VALIDATION FOUND {len(result['errors'])} ERROR(S):")
        for e in result["errors"]:
            print(f"  - {e}")
    else:
        print("Validation OK.")


def run_on_val(args: argparse.Namespace) -> None:
    ds = FilamentDataset(args.data_json, args.train_images_dir, image_ids=[])  # img_size unused here
    _, val_ids = group_split(ds.coco, val_fraction=args.val_fraction, seed=args.seed)
    print(f"Evaluating classical baseline on {len(val_ids)} held-out validation images (val_fraction={args.val_fraction}, seed={args.seed})...")

    per_image_results = []
    for image_id in tqdm(val_ids, desc="baseline val eval"):
        info = ds.coco.imgs[image_id]
        gray = cv2.imread(str(args.train_images_dir / info["file_name"]), cv2.IMREAD_GRAYSCALE)
        pred_instances = detect_filament_instances(gray, args.blur_sigma, args.min_area_px)
        gt_instances = ds.get_instance_masks(image_id)
        per_image_results.append(panoptic_quality(gt_instances, pred_instances))

    agg = aggregate_pq(per_image_results)
    print(f"mean-per-image PQ: {agg['mean_per_image_pq']:.4f}")
    print(f"pooled PQ:         {agg['pooled_pq']:.4f}")
    print(f"total TP={agg['total_tp']} FP={agg['total_fp']} FN={agg['total_fn']}")


def main() -> None:
    args = parse_args()
    if args.split == "test":
        run_on_test(args)
    else:
        run_on_val(args)


if __name__ == "__main__":
    main()
