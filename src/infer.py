#!/usr/bin/env python3
"""Run the trained tiny U-Net over test/ (or a held-out val split for local PQ),
producing per-image instance masks via the resize-then-threshold-then-CC discipline
from src/postprocess.py.

Usage:
    python -m src.infer --checkpoint outputs/checkpoints/mvp1_unet.pt --split test
    python -m src.infer --checkpoint outputs/checkpoints/mvp1_unet.pt --split val
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from src.dataset import FilamentDataset, group_split
from src.metrics import aggregate_pq, panoptic_quality
from src.model import build_model
from src.postprocess import prob_map_to_instances
from src.submission import build_submission, validate_submission
from src.train import resolve_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument(
        "--test-images-dir",
        type=Path,
        default=Path("data/raw/MAGFiLO_1.0_Kaggle_2026/test/test_images"),
    )
    p.add_argument(
        "--data-json",
        type=Path,
        default=Path("data/raw/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"),
        help="train COCO json, only used with --split val",
    )
    p.add_argument(
        "--train-images-dir",
        type=Path,
        default=Path("data/raw/MAGFiLO_1.0_Kaggle_2026/train/train_images"),
    )
    p.add_argument("--split", choices=["test", "val"], default="test")
    p.add_argument(
        "--val-fraction",
        type=float,
        default=None,
        help="override the val split fraction used for --split val; defaults to whatever the checkpoint was trained with",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="override the val split seed used for --split val; defaults to whatever the checkpoint was trained with",
    )
    p.add_argument("--prob-thresh", type=float, default=0.5)
    p.add_argument("--min-area-px", type=int, default=15)
    p.add_argument("--device", default=None)
    p.add_argument("--out", type=Path, default=Path("outputs/submissions/mvp1_unet.csv"))
    return p.parse_args()


def predict_prob_map(model: torch.nn.Module, gray: np.ndarray, img_size: int, device: torch.device) -> np.ndarray:
    img_rs = cv2.resize(gray, (img_size, img_size), interpolation=cv2.INTER_AREA)
    x = torch.from_numpy(img_rs).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0
    with torch.no_grad():
        prob = torch.sigmoid(model(x))[0, 0].cpu().numpy()
    return prob


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_model().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def run_on_test(args: argparse.Namespace, model: torch.nn.Module, img_size: int, device: torch.device) -> None:
    image_paths = sorted(args.test_images_dir.glob("*.jpeg"))
    print(f"Running U-Net inference over {len(image_paths)} test images...")

    image_stem_to_instances = {}
    n_instances_total = 0
    for path in tqdm(image_paths, desc="unet inference"):
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        prob = predict_prob_map(model, gray, img_size, device)
        instances = prob_map_to_instances(prob, gray.shape, prob_thresh=args.prob_thresh, min_area_px=args.min_area_px)
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


def run_on_val(args: argparse.Namespace, model: torch.nn.Module, img_size: int, device: torch.device, ckpt: dict) -> None:
    # Default to the checkpoint's own split params so "val" here is guaranteed to be
    # the exact set held out at training time, regardless of src/train.py's current
    # CLI defaults -- only override if the caller explicitly asks to evaluate a
    # different split.
    val_fraction = args.val_fraction if args.val_fraction is not None else ckpt.get("val_fraction", 0.15)
    seed = args.seed if args.seed is not None else ckpt.get("seed", 0)

    ds = FilamentDataset(args.data_json, args.train_images_dir, image_ids=[])
    _, val_ids = group_split(ds.coco, val_fraction=val_fraction, seed=seed)
    print(f"Evaluating U-Net on {len(val_ids)} held-out validation images (val_fraction={val_fraction}, seed={seed})...")

    per_image_results = []
    for image_id in tqdm(val_ids, desc="unet val eval"):
        info = ds.coco.imgs[image_id]
        gray = cv2.imread(str(args.train_images_dir / info["file_name"]), cv2.IMREAD_GRAYSCALE)
        prob = predict_prob_map(model, gray, img_size, device)
        pred_instances = prob_map_to_instances(prob, gray.shape, prob_thresh=args.prob_thresh, min_area_px=args.min_area_px)
        gt_instances = ds.get_instance_masks(image_id)
        per_image_results.append(panoptic_quality(gt_instances, pred_instances))

    agg = aggregate_pq(per_image_results)
    print(f"mean-per-image PQ: {agg['mean_per_image_pq']:.4f}")
    print(f"pooled PQ:         {agg['pooled_pq']:.4f}")
    print(f"total TP={agg['total_tp']} FP={agg['total_fp']} FN={agg['total_fn']}")


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    model, ckpt = load_model(args.checkpoint, device)
    img_size = ckpt["img_size"]
    print(f"Loaded checkpoint {args.checkpoint} (img_size={img_size}) on {device}")

    if args.split == "test":
        run_on_test(args, model, img_size, device)
    else:
        run_on_val(args, model, img_size, device, ckpt)


if __name__ == "__main__":
    main()
