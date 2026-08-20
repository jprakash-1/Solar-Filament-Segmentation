#!/usr/bin/env python3
"""EDA: sample N random annotated training images and render plain vs. annotated side by side.

Usage:
    python scripts/eda_side_by_side.py
    python scripts/eda_side_by_side.py --n 50 --seed 7 --out-dir outputs/eda/side_by_side
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.coco_utils import (  # noqa: E402
    CATEGORY_COLORS,
    TRAIN_ANNOTATIONS,
    TRAIN_IMAGES_DIR,
    category_lookup,
    draw_all_annotations,
    index_annotations_by_image,
    index_images_by_id,
    legend_handles,
    load_coco,
    load_image,
    sample_annotated_image_ids,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=100, help="number of random annotated images to sample")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--annotations", type=Path, default=TRAIN_ANNOTATIONS)
    p.add_argument("--images-dir", type=Path, default=TRAIN_IMAGES_DIR)
    p.add_argument("--out-dir", type=Path, default=Path("outputs/eda/side_by_side"))
    p.add_argument("--contact-sheet", type=Path, default=Path("outputs/eda/contact_sheet.png"))
    p.add_argument("--cols", type=int, default=10, help="columns in the contact-sheet overview grid")
    return p.parse_args()


def plot_pair(image, anns: list[dict], cat_names: dict[int, str], title: str, out_path: Path) -> None:
    fig, (ax_plain, ax_annotated) = plt.subplots(1, 2, figsize=(10, 5))
    for ax in (ax_plain, ax_annotated):
        ax.imshow(image)
        ax.axis("off")
    ax_plain.set_title("Plain (no annotation)")
    ax_annotated.set_title(f"Annotated ({len(anns)} filaments)")

    draw_all_annotations(ax_annotated, anns, show_bbox=True, show_segmentation=True)
    handles = legend_handles(cat_names)
    if handles:
        ax_annotated.legend(handles=handles, loc="upper right", fontsize=6, framealpha=0.6)

    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.contact_sheet.parent.mkdir(parents=True, exist_ok=True)

    coco = load_coco(args.annotations)
    cat_names = category_lookup(coco)
    ann_index = index_annotations_by_image(coco)
    img_index = index_images_by_id(coco)

    image_ids = sample_annotated_image_ids(coco, args.n, seed=args.seed)
    print(f"Sampled {len(image_ids)} annotated images (seed={args.seed})")

    thumbnails = []
    for i, image_id in enumerate(image_ids, start=1):
        img_meta = img_index[image_id]
        anns = ann_index[image_id]
        image = load_image(args.images_dir, img_meta["file_name"])

        stem = Path(img_meta["file_name"]).stem
        out_path = args.out_dir / f"{i:03d}_{stem}.png"
        plot_pair(image, anns, cat_names, title=img_meta["file_name"], out_path=out_path)
        thumbnails.append((image, anns))

        if i % 10 == 0 or i == len(image_ids):
            print(f"  {i}/{len(image_ids)} pairs saved")

    rows = (len(thumbnails) + args.cols - 1) // args.cols
    fig, axes = plt.subplots(rows, args.cols, figsize=(args.cols * 2, rows * 2))
    axes = axes.flatten()
    for ax, (image, anns) in zip(axes, thumbnails):
        ax.imshow(image)
        draw_all_annotations(ax, anns, show_bbox=False, show_segmentation=True)
        ax.axis("off")
    for ax in axes[len(thumbnails):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(args.contact_sheet, dpi=150)
    plt.close(fig)

    print(f"Saved {len(image_ids)} side-by-side pairs to {args.out_dir}")
    print(f"Saved contact-sheet overview to {args.contact_sheet}")


if __name__ == "__main__":
    main()
