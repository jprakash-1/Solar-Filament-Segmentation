#!/usr/bin/env python3
"""Plot one image in four panels: plain, bounding boxes, segmentation boundary, spine coordinates.

Usage:
    python scripts/plot_annotation_panels.py                          # random annotated image
    python scripts/plot_annotation_panels.py --image-id 040301-20140609195854Bh
    python scripts/plot_annotation_panels.py --n 5 --seed 3            # 5 random images, one figure each
"""

from __future__ import annotations

import argparse
import random
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
    draw_bbox,
    draw_segmentation,
    draw_spine,
    index_annotations_by_image,
    index_images_by_id,
    legend_handles,
    load_coco,
    load_image,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image-id", type=str, default=None, help="COCO image id; random annotated image if omitted")
    p.add_argument("--n", type=int, default=1, help="number of random images to plot (ignored if --image-id is set)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--annotations", type=Path, default=TRAIN_ANNOTATIONS)
    p.add_argument("--images-dir", type=Path, default=TRAIN_IMAGES_DIR)
    p.add_argument("--out-dir", type=Path, default=Path("outputs/eda/panels"))
    return p.parse_args()


def plot_panels(image, anns: list[dict], cat_names: dict[int, str], title: str, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    panel_titles = ["Plain image", "Bounding boxes", "Segmentation boundary", "Coordinates (spine)"]
    for ax, panel_title in zip(axes, panel_titles):
        ax.imshow(image)
        ax.set_title(panel_title, fontsize=11)
        ax.axis("off")

    for ann in anns:
        color = CATEGORY_COLORS.get(ann["category_id"], "yellow")
        label = cat_names.get(ann["category_id"], str(ann["category_id"]))
        draw_bbox(axes[1], ann, color, label=label)
        draw_segmentation(axes[2], ann, color)
        draw_spine(axes[3], ann, color)

    handles = legend_handles(cat_names)
    if handles:
        fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=9, bbox_to_anchor=(0.5, -0.03))

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    coco = load_coco(args.annotations)
    cat_names = category_lookup(coco)
    ann_index = index_annotations_by_image(coco)
    img_index = index_images_by_id(coco)

    if args.image_id is not None:
        image_ids = [args.image_id]
    else:
        rng = random.Random(args.seed)
        image_ids = rng.sample(sorted(ann_index.keys()), min(args.n, len(ann_index)))

    for image_id in image_ids:
        if image_id not in img_index:
            raise SystemExit(f"Unknown image_id: {image_id}")

        img_meta = img_index[image_id]
        anns = ann_index.get(image_id, [])
        image = load_image(args.images_dir, img_meta["file_name"])

        stem = Path(img_meta["file_name"]).stem
        out_path = args.out_dir / f"{stem}_panels.png"
        title = f"{img_meta['file_name']}  |  image_id={image_id}  |  {len(anns)} filaments"
        plot_panels(image, anns, cat_names, title, out_path)


if __name__ == "__main__":
    main()
