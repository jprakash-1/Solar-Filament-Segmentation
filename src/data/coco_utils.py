"""Loading and drawing helpers for the MAGFiLO COCO-style filament annotations."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import matplotlib.lines as mlines
import matplotlib.patches as patches
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data" / "raw" / "MAGFiLO_1.0_Kaggle_2026"
TRAIN_IMAGES_DIR = DATA_ROOT / "train" / "train_images"
TRAIN_ANNOTATIONS = DATA_ROOT / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
TEST_IMAGES_DIR = DATA_ROOT / "test" / "test_images"

# category_id -> color, fixed so the same class always renders the same color
CATEGORY_COLORS = {
    1: "#1f77b4",  # Left
    2: "#d62728",  # Right
    3: "#7f7f7f",  # Unidentifiable
    4: "#9467bd",  # Ambiguous
}


def load_coco(annotations_path: Path = TRAIN_ANNOTATIONS) -> dict[str, Any]:
    with open(annotations_path) as f:
        return json.load(f)


def category_lookup(coco: dict) -> dict[int, str]:
    return {c["id"]: c["name"] for c in coco["categories"]}


def index_annotations_by_image(coco: dict) -> dict[str, list[dict]]:
    index: dict[str, list[dict]] = {}
    for ann in coco["annotations"]:
        index.setdefault(ann["image_id"], []).append(ann)
    return index


def index_images_by_id(coco: dict) -> dict[str, dict]:
    return {img["id"]: img for img in coco["images"]}


def load_image(images_dir: Path, file_name: str) -> np.ndarray:
    with Image.open(images_dir / file_name) as im:
        return np.array(im.convert("RGB"))


def sample_annotated_image_ids(coco: dict, n: int, seed: int | None = None) -> list[str]:
    ann_index = index_annotations_by_image(coco)
    ids = sorted(ann_index.keys())
    rng = random.Random(seed)
    return rng.sample(ids, min(n, len(ids)))


def polygons_from_segmentation(segmentation: list[list[float]]) -> list[np.ndarray]:
    return [np.array(part).reshape(-1, 2) for part in segmentation]


def spine_points(spine: list[float]) -> np.ndarray:
    return np.array(spine).reshape(-1, 2)


def draw_bbox(ax, ann: dict, color: str, label: str | None = None) -> None:
    x, y, w, h = ann["bbox"]
    ax.add_patch(patches.Rectangle((x, y), w, h, linewidth=1.5, edgecolor=color, facecolor="none"))
    if label:
        ax.text(
            x, max(y - 4, 0), label, color="white", fontsize=6, va="bottom",
            bbox=dict(facecolor=color, alpha=0.7, pad=1.0, edgecolor="none"),
        )


def draw_segmentation(ax, ann: dict, color: str) -> None:
    for poly in polygons_from_segmentation(ann["segmentation"]):
        closed = np.vstack([poly, poly[0]])
        ax.plot(closed[:, 0], closed[:, 1], color=color, linewidth=1.2)


def draw_spine(ax, ann: dict, color: str) -> None:
    pts = spine_points(ann["spine"])
    ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=1.5, marker="o", markersize=1.5)


def draw_all_annotations(
    ax,
    anns: list[dict],
    colors: dict[int, str] = CATEGORY_COLORS,
    show_bbox: bool = True,
    show_segmentation: bool = True,
    show_spine: bool = False,
) -> None:
    for ann in anns:
        color = colors.get(ann["category_id"], "yellow")
        if show_segmentation:
            draw_segmentation(ax, ann, color)
        if show_bbox:
            draw_bbox(ax, ann, color)
        if show_spine:
            draw_spine(ax, ann, color)


def legend_handles(cat_names: dict[int, str], colors: dict[int, str] = CATEGORY_COLORS) -> list[mlines.Line2D]:
    return [
        mlines.Line2D([], [], color=colors[cid], label=cat_names[cid], linewidth=2)
        for cid in sorted(colors)
        if cid in cat_names
    ]
