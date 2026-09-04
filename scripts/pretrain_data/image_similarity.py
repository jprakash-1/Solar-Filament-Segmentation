"""Pairwise similarity between a small batch of images using classical
(non-deep-learning) CV techniques:

  - histogram: correlation between grayscale intensity histograms (cv2)
  - ssim:      structural similarity index on grayscale pixels (skimage)
  - orb:       fraction of ORB keypoints with a good mutual match (cv2)
  - combined:  average of histogram and ssim (default; cheap and robust
               for near-duplicate detection on similar-looking frames)

All scores are normalized to [0, 1], where 1.0 means identical.

CLI usage:
    python scripts/pretrain_data/image_similarity.py \
        --dir data/processed/gong_pretrain --n 8 --method combined \
        --output similarity.png
"""

import argparse
import os

import cv2
import numpy as np
from skimage.metrics import structural_similarity as skimage_ssim

DEFAULT_SIZE = (256, 256)


def load_gray(path: str, size: tuple[int, int] = DEFAULT_SIZE) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def histogram_similarity(img1: np.ndarray, img2: np.ndarray) -> float:
    hist1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([img2], [0], None, [256], [0, 256])
    cv2.normalize(hist1, hist1)
    cv2.normalize(hist2, hist2)
    corr = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return float(max(0.0, corr))


def ssim_similarity(img1: np.ndarray, img2: np.ndarray) -> float:
    score = skimage_ssim(img1, img2)
    return float(max(0.0, score))


def orb_similarity(img1: np.ndarray, img2: np.ndarray, max_features: int = 500) -> float:
    orb = cv2.ORB_create(nfeatures=max_features)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    if des1 is None or des2 is None or len(kp1) == 0 or len(kp2) == 0:
        return 0.0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    good = [m for m in matches if m.distance < 50]
    return float(len(good) / min(len(kp1), len(kp2)))


def combined_similarity(img1: np.ndarray, img2: np.ndarray) -> float:
    return float(0.5 * histogram_similarity(img1, img2) + 0.5 * ssim_similarity(img1, img2))


METHODS = {
    "histogram": histogram_similarity,
    "ssim": ssim_similarity,
    "orb": orb_similarity,
    "combined": combined_similarity,
}


def similarity_matrix(
    image_paths: list[str], method: str = "combined", size: tuple[int, int] = DEFAULT_SIZE
) -> np.ndarray:
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}; choose from {list(METHODS)}")
    score_fn = METHODS[method]
    imgs = [load_gray(p, size) for p in image_paths]
    n = len(imgs)
    matrix = np.eye(n, dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            score = score_fn(imgs[i], imgs[j])
            matrix[i, j] = matrix[j, i] = score
    return matrix


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", help="Directory of images to sample from")
    parser.add_argument("--files", nargs="+", help="Explicit list of image paths")
    parser.add_argument("--n", type=int, default=8, help="Number of images to compare (default 8)")
    parser.add_argument("--method", choices=list(METHODS) + ["combined"], default="combined")
    parser.add_argument("--output", help="Path to save a heatmap PNG (optional)")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.files:
        paths = args.files
    elif args.dir:
        names = sorted(
            f for f in os.listdir(args.dir) if f.lower().endswith((".jpeg", ".jpg", ".png"))
        )[: args.n]
        paths = [os.path.join(args.dir, f) for f in names]
    else:
        raise SystemExit("Provide either --dir or --files")

    if len(paths) < 2:
        raise SystemExit("Need at least 2 images to compare")

    matrix = similarity_matrix(paths, method=args.method)
    names = [os.path.basename(p) for p in paths]

    print(f"Pairwise '{args.method}' similarity (1.0 = identical):\n")
    header = " " * 22 + " ".join(f"{i:>6}" for i in range(len(names)))
    print(header)
    for i, name in enumerate(names):
        row = " ".join(f"{matrix[i, j]:6.3f}" for j in range(len(names)))
        print(f"{i:>2} {name[:18]:<18} {row}")

    if args.output:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis")
        ax.set_xticks(range(len(names)))
        ax.set_yticks(range(len(names)))
        ax.set_xticklabels(range(len(names)))
        ax.set_yticklabels([f"{i}: {n[:18]}" for i, n in enumerate(names)])
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="white", fontsize=7)
        fig.colorbar(im, ax=ax, label="similarity")
        fig.tight_layout()
        fig.savefig(args.output, dpi=150)
        print(f"\nSaved heatmap to {args.output}")


if __name__ == "__main__":
    main()
