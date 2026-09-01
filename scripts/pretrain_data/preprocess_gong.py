#!/usr/bin/env python3
"""Stage 2: disk-crop, limb-darkening correct, resize, dedup, and normalize
downloaded GONG H-Alpha FITS frames.

Verified against real files (see PRETRAIN_PLAN.md section 3): GONG's own
processing pipeline already re-registers every "haf" (reduced) frame to a
standardized geometry -- disk center at pixel (1024, 1024), radius 900px, in
a 2048x2048 frame (`FNDLMBXC`/`FNDLMBYC`/`FNDLMBMA`/`FNDLMBMI` header keys).
This holds across sites (checked Big Bear and Mauna Loa samples), so disk
detection here is a matter of *reading* that header geometry, not deriving it.
`CRPIX1`/`CRPIX2` + a flat `RADIUS=900` are kept as a fallback only, in case a
future frame's limb-fit keys are missing or malformed.

Usage:
    python scripts/pretrain_data/preprocess_gong.py \
        --raw-dir data/raw/gong_pretrain --out-dir data/processed/gong_pretrain
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from astropy.io import fits

IMG_SIZE = 384  # multiple of ViT patch size 16; revisit against the eventual encoder
CROP_MARGIN = 1.05  # small margin beyond the disk radius so limb pixels aren't clipped
DEDUP_THRESHOLD = 0.01  # mean abs difference is measured *after* per-image
# z-score normalization (mean 0, std 1), so this is on a normalized scale, not
# raw pixel counts. Calibrated against two real, genuinely-different-hour Big
# Bear frames, which differed by ~0.02-0.06 on this scale -- a stuck-camera
# repeat (byte-identical source data) differs by ~1e-7 (floating-point noise
# only), so 0.01 sits comfortably between the two without needing a higher
# threshold that would risk treating legitimately different hours as dups.


def read_disk_geometry(header: fits.Header) -> tuple[float, float, float]:
    cx = header.get("FNDLMBXC")
    cy = header.get("FNDLMBYC")
    major = header.get("FNDLMBMA")
    minor = header.get("FNDLMBMI")
    if cx is not None and cy is not None and major is not None and minor is not None:
        return float(cx), float(cy), float(major + minor) / 2.0
    # fallback -- not expected to trigger on real "haf" files, but don't crash if it does
    cx = header.get("CRPIX1", header["NAXIS1"] / 2)
    cy = header.get("CRPIX2", header["NAXIS2"] / 2)
    r = header.get("RADIUS", min(header["NAXIS1"], header["NAXIS2"]) / 2.2)
    return float(cx), float(cy), float(r)


def crop_to_disk(img: np.ndarray, cx: float, cy: float, r: float, margin: float = CROP_MARGIN) -> np.ndarray:
    half = r * margin
    x0, x1 = int(round(cx - half)), int(round(cx + half))
    y0, y1 = int(round(cy - half)), int(round(cy + half))
    h, w = img.shape
    pad_left, pad_top = max(0, -x0), max(0, -y0)
    pad_right, pad_bottom = max(0, x1 - w), max(0, y1 - h)
    if pad_left or pad_top or pad_right or pad_bottom:
        img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="constant")
        x0, x1, y0, y1 = x0 + pad_left, x1 + pad_left, y0 + pad_top, y1 + pad_top
    return img[y0:y1, x0:x1]


N_PROFILE_BINS = 32


def _rho_grid(shape: tuple[int, int], r: float) -> np.ndarray:
    h, w = shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0  # image is already disk-centered by crop_to_disk
    yy, xx = np.mgrid[0:h, 0:w]
    return np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / r


def compute_radial_profile(images: list[np.ndarray], r: float, n_bins: int = N_PROFILE_BINS) -> np.ndarray:
    """Empirical mean intensity per radial bin, pooled over a sample of
    (cropped, uncorrected) images. Used instead of a parametric limb-darkening
    law: a first attempt with the textbook photospheric-continuum formula
    1/(0.3 + 0.7*mu) badly *overcorrected* real GONG Halpha frames -- checked
    with check_limb_flatness() below, the "corrected" profile sloped upward
    toward the limb instead of flattening. Halpha center-to-limb variation is
    much milder than continuum limb darkening (it's chromospheric emission,
    not photospheric), so a generic continuum formula doesn't transfer; an
    empirical profile fit to this data does, by construction."""
    if not images:
        raise ValueError("need at least one image to compute a radial profile")
    rho = _rho_grid(images[0].shape, r)
    bins = np.linspace(0, 1, n_bins + 1)
    profile = np.empty(n_bins, dtype=np.float64)
    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        mask = (rho >= lo) & (rho < hi)
        vals = np.concatenate([img[mask] for img in images]) if mask.any() else np.array([np.nan])
        profile[i] = vals.mean()
    # guard against a noisy/near-empty outermost bin (few pixels right at the
    # crop edge) producing a wild correction factor
    profile = np.where(np.isnan(profile), profile[~np.isnan(profile)][-1], profile)
    return profile


def limb_darkening_correct(img: np.ndarray, r: float, profile: np.ndarray) -> np.ndarray:
    """Radial intensity normalization using an empirical profile (see
    compute_radial_profile). Off-disk background (rho >= 1) is left untouched
    -- the profile is only defined inside the solar disk, and applying a
    correction off-disk would amplify background noise instead of leaving it flat."""
    n_bins = len(profile)
    rho = _rho_grid(img.shape, r)
    on_disk = rho < 1.0
    bin_idx = np.clip((rho * n_bins).astype(int), 0, n_bins - 1)
    correction = profile[0] / profile[bin_idx]
    out = img.copy()
    out[on_disk] = img[on_disk] * correction[on_disk]
    return out


def check_limb_flatness(img: np.ndarray, r: float, n_bins: int = 10) -> list[float]:
    """Diagnostic: mean intensity per radial bin (0 = center, 1 = limb). A
    correctly limb-darkening-corrected image should show roughly flat values
    across bins; a rising/falling trend means the profile sample was too small
    or unrepresentative."""
    rho = _rho_grid(img.shape, r)
    bins = np.linspace(0, 1, n_bins + 1)
    means = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (rho >= lo) & (rho < hi)
        means.append(float(img[mask].mean()) if mask.any() else float("nan"))
    return means


def load_and_crop(fits_path: Path) -> tuple[np.ndarray, float, dict]:
    with fits.open(fits_path) as hdul:
        data_hdu = hdul[1] if len(hdul) > 1 else hdul[0]
        data = data_hdu.data.astype(np.float32)
        header = data_hdu.header
        cx, cy, r = read_disk_geometry(header)
    cropped = crop_to_disk(data, cx, cy, r)
    return cropped, r, {"cx": cx, "cy": cy, "r": r}


def process_one(fits_path: Path, profile: np.ndarray) -> tuple[np.ndarray, dict]:
    cropped, crop_r, meta = load_and_crop(fits_path)
    corrected = limb_darkening_correct(cropped, crop_r, profile)
    resized = cv2.resize(corrected, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    resized = (resized - resized.mean()) / (resized.std() + 1e-6)  # per-image normalize;
    # corpus-level mean/std for final pretraining normalization is computed once
    # over the full kept corpus, in main() below, not per-image here

    meta["resized_r"] = crop_r * IMG_SIZE / cropped.shape[0]  # radius rescaled
    # into the final IMG_SIZE frame -- what Stage 3's disk-radius-bounded crop
    # augmentation should use, not the original-resolution r
    return resized.astype(np.float32), meta


def dedup_consecutive(paths_sites_imgs: list[tuple[Path, str, np.ndarray]], threshold: float = DEDUP_THRESHOLD) -> list[Path]:
    """Per-site stuck-camera dedup: drop a frame if it's nearly identical to
    the immediately preceding *kept* frame from the same site. Caller must
    pass frames already sorted by (site, timestamp); `prev_img` is reset on
    every site change so the last frame of one site is never compared against
    the first frame of the next (sorting groups files by site folder first,
    so those two rows are adjacent in the input list -- comparing across that
    boundary would be comparing two unrelated sites' frames, not a real dup)."""
    kept = []
    prev_img = None
    prev_site = None
    for path, site, img in paths_sites_imgs:
        if site != prev_site or prev_img is None or np.abs(img - prev_img).mean() > threshold:
            kept.append(path)
            prev_img = img
            prev_site = site
    return kept


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-dir", type=Path, default=Path("data/raw/gong_pretrain"))
    p.add_argument("--out-dir", type=Path, default=Path("data/processed/gong_pretrain"))
    p.add_argument("--profile-sample-size", type=int, default=200,
                   help="number of raw frames used to build the empirical limb-darkening profile")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fits_files = sorted(args.raw_dir.rglob("*.fits.fz")) + sorted(args.raw_dir.rglob("*.fits"))
    if not fits_files:
        raise SystemExit(f"no .fits/.fits.fz files found under {args.raw_dir}")

    # Pass 1: build one empirical limb-darkening profile from an evenly-spaced
    # sample, shared across sites/dates -- see compute_radial_profile's
    # docstring for why this replaced a parametric formula.
    sample_stride = max(1, len(fits_files) // args.profile_sample_size)
    sample_paths = fits_files[::sample_stride][: args.profile_sample_size]
    sample_crops, sample_r = [], None
    for fp in sample_paths:
        try:
            cropped, r, _ = load_and_crop(fp)
        except Exception as e:
            print(f"skip (profile sample) {fp}: {e}")
            continue
        sample_crops.append(cropped)
        sample_r = r  # standardized geometry (section 3) -- same r across files
    profile = compute_radial_profile(sample_crops, sample_r)
    np.save(args.out_dir / "limb_profile.npy", profile)
    print(f"built limb-darkening profile from {len(sample_crops)} sample frames")
    print(f"profile flatness check (pre-correction): {[round(x, 1) for x in profile]}")

    # Pass 2: process every frame using that fixed profile.
    manifest_rows = []
    processed: list[tuple[Path, str, np.ndarray]] = []
    for fp in fits_files:
        try:
            img, meta = process_one(fp, profile)
        except Exception as e:
            print(f"skip {fp}: {e}")
            continue
        out_path = args.out_dir / (fp.stem.replace(".fits", "") + ".npy")
        np.save(out_path, img)
        site = fp.parent.name
        manifest_rows.append({"path": str(out_path), "site": site, "source": str(fp), **meta})
        processed.append((out_path, site, img))

    kept_paths = set(dedup_consecutive(processed))
    manifest_rows = [row for row in manifest_rows if Path(row["path"]) in kept_paths]

    stats = {"mean": 0.0, "std": 1.0, "n_images": 0}
    if manifest_rows:
        stacked = np.stack([np.load(row["path"]) for row in manifest_rows])
        stats = {"mean": float(stacked.mean()), "std": float(stacked.std()), "n_images": len(manifest_rows)}

    with open(args.out_dir / "manifest.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "site", "source", "cx", "cy", "r", "resized_r"])
        writer.writeheader()
        writer.writerows(manifest_rows)
    with open(args.out_dir / "dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"processed {len(fits_files)} raw files -> {len(manifest_rows)} kept "
          f"(dropped {len(processed) - len(manifest_rows)} near-duplicates)")
    print(f"corpus stats: {stats}")


if __name__ == "__main__":
    main()
