#!/usr/bin/env python3
"""Stage 2: limb-darkening correct, contrast-stretch to 8-bit, dedup, and save
downloaded GONG H-Alpha FITS frames as JPEGs -- keeping the full native
2048x2048, 1-channel frame, matching MAGFiLO's own input convention exactly
(verified: a real MAGFiLO training image is itself a 2048x2048, single-channel
(`L` mode) *JPEG*, with the same GONG-style filename and the same disk
center/radius framing -- off-disk corners read exactly 0, and the limb sits
right at radius ~900 around (1024, 1024), the same standardized geometry as
the raw "haf" product below). No resize/crop to a smaller pretraining
resolution, so there's no train/pretrain resolution mismatch to bridge at
Stage 4 fine-tuning time, and no format mismatch either.

Verified against real files (see PRETRAIN_PLAN.md section 3): GONG's own
processing pipeline already re-registers every "haf" (reduced) frame to that
standardized geometry -- disk center at pixel (1024, 1024), radius 900px, in a
2048x2048 frame (`FNDLMBXC`/`FNDLMBYC`/`FNDLMBMA`/`FNDLMBMI` header keys). This
holds across sites (checked Big Bear and Mauna Loa samples), so disk geometry
here is read from the header, not derived via CV. `CRPIX1`/`CRPIX2` + a flat
`RADIUS=900` are kept as a fallback only, in case a future frame's limb-fit
keys are missing or malformed.

The 8-bit contrast stretch (percentile clip over the disk *interior*, then
rescale to [0, 255]) reuses the exact approach already validated on this
project's `jp-analysis` branch (`scripts/apply_contrast_stretch.py`), which
found a naive full-disk percentile stretch anchors on the limb-brightening
artifact near the edge and closes a real per-site photometric gap once
restricted to the interior 85% of the disk radius. That script re-detects
disk geometry via CV; here the header-based (cx, cy, r) from section 3 is used
directly instead, since it's already known precisely.

Storing as JPEG (not .npy) matters at this corpus size too: a 2048x2048
float32 .npy is ~16.8MB/frame -- a 10K-image corpus would be ~168GB, clearly
not uploadable as a Kaggle Dataset. A real MAGFiLO JPEG at this resolution is
~700KB, i.e. ~7GB for 10K images.

Note for Stage 3: a ViT operating on the full 2048x2048 frame directly is not
feasible on 2xT4 -- at patch=16 that's 128x128=16384 patches (even after MAE's
75% masking, ~4096 visible tokens, ~28x the token count of a 384px/patch=16
setup). Stage 3's dataset should sample a disk-radius-bounded crop (e.g.
384/512px) from these full-resolution frames per training step instead of
feeding the whole frame to the encoder -- see PRETRAIN_PLAN.md section 4.

Usage:
    python scripts/pretrain_data/preprocess_gong.py \
        --raw-dir data/raw/gong_pretrain --out-dir data/processed/gong_pretrain
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from astropy.io import fits
from PIL import Image

DEDUP_THRESHOLD = 2.0  # mean abs difference on the final 8-bit [0,255] scale.
# Calibrated against 20 real downloaded frames (both sites): genuinely
# different-hour frames differed by ~20-42 on this scale for both Big Bear and
# Mauna Loa; an identical file compared against itself differs by exactly 0
# (a real stuck-camera repeat -- byte-identical source data -- would land
# here). 2.0 sits comfortably below the real-difference range without needing
# to be right at 0, leaving margin for JPEG re-encoding not being perfectly
# bit-exact run-to-run.

N_PROFILE_BINS = 32
CONTRAST_LOW_PCT = 1.0
CONTRAST_HIGH_PCT = 99.0
CONTRAST_INTERIOR_FRAC = 0.85  # fraction of disk radius treated as artifact-free
# interior for the percentile stretch -- excludes the outer annulus where
# limb-brightening lives, matching jp-analysis:scripts/apply_contrast_stretch.py


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


def _rho_grid(shape: tuple[int, int], cx: float, cy: float, r: float) -> np.ndarray:
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    return np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / r


def compute_radial_profile(images: list[np.ndarray], cx: float, cy: float, r: float,
                            n_bins: int = N_PROFILE_BINS) -> np.ndarray:
    """Empirical mean intensity per radial bin, pooled over a sample of
    (uncorrected) full frames sharing the same standardized (cx, cy, r). Used
    instead of a parametric limb-darkening law: a first attempt with the
    textbook photospheric-continuum formula 1/(0.3 + 0.7*mu) badly
    *overcorrected* real GONG Halpha frames -- checked with
    check_limb_flatness() below, the "corrected" profile sloped upward toward
    the limb instead of flattening. Halpha center-to-limb variation is much
    milder than continuum limb darkening (it's chromospheric emission, not
    photospheric), so a generic continuum formula doesn't transfer; an
    empirical profile fit to this data does, by construction."""
    if not images:
        raise ValueError("need at least one image to compute a radial profile")
    rho = _rho_grid(images[0].shape, cx, cy, r)
    bins = np.linspace(0, 1, n_bins + 1)
    profile = np.empty(n_bins, dtype=np.float64)
    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        mask = (rho >= lo) & (rho < hi)
        vals = np.concatenate([img[mask] for img in images]) if mask.any() else np.array([np.nan])
        profile[i] = vals.mean()
    # guard against a noisy/near-empty outermost bin producing a wild correction factor
    profile = np.where(np.isnan(profile), profile[~np.isnan(profile)][-1], profile)
    return profile


def limb_darkening_correct(img: np.ndarray, cx: float, cy: float, r: float, profile: np.ndarray) -> np.ndarray:
    """Radial intensity normalization using an empirical profile (see
    compute_radial_profile). Off-disk background (rho >= 1) is left untouched
    -- the profile is only defined inside the solar disk, and applying a
    correction off-disk would amplify background noise instead of leaving it flat.

    Interpolates the profile between bin centers (np.interp) rather than doing
    a nearest-bin lookup -- a nearest-bin version was tried first and produced
    visible concentric ring artifacts on real frames (confirmed by comparing
    against the uncorrected raw image, which has no rings) from the hard
    correction-factor jump at each bin boundary. Interpolation removes them."""
    n_bins = len(profile)
    rho = _rho_grid(img.shape, cx, cy, r)
    on_disk = rho < 1.0
    bin_centers = (np.arange(n_bins) + 0.5) / n_bins
    interpolated = np.interp(rho, bin_centers, profile)
    correction = profile[0] / interpolated
    out = img.copy()
    out[on_disk] = img[on_disk] * correction[on_disk]
    return out


def check_limb_flatness(img: np.ndarray, cx: float, cy: float, r: float, n_bins: int = 10) -> list[float]:
    """Diagnostic: mean intensity per radial bin (0 = center, 1 = limb). A
    correctly limb-darkening-corrected image should show roughly flat values
    across bins; a rising/falling trend means the profile sample was too small
    or unrepresentative."""
    rho = _rho_grid(img.shape, cx, cy, r)
    bins = np.linspace(0, 1, n_bins + 1)
    means = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (rho >= lo) & (rho < hi)
        means.append(float(img[mask].mean()) if mask.any() else float("nan"))
    return means


def load_frame(fits_path: Path) -> tuple[np.ndarray, dict]:
    """Full native 2048x2048, 1-channel frame (uncorrected) plus its disk geometry."""
    with fits.open(fits_path) as hdul:
        data_hdu = hdul[1] if len(hdul) > 1 else hdul[0]
        data = data_hdu.data.astype(np.float32)
        header = data_hdu.header
        cx, cy, r = read_disk_geometry(header)
    return data, {"cx": cx, "cy": cy, "r": r}


def contrast_stretch_to_uint8(img: np.ndarray, cx: float, cy: float, r: float,
                               low_pct: float = CONTRAST_LOW_PCT, high_pct: float = CONTRAST_HIGH_PCT,
                               interior_frac: float = CONTRAST_INTERIOR_FRAC) -> np.ndarray:
    """Per-image percentile clip + rescale to [0, 255], matching MAGFiLO's own
    8-bit JPEG format. Percentiles are computed only over the disk *interior*
    (rho < interior_frac), not the whole disk or frame -- anchoring on the
    outer annulus (where limb-brightening/the correction's own edge effects
    live) or the off-disk background would waste dynamic range on artifacts
    instead of the actual filament-relevant texture. Off-disk pixels are set
    to exactly 0 after stretching, matching real MAGFiLO images' corners."""
    rho = _rho_grid(img.shape, cx, cy, r)
    on_disk = rho < 1.0
    interior = on_disk & (rho < interior_frac)
    lo, hi = np.percentile(img[interior], [low_pct, high_pct])
    stretched = np.clip((img - lo) / max(hi - lo, 1e-6) * 255.0, 0, 255)
    out = np.zeros_like(stretched, dtype=np.uint8)
    out[on_disk] = stretched[on_disk].astype(np.uint8)
    return out


def process_one(fits_path: Path, profile: np.ndarray) -> tuple[np.ndarray, dict]:
    data, meta = load_frame(fits_path)
    corrected = limb_darkening_correct(data, meta["cx"], meta["cy"], meta["r"], profile)
    stretched = contrast_stretch_to_uint8(corrected, meta["cx"], meta["cy"], meta["r"])
    return stretched, meta


class PerSiteDeduper:
    """Streaming per-site stuck-camera dedup: drop a frame if it's nearly
    identical to the immediately preceding *kept* frame from the *same* site.
    Keeps only the last kept image per site in memory (not the whole corpus --
    at 2048x2048, holding every processed frame at once would use significant
    memory even at 1 byte/pixel for a 10K-image corpus)."""

    def __init__(self, threshold: float = DEDUP_THRESHOLD) -> None:
        self.threshold = threshold
        self._prev_by_site: dict[str, np.ndarray] = {}

    def is_duplicate(self, site: str, img: np.ndarray) -> bool:
        prev = self._prev_by_site.get(site)
        # cast to a signed dtype before differencing -- img/prev are uint8, and
        # unsigned subtraction wraps around (e.g. 5 - 200 -> 61, not -195)
        # instead of going negative, which would silently corrupt this check
        is_dup = prev is not None and np.abs(img.astype(np.int16) - prev.astype(np.int16)).mean() <= self.threshold
        if not is_dup:
            self._prev_by_site[site] = img
        return is_dup


class RunningStats:
    """Streaming mean/std (single-pass, float64 accumulators) so corpus-level
    stats don't require holding every image in memory simultaneously."""

    def __init__(self) -> None:
        self.sum = 0.0
        self.sumsq = 0.0
        self.count = 0
        self.n_images = 0

    def update(self, img: np.ndarray) -> None:
        self.sum += float(img.sum(dtype=np.float64))
        self.sumsq += float(np.square(img, dtype=np.float64).sum())
        self.count += img.size
        self.n_images += 1

    def finalize(self) -> dict:
        if self.count == 0:
            return {"mean": 0.0, "std": 1.0, "n_images": 0}
        mean = self.sum / self.count
        var = max(self.sumsq / self.count - mean**2, 0.0)
        return {"mean": mean, "std": var**0.5, "n_images": self.n_images}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-dir", type=Path, default=Path("data/raw/gong_pretrain"))
    p.add_argument("--out-dir", type=Path, default=Path("data/processed/gong_pretrain"))
    p.add_argument("--profile-sample-size", type=int, default=50,
                   help="number of raw frames used to build the empirical limb-darkening "
                        "profile -- kept small since each frame is a full 2048x2048 float32 array")
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
    sample_frames, sample_geom = [], None
    for fp in sample_paths:
        try:
            data, geom = load_frame(fp)
        except Exception as e:
            print(f"skip (profile sample) {fp}: {e}")
            continue
        sample_frames.append(data)
        sample_geom = geom  # standardized geometry (section 3) -- same for every file
    profile = compute_radial_profile(sample_frames, sample_geom["cx"], sample_geom["cy"], sample_geom["r"])
    np.save(args.out_dir / "limb_profile.npy", profile)
    print(f"built limb-darkening profile from {len(sample_frames)} sample frames")
    print(f"profile flatness check (pre-correction): {[round(x, 1) for x in profile]}")
    del sample_frames  # release the sampled full-res frames before pass 2

    # Pass 2: process every frame using that fixed profile, one at a time.
    manifest_rows = []
    deduper = PerSiteDeduper()
    stats = RunningStats()
    n_dropped = 0
    for fp in fits_files:
        try:
            img, meta = process_one(fp, profile)
        except Exception as e:
            print(f"skip {fp}: {e}")
            continue
        site = fp.parent.name
        if deduper.is_duplicate(site, img):
            n_dropped += 1
            continue
        out_path = args.out_dir / (fp.stem.replace(".fits", "") + ".jpeg")
        Image.fromarray(img, mode="L").save(out_path, quality=95)
        manifest_rows.append({"path": str(out_path), "site": site, "source": str(fp), **meta})
        stats.update(img)

    with open(args.out_dir / "manifest.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "site", "source", "cx", "cy", "r"])
        writer.writeheader()
        writer.writerows(manifest_rows)
    final_stats = stats.finalize()
    with open(args.out_dir / "dataset_stats.json", "w") as f:
        json.dump(final_stats, f, indent=2)

    print(f"processed {len(fits_files)} raw files -> {len(manifest_rows)} kept "
          f"(dropped {n_dropped} near-duplicates)")
    print(f"corpus stats: {final_stats}")


if __name__ == "__main__":
    main()
