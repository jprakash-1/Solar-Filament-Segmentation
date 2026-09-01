#!/usr/bin/env python3
"""Stage 2: convert downloaded GONG H-Alpha FITS frames to JPEG. No other
processing -- no limb-darkening correction, no contrast stretch beyond the
per-image min-max rescale needed to fit 16-bit FITS data into an 8-bit JPEG at
all, no dedup.

Keeps the full native 2048x2048, 1-channel frame, matching MAGFiLO's own
format exactly (verified: a real MAGFiLO training image is itself a
2048x2048, single-channel (`L` mode) JPEG, with the same GONG-style filename).

Disk center/radius are still read from the header (`FNDLMBXC`/`FNDLMBYC`/
`FNDLMBMA`/`FNDLMBMI` -- GONG's own standardized geometry, see git history of
this file for how that was verified) and carried into the manifest as plain
metadata, since they're free to read and may be useful later -- they are not
applied to the pixels here.

Usage:
    python scripts/pretrain_data/preprocess_gong.py \
        --raw-dir data/raw/gong_pretrain --out-dir data/processed/gong_pretrain
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from astropy.io import fits
from PIL import Image


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


def load_frame(fits_path: Path) -> tuple[np.ndarray, dict]:
    """Full native 2048x2048, 1-channel frame plus its disk geometry (metadata only)."""
    with fits.open(fits_path) as hdul:
        data_hdu = hdul[1] if len(hdul) > 1 else hdul[0]
        data = data_hdu.data.astype(np.float32)
        header = data_hdu.header
        cx, cy, r = read_disk_geometry(header)
    return data, {"cx": cx, "cy": cy, "r": r}


def to_uint8(data: np.ndarray) -> np.ndarray:
    """Linearly rescale this image's own min-max to [0, 255] -- the minimal
    step needed to fit 16-bit FITS data into an 8-bit JPEG at all. No disk
    masking, no percentile clipping, no per-region logic."""
    lo, hi = float(data.min()), float(data.max())
    scaled = (data - lo) / max(hi - lo, 1e-6) * 255.0
    return np.clip(scaled, 0, 255).astype(np.uint8)


def process_one(fits_path: Path) -> tuple[np.ndarray, dict]:
    data, meta = load_frame(fits_path)
    return to_uint8(data), meta


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-dir", type=Path, default=Path("data/raw/gong_pretrain"))
    p.add_argument("--out-dir", type=Path, default=Path("data/processed/gong_pretrain"))
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fits_files = sorted(args.raw_dir.rglob("*.fits.fz")) + sorted(args.raw_dir.rglob("*.fits"))
    if not fits_files:
        raise SystemExit(f"no .fits/.fits.fz files found under {args.raw_dir}")

    manifest_rows = []
    for fp in fits_files:
        try:
            img, meta = process_one(fp)
        except Exception as e:
            print(f"skip {fp}: {e}")
            continue
        site = fp.parent.name
        out_path = args.out_dir / (fp.stem.replace(".fits", "") + ".jpeg")
        Image.fromarray(img, mode="L").save(out_path, quality=95)
        manifest_rows.append({"path": str(out_path), "site": site, "source": str(fp), **meta})

    with open(args.out_dir / "manifest.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "site", "source", "cx", "cy", "r"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"converted {len(manifest_rows)}/{len(fits_files)} raw files to JPEG")


if __name__ == "__main__":
    main()
