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

Multi-process (CPU-bound: FITS decompression + array math + JPEG encoding,
not I/O-bound like Stage 1) via ProcessPoolExecutor. `--processes` defaults to
the machine's CPU count.

Usage:
    python scripts/pretrain_data/preprocess_gong.py \
        --raw-dir data/raw/gong_pretrain --out-dir data/processed/gong_pretrain
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from astropy.io import fits
from PIL import Image

logger = logging.getLogger("preprocess_gong")


def setup_logging(log_file: Path | None = None, level: int = logging.INFO) -> None:
    logger.setLevel(level)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)


def _init_worker() -> None:
    # Each worker process needs its own logging config -- it doesn't inherit
    # the main process's handlers under the 'spawn' start method (macOS/
    # Windows default). Console-only here; the main process aggregates
    # progress from returned results, so per-worker file logging isn't needed.
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(processName)s: %(message)s",
        datefmt="%H:%M:%S",
    )


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


def _process_and_save(fits_path_str: str, out_dir_str: str) -> dict | None:
    """Runs in a worker process: convert one FITS file and save it, returning
    only the lightweight manifest row (not the pixel array) back to the main
    process."""
    fits_path = Path(fits_path_str)
    out_dir = Path(out_dir_str)
    try:
        img, meta = process_one(fits_path)
    except Exception as e:
        logging.warning(f"skip {fits_path}: {e}")
        return None
    site = fits_path.parent.name
    out_path = out_dir / (fits_path.stem.replace(".fits", "") + ".jpeg")
    Image.fromarray(img, mode="L").save(out_path, quality=95)
    return {"path": str(out_path), "site": site, "source": str(fits_path), **meta}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-dir", type=Path, default=Path("data/raw/gong_pretrain"))
    p.add_argument("--out-dir", type=Path, default=Path("data/processed/gong_pretrain"))
    p.add_argument("--processes", type=int, default=os.cpu_count(), help="worker processes (CPU-bound work)")
    p.add_argument("--progress-every", type=int, default=50)
    p.add_argument("--log-file", type=Path, default=None)
    args = p.parse_args()
    setup_logging(args.log_file)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fits_files = sorted(args.raw_dir.rglob("*.fits.fz")) + sorted(args.raw_dir.rglob("*.fits"))
    if not fits_files:
        raise SystemExit(f"no .fits/.fits.fz files found under {args.raw_dir}")
    total = len(fits_files)
    logger.info(f"converting {total} files using {args.processes} processes")

    manifest_rows = []
    n_failed = 0
    completed = 0
    start_time = time.monotonic()

    with ProcessPoolExecutor(max_workers=args.processes, initializer=_init_worker) as executor:
        futures = {
            executor.submit(_process_and_save, str(fp), str(args.out_dir)): fp
            for fp in fits_files
        }
        for future in as_completed(futures):
            row = future.result()
            completed += 1
            if row is None:
                n_failed += 1
            else:
                manifest_rows.append(row)
            if completed % args.progress_every == 0 or completed == total:
                elapsed = time.monotonic() - start_time
                rate = elapsed / completed
                remaining = rate * (total - completed)
                logger.info(
                    f"[{completed}/{total}] converted={len(manifest_rows)} failed={n_failed} -- "
                    f"elapsed {elapsed / 60:.1f}min, ~{remaining / 60:.1f}min remaining"
                )

    with open(args.out_dir / "manifest.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "site", "source", "cx", "cy", "r"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    logger.info(f"converted {len(manifest_rows)}/{total} raw files to JPEG ({n_failed} failed)")


if __name__ == "__main__":
    main()
