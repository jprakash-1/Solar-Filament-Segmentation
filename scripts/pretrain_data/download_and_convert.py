#!/usr/bin/env python3
"""Combined Stage 1+2: download each manifested FITS file, convert it to
JPEG immediately, then delete the raw FITS file -- so raw data never
accumulates on disk beyond whatever's actively in flight.

Why this exists alongside the separate `gong_halpha.py download` +
`preprocess_gong.py` stages (both still work standalone, e.g. for the
existing downloaded corpus): at the current scale (up to 40K images across
6 sites / ~16 years), keeping the full raw corpus on disk would be
~100GB+ of FITS vs. ~20GB of JPEG output. This script keeps peak raw disk
usage bounded by concurrency, not by total corpus size.

Reuses existing, already-verified code rather than duplicating logic:
  - gong_halpha.check_connectivity / _get_with_retry (I/O download)
  - preprocess_gong.process_one (FITS -> uint8 conversion)

Concurrency: a ThreadPoolExecutor (--workers, I/O-bound) downloads files;
every download thread submits its file to one shared ProcessPoolExecutor
(--processes, CPU-bound) for conversion and blocks briefly on the result
(~0.1s, cheap relative to download latency) before starting its next
download. Deletion of the raw file happens only after the JPEG is
confirmed written -- a conversion failure deliberately leaves the raw file
in place for inspection rather than silently losing data.

Resume-safe the same way as gong_halpha.py's download: a row is skipped
entirely if its output JPEG already exists. The final manifest.csv merges
with whatever's already in --out-dir (by path), so running this
incrementally over successive manifest expansions accumulates rather than
overwrites.

A live tqdm progress bar renders on the terminal (postfix shows running
converted/skipped/failed counts); log lines (including download/convert
failures) print through tqdm.write() so they don't corrupt the bar.
--log-file additionally gets the periodic summary line, undisturbed by tqdm.

Usage:
    python scripts/pretrain_data/download_and_convert.py \
        --manifest manifests/gong_pretrain_manifest.csv \
        --raw-tmp-dir data/raw/gong_pretrain \
        --out-dir data/processed/gong_pretrain \
        --workers 8 --processes 8 \
        --log-file data/processed/gong_pretrain/download_convert.log
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from gong_halpha import _get_with_retry, check_connectivity
from preprocess_gong import _init_worker, process_one

logger = logging.getLogger("download_and_convert")


class _TqdmLoggingHandler(logging.Handler):
    """Routes log records through tqdm.write() instead of a plain stream --
    printing straight to stdout/stderr while a tqdm bar is active corrupts the
    bar's rendering (extra blank lines, the bar getting pushed down). This
    handler is what makes the progress bar and warning-level log lines (e.g.
    a download failure) coexist cleanly in an interactive terminal."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            tqdm.write(self.format(record))
        except Exception:
            self.handleError(record)


def setup_logging(log_file: Path | None = None, level: int = logging.INFO) -> None:
    logger.setLevel(level)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    console = _TqdmLoggingHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)


def _convert_and_cleanup(fits_path_str: str, out_dir_str: str, source_url: str, site: str) -> dict | None:
    """Runs in a worker process: convert one already-downloaded FITS file to
    JPEG, delete the raw file only on success, and return the lightweight
    manifest row (not the pixel array). `source` records the archive URL,
    not the local raw path -- that path is deleted right after this returns,
    so it wouldn't be a useful trail to keep."""
    fits_path = Path(fits_path_str)
    out_dir = Path(out_dir_str)
    try:
        img, meta = process_one(fits_path)
    except Exception as e:
        logging.warning(f"convert failed, keeping raw file {fits_path}: {e}")
        return None
    out_path = out_dir / (fits_path.stem.replace(".fits", "") + ".jpeg")
    Image.fromarray(img, mode="L").save(out_path, quality=95)
    fits_path.unlink()
    return {"path": str(out_path), "site": site, "source": source_url, **meta}


def _download_convert_one(row: dict, raw_tmp_dir: Path, out_dir: Path, process_pool: ProcessPoolExecutor) -> tuple[str, dict | None]:
    """Returns (status, manifest_row_or_None). status is one of: 'converted',
    'skipped' (output already exists), 'download_failed', 'convert_failed'.
    Never raises."""
    url = row["url"]
    site = row["site"]
    filename = Path(url).name
    out_path = out_dir / (filename.replace(".fits.fz", "").replace(".fits", "") + ".jpeg")
    if out_path.exists():
        return "skipped", None

    raw_dest = raw_tmp_dir / site / filename
    raw_dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        resp = _get_with_retry(url)
        if resp is None:
            return "download_failed", None
        resp.raise_for_status()
        raw_dest.write_bytes(resp.content)
    except Exception as e:
        logger.warning(f"download failed for {url}: {type(e).__name__}: {e}")
        return "download_failed", None

    future = process_pool.submit(_convert_and_cleanup, str(raw_dest), str(out_dir), url, site)
    try:
        result = future.result()
    except Exception:
        logger.exception(f"unexpected error converting {raw_dest}")
        result = None
    if result is None:
        return "convert_failed", None
    return "converted", result


def download_and_convert(
    manifest_csv: Path,
    raw_tmp_dir: Path,
    out_dir: Path,
    workers: int = 4,
    processes: int | None = None,
    progress_every: int = 50,
) -> None:
    check_connectivity()
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_tmp_dir.mkdir(parents=True, exist_ok=True)
    processes = processes or os.cpu_count()

    with open(manifest_csv) as f:
        rows = list(csv.DictReader(f))
    total = len(rows)
    logger.info(f"processing {total} files: {workers} download threads, {processes} convert processes")

    counts = {"converted": 0, "skipped": 0, "download_failed": 0, "convert_failed": 0}
    new_rows: list[dict] = []
    lock = threading.Lock()
    completed = 0
    start_time = time.monotonic()

    with ProcessPoolExecutor(max_workers=processes, initializer=_init_worker) as process_pool:
        with ThreadPoolExecutor(max_workers=workers) as thread_pool:
            futures = {
                thread_pool.submit(_download_convert_one, row, raw_tmp_dir, out_dir, process_pool): row
                for row in rows
            }
            with tqdm(total=total, desc="download+convert", unit="file") as pbar:
                for future in as_completed(futures):
                    try:
                        status, new_row = future.result()
                    except Exception:
                        logger.exception(f"unexpected error on {futures[future].get('url')}")
                        status, new_row = "download_failed", None
                    with lock:
                        counts[status] += 1
                        if new_row:
                            new_rows.append(new_row)
                        completed += 1
                        pbar.update(1)
                        pbar.set_postfix(counts, refresh=False)
                        if completed % progress_every == 0 or completed == total:
                            elapsed = time.monotonic() - start_time
                            rate = elapsed / completed
                            remaining = rate * (total - completed)
                            logger.info(
                                f"[{completed}/{total}] converted={counts['converted']} "
                                f"skipped={counts['skipped']} download_failed={counts['download_failed']} "
                                f"convert_failed={counts['convert_failed']} -- "
                                f"elapsed {elapsed / 60:.1f}min, ~{remaining / 60:.1f}min remaining"
                            )

    # Merge with whatever's already in out_dir/manifest.csv rather than
    # overwriting -- lets this be run incrementally over successive manifest
    # expansions (e.g. 2011-2016 now, 2017+ later) without losing earlier rows.
    manifest_out = out_dir / "manifest.csv"
    existing_rows: dict[str, dict] = {}
    if manifest_out.exists():
        with open(manifest_out) as f:
            for r in csv.DictReader(f):
                existing_rows[r["path"]] = r
    for r in new_rows:
        existing_rows[r["path"]] = r
    with open(manifest_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "site", "source", "cx", "cy", "r"])
        writer.writeheader()
        writer.writerows(existing_rows.values())

    logger.info(f"done: {counts} -- manifest now has {len(existing_rows)} total rows")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--raw-tmp-dir", type=Path, default=Path("data/raw/gong_pretrain"),
                   help="scratch space for a file between download and conversion -- files are deleted "
                        "right after successful conversion, so this never accumulates the full corpus")
    p.add_argument("--out-dir", type=Path, default=Path("data/processed/gong_pretrain"))
    p.add_argument("--workers", type=int, default=4, help="concurrent download threads")
    p.add_argument("--processes", type=int, default=None, help="convert worker processes (default: CPU count)")
    p.add_argument("--progress-every", type=int, default=50)
    p.add_argument("--log-file", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_file)
    download_and_convert(
        manifest_csv=args.manifest,
        raw_tmp_dir=args.raw_tmp_dir,
        out_dir=args.out_dir,
        workers=args.workers,
        processes=args.processes,
        progress_every=args.progress_every,
    )


if __name__ == "__main__":
    main()
