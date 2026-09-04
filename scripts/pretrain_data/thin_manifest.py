#!/usr/bin/env python3
"""Thin the Stage 2 output by keeping at most 2 frames per (site, date)
instead of every hourly frame, to cut near-duplicate redundancy before
Stage 3 pretraining.

Why: consecutive-hour frames from the same site are near-duplicates by
construction -- the Sun only rotates ~0.5deg/hour and filaments persist for
hours-to-days, so an hourly-cadence sequence barely moves. Measured directly
on this corpus (disk-cropped SSIM, `--day-stride 3` already applied across
dates): same-site consecutive-hour pairs average SSIM 0.91 (85% exceed 0.90),
vs. 0.79 for random cross-date pairs and 0.84 for same-site frames ~3 days
apart. The corpus's real diversity axis is across dates and sites, not within
a day, so thinning is done as *structured temporal subsampling* (targeted at
the known redundancy source) rather than generic perceptual-hash/pairwise
dedup over the full 40K-image corpus.

Selection per (site, date) group with N frames sorted by time:
  - N <= 2: keep all
  - N > 2: keep the frames nearest the 25th and 75th percentile positions,
    so the two kept frames are spread across the middle of the site's
    daylight window rather than picking the (possibly lower-quality,
    limb-grazing) very first/last frames of the day.

Non-destructive: only writes a new manifest CSV listing the kept subset --
does not move or delete any JPEGs. `--images-dir` (not the Stage 2
manifest.csv) is the source of truth for which files exist, since a prior
pipeline gap left `data/processed/gong_pretrain/manifest.csv` covering only
~16K of the ~40K JPEGs on disk; `--manifest` is consulted opportunistically
for cx/cy/r/source and falls back to GONG's standardized geometry
(1024, 1024, 900 -- see preprocess_gong.py) for files it doesn't cover.

Usage:
    python scripts/pretrain_data/thin_manifest.py \
        --images-dir data/processed/gong_pretrain \
        --manifest data/processed/gong_pretrain/manifest.csv \
        --out data/processed/gong_pretrain/manifest_thinned.csv \
        --per-day-site 2
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger("thin_manifest")

FNAME_RE = re.compile(r"^(\d{8})(\d{6})([A-Za-z])h\.jpe?g$")

# GONG's own processing pipeline re-registers every frame to this standardized
# geometry (verified in preprocess_gong.py) -- used as a fallback for files
# missing from --manifest.
DEFAULT_CX, DEFAULT_CY, DEFAULT_R = 1024.0, 1024.0, 900.0


def parse_filename(name: str) -> tuple[str, str, str] | None:
    """Returns (date, time, site) from a `<YYYYMMDDHHMMSS><site>h.jpeg` filename, or None."""
    m = FNAME_RE.match(name)
    if not m:
        return None
    return m.group(1), m.group(2), m.group(3)


def select_thinned(images_dir: Path, per_day_site: int) -> dict[tuple[str, str], list[Path]]:
    """Groups files by (date, site) and selects up to `per_day_site` frames per
    group, spread across the middle of the day. Returns {(date, site): [paths]}."""
    groups: dict[tuple[str, str], list[tuple[str, Path]]] = defaultdict(list)
    n_unparsed = 0
    for path in images_dir.glob("*.jpeg"):
        parsed = parse_filename(path.name)
        if parsed is None:
            n_unparsed += 1
            continue
        date, time_, site = parsed
        groups[(date, site)].append((time_, path))
    if n_unparsed:
        logger.warning(f"skipped {n_unparsed} files that didn't match the expected filename pattern")

    selected: dict[tuple[str, str], list[Path]] = {}
    for key, entries in groups.items():
        entries.sort(key=lambda e: e[0])
        n = len(entries)
        if n <= per_day_site:
            selected[key] = [p for _, p in entries]
            continue
        # Evenly spaced quantile positions across the sorted (by time) frames,
        # e.g. per_day_site=2 -> 25th/75th percentile; per_day_site=3 -> quartiles.
        idxs = sorted({round((i + 0.5) * n / per_day_site - 0.5) for i in range(per_day_site)})
        idxs = [min(max(i, 0), n - 1) for i in idxs]
        selected[key] = [entries[i][1] for i in idxs]
    return selected


def load_metadata(manifest_csv: Path | None) -> dict[str, dict]:
    """Maps JPEG path -> {site, source, cx, cy, r} from an existing manifest.csv, if given."""
    if manifest_csv is None or not manifest_csv.exists():
        return {}
    with open(manifest_csv) as f:
        return {row["path"]: row for row in csv.DictReader(f)}


def thin(images_dir: Path, manifest_csv: Path | None, out_csv: Path, per_day_site: int) -> None:
    metadata = load_metadata(manifest_csv)
    logger.info(f"loaded metadata for {len(metadata)} files from {manifest_csv}")

    selected = select_thinned(images_dir, per_day_site)
    total_files = sum(1 for _ in images_dir.glob("*.jpeg"))
    n_kept = sum(len(v) for v in selected.values())
    logger.info(
        f"{len(selected)} (date, site) groups; keeping {n_kept}/{total_files} frames "
        f"({n_kept / total_files:.1%}) at --per-day-site {per_day_site}"
    )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "site", "source", "cx", "cy", "r"])
        writer.writeheader()
        for (date, site), paths in selected.items():
            for path in paths:
                row = metadata.get(str(path))
                if row is not None:
                    writer.writerow(row)
                else:
                    writer.writerow({"path": str(path), "site": site, "source": "",
                                      "cx": DEFAULT_CX, "cy": DEFAULT_CY, "r": DEFAULT_R})
    logger.info(f"wrote {n_kept} rows to {out_csv}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images-dir", type=Path, default=Path("data/processed/gong_pretrain"))
    p.add_argument("--manifest", type=Path, default=Path("data/processed/gong_pretrain/manifest.csv"),
                   help="existing Stage 2 manifest, consulted for cx/cy/r/source where available")
    p.add_argument("--out", type=Path, default=Path("data/processed/gong_pretrain/manifest_thinned.csv"))
    p.add_argument("--per-day-site", type=int, default=2,
                   help="max frames to keep per (site, date) group")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    args = parse_args()
    thin(args.images_dir, args.manifest, args.out, args.per_day_site)


if __name__ == "__main__":
    main()
