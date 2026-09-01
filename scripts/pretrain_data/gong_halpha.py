#!/usr/bin/env python3
"""Stage 1: scrape + download GONG H-Alpha "reduced" FITS imagery.

Verified directly against the live archive (see PRETRAIN_PLAN.md section 2):
  https://gong2.nso.edu/HA/haf/<YYYYMM>/<YYYYMMDD>/<YYYYMMDDHHMMSS><site><h>.fits.fz
Directory listing is a plain Apache-style index -- parsed with BeautifulSoup,
filtering strictly to `.fits.fz` hrefs. The listing page also contains a
zero-font-size decoy link (a bot-trap) that must never be followed; the
`.fits.fz` suffix filter already excludes it, but don't loosen that filter.

Two-step usage, matching the "manifest before download" plan:
    python scripts/pretrain_data/gong_halpha.py manifest --out data/raw/gong_pretrain/manifest.csv
    python scripts/pretrain_data/gong_halpha.py download --manifest data/raw/gong_pretrain/manifest.csv \
        --out-dir data/raw/gong_pretrain
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://gong2.nso.edu/HA/haf"

# Single-letter site codes embedded in filenames, confirmed against a real
# directory listing (2022-03-18) -- not the two-letter codes used by the
# archive's query-form UI (bb/ml/le/ud/td/ct).
SITE_CODES = {
    "big_bear": "B",
    "mauna_loa": "M",
    "learmonth": "L",
    "udaipur": "U",
    "teide": "T",
    "cerro_tololo": "C",
}

USER_AGENT = "solar-filament-segmentation-2026-pretrain-data (research, low-rate)"
REQUEST_TIMEOUT = 30
REQUEST_DELAY_SECONDS = 0.5  # politeness delay between requests to a public archive


@dataclass
class FrameRef:
    site: str
    timestamp: datetime
    url: str


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


def list_day_files(session: requests.Session, site_letter: str, day: date) -> list[FrameRef]:
    """List one site's files for one UTC day. Returns [] if the day has no
    directory (e.g. too far in the future) or the site has no frames that day
    (down for maintenance, weather, etc.) -- both are normal, not errors."""
    yyyymm = day.strftime("%Y%m")
    yyyymmdd = day.strftime("%Y%m%d")
    url = f"{BASE_URL}/{yyyymm}/{yyyymmdd}/"
    resp = session.get(url, timeout=REQUEST_TIMEOUT)
    if resp.status_code == 404:
        return []
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    frames = []
    for a in soup.find_all("a"):
        href = a.get("href", "")
        if not href.endswith(".fits.fz"):
            continue  # excludes the hidden decoy link and any non-data entries
        stem = href[: -len(".fits.fz")]
        # YYYYMMDDHHMMSS (14) + <site letter> (1) + "h" (1) = 16 chars, e.g.
        # "20220318000050Bh"
        if len(stem) != 16 or stem[15] != "h" or not stem[14].isalpha():
            continue  # not the expected shape
        if stem[14] != site_letter:
            continue  # this day's listing is shared across all sites
        try:
            ts = datetime.strptime(stem[:14], "%Y%m%d%H%M%S")
        except ValueError:
            continue
        frames.append(FrameRef(site=site_letter, timestamp=ts, url=url + href))
    return frames


def nearest_per_hour(frames: list[FrameRef], tolerance_minutes: int = 20) -> list[FrameRef]:
    """Pick the single frame closest to each :00 UTC hour mark, skipping hours
    with nothing within tolerance (expected: the site's local nighttime)."""
    by_hour: dict[datetime, FrameRef] = {}
    best_delta: dict[datetime, float] = {}
    for f in frames:
        hour_mark = f.timestamp.replace(minute=0, second=0, microsecond=0)
        delta = abs((f.timestamp - hour_mark).total_seconds()) / 60.0
        if delta > tolerance_minutes:
            continue
        if hour_mark not in best_delta or delta < best_delta[hour_mark]:
            best_delta[hour_mark] = delta
            by_hour[hour_mark] = f
    return sorted(by_hour.values(), key=lambda f: f.timestamp)


def build_manifest(
    sites: list[str],
    date_ranges: list[tuple[date, date]],
    day_stride: int = 1,
    tolerance_minutes: int = 20,
    max_images: int | None = None,
) -> list[FrameRef]:
    """Walk the given date ranges (stepping every `day_stride` days) and
    collect ~1 frame/hour per site. Ground telescopes only see the Sun during
    local daytime, so expect well under 24 frames/site/day, not a full day's
    worth -- that's normal, not a bug."""
    session = _session()
    manifest: list[FrameRef] = []
    for start, end in date_ranges:
        day = start
        while day <= end:
            for site in sites:
                site_letter = SITE_CODES[site]
                frames = list_day_files(session, site_letter, day)
                manifest.extend(nearest_per_hour(frames, tolerance_minutes))
                time.sleep(REQUEST_DELAY_SECONDS)
            day += timedelta(days=day_stride)

    manifest.sort(key=lambda f: (f.site, f.timestamp))
    if max_images and len(manifest) > max_images:
        # even subsample across the whole manifest rather than truncating,
        # so the kept frames still span the full requested date range
        stride = len(manifest) / max_images
        manifest = [manifest[int(i * stride)] for i in range(max_images)]
    return manifest


def write_manifest_csv(manifest: list[FrameRef], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["site", "timestamp", "url"])
        for frame in manifest:
            writer.writerow([frame.site, frame.timestamp.isoformat(), frame.url])


def download_manifest(manifest_csv: Path, out_dir: Path) -> None:
    session = _session()
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(manifest_csv) as f:
        rows = list(csv.DictReader(f))

    for i, row in enumerate(rows):
        url = row["url"]
        dest = out_dir / row["site"] / Path(url).name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and dest.stat().st_size > 0:
            continue  # resume-safe: skip already-downloaded files
        try:
            resp = session.get(url, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
        except requests.RequestException as e:
            print(f"skip {url}: {e}")
        time.sleep(REQUEST_DELAY_SECONDS)
        if (i + 1) % 100 == 0:
            print(f"downloaded {i + 1}/{len(rows)}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("manifest", help="build a (site, timestamp, url) manifest without downloading")
    m.add_argument("--sites", nargs="+", default=["big_bear", "mauna_loa"], choices=list(SITE_CODES))
    m.add_argument("--date-ranges", nargs="+", default=["2019-06-01:2020-06-01", "2023-06-01:2024-06-01"],
                    help="one or more START:END dates (YYYY-MM-DD), e.g. a quiet-sun and an active-sun window")
    m.add_argument("--day-stride", type=int, default=3, help="sample every Nth day within each range")
    m.add_argument("--tolerance-minutes", type=int, default=20)
    m.add_argument("--max-images", type=int, default=10000)
    m.add_argument("--out", type=Path, default=Path("data/raw/gong_pretrain/manifest.csv"))

    d = sub.add_parser("download", help="download every file listed in a manifest CSV")
    d.add_argument("--manifest", type=Path, required=True)
    d.add_argument("--out-dir", type=Path, default=Path("data/raw/gong_pretrain"))

    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.cmd == "manifest":
        date_ranges = []
        for r in args.date_ranges:
            start_s, end_s = r.split(":")
            date_ranges.append((date.fromisoformat(start_s), date.fromisoformat(end_s)))
        manifest = build_manifest(
            sites=args.sites,
            date_ranges=date_ranges,
            day_stride=args.day_stride,
            tolerance_minutes=args.tolerance_minutes,
            max_images=args.max_images,
        )
        write_manifest_csv(manifest, args.out)
        print(f"wrote {len(manifest)} rows to {args.out}")
    elif args.cmd == "download":
        download_manifest(args.manifest, args.out_dir)


if __name__ == "__main__":
    main()
