#!/usr/bin/env python3
"""Stage 1: scrape + download GONG H-Alpha "reduced" FITS imagery.

Verified directly against the live archive (see PRETRAIN_PLAN.md section 2):
  https://gong2.nso.edu/HA/haf/<YYYYMM>/<YYYYMMDD>/<YYYYMMDDHHMMSS><site><h>.fits.fz
Directory listing is a plain Apache-style index -- parsed with BeautifulSoup,
filtering strictly to `.fits.fz` hrefs. The listing page also contains a
zero-font-size decoy link (a bot-trap) that must never be followed; the
`.fits.fz` suffix filter already excludes it, but don't loosen that filter.

Both `manifest` and `download` are multi-threaded (I/O-bound: network
requests, not CPU work) via a bounded ThreadPoolExecutor -- each thread gets
its own `requests.Session` (thread-local) rather than sharing one across
threads. `--workers` controls concurrency; keep it modest (default 4) since
this is hitting a shared public archive, not a CDN built for parallel clients.

Two-step usage, matching the "manifest before download" plan:
    python scripts/pretrain_data/gong_halpha.py manifest --out data/raw/gong_pretrain/manifest.csv
    python scripts/pretrain_data/gong_halpha.py download --manifest data/raw/gong_pretrain/manifest.csv \
        --out-dir data/raw/gong_pretrain
"""

from __future__ import annotations

import argparse
import csv
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
REQUEST_DELAY_SECONDS = 0.5  # politeness delay a single thread waits between its
# own requests -- with N worker threads the aggregate rate is roughly N/0.5 req/s,
# which is why --workers defaults to a modest 4 rather than something large
DAY_RETRIES = 2  # transient-failure retries for a single day's listing/download,
# not a substitute for the upfront connectivity check below -- a systemic outage
# (e.g. Kaggle's Internet toggle off) should fail fast with a clear message, not
# retry-and-skip silently across a whole multi-year date range doing nothing useful

logger = logging.getLogger("gong_halpha")


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


@dataclass
class FrameRef:
    site: str
    timestamp: datetime
    url: str


_thread_local = threading.local()


def _session() -> requests.Session:
    """One requests.Session per worker thread (thread-local), not one shared
    session across threads -- sidesteps any question of whether concurrent use
    of a single Session is safe, at the cost of one extra connection pool per
    thread (negligible at --workers ~4-8)."""
    if not hasattr(_thread_local, "session"):
        s = requests.Session()
        s.headers.update({"User-Agent": USER_AGENT})
        _thread_local.session = s
    return _thread_local.session


def check_connectivity() -> None:
    """Fail fast with an actionable message if the archive is unreachable at
    all, instead of only discovering it after retrying every single day in a
    multi-year date range. A `ConnectTimeout` here (not a slow response, a
    failure to even open the TCP connection) on Kaggle can mean the notebook's
    Internet toggle is off (Settings -> Internet -> On, requires phone
    verification on the account) -- but if general internet access otherwise
    works (e.g. `!wget google.com` succeeds), this instead points at this
    specific host being unreachable from that network (e.g. an IP-range block
    on the archive's side), not a local settings problem."""
    try:
        _session().get(BASE_URL, timeout=REQUEST_TIMEOUT)
    except requests.RequestException as e:
        raise RuntimeError(
            f"Could not reach {BASE_URL} ({e}). If general internet access "
            "works otherwise (e.g. Kaggle's Internet toggle is on and "
            "google.com is reachable), this points at this specific host "
            "being blocked/unreachable from this network -- try downloading "
            "elsewhere and uploading the result as a Kaggle Dataset instead. "
            "If general internet access doesn't work either, enable it under "
            "Settings -> Internet first."
        ) from e


def _get_with_retry(url: str, retries: int = DAY_RETRIES) -> requests.Response | None:
    """Retry a transient connection failure a couple of times before giving up
    on just this one URL (the whole run already passed check_connectivity, so
    this is about a single flaky request, not a systemic outage)."""
    for attempt in range(retries + 1):
        try:
            return _session().get(url, timeout=REQUEST_TIMEOUT)
        except (requests.ConnectionError, requests.Timeout) as e:
            if attempt == retries:
                logger.warning(f"giving up on {url} after {retries + 1} attempts ({e})")
                return None
            time.sleep(2 * (attempt + 1))
    return None


def list_day_files(site_letter: str, day: date) -> list[FrameRef]:
    """List one site's files for one UTC day. Returns [] if the day has no
    directory (e.g. too far in the future), the site has no frames that day
    (down for maintenance, weather, etc.), or the request failed even after
    retries -- all three are non-fatal so one bad day doesn't abort the whole
    date range."""
    yyyymm = day.strftime("%Y%m")
    yyyymmdd = day.strftime("%Y%m%d")
    url = f"{BASE_URL}/{yyyymm}/{yyyymmdd}/"
    resp = _get_with_retry(url)
    time.sleep(REQUEST_DELAY_SECONDS)
    if resp is None:
        return []
    if resp.status_code == 404:
        return []
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        # a non-2xx/404 status (500, 503, etc.) is treated the same as "no
        # data this day" rather than propagating -- an uncaught exception
        # here would reach build_manifest's future.result() and silently
        # break its counting/progress-logging for every remaining future
        logger.warning(f"skipping {url}: {e}")
        return []

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


def thin_per_day(frames: list[FrameRef], per_day: int | None) -> list[FrameRef]:
    """Keep at most `per_day` frames out of one (site, day)'s hourly frames,
    picked at evenly-spaced quantile positions (e.g. per_day=2 -> the frames
    nearest the 25th/75th percentile time) so they're spread across the
    middle of the site's daylight window rather than the possibly
    limb-grazing first/last hour. `frames` is assumed already sorted by
    timestamp (true of `nearest_per_hour`'s output) and to be a single (site,
    day)'s frames, since this is called per listing task in `build_manifest`.

    Mirrors `thin_manifest.py`'s post-hoc selection logic (see
    PRETRAIN_PLAN.md for the SSIM measurements motivating it), but applied at
    manifest-build time so a fresh download doesn't fetch hourly frames only
    to discard most of them afterward -- `per_day=None` (the default) keeps
    the old ~1/hour behavior for callers that still want the full cadence."""
    n = len(frames)
    if per_day is None or n <= per_day:
        return frames
    idxs = sorted({round((i + 0.5) * n / per_day - 0.5) for i in range(per_day)})
    idxs = [min(max(i, 0), n - 1) for i in idxs]
    return [frames[i] for i in idxs]


def _enumerate_tasks(sites: list[str], date_ranges: list[tuple[date, date]], day_stride: int) -> list[tuple[str, date]]:
    tasks = []
    for start, end in date_ranges:
        day = start
        while day <= end:
            for site in sites:
                tasks.append((SITE_CODES[site], day))
            day += timedelta(days=day_stride)
    return tasks


def build_manifest(
    sites: list[str],
    date_ranges: list[tuple[date, date]],
    day_stride: int = 1,
    tolerance_minutes: int = 20,
    max_images: int | None = None,
    checkpoint_path: Path | None = None,
    progress_every: int = 20,
    workers: int = 4,
    per_day_site: int | None = 2,
) -> list[FrameRef]:
    """Walk the given date ranges (stepping every `day_stride` days) and
    collect up to `per_day_site` frames/day per site (default 2, spread
    across the day -- see `thin_per_day`), `workers` (site, day) listings at
    a time. Ground telescopes only see the Sun during local daytime, so
    expect well under 24 frames/site/day even before this thinning, not a
    full day's worth -- that's normal, not a bug.

    `per_day_site=2` is the default (not the old ~1/hour) because measuring
    the already-downloaded corpus showed same-site consecutive-hour frames
    average SSIM 0.91 (85% exceed 0.90) -- see PRETRAIN_PLAN.md. Fetching
    hourly and thinning after the fact wastes the download itself; passing
    `per_day_site=None` restores the old full-cadence behavior.

    Logs a running "N/total" progress line every `progress_every` completed
    listings (wall-clock alone isn't a reliable progress signal -- any single
    request can silently retry/backoff). If `checkpoint_path` is given, the
    manifest built *so far* (pre-subsampling) is written there at the same
    cadence, so a killed run leaves a real, inspectable partial CSV.

    Genuinely resumable, not just checkpointed for visibility: a sidecar
    `<checkpoint_path>.progress` file records one "site,day" line per
    *completed* (site, day) task, flushed immediately after each one (not
    just at the progress_every cadence). On start, any already-completed
    tasks found there are skipped entirely, and the matching checkpoint
    manifest CSV (if present) is loaded to seed the running result -- a large
    multi-hour pull that gets killed partway (e.g. a ~11K-task, ~6-site,
    15-year pull was killed by the environment after ~63 minutes / ~26% done
    in practice) picks up where it left off instead of re-querying everything
    from scratch."""
    check_connectivity()
    tasks = _enumerate_tasks(sites, date_ranges, day_stride)
    total = len(tasks)

    progress_log_path = Path(str(checkpoint_path) + ".progress") if checkpoint_path else None
    manifest: list[FrameRef] = []
    done_tasks: set[tuple[str, date]] = set()
    # Seed from an existing checkpoint CSV whenever one is present -- even
    # without a matching .progress file (e.g. a checkpoint left by a run from
    # before this resumability was added), those frames are still real and
    # shouldn't be thrown away. The URL-based de-dupe at the end of this
    # function makes it safe to seed frames without also knowing exactly
    # which tasks produced them.
    if checkpoint_path and checkpoint_path.exists():
        with open(checkpoint_path) as f:
            for row in csv.DictReader(f):
                manifest.append(FrameRef(row["site"], datetime.fromisoformat(row["timestamp"]), row["url"]))
    if progress_log_path and progress_log_path.exists():
        with open(progress_log_path) as f:
            for line in f:
                site_letter, day_iso = line.strip().split(",")
                done_tasks.add((site_letter, date.fromisoformat(day_iso)))
    if manifest or done_tasks:
        logger.info(f"resuming: {len(done_tasks)} tasks marked done, {len(manifest)} frames already collected")

    remaining_tasks = [t for t in tasks if t not in done_tasks]
    logger.info(
        f"listing {len(remaining_tasks)}/{total} remaining (site, day) combinations "
        f"using {workers} threads"
    )

    lock = threading.Lock()
    completed = len(done_tasks)
    start_time = time.monotonic()
    progress_log = open(progress_log_path, "a") if progress_log_path else None

    def _worker(site_letter: str, day: date) -> list[FrameRef]:
        hourly = nearest_per_hour(list_day_files(site_letter, day), tolerance_minutes)
        return thin_per_day(hourly, per_day_site)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_worker, site_letter, day): (site_letter, day) for site_letter, day in remaining_tasks}
        for future in as_completed(futures):
            site_letter, day = futures[future]
            try:
                frames = future.result()
            except Exception as e:
                logger.warning(f"failed listing site={site_letter} day={day}: {e}")
                frames = []
            with lock:
                manifest.extend(frames)
                completed += 1
                if progress_log:
                    progress_log.write(f"{site_letter},{day.isoformat()}\n")
                    progress_log.flush()
                if completed % progress_every == 0 or completed == total:
                    elapsed = time.monotonic() - start_time
                    done_this_run = completed - len(done_tasks)
                    rate = elapsed / max(done_this_run, 1)
                    remaining = rate * (total - completed)
                    logger.info(
                        f"[{completed}/{total}] {len(manifest)} frames so far -- "
                        f"elapsed {elapsed / 60:.1f}min, ~{remaining / 60:.1f}min remaining"
                    )
                    if checkpoint_path:
                        # de-dupe before every checkpoint write, not just the
                        # final return -- without this, repeated kill/resume
                        # cycles (seeding from the last checkpoint, then
                        # appending newly-found frames) accumulate duplicate
                        # rows in the checkpoint itself. Hit this for real:
                        # one paused multi-hour pull had 25,156 duplicate rows
                        # out of 62,298 (~40%) after several kill/resume cycles,
                        # since the checkpoint was never deduped, only the
                        # (never-reached, this run kept getting interrupted)
                        # final return value was.
                        deduped = list({f.url: f for f in manifest}.values())
                        write_manifest_csv(sorted(deduped, key=lambda f: (f.site, f.timestamp)), checkpoint_path)

    if progress_log:
        progress_log.close()

    # De-dupe by URL -- a resumed run without a matching .progress file for an
    # older, pre-resumability checkpoint can re-query some already-covered
    # (site, day) tasks, which would otherwise double-count those frames
    manifest = list({f.url: f for f in manifest}.values())
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


def _download_one(row: dict, out_dir: Path) -> str:
    """Returns one of 'downloaded', 'skipped' (already on disk), 'failed'.
    Never raises -- any exception here (network, filesystem, whatever) would
    otherwise propagate through future.result() in download_manifest's loop
    and silently break its counting/progress-logging for every remaining
    future (the ThreadPoolExecutor keeps running already-submitted downloads
    in the background regardless, so files kept landing on disk with zero
    further log output -- exactly what happened on a real run before this
    except clause was widened from requests.RequestException to Exception)."""
    url = row["url"]
    dest = out_dir / row["site"] / Path(url).name
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and dest.stat().st_size > 0:
            return "skipped"
        resp = _get_with_retry(url)
        if resp is None:
            return "failed"
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        return "downloaded"
    except Exception as e:
        logger.warning(f"skip {url}: {type(e).__name__}: {e}")
        return "failed"


def download_manifest(manifest_csv: Path, out_dir: Path, workers: int = 4, progress_every: int = 50) -> None:
    check_connectivity()
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(manifest_csv) as f:
        rows = list(csv.DictReader(f))
    total = len(rows)
    logger.info(f"downloading {total} files using {workers} threads")

    counts = {"downloaded": 0, "skipped": 0, "failed": 0}
    lock = threading.Lock()
    completed = 0
    start_time = time.monotonic()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_download_one, row, out_dir): row for row in rows}
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception:
                # _download_one shouldn't raise (it catches Exception itself),
                # but this loop's counting/logging must never silently break
                # again if something unexpected still slips through
                logger.exception(f"unexpected error downloading {futures[future].get('url')}")
                result = "failed"
            with lock:
                counts[result] += 1
                completed += 1
                if completed % progress_every == 0 or completed == total:
                    elapsed = time.monotonic() - start_time
                    rate = elapsed / completed
                    remaining = rate * (total - completed)
                    logger.info(
                        f"[{completed}/{total}] downloaded={counts['downloaded']} "
                        f"skipped={counts['skipped']} failed={counts['failed']} -- "
                        f"elapsed {elapsed / 60:.1f}min, ~{remaining / 60:.1f}min remaining"
                    )

    logger.info(f"done: {counts}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("manifest", help="build a (site, timestamp, url) manifest without downloading")
    m.add_argument("--sites", nargs="+", default=["big_bear", "mauna_loa"], choices=list(SITE_CODES))
    m.add_argument("--date-ranges", nargs="+", default=["2019-06-01:2020-06-01", "2023-06-01:2024-06-01"],
                    help="one or more START:END dates (YYYY-MM-DD), e.g. a quiet-sun and an active-sun window")
    m.add_argument("--day-stride", type=int, default=3, help="sample every Nth day within each range")
    m.add_argument("--tolerance-minutes", type=int, default=20)
    m.add_argument("--per-day-site", type=int, default=2,
                    help="max frames to keep per (site, day), spread across the day (pass 0 to disable "
                         "and keep the old ~1/hour cadence -- see thin_per_day)")
    m.add_argument("--max-images", type=int, default=10000)
    m.add_argument("--out", type=Path, default=Path("data/raw/gong_pretrain/manifest.csv"))
    m.add_argument("--workers", type=int, default=4, help="concurrent listing requests")
    m.add_argument("--log-file", type=Path, default=None)

    d = sub.add_parser("download", help="download every file listed in a manifest CSV")
    d.add_argument("--manifest", type=Path, required=True)
    d.add_argument("--out-dir", type=Path, default=Path("data/raw/gong_pretrain"))
    d.add_argument("--workers", type=int, default=4, help="concurrent downloads")
    d.add_argument("--log-file", type=Path, default=None)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_file)
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
            checkpoint_path=args.out,
            workers=args.workers,
            per_day_site=args.per_day_site or None,
        )
        write_manifest_csv(manifest, args.out)
        logger.info(f"wrote {len(manifest)} rows to {args.out}")
    elif args.cmd == "download":
        download_manifest(args.manifest, args.out_dir, workers=args.workers)


if __name__ == "__main__":
    main()
