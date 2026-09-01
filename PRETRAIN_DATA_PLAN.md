# Pretraining Data Preparation Plan

## Goal

Build a curated GONG H-Alpha image corpus for self-supervised pretraining (e.g. MAE /
DINO on a ViT encoder), to later transfer into the MAGFiLO filament segmentation
model trained on `jp-mvp1`. This document covers **step 1 only: data preparation**,
before any pretraining code is written.

## Scope (as specified)

- **Sites**: Big Bear (BBSO) + Mauna Loa (MLSO) — two of GONG's six stations, chosen
  for day/night complementary coverage.
- **Volume**: ~5-10K images total, not the full archive.
- **Cadence**: ~1 frame/hour.
- **Time span**: multiple years, deliberately sampling both quiet-sun and active-sun
  periods of solar cycle 25 (started ~Dec 2019; active/max period ~2023-2025) so the
  corpus isn't skewed toward one filament-density regime.

## Open questions to resolve before pulling data at scale

- **Exact GONG H-Alpha archive access path and file format.** Confirm the archive URL
  structure/API (NSO GONG public archive) and whether files are FITS or another
  format, by pulling one sample file per site first.
- **FITS header keys for disk center/radius.** GONG headers are expected to carry
  something like `CRPIX1/2` + a solar-radius key (e.g. `SOLAR_R` / `RSUN_OBS` /
  a GONG-specific key) — confirm actual key names on a real sample file before
  writing the crop logic; don't assume the key name until verified.
- **Target resolution** for the ViT patch size chosen elsewhere in the pretraining
  plan (not yet decided here) — 512 is a reasonable default (patch=16 → 32×32
  tokens; also a clean 2048/4 downsample of MAGFiLM's native 2048×2048, so eventual
  fine-tuning crops/tiling map back cleanly). Revisit once the encoder architecture
  is chosen.

## Pipeline steps

### 1. Manifest before download

Build a list of `(site, timestamp)` pairs to pull — one per site per ~hour — across
the chosen multi-year, quiet+active date ranges, **before** downloading anything.
This lets the ~5-10K image budget be planned and reviewed up front rather than
discovered after a bulk pull.

### 2. Download

Pull only the manifested FITS files (resumable, rate-limited, logged). Store raw
files under `data/raw/gong_pretrain/<site>/<year>/` — gitignored, same convention as
the existing `data/raw/`.

### 3. Disk detection + centering/cropping

Read disk center/radius directly from FITS header metadata and crop to a
disk-centered square with a small margin. Do **not** re-derive center/radius via CV
(Hough circle fit, thresholding, etc.) — the header metadata is authoritative and
cheaper. If a frame's header metadata is missing or malformed, drop the frame and log
it to a rejected-frames manifest rather than falling back to a CV heuristic.

### 4. Limb darkening correction

Apply standard radial intensity normalization (I(r)/I(0) profile) using the center/
radius from step 3. Without this, the network spends encoder capacity on the radial
brightness gradient instead of filament-relevant texture. Validate by checking that
mean intensity vs. radius is flat after correction on a sample of frames.

### 5. Resize to fixed resolution

Resize the (already-square) disk-cropped image to the target resolution decided
above (default 512, revisit against the eventual ViT patch size). No padding should
be needed since the crop is already square.

### 6. Dataset-level normalization

Compute mean/std over the final preprocessed corpus itself (not ImageNet stats),
once the full corpus is assembled. Store the result (e.g.
`configs/pretrain_data_stats.json`) for the dataloader to consume later.

### 7. Dedup near-identical frames

At ~1 frame/hour the corpus is already coarse, so dedup here is mainly about
data-quality duplicates — e.g. a stuck camera repeating an identical frame during an
outage — rather than natural minute-to-minute similarity. Use a perceptual hash
(pHash/dHash) per frame; drop frames within a similarity threshold of an
already-kept frame, keeping the manifest's reason for every drop.

### 8. QA pass

Before committing to a full run: visualize a random sample of frames at each pipeline
stage (raw → disk-cropped → limb-corrected → resized) to catch a wrong header key or
a broken crop early, when it's cheap to fix.

## Deliverables / layout

```
scripts/pretrain_data/
  build_manifest.py         # (site, timestamp) sampling plan, before any download
  download_gong.py          # pulls manifested FITS, resumable/rate-limited
  preprocess_gong.py        # disk-crop (header-driven) -> limb-darkening -> resize
  dedup.py                  # perceptual-hash near-duplicate removal
  compute_dataset_stats.py  # corpus-level mean/std
configs/
  pretrain_data.yaml        # sites, date ranges, cadence, target resolution, paths
data/
  raw/gong_pretrain/        # gitignored -- raw FITS per site/year
  processed/gong_pretrain/  # gitignored -- final images + manifest.csv
```

## Definition of done for this phase

- [ ] Manifest of ~5-10K `(site, timestamp)` pairs spanning quiet+active solar-cycle
      date ranges, reviewed before any download.
- [ ] All manifested FITS downloaded; disk-cropped via header metadata (not CV);
      limb-darkening corrected; resized to the target resolution.
- [ ] Corpus-level mean/std computed and saved.
- [ ] Near-duplicate/stuck-camera frames removed via hash-based dedup, with reasons
      logged.
- [ ] `manifest.csv` (path, site, timestamp, center/radius used, hash, kept/dropped
      reason) plus a handful of per-stage visualizations checked for correctness.

Pretraining code itself (model, SSL objective, training loop) is out of scope for
this document — this covers data preparation only.
