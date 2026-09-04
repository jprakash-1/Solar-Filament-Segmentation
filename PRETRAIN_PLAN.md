# Hα Filament Segmentation — Self-Supervised Backbone Pretraining Plan

Implementation plan for MAE pretraining on public GONG/BBSO Hα imagery, targeting
Kaggle's 2×T4 multi-GPU notebooks under session/quota constraints. Supersedes the
earlier data-only outline (`PRETRAIN_DATA_PLAN.md`) — stages 1-2 below cover the
same ground, folded into the full pipeline through fine-tuning handoff.

---

## 0. Constraints driving this design

- Kaggle GPU notebooks: 2×T4 (16GB each), ~9-12h session wall-clock cap, weekly
  quota limit.
- No guaranteed persistence of notebook output between sessions unless explicitly
  versioned as a **Kaggle Dataset**.
- Internet access is available but should be used once for data acquisition, not
  repeatedly per training session.

**Design consequence:** everything downstream of data acquisition assumes
checkpoint-and-resume across multiple sessions, with data and checkpoints persisted
as Kaggle Datasets. Pretraining must also *actively* stop itself with margin before
the hard session cap, not rely on a human noticing in time — see 4.3.

---

## 1. Pipeline stages (run as separate Kaggle notebooks/sessions)

1. **Data acquisition** (one-time) → raw FITS dump → Kaggle Dataset `halpha-raw`
2. **Preprocessing** (one-time) → limb-corrected 8-bit JPEG + manifest CSV → Kaggle Dataset `halpha-preprocessed`
3. **MAE pretraining** (multi-session, resumable) → mounts `halpha-preprocessed`, checkpoints to a `halpha-mae-ckpt` dataset each session
4. **Segmentation fine-tuning** (separate notebook, on `jp-mvp1`-style pipeline) → mounts the final `halpha-mae-ckpt`, fine-tunes on labeled MAGFiLO data

---

## 2. Stage 1 — Data acquisition

**Target:** ~5,000-10,000 images across 2 sites (Big Bear + Mauna Loa, for
day/night complementary coverage), ~1 frame/hour, spanning multiple years so the
corpus covers both quiet-sun and active-sun phases of solar cycle 25 (started
~Dec 2019; recent active/max period ~2023-2025) rather than skewing toward one
filament-density regime.

**Verified directly against the live archive** (superseding the "browse it
manually first" open item from the previous revision of this plan):

- Base URL: `https://gong2.nso.edu/HA/haf/<YYYYMM>/<YYYYMMDD>/` — a plain
  Apache-style directory listing, confirmed with a real `requests.get()`, no
  authentication needed.
- Filenames: `<YYYYMMDDHHMMSS><site-letter>h.fits.fz`, e.g.
  `20220318000050Bh.fits.fz`. Single-letter site codes (confirmed from a real
  listing, **not** the two-letter codes used by the archive's separate
  query-form UI): `B`=Big Bear, `M`=Mauna Loa, `L`=Learmonth, `U`=Udaipur,
  `T`=Teide.
- The listing page also embeds a zero-font-size decoy link (a bot-trap) —
  `scripts/pretrain_data/gong_halpha.py` filters strictly to `.fits.fz` hrefs
  matching the expected filename shape, which already excludes it; don't loosen
  that filter to something broader.
- Per-site cadence within the archive is roughly ~1/minute during that site's
  local daytime — nothing near-continuous to worry about, and nothing at all
  outside daylight hours (these are ground telescopes; a site only sees the Sun
  during its own local day). Big Bear and Mauna Loa's daylight windows overlap
  only partially, which is the actual mechanism behind "day/night complementary
  coverage."

Implementation: `scripts/pretrain_data/gong_halpha.py`, run in two steps (manifest
first, review it, then download):

```bash
python scripts/pretrain_data/gong_halpha.py manifest \
    --sites big_bear mauna_loa \
    --date-ranges 2019-06-01:2020-06-01 2023-06-01:2024-06-01 \
    --day-stride 3 --max-images 10000 --workers 6 \
    --out data/raw/gong_pretrain/manifest.csv \
    --log-file data/raw/gong_pretrain/manifest_build.log

python scripts/pretrain_data/gong_halpha.py download \
    --manifest data/raw/gong_pretrain/manifest.csv \
    --out-dir data/raw/gong_pretrain --workers 6
```

`--day-stride 3` (sample every 3rd day) is the default because 2 sites' combined
daylight coverage is ~15-20 frames/day, and continuous daily coverage across two
full-year windows would produce ~20-25K frames — well over the 5-10K target.
Tune it once the manifest's row count comes back (rerun `manifest` with a
different stride until the count lands in range; it's cheap, no download yet).
Downloading is resume-safe (skips files already on disk) so a killed Kaggle
session can just rerun the same command.

**Concurrency and logging**: both `manifest` and `download` are multi-threaded
(`--workers`, default 4) — this is I/O-bound (network requests), not CPU work,
so threads rather than processes. Each worker thread gets its own
`requests.Session` (thread-local) rather than sharing one. This mattered in
practice: a first real run of the full date range (sequential, before
threading) sat for nearly an hour with no visible progress, because the only
progress signal was the final row count printed at completion — there was no
way to tell whether it was almost done or stuck. `--log-file` now streams a
running `[completed/total] N frames so far -- elapsed Xmin, ~Ymin remaining`
line (also written to the manifest CSV path as a periodic checkpoint during
`manifest`, so a killed run still leaves a real partial result instead of
nothing). Individual per-request latency to this archive can be surprisingly
high (~15-20s observed on one run, vs. sub-second on earlier small manual
tests) — not a bug in this script, just the live archive's actual response
time varying; threading absorbs that by overlapping requests rather than
paying the full latency serially.

Once downloaded: in the Kaggle UI, **New Dataset → upload the output directory →
version it** as `halpha-raw`. This is the only stage that needs internet
access — every later stage mounts a Kaggle Dataset instead.

**At larger scale (6 sites, ~16 years), keeping the full raw corpus on disk
stops being practical** — raw FITS would be ~100GB+ vs. ~20GB of JPEG output.
`scripts/pretrain_data/download_and_convert.py` combines Stage 1+2 into one
per-file pipeline instead: download → convert to JPEG immediately → delete the
raw file, so raw data never accumulates beyond whatever's actively in flight.
It reuses `gong_halpha`'s download logic and `preprocess_gong`'s conversion
logic rather than duplicating either. The separate `download` + `preprocess`
stages above still work standalone (already used for the first 3,033-image
corpus); this combined script is for downloading additional data at a scale
where the two-phase approach would need too much scratch disk. Resume-safe
(skips a row if its JPEG already exists) and incremental (merges with
whatever's already in `--out-dir`, so it can be run again as the source
manifest grows without losing earlier results).

---

## 3. Stage 2 — Preprocessing

**Deliberately minimal by request: FITS → JPEG, no other processing.** An
earlier revision of this script did limb-darkening correction, a per-image
percentile contrast stretch, and per-site dedup (all verified against real
data — see the git history of `scripts/pretrain_data/preprocess_gong.py` for
that work, including two real bugs it caught: a ring artifact from a
nearest-bin profile lookup, and a genuine per-site graininess difference
between Mauna Loa and Big Bear traced to the source data itself). All of that
was explicitly dropped to keep this stage to the simplest possible format
conversion. If any of it turns out to be needed later (e.g. Stage 3 wanting
corpus-level normalization stats, or the per-site quality gap mattering enough
to address), it's sitting in that history to bring back rather than
reinvent — not lost, just not run by default.

What's still verified and kept as free metadata (read, not applied to
pixels): GONG's own processing pipeline (the "H Alpha Laminator") re-registers
every "haf" (reduced) product to a **standardized geometry** — disk center at
pixel `(1024, 1024)`, radius `900` px, in every `2048×2048` frame, confirmed
identical across both sites' samples via the
`FNDLMBXC`/`FNDLMBYC`/`FNDLMBMA`/`FNDLMBMI` header keys. `CRPIX1`/`CRPIX2` + a
flat `RADIUS=900` are kept as a fallback only, for the rare frame where the
limb-fit keys are missing or malformed.

**Output: full native `2048×2048`, 1-channel, 8-bit JPEG.** Verified this
matches MAGFiLO's own input convention exactly, not just approximately: a real
MAGFiLO training image is itself a `2048×2048`, single-channel (`L` mode)
*JPEG*, with the same GONG-style filename and the *same* disk framing — its
off-disk corners read exactly `0`, and its limb sits right at radius ~900
around `(1024, 1024)`, identical to the raw "haf" product's standardized
geometry above. Matching format as well as resolution means there's no
train/pretrain mismatch of any kind to bridge at Stage 4 fine-tuning time.
JPEG also matters for corpus size: a `2048×2048` float32 `.npy` (the earlier
revision's format) is ~16.8MB — a 10K-image corpus would be ~168GB, not
uploadable as a Kaggle Dataset. A real MAGFiLO JPEG at this resolution is
~700KB (~7GB for 10K images); this script's own 8-bit-min-max output measured
similarly (~500KB/frame on 20 real files).

**Consequence for Stage 3**: a ViT can't operate on the full `2048×2048` frame
directly — at patch=16 that's 128×128=16,384 patches (even after MAE's 75%
masking, ~4,096 visible tokens, ~28× the token count of a 384px/patch=16
setup, quadratic in attention cost). Stage 3's dataset still needs to sample a
disk-radius-bounded crop from these full-resolution frames per training
step — see section 4.1. Since there's no `dataset_stats.json` anymore either,
Stage 3's normalization needs a different source (compute it once itself, or
just divide by 255) — noted in the open items list.

Implementation: `scripts/pretrain_data/preprocess_gong.py`, run once over the
downloaded corpus:

```bash
python scripts/pretrain_data/preprocess_gong.py \
    --raw-dir data/raw/gong_pretrain --out-dir data/processed/gong_pretrain \
    --processes 8 --log-file data/processed/gong_pretrain/preprocess.log
```

One pass, per frame: read the FITS data → linearly rescale *that image's own*
min/max to `[0, 255]` (the only unavoidable step to fit 16-bit data into an
8-bit JPEG at all — not an adaptive/astronomy-specific stretch) → save as
JPEG (quality 95) → record `(cx, cy, r)` in `manifest.csv` as metadata. Ran
against 20 real downloaded frames (both sites) — all 20 converted, off-disk
background reads exactly 0, filament threads and plage visible, no artifacts.

**Multi-process, not multi-threaded**, unlike Stage 1: this is CPU-bound work
(FITS decompression, array rescaling, JPEG encoding), so a `ProcessPoolExecutor`
(`--processes`, default = CPU count) actually uses multiple cores, where
threads wouldn't (the GIL). Each worker returns only the lightweight manifest
row, not the pixel array, back to the main process. `--log-file` gives the
same running progress line as Stage 1.

Save output as Kaggle Dataset `halpha-preprocessed` — every future pretraining
session mounts this read-only, no re-downloading or re-processing.

**Post-hoc thinning (added after the 6-site/16-year corpus reached 40,174
images, well over the original 5-10K target):** consecutive-hour frames from
the same site turned out to be near-duplicates by construction — the Sun
only rotates ~0.5°/hour and filaments persist for hours-to-days. Measured
directly on the corpus (disk-cropped SSIM at native `(1024, 1024, 900)`
geometry): same-site consecutive-hour pairs average SSIM 0.91 (85% exceed
0.90), vs. 0.79 for random cross-date pairs and 0.84 for same-site frames
~3 days apart (the existing `--day-stride 3`). The corpus's real diversity
axis is across dates/sites, not within a day, so `scripts/pretrain_data/thin_manifest.py`
keeps at most 2 frames per (site, date) — the ones nearest the 25th/75th
percentile time, spread across the middle of the day rather than the
possibly-noisier limb-grazing first/last frames — producing
`manifest_thinned.csv` (8,557 rows, 21% of the full corpus, back in the
original target range). This is structured temporal subsampling targeted at
the known redundancy source, not generic perceptual-hash/pairwise dedup, and
is non-destructive: it only writes a new manifest, the full 40,174-image
corpus stays on disk untouched. Stage 3 should point `MANIFEST` at
`manifest_thinned.csv` rather than `manifest.csv`.

(Note found while building this: `data/processed/gong_pretrain/manifest.csv`
itself only covers ~16K of the 40,174 on-disk JPEGs — some prior run(s) added
files without updating it. `thin_manifest.py` works around this by using the
JPEG directory listing, not the manifest, as the source of truth for which
files exist, falling back to the standardized `(1024, 1024, 900)` geometry
for files the manifest doesn't cover. Worth a proper fix in the Stage 1/2
scripts' manifest-merge logic if `manifest.csv` needs to be authoritative for
something else later.)

---

## 4. Stage 3 — MAE pretraining (DDP, 2×T4, resumable)

### 4.1 Dataset / augmentation

Stage 2's files are the full native `2048×2048`, 8-bit JPEG frame (see section
3), so this dataset's job includes sampling the actual ViT-sized training crop
and normalizing with the corpus stats, not just loading a pre-sized image:

```python
# dataset.py
import numpy as np, pandas as pd, torch, random
from PIL import Image
from torch.utils.data import Dataset

class HalphaMAEDataset(Dataset):
    def __init__(self, manifest_csv, crop_size=384, train=True):
        manifest = pd.read_csv(manifest_csv)
        self.paths = manifest["path"].tolist()
        self.cx = manifest["cx"].tolist()
        self.cy = manifest["cy"].tolist()
        self.r = manifest["r"].tolist()  # disk radius, in the *native* 2048x2048
        # frame -- Stage 2 doesn't resize, so no rescaling needed here
        self.crop_size = crop_size
        self.train = train

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = np.array(Image.open(self.paths[idx]), dtype=np.float32) / 255.0
        # Stage 2 no longer computes corpus-level mean/std (it does nothing but
        # convert FITS -> JPEG now), so there's no dataset_stats.json to load
        # here -- dividing by 255 is a placeholder until Stage 3 either computes
        # its own corpus stats or this normalization gets revisited.
        img = self._disk_bounded_crop(img, self.cx[idx], self.cy[idx], self.r[idx])
        if self.train:
            img = self._augment(img)
        return torch.from_numpy(img).unsqueeze(0)  # [1, crop_size, crop_size]

    def _disk_bounded_crop(self, img, cx, cy, r):
        # Sample the crop's *center* uniformly within the solar disk (not an
        # unbounded crop over the full 2048x2048 frame) -- an unbounded crop
        # can land mostly off-disk (pure background), which would let the MAE
        # pretext task shortcut on "is this patch background" rather than
        # learning filament-relevant texture. The crop can still extend past
        # the limb near its edges (real limb-adjacent frames are valid too),
        # just not be centered on pure background.
        half = self.crop_size / 2
        for _ in range(10):  # a handful of rejection-sample attempts is plenty
            angle = random.uniform(0, 2 * np.pi)
            radius = r * (random.random() ** 0.5)  # uniform over disk *area*, not radius
            cx_s = cx + radius * np.cos(angle)
            cy_s = cy + radius * np.sin(angle)
            x0, y0 = int(cx_s - half), int(cy_s - half)
            if 0 <= x0 and x0 + self.crop_size <= img.shape[1] and 0 <= y0 and y0 + self.crop_size <= img.shape[0]:
                return img[y0:y0 + self.crop_size, x0:x0 + self.crop_size]
        # fallback: disk-centered crop, always in-bounds given r=900, crop=384
        cx_i, cy_i = int(cx), int(cy)
        return img[cy_i - int(half):cy_i + int(half), cx_i - int(half):cx_i + int(half)]

    def _augment(self, img):
        # rotation is valid here -- no canonical "up" on the Sun
        k = random.randint(0, 3)
        img = np.rot90(img, k).copy()
        # intensity/contrast perturbation as the grayscale stand-in for color jitter
        if random.random() < 0.5:
            gamma = random.uniform(0.8, 1.2)
            img = np.sign(img) * (np.abs(img) ** gamma)
        return img
```

Not yet run against real data (Stage 3 isn't implemented as actual code yet,
unlike Stages 1-2) — verify `_disk_bounded_crop` against a handful of real
`.npy` frames once Stage 3 is built, the same way Stage 2's limb-darkening
correction was checked here before being trusted.

### 4.2 Model (HuggingFace ViTMAE, 1-channel)

```python
# model.py
from transformers import ViTMAEConfig, ViTMAEForPreTraining

def build_vitmae_1ch(img_size=384, patch_size=16, mask_ratio=0.75, model_scale="small"):
    scale_cfg = {
        "small": dict(hidden_size=384, num_hidden_layers=12, num_attention_heads=6),
        "base":  dict(hidden_size=768, num_hidden_layers=12, num_attention_heads=12),
    }[model_scale]

    config = ViTMAEConfig(
        image_size=img_size,
        patch_size=patch_size,
        num_channels=1,          # grayscale -- key change from the pretrained-3ch default
        mask_ratio=mask_ratio,
        decoder_hidden_size=256,
        decoder_num_hidden_layers=4,
        decoder_num_attention_heads=8,
        **scale_cfg,
    )
    return ViTMAEForPreTraining(config)
```

Decide ViT-Small vs. ViT-Base from the **first session's** measured images/sec at
each scale on 2×T4, not up front — see the open items list.

### 4.3 DDP training loop with checkpoint-resume

Launched via `torchrun`, not `mp.spawn` from inside the script — `jp-mvp1`'s
existing DDP training already had to debug this exact environment (Kaggle 2×T4)
and settled on `torchrun` because it sets `RANK`/`LOCAL_RANK`/`WORLD_SIZE`
automatically (a hand-rolled `mp.spawn` setup would still need `MASTER_ADDR`/
`MASTER_PORT` wired up manually and would be rediscovering solved problems). The
two NCCL fixes below (`device_id` passed explicitly, `NCCL_P2P_DISABLE=1`) are
carried over directly from that debugging, not re-derived here — see
`jp-mvp1:src/distributed.py` and its README "Known gotchas" section for the
original writeup (NCCL watchdog timeout + one GPU idling at 0%).

```python
# distributed.py -- identical helper to jp-mvp1:src/distributed.py
from __future__ import annotations
import os
import torch
import torch.distributed as dist

def setup_distributed() -> tuple[int, int, int]:
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl", rank=rank, world_size=world_size,
        device_id=torch.device(f"cuda:{local_rank}"),  # avoids NCCL "guessing device
        # ID" hang / one-GPU-idle-at-0% observed on Kaggle T4 x2
    )
    return local_rank, rank, world_size

def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()
```

```python
# train_ddp.py
import os, time, csv
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler

from dataset import HalphaMAEDataset
from model import build_vitmae_1ch
from distributed import setup_distributed, cleanup_distributed

MANIFEST = "/kaggle/input/halpha-preprocessed/manifest_thinned.csv"
CKPT_DIR = "/kaggle/working/checkpoints"
RESUME_CKPT = "/kaggle/input/halpha-mae-ckpt/latest.pt"  # None on the very first session

TOTAL_EPOCHS = 200          # planned across ALL sessions, not per-session
BASE_LR = 1.5e-4
PER_GPU_BATCH = 64
WARMUP_EPOCHS = 20
SESSION_BUDGET_SECONDS = 8 * 3600  # stay well under Kaggle's ~9-12h cap; leaves
                                    # margin for the final checkpoint write + upload

def cosine_lr(epoch, total_epochs, warmup_epochs, base_lr):
    if epoch < warmup_epochs:
        return base_lr * epoch / warmup_epochs
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    return 0.5 * base_lr * (1 + torch.cos(torch.tensor(progress * 3.14159265)))

def main():
    local_rank, rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    session_start = time.time()

    model = build_vitmae_1ch().to(device)
    model = DDP(model, device_ids=[local_rank])
    optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=0.05)
    scaler = GradScaler()

    start_epoch = 0
    if RESUME_CKPT and os.path.exists(RESUME_CKPT):
        ckpt = torch.load(RESUME_CKPT, map_location=device)
        model.module.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        if rank == 0:
            print(f"Resumed from epoch {start_epoch}")

    dataset = HalphaMAEDataset(MANIFEST, train=True)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    loader = DataLoader(dataset, batch_size=PER_GPU_BATCH, sampler=sampler,
                         num_workers=4, pin_memory=True, drop_last=True)

    stop_early = torch.zeros(1, device=device)  # all-reduced so every rank agrees
    # on when to stop -- checking wall-clock independently per rank could let one
    # rank exit while another keeps iterating, hanging DDP's collective ops

    for epoch in range(start_epoch, TOTAL_EPOCHS):
        sampler.set_epoch(epoch)
        lr = cosine_lr(epoch, TOTAL_EPOCHS, WARMUP_EPOCHS, BASE_LR)
        for g in optimizer.param_groups:
            g["lr"] = lr

        model.train()
        epoch_loss = 0.0
        for imgs in loader:
            imgs = imgs.to(device, non_blocking=True)
            optimizer.zero_grad()
            with autocast():
                outputs = model(pixel_values=imgs)
                loss = outputs.loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()

        if rank == 0:
            avg_loss = epoch_loss / len(loader)
            print(f"epoch {epoch} | loss {avg_loss:.4f} | lr {lr:.2e}")
            os.makedirs(CKPT_DIR, exist_ok=True)
            torch.save({
                "model": model.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "loss": avg_loss,
            }, f"{CKPT_DIR}/latest.pt")
            with open(f"{CKPT_DIR}/loss_log.csv", "a") as f:
                csv.writer(f).writerow([epoch, avg_loss, lr])

            if time.time() - session_start > SESSION_BUDGET_SECONDS:
                stop_early += 1

        # broadcast rank 0's stop decision to every rank so all exit the epoch
        # loop together -- rank 0 is the only one that knows wall-clock elapsed
        # relative to the checkpoint write it just did
        torch.distributed.broadcast(stop_early, src=0)
        if stop_early.item() > 0:
            if rank == 0:
                print(f"Session budget reached at epoch {epoch}, stopping cleanly")
            break

    cleanup_distributed()

if __name__ == "__main__":
    main()
```

Launch cell:

```bash
!NCCL_P2P_DISABLE=1 torchrun --nproc_per_node=$(python -c "import torch; print(max(1, torch.cuda.device_count()))") train_ddp.py
```

### 4.4 Session checklist (repeat each pretraining session)

1. Mount `halpha-preprocessed` (data) and `halpha-mae-ckpt` (latest checkpoint, if
   any) as notebook inputs.
2. Set `RESUME_CKPT` to the mounted checkpoint path (or leave it pointing at a
   nonexistent path for the very first session — the script already checks
   `os.path.exists`).
3. Run the launch cell above; monitor `loss_log.csv` growth.
4. The loop stops itself once `SESSION_BUDGET_SECONDS` elapses (checkpointing
   every epoch along the way, so an unplanned kill loses at most one epoch) — no
   need to watch the clock and stop it manually.
5. **New Dataset version** from `/kaggle/working/checkpoints` → this becomes next
   session's `halpha-mae-ckpt` input.
6. Periodically (every few sessions): visually inspect masked-reconstruction
   outputs on held-out frames — check whether filament-like thin structures are
   reconstructed or blurred away. This is the earliest quality signal, well before
   any downstream segmentation metric is available.

### 4.5 Held-out split for pretraining itself (needed before trusting the loss curve)

`HalphaMAEDataset`/`train_ddp.py` above train on the *entire* thinned manifest with
no held-out split — there is currently no way to tell whether `loss_log.csv`
reflects real generalization or Stage 3 memorizing the corpus (~8.5K rows in
`manifest_thinned.csv` as of the post-hoc thinning in section 3, though the raw
on-disk JPEG count has since grown past that) over up to 200 epochs. Before
trusting "the loss plateaued" as a stopping signal in section 4.4:

- Hold out ~3-5% of `manifest_thinned.csv` rows as a fixed validation set, grouped
  by **date** (not row) so a held-out day's frames aren't near-duplicates of a
  training day's — the same SSIM redundancy the thinning step itself was built to
  address (section 3) would otherwise leak straight across this split too.
- Log val MAE loss every few epochs (a rank-0-only, no-grad pass — doesn't need DDP).
- Train loss still dropping while val loss flattens or rises is overfitting the
  pretext task specifically. Fix: smaller model, higher mask ratio, or more corpus
  diversity — not more epochs.
- This risk scales with model size against a fixed corpus: ViT-Base (~86M params)
  is more likely to overfit an 8.5-50K-image single-domain corpus than ViT-Small
  (~22M) is. Start with **Small** regardless of the throughput-driven pick in 4.2's
  open item, and only move to Base if val loss shows no overfitting signal at Small.

---

## 5. Stage 4 — Segmentation fine-tuning

**Scale reality check driving every choice below:** the labeled side of this
problem is `MAGFiLO_1.0_Kaggle_2026/train` — verified directly from the COCO JSON
(`MAGFiLO_1.0_Annotations_kaggle2026_train.json`): **707 unique underlying images**
(1,154 annotated `image_id`s, because the same physical frame can carry multiple
independent annotator passes — `kaggle.md`'s "1,039" figure was an earlier EDA
pass; only 20 images are downloaded locally right now as a sample either way),
**8,199 instance annotations**, median **7 instances/image** (up to 26), median
instance area **1,228 px²** (p10 409 px²) against the full 2048×2048 frame, median
bbox aspect ratio **1.66** (p90 3.2, max 12.8 — elongated, not extreme). Grouping
for any split must be by `file_name`, never `image_id` (`jp-mvp1:src/dataset.py`'s
`group_split` already does this). That label count is roughly **2 orders of
magnitude smaller** than even the thinned pretraining corpus, and the instance
geometry (small, elongated — *not* densely adjacent, see §5.1's corrected
measurement) is what drives the head/architecture choice below, not just the
label count.

### 5.1 Segmentation model / head: what actually separates instances

**This section was revised after review caught two errors in the original
version below — kept visible rather than silently rewritten, because the
correction is more instructive than the clean version would be.** Original
claim: filament geometry is "small, elongated, *dense*," therefore instance
*separation* (specifically: clustering nearby/touching instances apart) is the
bottleneck, therefore a learned separation mechanism (embedding clustering)
should be preferred. Two things were wrong with that:

1. **"Don't crop to a box" and "cluster in embedding space" got conflated.**
   The box-cropping complaint actually decomposes into two independent
   failures: (a) an axis-aligned box around a thin diagonal curve is mostly
   background, so RoIAlign hands the mask head a bad crop — a *fill-ratio*
   problem, true regardless of how close other instances are; (b) box-IoU/NMS
   suppression when instances are near each other — an *adjacency* problem.
   Embedding clustering fixes both, but so does anything that predicts masks
   over the full feature map without cropping at all (CondInst, Mask2Former,
   Panoptic-DeepLab's center+offset scheme). The geometry argument licenses
   "don't crop to a box," not "cluster in embedding space" specifically — those
   are different commitments with different implementation costs, and the
   original write-up jumped from the first to the second without ruling out
   the middle ground.

2. **"Dense" was asserted, not measured — and the measurement contradicts it.**
   Checked directly against `MAGFiLO_1.0_Annotations_kaggle2026_train.json`:
   for every instance in the 1,051 multi-instance images, distance from its
   mask to the nearest *other* instance's mask. **69% of instances have no
   other instance within 60px at all**; only **0.3% touch or overlap (gap
   ≤2px)**, **2.7% are within 10px**, **5.5% within 20px** — median gap for
   instances that do have a neighbor within 60px is 46.7px. That is not a
   dense, mutually-adjacent field of objects (the EmbedSeg-style regime this
   was implicitly modeled on — hundreds of touching nuclei — where clustering
   resolves genuine ambiguity). It's mostly-isolated objects. A correct
   semantic mask fed through plain connected components would already separate
   the large majority of instances correctly, because they're mostly just not
   touching. Whatever drives `kaggle.md`'s "crowded images under-segment"
   finding, it is very unlikely to be geometric adjacency between distinct
   filaments at a rate that would justify embedding clustering as the fix —
   self-fragmentation of one winding filament (its probability map dipping
   below threshold somewhere along its length) is a more plausible candidate,
   and it's a *different* failure mode that clustering doesn't obviously
   address at all.

**What still holds:** filaments are thin and non-compact, so cropping to a box
degrades the mask head. That's real and doesn't depend on the adjacency
measurement above. **What doesn't hold:** the specific preference for embedding
clustering over other box-free options, which was leaning on "dense" and on
a citation (the in-domain solar filament paper using CondInst) that actually
argues for the *opposite, cheaper* fix — CondInst keeps a box-based FCOS
detection front end and only avoids RoIAlign for the mask itself, i.e. it
fixes failure (a) above while keeping detection, which is evidence for "fix
the mask head," not "add clustering."

**The right evaluation frame is PQ's own decomposition, not shape adjectives.**
PQ = SQ (segmentation quality: mean IoU over *matched* instances) × RQ
(recognition quality: precision/recall of the Hungarian match at IoU>0.5).
Different failure modes hit different terms, so candidates should be compared
by which term they actually move:

- **Missed detection** (a faint/low-contrast filament never predicted at all)
  → pure RQ loss (a false negative). Given the adjacency finding, this is
  currently the more plausible dominant error than merges.
- **Self-fragmentation / over-segmentation** (one filament predicted as several
  pieces) → RQ loss, double-counted (a spurious extra prediction *and* a
  degraded match on the real instance).
- **Merge of two distinct instances** → RQ loss (one match lost) — but per the
  adjacency measurement, the *opportunity* for this specific error is rare
  (≤5.5% of instances even have a neighbor within 20px), so this is unlikely to
  be the dominant term regardless of separation mechanism.
- **Boundary/mask-quality error on an otherwise-correct instance** → SQ loss
  directly, and *discontinuously* becomes an RQ loss too if it drags IoU below
  the 0.5 matching threshold.

This is a hypothesis about where the error mass currently sits, not yet a
finding — it needs `kaggle.md`'s own Month-2 Step 3 per-image error-analysis
tooling run against the current `jp-mvp1` baseline's real predictions,
classified by failure type (missed / fragmented / merged / poor-boundary) and
split into SQ vs. RQ contribution.

**Candidates, now framed as "which PQ term does each one actually move,"
none preferred yet:**

- **(A) Semantic mask + watershed (current baseline).** Already working;
  postprocessing-only tuning already produced an 8x local PQ jump
  (`kaggle.md`), proof this axis has headroom on top of this exact model
  before any architecture change. No shape prior in the split step, so
  self-fragmentation (the more plausible failure mode per above) isn't
  addressed by it — but neither is it addressed by embedding clustering,
  which targets adjacency-driven merges instead.
- **(B) Semantic mask + pixel-embedding instance head.** Fixes the box-crop
  fill-ratio problem (still valid) and would fix adjacency-driven merges — but
  the adjacency measurement suggests that specific benefit applies to a small
  minority of instances here. Adds a new loss and a new inference-time
  clustering-bandwidth hyperparameter — real implementation cost for a
  mechanism that may not target the dominant error.
- **(C) Mask R-CNN + ResNet-FPN.** Median instance area (1,228px²) sits below
  COCO's own "small object" threshold (1,024px²) at native resolution, further
  shrunk by the standard ~800px-short-side resize; fixed 28×28 RoI mask grid
  discards fine structure for AR-up-to-12.8 shapes. Still a weak fit
  regardless of the adjacency correction — this conclusion doesn't change.
- **(D) CondInst-style dynamic-convolution instance segmentation.** Keeps a
  box-based (FCOS) detection front end for instance identification but
  generates each mask via per-instance dynamic convolutions over the *full*
  feature map — no RoIAlign crop. This is the architecture the in-domain solar
  filament paper (§ "directly in-domain" evidence, `learning.md`) actually
  used, successfully, on this exact modality. Fixes the fill-ratio problem
  directly; a box-based front end could still under-count faint filaments if
  its objectness/centerness scoring is miscalibrated for low-contrast
  targets — a detection-recall risk, distinct from a separation risk.
- **(E) Mask2Former-style mask-classification transformer.** Previously
  shelved as "too data-hungry for 707 labels" — that was asserted, not
  measured, and asserted inconsistently: the parent plan is a whole
  self-supervised pretraining program built to solve label scarcity for the
  embedding path's encoder, and COCO/ADE-pretrained Mask2Former checkpoints
  exist off the shelf, requiring no domain SSL investment to try at all. It's
  also the architecture panoptic segmentation's own leaderboards actually
  converged on. Given the adjacency finding, its query-based grouping isn't
  even solving a hard problem in this data — meaning if it works, it's likely
  via detection quality (RQ, recall on faint filaments) and mask quality (SQ),
  not because grouping was hard. **Reopened as a live candidate, not shelved.**

**No preference stated here.** The next step is measurement, not another round
of architecture comparison from first principles: run the per-image
error-analysis pass (failure-type × SQ/RQ breakdown) on the current baseline's
real predictions. Concrete falsification criteria to apply once that exists:

- If merge-errors are a small fraction (rough working threshold: <10%) of
  total RQ loss → do not adopt embedding clustering for its core mechanism;
  its main selling point doesn't target the actual error source.
- If boundary IoU is the dominant SQ loss even on correctly-matched instances
  → prioritize a box-free *mask head* (CondInst or Mask2Former) over pure
  clustering, since that's a mask-quality problem, not a separation problem.
- If missed low-contrast filaments dominate RQ loss → prioritize detection
  recall (confidence threshold, hard-negative mining, or a strong pretrained
  detector/transformer) over any instance-separation mechanism at all —
  separation can't help with an instance that was never detected.

Target: run this measurement before committing further engineering to any of
(B)/(D)/(E) — this is a cheap, existing-tooling pass, not a new architecture.

### 5.2 Encoder: ResNet vs. ViT

**ResNet (CNN).**
- *Pros:* convolutional inductive bias (locality, translation equivariance) is a
  strong prior that directly compensates for a small labeled set (707 images);
  mature ImageNet-pretrained weights; already working with 1-channel input via a
  straightforward stem adaptation (`jp-mvp1`'s `smp.Unet` does this today);
  cheaper to train, faster iteration.
- *Cons:* lower ceiling than a well-pretrained ViT *if* the domain-specific SSL
  corpus (GONG Hα) carries transferable signal ImageNet genuinely lacks; weaker
  at modeling long-range context (e.g., relating distant parts of one curving
  filament) without deep stacking or dilation.

**ViT (Transformer), MAE-pretrained.**
- *Pros:* global self-attention naturally models long-range structure, which
  could matter for elongated, curving filaments; can absorb the large *unlabeled*
  domain corpus (8.5K-49K images) directly via MAE, a pretraining story ResNet
  doesn't have as cleanly.
- *Cons:* no built-in locality/equivariance prior — needs much more labeled data
  or very strong pretraining to compensate, and 707 labeled images is not much;
  materially higher overfitting/forgetting risk during fine-tuning (5.6); still
  unproven for this project specifically — Stage 3's MAE pretraining hasn't been
  run yet, so this option carries real unresolved technical risk (does the
  pretrain even converge well on this corpus?) that ResNet+ImageNet doesn't.

**Preference: ResNet (`resnet34`) as the primary/production path; ViT-MAE stays a
gated secondary track**, promoted only if the mandatory ablation (5.3) shows it
beating the ResNet baseline's *real* local PQ — not just beating a scratch or
ImageNet-init ViT, which would only prove pretraining helped some ViT, not that
ViT beats the simpler, safer CNN.

### 5.3 Pretraining strategy per encoder

**For ResNet:**

*Option 1 — ImageNet weights only (current `jp-mvp1` default).*
Pros: zero extra engineering/compute, already the working starting point, and
ImageNet features are already known to transfer reasonably to this grayscale
domain (the existing baseline runs on exactly this). Cons: no domain adaptation
to Hα-specific texture (limb darkening, plage, filament threading) that a
domain-specific SSL pass could target directly.

*Option 2 — Domain SSL on the GONG Hα corpus (BYOL/MoCo contrastive, or a
CNN-adapted masked-modeling method like SparK), starting from ImageNet weights.*
Pros: could close the same kind of domain gap MAE is meant to close for ViT,
giving a fair "does domain pretraining help at all" comparison across both
encoder families instead of only ViT getting that investment. Cons: a second,
CNN-specific SSL pipeline, distinct from Stage 3's ViTMAE code — real engineering
cost for a benefit that's unproven until measured.

**Preference: start with Option 1 for ResNet — it's free and already the working
baseline.** Only build Option 2 if the ViT-MAE ablation shows domain pretraining
is genuinely valuable for this corpus/task at all; don't build two speculative SSL
pipelines before either is proven once.

**For ViT:**

*Option 1 — MAE pretraining on the GONG Hα corpus (Stage 3, already planned).*
Pros: domain-adapted features, potentially large gain if ImageNet-style features
transfer poorly to full-disk Hα imagery; the corpus itself (8.5K-49K images) is
already a real, in-progress asset. Cons: unproven — Stage 3 isn't implemented/run
yet; real risk the pretrain doesn't pay off or overfits its own pretext task
(4.5); the most compute-expensive of the three (multi-session Kaggle DDP).

*Option 2 — ImageNet-pretrained ViT, channel-adapted to 1-channel input.*
Pros: cheap, immediate, tests whether ViT-as-architecture is viable at all before
committing to Stage 3's investment. Cons: 3-channel-to-1-channel patch-embed
adaptation is lossier for a ViT than a ResNet stem conv (no equivalent to `smp`'s
clean channel-averaging trick in standard ViT loaders); still doesn't close the
domain gap.

*Option 3 — Scratch ViT, no pretraining.*
Pros: simplest to implement, serves as the ablation's own negative control. Cons:
essentially guaranteed to underperform given ~707 labeled images and no inductive
bias — its only purpose is calibrating how much Options 1-2 actually contribute.

**Preference: run all three as the mandatory ablation already specified** — but
sequence it *after* the ResNet baseline (with whichever segmentation head §5.1's
error-analysis measurement actually selects — not necessarily the embedding
head) has a real local PQ number, since that's the number every ViT variant
actually has to beat to be worth adopting, not just an academic comparison
among ViT variants themselves.

### 5.4 Loss function

The current MVP1 target is a single binary semantic mask (union of instance
polygons; instance separation happens later via postprocessing — see
`jp-mvp1:src/dataset.py`'s `FilamentDataset` docstring), and filaments are thin,
elongated, and a small minority of pixels per frame. Plain BCE or plain Dice each
have known failure modes here (BCE: dominated by the easy background majority;
Dice alone: unstable gradients when a batch's positive mask is tiny or empty,
which happens here — MVP1 deliberately keeps filament-free frames as valid
negatives).

- **Primary: Tversky loss + BCE**, `loss = 0.5 * BCE + Tversky(alpha=0.3, beta=0.7)`.
  Tversky generalizes Dice with independent false-positive/false-negative weights;
  `alpha < beta` biases the loss toward recall, which matters because thin
  structures are the ones a model under-predicts first, and per `kaggle.md`'s own
  error-analysis finding, false negatives/positives here were shape-indistinguishable
  — meaning the model's discrimination, not obviously the loss's FP/FN balance, was
  the deeper issue; still, starting recall-biased is the safer default for thin
  positives and is cheap to re-tune once real error-analysis tooling (kaggle.md
  Month 2 Step 3) is in place.
- **Add a boundary-aware term once the plain Tversky+BCE baseline is running**:
  a distance-transform-weighted BCE, or clDice (centerline Dice — designed
  specifically for thin/tubular structures like vessels, directly applicable to
  filaments) as a secondary term. This is a `kaggle.md` Month-2-Step-5 item
  (architecture/loss changes justified by a specific finding), not a day-one
  requirement — don't add it before the plain baseline's error analysis says
  boundary localization specifically is the gap.
- Keep loss changes isolated per `kaggle.md`'s Step 4: don't change the loss and
  the architecture in the same run, or a PQ delta can't be attributed to either.

### 5.5 Optimizer, LR schedule, and layer-wise decay

Both branches share the same shape — **discriminative (layer-wise) LR decay**, a
short warmup, then cosine decay to 0, with the pretrained encoder kept an order of
magnitude "colder" than the fresh decoder head so the head's initially-random
gradients don't blow away pretrained features before it has learned anything
useful to backprop:

```python
def layerwise_lr(depth_from_input, num_layers, head_lr, decay=0.75):
    # depth_from_input=0 is the stem/first block; num_layers-1 is the block
    # nearest the decoder. Deeper (more task-specific) layers get less decay.
    return head_lr * (decay ** (num_layers - 1 - depth_from_input))
```

- **ResNet + U-Net branch**: `head_lr = 1e-3` (decoder + final layer), encoder LR
  decayed per ResNet stage (`decay ≈ 0.7-0.8`, 5 stages) down to roughly `1e-5` at
  the stem. AdamW, `weight_decay = 1e-4`. Warmup 3-5 epochs (linear), then cosine
  to 0. Total budget ~40-60 epochs, but gate on early stopping (below), not a
  fixed count — 1,039 images trains fast enough per epoch that overshooting the
  useful range costs wall-clock, not much else.
- **ViT-MAE branch**: `head_lr = 1e-3`, `decay ≈ 0.65-0.75` across the 12 ViT
  blocks (this range matches published MAE fine-tuning recipes, which is exactly
  the setting this schedule is adapted from), floor around `1e-5` at the earliest
  blocks. AdamW, `weight_decay = 0.05` (kept consistent with Stage 3's own
  pretraining wd). Warmup ~5-10% of total epochs. Transformers typically need a
  longer fine-tune schedule than a CNN to reach the same point on a small dataset,
  but weigh that against section 5.6's overfitting risk at this label count —
  don't just extend epochs to compensate without watching val loss.
- **Batch size**: moderate (16-32), not maximized — very small batches on a
  1,039-image dataset add useful gradient noise as a mild regularizer, very large
  batches don't help here and just reduce update count per epoch.
- Mixed precision (already used in Stage 3) + gradient clipping (`max_norm=1.0`)
  for stability — thin positive-heavy loss terms like Tversky can spike on a batch
  with unusually large filament coverage.

### 5.6 Avoiding overfitting and catastrophic forgetting at 707 labeled images

This is the actual risk this plan needs to manage, more than picking an
architecture:

- **Encoder freeze warmup**: freeze the pretrained encoder entirely for the first
  2-5 epochs, training only the fresh decoder/head. This lets the head reach a
  sane starting point before its initially-large gradients start flowing back
  into (and potentially wrecking) the pretrained encoder once unfrozen — cheap
  insurance against forgetting in the first few unstable epochs.
- **Layer-wise LR decay (5.5) is itself the primary forgetting guard** — it's not
  just an optimization nicety here, it's what keeps early/mid encoder layers
  close to their pretrained values while the decoder does most of the adapting.
- **Linear-probe baseline as a forgetting tripwire**: before running the full
  unfrozen fine-tune, train *only* a linear/shallow head on frozen pretrained
  features and record that PQ/Dice number. If full fine-tuning ends up *below*
  the frozen linear-probe result, that's a direct signal the unfrozen fine-tune is
  destructively overwriting useful pretrained features rather than improving on
  them — not just noise to shrug off.
- **Augmentation, kept label-preserving**: rotation (any multiple of 90°, plus
  free-angle) and flips are valid — there's no canonical "up" on the Sun, same
  reasoning Stage 3 already uses (section 4.1). Gamma/contrast jitter, mild
  Gaussian noise, coarse dropout patches. Avoid strong elastic/geometric warps
  applied independently of the mask — filaments are thin curvilinear structures
  and any augmentation must warp image and mask *jointly*, or a few pixels of
  misalignment corrupts a large fraction of a thin positive region.
- **Weight decay as an explicit anchor**, not just a generic regularizer — at this
  label count, decay pulling weights toward zero also indirectly limits how far
  they can drift from their (already-good) pretrained initialization within a
  short fine-tune. If forgetting still shows up despite 5.3-5.4's other guards,
  the next lever is literally penalizing distance from the pretrained weights
  (L2-SP) rather than distance from zero — a documented technique, not implemented
  here yet, worth trying only if the simpler guards above prove insufficient.
- **Model selection: k-fold, not one split, once comparing architectures/losses.**
  `kaggle.md` flags this for Month 3 in general, but it applies earlier here
  specifically *because* the dataset is small enough that a single grouped
  85/15 split's noise can plausibly exceed the gap between e.g. ResNet18 vs.
  ResNet34, or Tversky vs. Tversky+boundary-term. Use `jp-mvp1`'s existing
  `group_split` grouping logic, repeated across ≥3-5 folds/seeds, before trusting
  a comparison enough to act on it.
- **Early stopping on local PQ, not train loss or even val Dice** — `jp-mvp1`
  already has a real PQ implementation (`src/metrics.py`); use it as the actual
  selection criterion, since Dice/BCE convergence doesn't guarantee good instance
  separation after postprocessing (the postprocessing step itself already moved
  PQ by 8x in `kaggle.md`'s own account, independent of the model).

### 5.7 Resolution mismatch to resolve before the ablation is fair

`jp-mvp1:src/dataset.py`'s `FilamentDataset` currently defaults to `img_size=256`;
Stage 3's MAE pretraining crops at `384` (section 4.1). Running the ResNet-vs-ViT
ablation at two different input resolutions would confound architecture with
resolution, exactly the single-variable violation `kaggle.md` Step 4 warns about.
Pick one fine-tuning resolution for the ablation (384, to match what the ViT
encoder was actually pretrained at — a ViT's positional embeddings are tied to
patch grid size, so mismatching resolution here also means interpolating position
embeddings, an extra confound) and hold it fixed across both branches; a
resolution sweep is a legitimate follow-up (per `kaggle.md`'s resolution/VRAM
notes) but only after the architecture question itself is settled on equal footing.

---

## 6. Open items to fill in before running

- [x] GONG archive URL/directory structure and site codes — verified live, see
      section 2.
- [x] FITS header keys for disk center/radius — verified against real Big Bear
      and Mauna Loa frames, see section 3.
- [x] Limb-darkening correction, contrast stretch, dedup — all implemented and
      verified against real data, then deliberately dropped by request to keep
      Stage 2 to a plain FITS→JPEG conversion. See section 3 and this file's
      git history if any of it needs reviving.
- [x] Output resolution/channels/format — kept native `2048×2048`, 1-channel,
      JPEG, verified to match MAGFiLO's own training images exactly, see
      section 3.
- [ ] Decide ViT-Small vs. ViT-Base from the first session's measured
      images/sec at each scale on 2×T4, not up front.
- [ ] Pick a normalization source for Stage 3 now that Stage 2 doesn't compute
      corpus stats — `HalphaMAEDataset` above just divides by 255 as a
      placeholder; decide whether that's good enough or Stage 3 should compute
      its own mean/std pass first.
- [ ] Verify `HalphaMAEDataset._disk_bounded_crop` (section 4.1) against real
      JPEG frames once Stage 3 is actually implemented — it's still
      illustrative pseudocode, unlike Stages 1-2's tested scripts.
- [ ] Pick a `crop_size` (default 384 above) against the ViT patch size chosen
      in section 4.2 — this is now decoupled from Stage 2's output resolution,
      so it can change without touching the preprocessed corpus.
- [ ] Pick `SESSION_BUDGET_SECONDS` against the actual per-session cap Kaggle grants
      the account (varies by verification tier) rather than assuming 8h flat.
- [ ] Tune `--day-stride` / `--max-images` once a real manifest is built, so the
      final downloaded corpus actually lands in the 5-10K target range.
- [ ] Add the section 4.5 held-out date-grouped split to `HalphaMAEDataset`/
      `train_ddp.py` — currently trains on the full manifest with no val signal.
- [ ] Download the full 707-unique-image MAGFiLO train set locally (only 20 sample
      images present as of this plan revision) before any Stage 4 fine-tuning run.
- [ ] **Blocking the rest of 5.1's decision:** run `kaggle.md`'s Month-2 Step 3
      per-image error-analysis tooling against the current `jp-mvp1` baseline's
      real predictions, classified by failure type (missed / fragmented /
      merged / poor-boundary) and split into SQ vs. RQ contribution. Nothing
      below should be built until this exists — see 5.1's falsification
      criteria for what result points at which candidate.
- [ ] Once that measurement picks a segmentation head (embedding head, CondInst,
      or Mask2Former per 5.1), implement it on top of `jp-mvp1`'s existing
      decoder output (or replace the decoder entirely, if Mask2Former) — new
      code, not yet built or validated against real data either way.
- [ ] Bump `jp-mvp1:src/model.py`'s baseline from `resnet18` to `resnet34` and
      re-run it (with whichever head 5.1 selects) as the number the ViT-MAE
      ablation (5.2, 5.3) actually has to beat.
- [ ] Resolve the 256px (`jp-mvp1` default) vs. 384px (Stage 3 crop) mismatch
      (5.7) before running the ablation.
- [ ] Implement the linear-probe forgetting tripwire (5.6) alongside the first
      full fine-tune run, not after — it's cheap and is the earliest signal that
      something in 5.5's LR schedule is off.

Pretraining is complete once Stage 3's masked-reconstruction quality plateaus
(loss curve + val loss from 4.5 not diverging from train) and the Stage 4
ablation (5.2, 5.3) shows the MAE-pretrained encoder beating both scratch/
ImageNet-init ViT *and* the ResNet34 baseline (whichever head 5.1's measurement
selects) on local PQ — at that point the resulting encoder checkpoint is the
deliverable this whole plan exists to produce. If it only beats scratch/
ImageNet-init ViT but not the ResNet baseline, that's a valid outcome too — it
means ship the ResNet baseline and treat the SSL investment as a documented
negative result, not a reason to keep sinking compute into it.
