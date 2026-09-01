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
    --day-stride 3 --max-images 10000 \
    --out data/raw/gong_pretrain/manifest.csv

python scripts/pretrain_data/gong_halpha.py download \
    --manifest data/raw/gong_pretrain/manifest.csv \
    --out-dir data/raw/gong_pretrain
```

`--day-stride 3` (sample every 3rd day) is the default because 2 sites' combined
daylight coverage is ~15-20 frames/day, and continuous daily coverage across two
full-year windows would produce ~20-25K frames — well over the 5-10K target.
Tune it once the manifest's row count comes back (rerun `manifest` with a
different stride until the count lands in range; it's cheap, no download yet).
Downloading is resume-safe (skips files already on disk) so a killed Kaggle
session can just rerun the same command.

Once downloaded: in the Kaggle UI, **New Dataset → upload the output directory →
version it** as `halpha-raw`. This is the only stage that needs internet
access — every later stage mounts a Kaggle Dataset instead.

---

## 3. Stage 2 — Preprocessing

**Verified against two real downloaded frames** (Big Bear and Mauna Loa,
2022-03-18), superseding the "confirm header keys" open item:

- GONG's own processing pipeline (the "H Alpha Laminator") already
  re-registers every "haf" (reduced) product to a **standardized geometry**:
  disk center at pixel `(1024, 1024)`, radius `900` px, in every `2048×2048`
  frame — confirmed identical across both sites' samples via the
  `FNDLMBXC`/`FNDLMBYC`/`FNDLMBMA`/`FNDLMBMI` header keys (found-limb center/
  major-axis/minor-axis). Disk detection is therefore a matter of *reading*
  that header geometry, not deriving it via CV — exactly the plan's original
  intent, now with the real key names filled in. `CRPIX1`/`CRPIX2` + a flat
  `RADIUS=900` are kept as a fallback only, for the rare frame where the
  limb-fit keys are missing or malformed.
- The first limb-darkening attempt used the textbook photospheric-continuum
  formula `1/(0.3 + 0.7·mu)` — checked against real data with a
  mean-intensity-per-radial-bin diagnostic, it **overcorrected badly** (the
  "corrected" profile sloped upward toward the limb instead of flattening).
  Hα center-to-limb variation is much milder than continuum limb darkening
  (chromospheric emission, not photospheric), so a generic continuum formula
  doesn't transfer. Replaced with an **empirical radial profile** — mean
  intensity per radial bin, computed once from a sample of raw frames — which
  measurably flattens a real held-out frame's profile (checked: pre-correction
  ~3206→2457 center-to-limb, post-correction flat at ~3200 throughout).
- **Output: full native `2048×2048`, 1-channel, 8-bit JPEG — no crop/resize to
  a smaller pretraining resolution, and no `.npy`.** Verified this matches
  MAGFiLO's own input convention exactly, not just approximately: a real
  MAGFiLO training image is itself a `2048×2048`, single-channel (`L` mode)
  *JPEG*, with the same GONG-style filename and the *same* disk framing — its
  off-disk corners read exactly `0`, and its limb sits right at radius ~900
  around `(1024, 1024)`, identical to the raw "haf" product's standardized
  geometry above. Matching format as well as resolution means there's no
  train/pretrain mismatch of any kind to bridge at Stage 4 fine-tuning time.
  JPEG also matters for corpus size: a `2048×2048` float32 `.npy` is ~16.8MB —
  a 10K-image corpus would be ~168GB, not uploadable as a Kaggle Dataset. A
  real MAGFiLO JPEG at this resolution is ~700KB (~7GB for 10K images).
  **Consequence for Stage 3**: a ViT can't operate on the full `2048×2048`
  frame directly — at patch=16 that's 128×128=16,384 patches (even after MAE's
  75% masking, ~4,096 visible tokens, ~28× the token count of a 384px/patch=16
  setup, quadratic in attention cost). Stage 3's dataset instead samples a
  disk-radius-bounded crop from these full-resolution frames per training
  step — see section 4.1.
- **8-bit conversion reuses this project's own prior work**, not a new scheme:
  `jp-analysis:scripts/apply_contrast_stretch.py` already found (while
  investigating a per-site photometric gap on the actual MAGFiLO images) that
  a per-image `[1st, 99th]` percentile clip, computed only over the disk
  *interior* (rho < 0.85, excluding the limb-brightening annulus) and rescaled
  to `[0, 255]`, closes that gap. Reused directly here, with the header-based
  `(cx, cy, r)` from above in place of that script's CV-based disk detection
  (already known precisely, no need to re-derive it).
- **Checked real output visually and found real per-site graininess**: Mauna
  Loa frames come out visibly grainier than Big Bear's. Traced this by
  comparing the *raw* frame (before any correction) stretched the same way —
  the graininess is already present pre-correction and pre-limb-darkening, so
  it's a genuine per-site noise/dynamic-range characteristic of this data
  (consistent with the photometric gap `apply_contrast_stretch.py` was built
  to investigate), not a bug introduced by this script. Not addressed further
  here — flagging it in case Stage 3/4 want a per-site quality weighting or a
  Mauna-Loa-specific denoise step later.

Implementation: `scripts/pretrain_data/preprocess_gong.py`, run once over the
downloaded corpus:

```bash
python scripts/pretrain_data/preprocess_gong.py \
    --raw-dir data/raw/gong_pretrain --out-dir data/processed/gong_pretrain
```

Two passes: (1) samples `--profile-sample-size` frames (default 50 — kept
small since each sampled frame is a full `2048×2048` float32 array, ~16.8MB) to
build one shared empirical limb-darkening profile (`limb_profile.npy`, a small
1D array — the only remaining `.npy` in this pipeline), printing the
pre-correction profile so a wildly non-monotonic result is visible
immediately; (2) processes every frame — limb-darkening correct (at full
native resolution, using the header's own `cx`/`cy`/`r`, interpolating the
profile between bin centers rather than a nearest-bin lookup, which produced
visible concentric ring artifacts when first tried) → percentile-clip
contrast-stretch to 8-bit → per-site stuck-camera dedup (frame-differencing on
the final `[0,255]` scale, threshold tuned empirically against real
distinct-hour frames — see the script's `DEDUP_THRESHOLD` comment) →
corpus-level mean/std over the kept set (`dataset_stats.json`, now over the
`[0,255]` domain), which Stage 3's dataloader should normalize with instead of
ImageNet stats. Both the dedup and the corpus-stats pass are streaming (one
frame in memory at a time, plus one running "previous frame" per site) rather
than stacking the whole corpus into one array. Output: one JPEG per kept frame
(native `2048×2048`, 8-bit, quality 95) plus `manifest.csv` (path, site, disk
center/radius in the frame's own native pixel coordinates).

Save output as Kaggle Dataset `halpha-preprocessed` — every future pretraining
session mounts this read-only, no re-downloading or re-processing.

---

## 4. Stage 3 — MAE pretraining (DDP, 2×T4, resumable)

### 4.1 Dataset / augmentation

Stage 2's files are the full native `2048×2048`, 8-bit JPEG frame (see section
3), so this dataset's job includes sampling the actual ViT-sized training crop
and normalizing with the corpus stats, not just loading a pre-sized image:

```python
# dataset.py
import json
import numpy as np, pandas as pd, torch, random
from PIL import Image
from torch.utils.data import Dataset

class HalphaMAEDataset(Dataset):
    def __init__(self, manifest_csv, stats_json, crop_size=384, train=True):
        manifest = pd.read_csv(manifest_csv)
        self.paths = manifest["path"].tolist()
        self.cx = manifest["cx"].tolist()
        self.cy = manifest["cy"].tolist()
        self.r = manifest["r"].tolist()  # disk radius, in the *native* 2048x2048
        # frame -- Stage 2 doesn't resize, so no rescaling needed here
        stats = json.load(open(stats_json))  # dataset_stats.json from Stage 2 --
        self.mean, self.std = stats["mean"], stats["std"]  # over [0,255], not
        # ImageNet stats -- these are this corpus's own values
        self.crop_size = crop_size
        self.train = train

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = np.array(Image.open(self.paths[idx]), dtype=np.float32)
        img = (img - self.mean) / (self.std + 1e-6)
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

MANIFEST = "/kaggle/input/halpha-preprocessed/manifest.csv"
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

---

## 5. Stage 4 — Handoff to segmentation fine-tuning (separate notebook)

- Load `model.module.vit` (encoder only) from the final MAE checkpoint; discard the
  MAE decoder.
- Attach a segmentation decoder (UperNet-style head pulling features from multiple
  ViT layer depths, or a simpler FPN head).
- Fine-tune end-to-end on labeled MAGFiLO data (the `jp-mvp1` pipeline's
  `src/dataset.py` COCO parsing / group-aware split can be reused directly) with a
  Tversky/boundary-aware loss, using layer-wise LR decay (lower LR on the
  pretrained encoder, higher on the fresh decoder head).
- Ablation to run before trusting the SSL investment: MAE-pretrained encoder vs.
  ImageNet-initialized encoder vs. scratch, same decoder/loss held fixed — this
  quantifies whether pretraining is actually paying off before sinking further
  compute into longer pretraining runs.

---

## 6. Open items to fill in before running

- [x] GONG archive URL/directory structure and site codes — verified live, see
      section 2.
- [x] FITS header keys for disk center/radius — verified against real Big Bear
      and Mauna Loa frames, see section 3.
- [x] Limb-darkening correction — the parametric formula failed a real flatness
      check and was replaced with an empirical profile, see section 3.
- [x] Output resolution/channels — kept native `2048×2048`, 1-channel,
      verified to match MAGFiLO's own training images exactly, see section 3.
- [ ] Decide ViT-Small vs. ViT-Base from the first session's measured
      images/sec at each scale on 2×T4, not up front.
- [ ] Verify `HalphaMAEDataset._disk_bounded_crop` (section 4.1) against real
      `.npy` frames once Stage 3 is actually implemented — it's still
      illustrative pseudocode, unlike Stages 1-2's tested scripts.
- [ ] Pick a `crop_size` (default 384 above) against the ViT patch size chosen
      in section 4.2 — this is now decoupled from Stage 2's output resolution,
      so it can change without touching the preprocessed corpus.
- [ ] Pick `SESSION_BUDGET_SECONDS` against the actual per-session cap Kaggle grants
      the account (varies by verification tier) rather than assuming 8h flat.
- [ ] Tune `--day-stride` / `--max-images` once a real manifest is built, so the
      final downloaded corpus actually lands in the 5-10K target range.

Pretraining is complete once Stage 3's masked-reconstruction quality plateaus
(loss curve + visual check) and the Stage 4 ablation shows the MAE-pretrained
encoder beating scratch/ImageNet-init on local Dice/PQ — at that point the
resulting encoder checkpoint is the deliverable this whole plan exists to produce.
