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
2. **Preprocessing** (one-time) → normalized `.npy` + manifest CSV → Kaggle Dataset `halpha-preprocessed`
3. **MAE pretraining** (multi-session, resumable) → mounts `halpha-preprocessed`, checkpoints to a `halpha-mae-ckpt` dataset each session
4. **Segmentation fine-tuning** (separate notebook, on `jp-mvp1`-style pipeline) → mounts the final `halpha-mae-ckpt`, fine-tunes on labeled MAGFiLO data

---

## 2. Stage 1 — Data acquisition

**Target:** ~5,000-10,000 images across 2 sites (Big Bear + Mauna Loa, for
day/night complementary coverage), ~1 frame/hour, spanning multiple years so the
corpus covers both quiet-sun and active-sun phases of solar cycle 25 (started
~Dec 2019; recent active/max period ~2023-2025) rather than skewing toward one
filament-density regime.

**Action items to resolve before running this at scale** (skeleton below has
placeholders for exactly these):

- Browse the real GONG/BBSO Hα archive tree for `big_bear_halpha/` and
  `mauna_loa_halpha/` once manually to confirm the actual year/month/day folder
  structure and filename pattern — layouts differ slightly between GONG products
  and sites, so don't assume the skeleton's path guess is right.
- Confirm the `sunpy.net.Fido` alternative resolves GONG Hα queries cleanly with a
  small date-range test query before committing to a bulk pull — if it works
  cleanly, prefer it over hand-rolled directory scraping.

```python
# acquire_gong_halpha.py
import os, requests
from pathlib import Path
from bs4 import BeautifulSoup

SITES = ["big_bear_halpha", "mauna_loa_halpha"]  # start with 2 sites
BASE_URL = "https://iswa.ccmc.gsfc.nasa.gov/iswa_data_tree/observation/solar/gong/{site}/"
OUT_DIR = Path("/kaggle/working/halpha_raw")
CADENCE_HOURS = 1  # subsample -- GONG native cadence is ~20s, don't pull all of it

# quiet-sun and active-sun windows of solar cycle 25 -- adjust once the corpus
# budget (5-10K images / 2 sites / ~1 frame-per-hour) is actually planned out
DATE_RANGES = [
    ("2019-06-01", "2020-06-01"),  # quiet
    ("2023-06-01", "2024-06-01"),  # active
]

def list_dir(url: str) -> list[str]:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    return [a["href"] for a in soup.find_all("a") if a["href"].endswith(("/", ".fits", ".jpg"))]

def download_site(site: str, date_ranges: list[tuple[str, str]]) -> None:
    (OUT_DIR / site).mkdir(parents=True, exist_ok=True)
    base = BASE_URL.format(site=site)
    for entry in list_dir(base):
        # directory structure is typically year/month/day -- confirm the real
        # listing once (see action items above) and fill in the date-filtered
        # recursive walk here, respecting CADENCE_HOURS and DATE_RANGES.
        pass  # skeleton -- do not run against date ranges/paths until verified

if __name__ == "__main__":
    for site in SITES:
        download_site(site, DATE_RANGES)
```

Once populated: run it, then in the Kaggle UI **New Dataset → upload
`/kaggle/working/halpha_raw` → version it** as `halpha-raw`. This is the only stage
that needs internet access — every later stage mounts a Kaggle Dataset instead.

---

## 3. Stage 2 — Preprocessing

```python
# preprocess_halpha.py
import numpy as np, pandas as pd, os
from astropy.io import fits
from pathlib import Path
import cv2

RAW_DIR = "/kaggle/input/halpha-raw"
OUT_DIR = "/kaggle/working/halpha_preprocessed"
IMG_SIZE = 384  # multiple of ViT patch size (16) -- revisit against the eventual
                # encoder's patch size before running at scale

def limb_darkening_correct(img, cx, cy, r):
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    rho = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / r
    rho = np.clip(rho, 0, 0.999)
    mu = np.sqrt(1 - rho ** 2)
    correction = 1.0 / (0.3 + 0.7 * mu)  # simple linear limb-darkening model;
    # tune the 0.3/0.7 coefficients against a handful of real frames -- check that
    # mean intensity vs. radius is flat post-correction before trusting this at scale
    return img * correction

def process_one(fits_path: Path, out_dir: str, manifest_rows: list[dict]) -> None:
    with fits.open(fits_path) as hdul:
        data = hdul[0].data.astype(np.float32)
        hdr = hdul[0].header
        # CRPIX1/2, SOLAR_R are placeholders -- confirm the actual header keys on
        # real pulled files before trusting these; GONG's disk-center/radius key
        # names are not yet verified against a real sample in this environment.
        cx = hdr.get("CRPIX1", data.shape[1] / 2)
        cy = hdr.get("CRPIX2", data.shape[0] / 2)
        r = hdr.get("SOLAR_R", min(data.shape) / 2.2)

    data = limb_darkening_correct(data, cx, cy, r)
    data = cv2.resize(data, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    data = (data - data.mean()) / (data.std() + 1e-6)  # per-image normalize here;
    # dataset-level mean/std for final pretraining normalization is computed
    # separately below, over the *kept* (post-dedup) corpus, not this per-image pass

    out_path = Path(out_dir) / (fits_path.stem + ".npy")
    np.save(out_path, data.astype(np.float32))
    manifest_rows.append({"path": str(out_path), "cx": cx, "cy": cy, "r": r})
    # r is carried into the manifest so Stage 3's augmentation can bound random
    # crops to the disk radius instead of risking a crop that's mostly off-disk

def dedup_near_identical(paths: list[str], threshold: float = 2.0) -> list[str]:
    # frame-differencing dedup -- fine at ~1 frame/hour where "near-identical" is
    # really about a stuck camera repeating a frame during an outage, not natural
    # minute-to-minute similarity. Switch to a perceptual hash (imagehash pHash/
    # dHash) if the corpus grows to a cadence where naive diffing gets slow.
    kept = []
    prev = None
    for p in paths:
        img = np.load(p)
        if prev is None or np.abs(img - prev).mean() > threshold:
            kept.append(p)
            prev = img
    return kept

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    manifest_rows: list[dict] = []
    for f in Path(RAW_DIR).rglob("*.fits"):
        try:
            process_one(f, OUT_DIR, manifest_rows)
        except Exception as e:
            print(f"skip {f}: {e}")

    df = pd.DataFrame(manifest_rows)
    kept_paths = dedup_near_identical(df["path"].tolist())
    df = df[df["path"].isin(kept_paths)]

    mean, std = 0.0, 1.0
    if len(df):
        stacked = np.stack([np.load(p) for p in df["path"]])
        mean, std = float(stacked.mean()), float(stacked.std())
    df.to_csv(f"{OUT_DIR}/manifest.csv", index=False)
    with open(f"{OUT_DIR}/dataset_stats.json", "w") as f:
        import json
        json.dump({"mean": mean, "std": std, "n_images": len(df)}, f)
    print(f"Final dataset: {len(df)} images, mean={mean:.4f}, std={std:.4f}")
```

Save output as Kaggle Dataset `halpha-preprocessed` — every future pretraining
session mounts this read-only, no re-downloading or re-processing.

---

## 4. Stage 3 — MAE pretraining (DDP, 2×T4, resumable)

### 4.1 Dataset / augmentation

```python
# dataset.py
import numpy as np, pandas as pd, torch, random
from torch.utils.data import Dataset

class HalphaMAEDataset(Dataset):
    def __init__(self, manifest_csv, img_size=384, train=True):
        manifest = pd.read_csv(manifest_csv)
        self.paths = manifest["path"].tolist()
        self.radii = manifest["r"].tolist()  # disk radius in the *original* frame;
        # rescaled below since Stage 2 already resized to img_size before saving
        self.img_size = img_size
        self.train = train

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = np.load(self.paths[idx]).astype(np.float32)
        if self.train:
            img = self._augment(img)
        return torch.from_numpy(img).unsqueeze(0)  # [1, H, W]

    def _augment(self, img):
        # rotation is valid here -- no canonical "up" on the Sun
        k = random.randint(0, 3)
        img = np.rot90(img, k).copy()
        # intensity/contrast perturbation as the grayscale stand-in for color jitter
        if random.random() < 0.5:
            gamma = random.uniform(0.8, 1.2)
            img = np.sign(img) * (np.abs(img) ** gamma)
        # deliberately no random-resized-crop here: an unbounded crop can land
        # mostly off-disk (pure background) or straddle the limb, which would let
        # the MAE pretext task shortcut on limb position rather than learning
        # filament-relevant texture. A disk-radius-bounded crop needs the radius
        # carried through Stage 2's manifest (done above) -- implement before
        # adding cropping augmentation, don't add unbounded RandomResizedCrop.
        return img
```

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

- [ ] Confirm the actual GONG/BBSO directory and date structure by browsing
      `big_bear_halpha/` once manually (or verify the `sunpy.net.Fido` path works
      cleanly instead).
- [ ] Confirm the real FITS header keys for disk center/radius on actual pulled
      files (`CRPIX1/2`, `SOLAR_R` above are placeholders).
- [ ] Tune the limb-darkening correction coefficients against a few sample images
      visually (check mean-intensity-vs-radius is flat post-correction).
- [ ] Decide ViT-Small vs. ViT-Base from the first session's measured
      images/sec at each scale on 2×T4, not up front.
- [ ] Implement the disk-radius-bounded crop in `_augment()` — the manifest already
      carries `r` per image (Stage 2) specifically so this doesn't need a second
      data pass later.
- [ ] Pick `SESSION_BUDGET_SECONDS` against the actual per-session cap Kaggle grants
      the account (varies by verification tier) rather than assuming 8h flat.

Pretraining is complete once Stage 3's masked-reconstruction quality plateaus
(loss curve + visual check) and the Stage 4 ablation shows the MAE-pretrained
encoder beating scratch/ImageNet-init on local Dice/PQ — at that point the
resulting encoder checkpoint is the deliverable this whole plan exists to produce.
