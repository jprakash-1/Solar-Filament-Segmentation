# ResNet50 Domain SSL Pretraining Plan (Hα, BYOL)

This is the concrete execution plan for the path decided in `PRETRAIN_PLAN.md`
§5.3 "Option 2" for the ResNet encoder: instead of leaving ResNet50 on generic
ImageNet weights, run a **domain-specific self-supervised pretraining pass on the
GONG Hα corpus** before handing the encoder to Stage 4 segmentation fine-tuning.
It sits parallel to `PRETRAIN_PLAN.md`'s Stage 3 (ViT-MAE) — same corpus, same
Kaggle 2×T4 infra constraints, different method because ResNet needs a
CNN-native SSL objective, not ViT-style patch masking.

**Backbone note (revised from an earlier ResNet34 draft of this plan):**
ResNet50, not ResNet34, specifically because CondInst — the current
best-supported segmentation-head candidate in `PRETRAIN_PLAN.md` §5.1 — ships
tested, stock configs (AdelaiDet and mmdetection) only for ResNet-50/101
`Bottleneck` backbones (`[256,512,1024,2048]` FPN channel dims). ResNet34 uses
`BasicBlock`s (`[64,128,256,512]`), which don't match those FPN necks and would
need a custom config rather than dropping into an existing one. Pretraining
ResNet50 means this checkpoint loads straight into CondInst's stock config in
place of ImageNet init — no FPN-channel surgery required.

**Sequencing note:** `PRETRAIN_PLAN.md` §5.3 originally recommended *not* building
this until the ViT-MAE ablation proved domain pretraining was worth it at all.
This plan proceeds directly instead, per direction to prioritize the ResNet path
now. That's a legitimate call — just documented here so it's clear this is a
deliberate reordering, not a forgotten gate. The ablation in §7 below still
exists; it now validates this investment after the fact rather than gating it
beforehand.

---

## 0. Corpus and infra this reuses (nothing new to build here)

- **Images:** `data/processed/gong_pretrain/*.jpeg` — **49,243 images on disk**
  (measured directly; close to the ~55K figure this plan started from, and still
  growing per `PRETRAIN_PLAN.md` §2's incremental download). Native 2048×2048,
  1-channel, 8-bit JPEG, disk-standardized geometry `(cx, cy, r)` recorded per-row
  in `manifest.csv`/`manifest_thinned.csv` (see §3 for which one to use here).
- **DDP helpers:** `distributed.py`'s `setup_distributed()`/`cleanup_distributed()`
  from `PRETRAIN_PLAN.md` §4.3 — identical `torchrun`-based launch, same
  `device_id`-explicit NCCL fix, same `NCCL_P2P_DISABLE=1` workaround. Don't
  re-derive this; import it as-is.
- **Session-budget + checkpoint-resume pattern:** `PRETRAIN_PLAN.md` §4.3's
  `SESSION_BUDGET_SECONDS` self-stopping loop and §4.4's session checklist
  (mount data + last checkpoint, run, new Dataset version, repeat) apply
  unchanged — this is a multi-session Kaggle run just like Stage 3.

---

## 1. Method: BYOL, not SimCLR/MoCo/SparK

- **No negative pairs needed.** SimCLR/MoCo treat every other image in a batch
  (or memory bank) as a negative — which is a real problem for *this specific
  corpus*: `PRETRAIN_PLAN.md` §3 measured same-site consecutive-hour frames at
  **SSIM 0.91** (85% exceed 0.90), i.e. near-duplicates by construction (the Sun
  rotates ~0.5°/hour, filaments persist hours-to-days). A contrastive method
  would routinely push apart representations of two frames that are almost the
  same image — actively counterproductive. BYOL has no negative-pair mechanism,
  so this redundancy is a non-issue, maybe even mildly helpful (more
  near-consistent samples to average over).
- **No large-batch requirement.** SimCLR-style methods need large batches (or a
  memory bank) for enough negatives to be useful; BYOL's target-network
  bootstrapping doesn't, which matters on 2×T4/16GB.
- **Proven for plain CNN backbones** — no exotic ops (unlike SparK's sparse
  masked convolutions, which are less mature tooling, or MAE's patch-masking,
  which is ViT-native and doesn't transfer to a ResNet's dense conv stem).

---

## 2. Channel adaptation: 1-channel ResNet50 from ImageNet stem weights

Average the pretrained stem conv's 3 input-channel filters into 1, rather than
re-initializing the stem randomly — this keeps the stem's learned edge/texture
detectors as a meaningful starting point instead of throwing away exactly the
layer ImageNet pretraining helps most. ResNet50's `conv1` is the same shape as
every other torchvision ResNet's (`[64, 3, 7, 7]`), so this is identical
regardless of depth/block type — only the backbone's final feature width (§5)
depends on that:

```python
import torch
import torchvision.models as tvm

def resnet50_1ch(pretrained=True):
    m = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
    old_conv = m.conv1  # [64, 3, 7, 7]
    new_conv = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    if pretrained:
        with torch.no_grad():
            new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
    m.conv1 = new_conv
    m.fc = torch.nn.Identity()  # backbone only; BYOL adds its own projector head
    return m
```

This is the same trick `segmentation_models_pytorch` already applies under the
hood for `in_channels=1` in `jp-mvp1:src/model.py` — reimplemented explicitly
here because BYOL needs the raw backbone (no `smp` decoder wrapper) as the base
for the projector/predictor heads in §5.

---

## 3. Which manifest: full corpus, not the thinned one — with a caveat

`PRETRAIN_PLAN.md` §3 built `manifest_thinned.csv` (8,557 rows) specifically
because same-site consecutive-hour redundancy wasn't useful for the corpus's
*original* diversity goal. For BYOL specifically, that redundancy isn't harmful
(§1) — so there's no correctness reason to thin here, and using the full
**49,243-image** corpus gives BYOL more raw augmentation diversity per epoch.

Caveat this doesn't resolve on its own: `manifest.csv` itself only covers 25,191
of the 49,243 on-disk JPEGs (a known gap — see `PRETRAIN_PLAN.md` §3's note on
`thin_manifest.py` working around this via directory listing). This plan's
dataset loader needs the same workaround: build the file list from the JPEG
directory listing, falling back to the standardized `(1024, 1024, 900)` disk
geometry for any file missing from `manifest.csv`, exactly as `thin_manifest.py`
already does — don't silently train on only the 25,191 rows the manifest happens
to cover.

**Open sizing question, not resolved here:** full corpus means longer epochs.
Whether that's worth it vs. a partially-thinned set is a throughput call to make
once real images/sec is measured on 2×T4 (same "measure before deciding" pattern
`PRETRAIN_PLAN.md` §4.2 uses for ViT-Small vs. Base).

---

## 4. Augmentation: two domain-adapted views per image

BYOL needs two independently augmented crops per source image. Reuse the
disk-bounded crop from `PRETRAIN_PLAN.md` §4.1 (so a view isn't centered on pure
off-disk background) as the base, then apply the domain-adapted equivalent of
BYOL's original view pipeline — grayscale removes the need for color jitter, but
everything else has a direct analog:

```python
import random
import numpy as np
from scipy.ndimage import gaussian_filter

def make_view(img, cx, cy, r, crop_size=224, blur_p=0.5):
    crop = disk_bounded_crop(img, cx, cy, r, crop_size)  # from PRETRAIN_PLAN.md 4.1
    # rotation: any angle valid, no canonical "up" on the Sun (same reasoning
    # PRETRAIN_PLAN.md 4.1 already uses for the MAE views)
    k = random.randint(0, 3)
    crop = np.rot90(crop, k).copy()
    if random.random() < 0.5:
        crop = np.fliplr(crop).copy()
    # grayscale analog of color jitter: gamma + brightness/contrast
    if random.random() < 0.8:
        gamma = random.uniform(0.7, 1.3)
        crop = np.clip(crop, 0, 1) ** gamma
    if random.random() < 0.5:
        contrast = random.uniform(0.8, 1.2)
        crop = np.clip((crop - 0.5) * contrast + 0.5, 0, 1)
    # BYOL's asymmetric blur: view 1 gets it more often than view 2 in the
    # original recipe (1.0 vs 0.1 probability) -- pass blur_p per call site
    if random.random() < blur_p:
        crop = gaussian_filter(crop, sigma=random.uniform(0.1, 1.0))
    return crop.astype(np.float32)
```

Call `make_view(..., blur_p=1.0)` for view 1 and `make_view(..., blur_p=0.1)` for
view 2 per image, matching BYOL's original asymmetric-blur recipe (verified
detail from the paper — the asymmetry is intentional, not arbitrary).

**Resolution: 224×224**, not Stage 3's 384px MAE crop. ResNet is
resolution-agnostic (global average pool before the head), so this doesn't force
a mismatch the way a ViT's positional embeddings would — but pretraining and
Stage 4 fine-tuning resolutions should still ideally match to keep BatchNorm
running statistics well-calibrated; if Stage 4 ends up at a different resolution,
recalibrate BN stats with a few forward-only passes at the new resolution before
trusting early fine-tune metrics (a known, cheap fix — flagged as an open item,
not solved here).

---

## 5. Model: online network + momentum target network

```python
import copy
import torch.nn as nn

def mlp(in_dim, hidden_dim, out_dim):
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, out_dim),
    )

class BYOL(nn.Module):
    def __init__(self, encoder_dim=2048, proj_dim=256, hidden_dim=4096):
        # encoder_dim=2048: ResNet50's Bottleneck final-stage width, not 512 --
        # that would be ResNet34/18's BasicBlock width instead.
        super().__init__()
        self.online_encoder = resnet50_1ch(pretrained=True)
        self.online_projector = mlp(encoder_dim, hidden_dim, proj_dim)
        self.online_predictor = mlp(proj_dim, hidden_dim, proj_dim)  # online branch only

        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.target_projector = copy.deepcopy(self.online_projector)
        for p in self.target_encoder.parameters(): p.requires_grad = False
        for p in self.target_projector.parameters(): p.requires_grad = False

    @torch.no_grad()
    def update_target(self, tau):
        for online, target in [(self.online_encoder, self.target_encoder),
                                (self.online_projector, self.target_projector)]:
            for po, pt in zip(online.parameters(), target.parameters()):
                pt.data = tau * pt.data + (1 - tau) * po.data

    def forward(self, view1, view2):
        def online_out(v):
            return self.online_predictor(self.online_projector(self.online_encoder(v)))
        with torch.no_grad():
            def target_out(v):
                return self.target_projector(self.target_encoder(v))
            t1, t2 = target_out(view1), target_out(view2)
        o1, o2 = online_out(view1), online_out(view2)
        return o1, o2, t1.detach(), t2.detach()
```

Only `online_encoder` (the plain `resnet50_1ch`) survives to Stage 4 — the
projector, predictor, and the entire target network are discarded after
pretraining, same as MAE's decoder is discarded in `PRETRAIN_PLAN.md` §5.

## 6. Loss

Symmetrized normalized MSE — predict each view's target from the *other* view's
online output, average both directions:

```python
import torch.nn.functional as F

def byol_loss(o1, o2, t1, t2):
    def d(o, t):
        o, t = F.normalize(o, dim=-1), F.normalize(t, dim=-1)
        return 2 - 2 * (o * t).sum(dim=-1)
    return (d(o1, t2) + d(o2, t1)).mean()
```

---

## 7. Optimizer, LR schedule, EMA momentum schedule

- **Optimizer:** SGD, momentum 0.9, `weight_decay=1e-6` — BYOL's original recipe
  uses LARS at large (4096+) batch; at the batch sizes realistic on 2×T4
  (likely still 96-256 per GPU at 224px — ResNet50 is heavier than ResNet34
  but still much lighter than the ViT plan's per-GPU 64 at 384px with a decoder;
  re-measure per §11's open item rather than assuming the 128-256 range holds
  exactly), plain SGD with a linearly-scaled LR is the standard smaller-scale
  substitute, not LARS.
- **LR:** `base_lr = 0.2 * (batch_size / 256)`, linear warmup (~10 epochs), then
  cosine decay to 0 — same shape as Stage 3's schedule, different base rate
  (BYOL's own recipe, not MAE's).
- **EMA target momentum (`tau`):** starts at `0.996`, cosine-annealed up to `1.0`
  over training — the target network should track the online network more
  loosely early on and converge to near-frozen by the end. Don't use a fixed
  `tau`; the schedule is part of BYOL's actual recipe, not an optional refinement.
- **Total epochs:** start around 100-150 given a domain-specific 49K-image corpus
  (well short of BYOL's original 300-1000 epochs on full ImageNet, which had ~26x
  more images) — budget across multiple Kaggle sessions exactly like Stage 3,
  and let §8's health checks (not a fixed count) decide when to stop.

---

## 8. Held-out validation and representation health checks

Unlike MAE, there's no reconstruction to eyeball — BYOL's training loss can keep
dropping even while representations collapse to a near-constant output (a known
failure mode of negative-free SSL methods if something's misconfigured), so loss
alone isn't a trustworthy stopping signal here. Concretely:

- **Held-out split:** same date-grouped ~3-5% holdout as `PRETRAIN_PLAN.md` §4.5,
  reused for this corpus too.
- **Embedding collapse check:** every few epochs, compute the per-dimension
  standard deviation of L2-normalized online-encoder embeddings over a batch of
  held-out images. A healthy run keeps this well above zero throughout; a value
  collapsing toward zero is the earliest, cheapest signal something is wrong
  (predictor/target update bug, LR too high, etc.) — check this before spending
  a full session assuming the loss curve alone means training is working.
- **Nearest-neighbor sanity check:** periodically embed a handful of held-out
  frames, pull each one's top-k cosine-nearest neighbors from the rest of the
  held-out set, and visually confirm the neighbors are astronomically similar
  (comparable filament density/activity level, not random frames) — the BYOL
  equivalent of Stage 3 §4.4's "inspect masked-reconstruction quality" check.

---

## 9. Session checklist (Kaggle multi-session, mirrors `PRETRAIN_PLAN.md` §4.4)

1. Mount `halpha-preprocessed` (or a corpus-specific dataset if this ends up
   packaged separately) and the latest BYOL checkpoint dataset as notebook inputs.
2. Resume from the mounted checkpoint if present (`online_encoder`,
   `online_projector`, `online_predictor`, `target_encoder`, `target_projector`,
   optimizer state, epoch, `tau` schedule position — all of it, not just weights,
   for the same reason `kaggle.md` flags naive resume as silently distorting the
   LR/momentum schedule).
3. Run; monitor loss **and** the §8 embedding-std check, not loss alone.
4. Self-stop at `SESSION_BUDGET_SECONDS`, checkpoint every epoch.
5. New Dataset version from the checkpoint dir → next session's input.
6. Every few sessions: run the §8 nearest-neighbor visual check.

---

## 10. Handoff to Stage 4 segmentation fine-tuning

This plan produces exactly one deliverable: a domain-pretrained `resnet50_1ch`
`online_encoder` state dict. Everything after that is already specified in
`PRETRAIN_PLAN.md` and applies unchanged:

- Attach whichever segmentation head `PRETRAIN_PLAN.md` §5.1's error-analysis
  measurement actually selects (embedding head, CondInst, or Mask2Former —
  **not settled yet**, see §5.1's revision history; do not assume the
  embedding head by default). This plan's encoder is agnostic to that choice
  either way — the ResNet50 pick specifically keeps the CondInst path cheap
  (drops into its stock config, per the backbone note at the top) without
  costing anything if a different head is selected instead (`smp.Unet` and
  most other segmentation decoders support ResNet50 just as readily).
- Fine-tune with the loss, layer-wise LR decay, freeze-warmup, linear-probe
  forgetting tripwire, and k-fold model selection already specified in §5.4-§5.6.
- Resolution: keep the §4-recommended 224px (or whatever this pretraining
  actually lands on) consistent through fine-tuning per §7's BatchNorm-matching
  note, not `PRETRAIN_PLAN.md` §5.7's 384px (that mismatch note is specifically
  about the ViT branch's positional embeddings, which don't apply here).

**Mandatory ablation before trusting this investment**, mirroring
`PRETRAIN_PLAN.md`'s ViT ablation structure: BYOL-pretrained `resnet50` vs.
plain-ImageNet-init `resnet50` vs. scratch `resnet50`, same segmentation
head/loss/schedule held fixed (whichever head §5.1 selects). This is what
actually answers the question this plan's sequencing note deferred — whether
domain SSL was worth building for the ResNet branch at all.

---

## 11. Open items

**Now implemented as real code** — `scripts/pretrain_resnet/` (`distributed.py`,
`dataset.py`, `model.py`, `health_checks.py`, `train_byol.py`) and
`pretrain_resnet_kaggle.ipynb`, verified locally (CPU, real corpus, small subset)
before this update. DDP/multi-GPU path (`SyncBatchNorm` + `torchrun`) is written
per §5/§9 but not yet exercised on real 2×T4 hardware — that's the notebook's
first real job.

- [x] Verify `resnet50_1ch`'s stem-averaging trick against `smp`'s existing
      `in_channels=1` behavior (`jp-mvp1:src/model.py`). **Measured result: they
      do NOT match** (`model.py`'s `verify_stem_averaging()`, max_abs_diff ~2.45).
      Two independent causes, confirmed by reading `smp`'s source
      (`segmentation_models_pytorch/encoders/_utils.py:patch_first_conv`): (1)
      smp **sums** the 3 input-channel filters rather than averaging them — a
      real 3× scale difference; this plan's `resnet50_1ch()` deliberately uses
      `.mean()` instead, since summing would leave the stem's output ~3× the
      scale the pretrained downstream BN/ReLU stack was calibrated for. (2) smp's
      `encoder_weights="imagenet"` pulls a different pretrained checkpoint (its
      own HuggingFace hub, `smp-hub/resnet50.imagenet`) than
      `tvm.ResNet50_Weights.IMAGENET1K_V2` used here. Conclusion: both are
      reasonable, independently-justified choices, but not interchangeable —
      don't assume `jp-mvp1`'s `smp.Unet(in_channels=1)` and this file's
      `resnet50_1ch()` would produce the same stem.
- [x] Implement the `manifest.csv`-gap workaround from §3 (directory listing as
      source of truth) in this plan's dataset loader —
      `scripts/pretrain_resnet/dataset.py`'s `discover_images()`, same pattern as
      `thin_manifest.py`. Verified against the real corpus: 49,243 on-disk images
      discovered, manifest.csv covered 25,189 of them (24,054 fell back to
      standardized geometry) — matches `PRETRAIN_PLAN.md` §3's ~25,191 figure.
- [ ] Measure real images/sec on 2×T4 at 224px/batch 128-256 to firm up §7's
      batch size and total-epoch budget, same "measure before committing" pattern
      as `PRETRAIN_PLAN.md` §4.2 — needs the notebook's actual first Kaggle run,
      not resolvable locally (no multi-GPU hardware here).
- [ ] Decide full-corpus (49,243) vs. some intermediate thinning once real
      per-epoch wall-clock is known (§3's open sizing question) — same
      Kaggle-run dependency as above.
- [x] Build the §8 embedding-std and nearest-neighbor health-check tooling before
      the first real multi-session run — `scripts/pretrain_resnet/health_checks.py`
      (`embedding_std`, `nearest_neighbors` + saved visual grid), wired into
      `train_byol.py`'s periodic logging. Verified it runs end-to-end on a local
      smoke run (real images, random-init weights, so the numbers themselves
      aren't meaningful yet — only that the plumbing produces valid output).
- [ ] Run the §10 ablation once both this pretraining and the Stage 4
      embedding-head fine-tuning pipeline exist — currently blocked on both.
