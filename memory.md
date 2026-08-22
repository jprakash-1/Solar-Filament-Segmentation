# Experiment log

Living record of tuning experiments for this project -- what was tried, why, and what
happened. Append new sections as experiments run; don't delete old ones.

**Note (2026-08-22)**: `scripts/train.py` now writes each run to its own auto-named
subfolder under `checkpoint_dir`/`log_dir` (e.g.
`outputs/checkpoints/20260822_184754_img1536_bs12_gc/`) instead of always overwriting
the same fixed path -- this is what prevented the 1536px+checkpointing run below from
clobbering the 768px epoch-27 baseline. `--resume` auto-detects and continues the most
recently modified run folder rather than starting a new one.

---

## 2026-08-22: Postprocessing tuning fixed instance metrics (no retraining needed)

**Symptom**: epoch-27 checkpoint (trained at `image_size=768`) had decent semantic
accuracy but near-zero instance-level metrics:

| metric | before | after |
|---|---|---|
| mean Dice (semantic) | 0.6717 | 0.6717 (unaffected -- postprocessing doesn't touch this) |
| mean IoU (semantic) | 0.5209 | 0.5209 (unaffected) |
| mean Panoptic Quality | 0.0394 | **0.3392** |
| mAP@[.5:.95] | 0.0012 | **0.1121** |
| AP@0.50 | 0.0056 | 0.3405 |
| AP@0.75 | 0.0001 | 0.0351 |

**Diagnosis**: predicted-vs-GT instance counts (4321 predicted vs 898 ground truth,
~4.8x over-segmentation) pointed at watershed over-splitting thin/elongated/curvy
filament shapes -- the distance-transform has multiple local peaks along a single
filament's length, so a small `watershed_min_distance` picks up several as separate
seeds and fragments one filament into pieces.

**Fix**: `scripts/evaluate.py --sweep-watershed-min-distance` (predicts once, tries many
values cheaply against cached predictions) found mean PQ peaks in a broad plateau around
**`watershed_min_distance=130-150`** (0.3383 vs 0.3381, effectively tied), clearly
declining by 170+. Settled on **140** (middle of the plateau, more robust than the exact
edge value) with `min_instance_area=40`. This is a postprocessing-only change --
verified on the *same* trained weights, no retraining involved.

**Still open**: `AP@0.75` stayed low (0.0351) even after this fix -- most matched
instances land in the 0.5-0.75 IoU range, not tighter. That gap isn't something
postprocessing tuning can close; see the next experiment below.

---

## 2026-08-22: Diagnosing the loss plateau -- underfitting, not overfitting

Looked at `outputs/logs/train_log.csv` (epochs 1-35 of the same run) to understand why
semantic Dice/IoU capped out around 0.67/0.52.

- **Train and val loss track within ~0.01-0.02 of each other the entire time**, no
  divergence anywhere from epoch ~12 to 35 (e.g. epoch 30: train=0.325, val=0.318;
  epoch 20: train=0.341, val=0.323). Rules out overfitting cleanly -- if it were
  overfitting, train_loss would keep dropping while val_loss stalled or rose.
- **The LR schedule fully decayed to ~1.3e-5 by epoch 35 with zero corresponding
  improvement** -- train_loss was 0.383 at epoch 10 and still 0.317 at epoch 35, barely
  moved despite the LR dropping ~20,000x over that span. The cosine schedule ran its
  full course; more epochs at this point would not have helped.
- **Conclusion**: this is a genuine capacity/information ceiling (underfitting), not a
  training-duration or overfitting problem.

**Leading hypothesis**: `image_size` was cut from the competition's native 2048px down
to 768px purely to fit T4 VRAM during this session's multi-GPU tuning (see git history
around the DDP rollout). For thin, fine-boundary filament shapes, that's a lot of lost
spatial detail -- consistent with both the loss plateau and the low `AP@0.75` above.

---

## 2026-08-22: Gradient checkpointing + higher resolution (in progress)

**Goal**: test whether recovering resolution moves the loss floor. If it does,
resolution was the bottleneck. If train_loss plateaus at roughly the same ~0.30-0.32
floor even at higher resolution, resolution wasn't the limiting factor -- the next lever
is a bigger encoder (e.g. `convnext_small` instead of `convnext_tiny`), not more
resolution or more epochs.

**Changes**:
- `src/models/unet_convnext.py`: new `gradient_checkpointing` param on `build_model()`.
  When enabled, calls `model.encoder.model.set_grad_checkpointing(enable=True)` --
  recomputes encoder activations during backward instead of storing them, trading
  compute time for VRAM headroom. Encoder-only: `smp`'s U-Net decoder has no
  checkpointing support, so decoder activations (non-trivial at high resolution) are
  still fully materialized.
  - Verified locally (no CUDA available on this dev machine, so this only confirms
    *correctness*, not memory savings): `model.encoder.model.grad_checkpointing` is
    `True` when enabled, `False` by default; a 10-step training loop with it enabled
    trains normally and loss decreases.
  - Technical note: `smp`'s `tu-convnext_tiny` encoder goes through timm's
    `features_only=True` wrapper (`FeatureListNet`/`FeatureDictNet`), whose own
    `_collect()` method does the actual checkpointing (wrapping each stage's forward in
    `torch.utils.checkpoint.checkpoint`) -- not the plain `ConvNeXt` class's internal
    per-block `checkpoint_seq`, which is a separate, unused code path in this
    configuration. Both default to **non-reentrant** checkpointing
    (`use_reentrant=False`), which is the DDP-safe mode (historical DDP + checkpointing
    breakage was specifically a reentrant-mode problem) -- confirmed via
    `timm.models._manipulate.use_reentrant_ckpt()`.
- `configs/config_kaggle.yaml`:
  - `model.gradient_checkpointing: true`
  - `data.image_size: 768 -> 1536`
  - `train.batch_size: 12 -> 4` (total across GPUs, ~2/GPU) -- conservative starting
    point. Untested on real hardware: 1536px is 4x the pixels of 768px, and the decoder
    isn't checkpointed, so the actual ceiling with checkpointing enabled is unknown
    until run on Kaggle's T4x2. Drop to 2 if this OOMs; if it fits with headroom, raise
    it back up (this is the first thing to try).

**Status**: not yet run on Kaggle. Needs `git pull` there to pick this up, then:
```
torchrun --nproc_per_node=$(python -c "import torch; print(max(1, torch.cuda.device_count()))") \
    scripts/train.py --config configs/config_kaggle.yaml
```

**Results** (fill in after running):

| image_size | gradient_checkpointing | batch_size | fits? | train_loss floor | val_dice_tm | notes |
|---|---|---|---|---|---|---|
| 768 | no | 12 | yes | ~0.30-0.32 | ~0.65 | baseline (epoch 27 checkpoint, this file's earlier sections) |
| 1536 | yes | 4 (start) | ? | ? | ? | this experiment |

**Decision criteria**:
- Train_loss floor drops meaningfully below ~0.30-0.32 -> resolution was (at least part
  of) the bottleneck. Worth pushing further (try 2048 next, or more headroom at 1536 if
  batch_size=4 leaves memory to spare).
- Train_loss floor stays roughly the same at 1536px -> resolution wasn't the limiting
  factor. Move to a bigger encoder (`convnext_small`/`convnext_base`) instead, per the
  fallback plan -- don't keep pushing resolution or training duration further.
