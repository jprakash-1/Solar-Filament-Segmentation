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

**Status update (2026-08-22, mid-run)**:
- `batch_size=4` fit with huge headroom (4.3GB/15GB per GPU at 100% utilization) --
  raised to `batch_size=12` (~6/GPU), matching the old 768px known-good value.
- Hit a real bug on the first run: `val_loader` was sized with `global_batch_size`
  instead of `per_device_batch_size`. Validation runs on rank 0 alone (not split across
  ranks), so it was trying 2x the per-GPU batch that training just proved fits on a
  single GPU -- OOM'd immediately on the first validation batch. This bug existed since
  DDP was introduced (the two batch sizes are identical in non-distributed mode, so it
  never showed up locally, and 768px had enough headroom to mask it there too). Fixed in
  `scripts/train.py`.
- Epoch-20 comparison (same epoch count both runs): `train_loss` 0.3294 (1536px) vs
  0.3406 (768px) -- a modest ~3% improvement, but **not yet conclusive**: the 1536px
  run is still trending down (not plateaued) and its LR is still ~56% of peak at epoch
  20, versus the 768px comparison which required the LR fully decayed (epoch 35) to
  confirm the plateau. Need to let this run reach a similarly LR-decayed point before
  making the real call.
- Postprocessing re-tuned for this resolution: `--sweep-watershed-min-distance
  "40,60,80,100,120,140"` against an early/mid-training checkpoint (~epoch 12) found PQ
  still climbing at the top of the range (140 best, 0.3364) -- same pattern as the 768px
  sweep. Decided **not** to chase the exact peak further right now since this checkpoint
  isn't converged yet (the optimum may shift once training finishes) -- locked in
  `watershed_min_distance=140`, `min_instance_area=40` as the new config defaults
  (`configs/config_kaggle.yaml`), matching the 768px-tuned values. Full metrics at this
  setting on the ~epoch-12 checkpoint: Dice=0.6735, IoU=0.5223, PQ=0.3364,
  mAP@[.5:.95]=0.1135, AP@0.50=0.3574, AP@0.75=0.0261 -- close to the 768px baseline
  already, on a checkpoint that isn't even converged yet. Re-sweep once training
  finishes and the final best checkpoint is available.

**Results** (fill in as the run progresses):

| image_size | gradient_checkpointing | batch_size | fits? | epoch | train_loss | val_dice_tm | notes |
|---|---|---|---|---|---|---|---|
| 768 | no | 12 | yes | 27 (of ~35) | ~0.329 (plateaued) | ~0.66 | baseline, LR fully decayed |
| 1536 | yes | 12 | yes | 20 (of 40) | 0.3294 (still dropping) | 0.640 | in progress, LR still ~56% of peak |

**Decision criteria** (apply once the 1536px run's LR has fully decayed, not before):
- Train_loss floor drops meaningfully below ~0.30-0.32 -> resolution was (at least part
  of) the bottleneck. Worth pushing further (try 2048 next, or more headroom at 1536 if
  there's memory to spare).
- Train_loss floor stays roughly the same at 1536px -> resolution wasn't the limiting
  factor. Move to a bigger encoder (`convnext_small`/`convnext_base`) instead, per the
  fallback plan -- don't keep pushing resolution or training duration further.
