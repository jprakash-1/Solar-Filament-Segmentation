# Stage 4: Supervised Fine-Tuning Plan (ResNet50 BYOL encoder → segmentation)

This is `PRETRAIN_PLAN.md`'s "Stage 4" and `RESNET_PRETRAIN_PLAN.md` section 10's
handoff target, made concrete: fine-tune the GONG Hα BYOL-pretrained ResNet50
encoder (from the `jp-pretraining-data-prep` branch) on the labeled MAGFiLO
filament data, using `jp-mvp1`'s existing supervised pipeline as the base rather
than building a new one.

---

## 0. Decisions made before writing any code

Two things `PRETRAIN_PLAN.md` explicitly leaves open were resolved by direct
project direction, not assumed:

- **Segmentation head: unchanged from `jp-mvp1`** — `smp.Unet` semantic mask +
  connected-components postprocessing. `PRETRAIN_PLAN.md` §5.1 gates CondInst /
  embedding-head / Mask2Former on an error-analysis measurement that hasn't been
  run yet; building any of those now would be scope beyond what the evidence
  currently justifies. This plan only swaps the **encoder**.
- **Resolution: `--img-size 2048`**, native, no downsampling — an explicit choice,
  not derived from either reference point on record: not the 224px the BYOL
  encoder was pretrained at (`RESNET_PRETRAIN_PLAN.md` §4), and not the 1024px
  `jp-mvp1`'s one real Kaggle run used (`mvp1-solar.ipynb`, val PQ=0.3071 —
  the only real baseline PQ number in this repo). `RESNET_PRETRAIN_PLAN.md` §7
  flags a pretrain/fine-tune resolution mismatch as a BatchNorm-calibration risk
  but treats it as something that resolves itself over a few epochs at the new
  resolution, not as disqualifying — accepted here as a known, documented
  tradeoff. Consequence: a ResNet50 U-Net at full 2048×2048 is far heavier than
  either reference point, so this branch adds mixed precision (`--amp`) and
  ships with a batch size that is a documented placeholder, not a measured
  value (see §5 and `configs/finetune_resnet50_kaggle.yaml`).
- **Branch: forked fresh off `jp-mvp1`**, not off `jp-pretraining-data-prep` —
  this task needs `jp-mvp1`'s working `src/` pipeline as its base.
- **Encoder checkpoint: a Kaggle Dataset, exact slug not yet decided** —
  `finetune_resnet50_kaggle.ipynb` auto-detects it under `/kaggle/input` by
  globbing for a `*.pt` file with `encoder` in its name, same pattern
  `train_mvp1_kaggle.ipynb` already uses for the competition data.

---

## 1. The stem/checkpoint compatibility landmine, and how it's actually avoided

`RESNET_PRETRAIN_PLAN.md` §11 already measured that `resnet50_1ch()`'s stem
averaging (`.mean()` over the 3 ImageNet channel filters) does **not** match
`smp.Unet(encoder_weights="imagenet", in_channels=1)`'s own stem adaptation (smp
**sums**, not averages, and pulls a different pretrained checkpoint entirely —
`smp-hub/resnet50.imagenet` vs. `torchvision.ResNet50_Weights.IMAGENET1K_V2`).

`src/model.py`'s `build_model(encoder_checkpoint=...)` sidesteps this rather than
reconciling it: build the encoder with `encoder_weights=None` (skip smp's stem
adaptation entirely) and `load_state_dict` the BYOL checkpoint directly, so every
encoder weight — stem included — comes from the checkpoint. This only works if
the two encoders' state dicts line up key-for-key, which was verified directly,
not assumed:

- `smp.Unet(encoder_name="resnet50", encoder_weights=None, in_channels=1).encoder`
  is `smp`'s `ResNetEncoder`, which subclasses `torchvision.models.resnet.ResNet`
  and only `del self.fc` / `del self.avgpool` — both parameterless, so their
  absence doesn't create a key mismatch against `resnet50_1ch()` (which keeps
  `avgpool` and sets `fc = nn.Identity()`, also parameterless).
- Measured directly (see `scripts/verify_encoder_checkpoint.py`, and the
  interactive check this plan's implementation ran before trusting it): 318/318
  keys identical, all shapes identical, and a forward pass with synced weights
  produces `max_abs_diff = 0.0` between the plain `resnet50_1ch()` and `smp`'s
  encoder (pooled).
- **A second, separate bug found the same way**: `ResNetEncoder.load_state_dict()`
  pops stray `fc.*` keys before delegating to `nn.Module`'s implementation, but
  doesn't `return` the result — it always returns `None`, regardless of `strict=`
  or real key mismatches. Relying on its return value (as an earlier draft of
  `build_model()` did) raises `TypeError: cannot unpack non-iterable NoneType
  object` instead of ever reaching a missing/unexpected-keys check. Fixed by
  computing the missing/unexpected key sets manually (against the target
  encoder's own `state_dict().keys()`) *before* calling `load_state_dict` at all,
  then calling it with `strict=True` only to catch shape mismatches. Confirmed
  against both a matching and a deliberately-broken dummy checkpoint.

Run `scripts/verify_encoder_checkpoint.py --checkpoint <path>` once against the
real exported checkpoint before trusting a fine-tune build on it — cheap, and
catches the exact class of bug §11 already found once for the ImageNet stem path.

---

## 2. What changed in `src/`

- **`src/model.py`**: `build_model()` gains `encoder_checkpoint`; unchanged
  behavior when omitted (still plain `smp.Unet(encoder_weights="imagenet")`,
  i.e. `jp-mvp1`'s original recipe).
- **`src/dataset.py`**: `FilamentDataset` gains an optional `transform` hook
  (an `albumentations.Compose` applied to image+mask jointly, right after the
  resize) — `None` by default, so `jp-mvp1`'s own behavior is unchanged unless a
  caller opts in.
- **`src/train.py`**: the real extension —
  - `--encoder-name` / `--encoder-checkpoint` / `--encoder-weights` wire into
    `build_model()`.
  - `--loss {bce_dice, tversky_bce}`, default `tversky_bce` —
    `0.5·BCE + Tversky(α=0.3, β=0.7)` per `PRETRAIN_PLAN.md` §5.4, recall-biased
    for thin filament structures. `bce_dice` (the exact `jp-mvp1` loss) stays
    available for direct comparison.
  - Layer-wise LR: `build_param_groups()` builds 5 depth-ordered groups (stem,
    `layer1`–`layer4`) plus a 6th "everything else" (decoder+head) group, per
    §5.5's `layerwise_lr()` formula — `--head-lr` (default `1e-3`) and
    `--encoder-lr-decay` (default `0.75`) control it. Only takes effect when
    `--encoder-checkpoint` is set; otherwise falls back to `jp-mvp1`'s plain
    `Adam(lr)` over all parameters unchanged.
  - Freeze warmup: `--freeze-epochs` (default 3) — `set_encoder_requires_grad()`
    toggles the whole encoder's `requires_grad` at the start of each epoch;
    `--linear-probe-only` forces this to cover the entire run (the §5.6
    forgetting-tripwire baseline — run once, record its best val PQ, compare
    manually against the full fine-tune's).
  - Augmentation: `build_train_transform()` — `RandomRotate90`/`HorizontalFlip`/
    `VerticalFlip` (label-preserving, no canonical "up" on the Sun),
    `RandomGamma`/`RandomBrightnessContrast`/`GaussNoise`/`CoarseDropout`, all
    joint image+mask via the new `FilamentDataset.transform` hook, train split
    only.
  - Mixed precision: `--amp {auto,on,off}` (default `auto`, CUDA-only in
    practice) — `torch.autocast` + `torch.amp.GradScaler`, needed at
    `img_size=2048`.
  - Model selection: checkpointing criterion switched from **val Dice** to
    **val PQ** (§5.6 — Dice/BCE convergence doesn't guarantee good instance
    separation). `validate()` now runs one merged pass per epoch computing
    loss/Dice *and* PQ together (`src/postprocess.mask_to_instances` +
    `src/metrics.panoptic_quality`/`aggregate_pq`, both reused unchanged) — at
    `img_size=2048` the postprocessing "resize to native" step is a no-op, so
    this is exact, not an approximation; at any smaller training resolution it's
    an approximation of the authoritative number `src/infer.py --split val`
    still produces at true native resolution.
  - Checkpoint now persists `encoder_name` (and `loss`), not just `img_size` —
    needed by `src/infer.py` to reconstruct the right architecture.
  - `find_unused_parameters=True` on the DDP wrapper whenever `fine_tuning` is
    True — required because the freeze-warmup schedule changes which
    parameters receive gradients from epoch to epoch, which DDP's default
    (`find_unused_parameters=False`) doesn't tolerate. **Found the hard way**:
    this crashed a real Kaggle T4 x2 run on the very first training step
    (`RuntimeError: Expected to have finished reduction in the prior
    iteration...`, every frozen encoder parameter listed as unused) — fixed
    directly from that traceback, which already named the fix.
  - `--grad-accum-steps` (default 1) — accumulates gradients over N
    micro-batches per optimizer step, recovering a larger effective batch size
    at a smaller `--batch-size`'s peak memory. Under DDP, all but the last
    micro-batch of each cycle runs inside `model.no_sync()` so gradients only
    all-reduce once per optimizer step, not once per micro-batch (PyTorch's own
    documented pattern, not novel) — this is the lever `configs/finetune_resnet50_kaggle.yaml`
    now recommends reaching for on OOM instead of lowering `--batch-size`
    outright.
  - `--early-stopping-patience` (default 0, disabled) — stops training if val
    PQ hasn't improved for that many consecutive epochs. Only rank 0 runs
    validation, so its stop decision is broadcast to every rank before any rank
    acts on it (`dist.broadcast` after the existing barrier) — the same
    per-rank-independent-decision hazard this file already avoided for the
    freeze-warmup fix and that `PRETRAIN_PLAN.md`'s `train_ddp.py` sketch flags
    for wall-clock-based stopping; letting rank 0 break out alone would hang
    every other rank at its next collective op.
- **`src/infer.py`**: `load_model()` reads `encoder_name` back out of the
  checkpoint (defaults to `"resnet18"` for older `jp-mvp1` checkpoints saved
  before this field existed) and builds with `encoder_weights=None` (every
  weight gets overwritten by the checkpoint anyway, so there's no reason to
  fetch ImageNet weights first — same reasoning `export_encoder.py` already
  uses on the pretraining branch).

---

## 3. Verification performed

- `scripts/verify_encoder_checkpoint.py` run both without `--checkpoint`
  (structural-only: random-init `resnet50_1ch()` vs. `smp`'s `resnet50` encoder)
  and with a dummy checkpoint shaped like a real `export_encoder.py` output —
  both pass; a deliberately-broken checkpoint (one missing key) correctly fails
  loudly instead of silently loading a partial encoder.
- Full `src/train.py` → checkpoint → `src/infer.py` round trip run end-to-end
  against a synthetic 8-image COCO dataset (small resolution, CPU, 2 epochs,
  `--encoder-name resnet50 --encoder-checkpoint <dummy>`), confirming: the
  encoder loads, the freeze→unfreeze transition happens at the right epoch (
  logged as `[encoder frozen]`), layer-wise LR groups compute to the expected
  values, val PQ is computed and used for checkpoint selection, and
  `src/infer.py` reconstructs the resnet50 architecture from the checkpoint's
  `encoder_name` field correctly for both `--split val` and `--split test`.
- `--grad-accum-steps` verified on a synthetic dataset sized to force a
  trailing partial accumulation cycle (7 train images, batch-size 2,
  grad-accum-steps 3 → micro-batch sizes `[2,2,2,1]`) — runs cleanly across
  epochs. `--early-stopping-patience` verified to stop training at the correct
  epoch (patience=1 with a frozen/non-improving run stopped after 2 epochs of a
  10-epoch budget, logging why).
- **A second bug found while re-testing the notebook after these additions**:
  `finetune_resnet50_kaggle.ipynb`'s dependency-install cell had `\\b` (two
  literal backslashes) instead of `\b` (a regex word boundary) in its
  `grep -v -E "^(torch|torchvision)\b"` pattern — a Python string-escaping slip
  in how this file generated the notebook, not present in `jp-mvp1`'s original
  `train_mvp1_kaggle.ipynb` this was modeled on. With the extra backslash, the
  pattern would have matched `torchmetrics` too (no working word-boundary),
  silently skipping its install. Fixed by copying the exact, already-proven
  cell source from `train_mvp1_kaggle.ipynb` and confirming the corrected
  pattern excludes exactly `torch`/`torchvision` while keeping `torchmetrics`.
- **Not yet run**: anything at real scale (real MAGFiLO data, real BYOL
  checkpoint, real GPU) — both currently exist only on Kaggle (data) and either
  on Kaggle or not yet at all (checkpoint; the BYOL pretraining run's own
  completion status is unconfirmed as of this plan). `finetune_resnet50_kaggle.ipynb`
  is built to accept both via auto-detection; it hasn't been executed.

---

## 4. Open items

- [ ] Run `scripts/verify_encoder_checkpoint.py --checkpoint <real path>` once
      the real exported BYOL encoder checkpoint exists, before the first real
      fine-tune.
- [ ] Measure real batch size / throughput at `img_size=2048` on actual Kaggle
      GPU hardware — `configs/finetune_resnet50_kaggle.yaml`'s `batch_size: 2` is
      a conservative placeholder, not a measured value. Gradient checkpointing
      is the next lever if batch-size-1 with AMP still OOMs; not implemented
      preemptively.
- [ ] Watch for the early-epoch BatchNorm mismatch `RESNET_PRETRAIN_PLAN.md` §7
      implies (pretrained at 224px, fine-tuned at 2048px) — expected to resolve
      itself within a few epochs; check the val PQ curve for it rather than
      assuming it's fine.
- [ ] Run the three-way ablation `RESNET_PRETRAIN_PLAN.md` §10 asks for (BYOL
      resnet50 vs. plain-ImageNet resnet50 vs. `jp-mvp1`'s resnet18 baseline,
      same head/loss/schedule/resolution) once a real checkpoint and real GPU
      time are available — see `configs/finetune_resnet50_kaggle.yaml`'s
      `ablation:` section for the exact three runs.
- [ ] `PRETRAIN_PLAN.md` §5.1's error-analysis measurement (failure-type × SQ/RQ
      breakdown) is still not done — this fine-tune is explicitly "the ResNet
      baseline with the *current* head," not a claim that semantic-mask+CC is
      the final architecture.
