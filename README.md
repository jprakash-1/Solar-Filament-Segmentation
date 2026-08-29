# Solar Filament Segmentation Challenge 2026 — MVP1

Kaggle competition: [Solar Filament Segmentation Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026)
Task: class-agnostic instance segmentation of solar filaments in GONG H-Alpha
full-disk imagery (2048×2048, 8-bit grayscale).

## MVP1 goal

**Not about score.** MVP1 proves every interface boundary in the pipeline is
correct while the cost of being wrong is cheap:

```
raw JPEG + COCO JSON -> Dataset/DataLoader -> tiny/fast model -> raw prediction
  -> postprocess (instance extraction, resize back to 2048x2048)
  -> RLE (pycocotools) encode -> submission.csv -> validate
```

## Repo layout

```
configs/
  mvp1.yaml                # local settings record (documentation, not live-loaded)
  mvp1_kaggle.yaml           # Kaggle GPU settings record (bigger batch/epochs, same architecture)
src/
  dataset.py             # COCO parsing, group-aware split, FilamentDataset
  model.py                # tiny U-Net (segmentation_models_pytorch)
  train.py                 # training loop (python -m src.train)
  infer.py                  # U-Net inference -> instances -> submission (python -m src.infer)
  postprocess.py             # prob map -> per-instance masks (upsample-then-threshold-then-CC)
  rle_utils.py                # mask <-> RLE (counts-only) helpers
  submission.py                # build + validate submission.csv
  metrics.py                    # local Dice + Panoptic Quality
scripts/
  baseline_classical.py    # zero-training CV baseline (Option A)
notebooks/
  00_eda.ipynb              # Step-0 sanity check -- run before trusting anything else
outputs/
  checkpoints/, logs/, submissions/   # gitignored except .gitkeep
train_mvp1_kaggle.ipynb   # repo root, not notebooks/ -- this is the file you upload to Kaggle directly.
                          # Clones this repo + runs src/train.py & src/infer.py on a Kaggle GPU.
```

## Quickstart

```bash
source .venv/bin/activate

# Step 0: sanity-check the data before trusting any downstream code
jupyter nbconvert --to notebook --execute --inplace notebooks/00_eda.ipynb

# Option A: classical CV baseline, zero training, first submission
python scripts/baseline_classical.py --split val    # local PQ check
python scripts/baseline_classical.py --split test   # writes outputs/submissions/baseline_classical.csv

# Option B: tiny U-Net
python -m src.train --epochs 10
python -m src.infer --checkpoint outputs/checkpoints/mvp1_unet.pt --split val   # local PQ check
python -m src.infer --checkpoint outputs/checkpoints/mvp1_unet.pt --split test  # writes outputs/submissions/mvp1_unet.csv

# Validate any submission independently before uploading
python -m src.submission --validate outputs/submissions/<file>.csv
```

### Training on Kaggle instead of locally

Local training runs on CPU/MPS, which is slow enough to matter for iteration speed.
`train_mvp1_kaggle.ipynb` (repo root -- this is the file to upload to Kaggle
directly) clones this repo's `jp-mvp1` branch inside a Kaggle kernel and runs the
identical `src/train.py` / `src/infer.py` on a real GPU (settings in
`configs/mvp1_kaggle.yaml`). **Requires `jp-mvp1` to be pushed to `origin` first** —
the notebook's `git clone -b jp-mvp1 ...` step will fail otherwise. Upload it to
Kaggle, edit/attach the competition dataset there, then run; it has not been
executed in this environment (no `/kaggle/input` mount or GPU available locally to
test against).

## Definition of done for MVP1

- [x] `notebooks/00_eda.ipynb` runs end-to-end, confirms 2048x2048 images, correct
      polygon->mask axis alignment, and the train/test filename schemes.
- [x] `scripts/baseline_classical.py` produces a validated `submission.csv` with zero
      training (proves postprocess -> RLE -> submission format independent of any
      model bug).
- [x] `src/train.py` + `src/infer.py` run end-to-end (tiny U-Net, `img_size=256`,
      few epochs) and produce a second validated `submission.csv`.
- [x] `src/submission.py --validate` passes on both submissions with zero errors
      (unique `filament_id`s, every RLE round-trips to `(2048, 2048)` and is
      non-empty); a handful of test images with zero predicted rows is reported as
      a warning, not a failure, since that's legitimate for a genuinely
      filament-free frame.
- [x] Local Dice/PQ (`src/metrics.py`) computed on a `file_name`-grouped held-out
      split for both approaches.
- [ ] Both submissions uploaded to Kaggle and leaderboard PQ compared against local
      PQ (not yet done in this environment -- requires Kaggle credentials/upload,
      a user action).

Everything beyond this list (augmentation, LR scheduling, k-fold CV, higher
resolution, instance-aware architectures, `spine`-based auxiliary supervision) is
explicitly post-MVP1 iteration -- see `MVP1_PLAN.md` section 6.

## Known gotchas already hit and fixed here

- **`image_id` is a string, not an int, and duplicate-annotates**: the same
  underlying JPEG can appear under multiple `image_id`s (one per independent
  annotator pass, e.g. `010101-...` / `010102-...`). `src/dataset.group_split`
  splits by `file_name`, never by `image_id`, to avoid leaking the same pixels
  across train/val under a different annotator's polygons.
- **`pycocotools.coco.getAnnIds(imgIds=<bare string>)` silently returns `[]`** —
  `_isArrayLike` treats a string as already array-like (it has `__iter__` and
  `__len__`), so it iterates character-by-character instead of treating the id as
  one value, and every character-lookup misses. Always call
  `getAnnIds(imgIds=[image_id])` — a one-element **list**, never a bare string. Hit
  and fixed everywhere in `src/`; documented with a live before/after demo in
  `notebooks/00_eda.ipynb`.
- **RLE discipline**: always `pycocotools.mask.encode` a mask at native 2048x2048 —
  never resize a mask after encoding, and never resize the *binary* mask instead of
  the *probability* map before thresholding (blocky artifacts, spurious merges,
  directly hurts PQ's RQ term). See `src/postprocess.py`.
- **Classical baseline over-detection**: Otsu thresholding on a
  background-flattened residual let granulation texture bridge into one 300k+ pixel
  blob per image and ~3000+ spurious tiny components, which also made a naive
  O(n_gt × n_pred) full-resolution PQ computation pathologically slow. Fixed by (a)
  thresholding at a fixed percentile of the residual distribution (grounded in the
  known ~0.4-0.5% foreground fraction from prior EDA) instead of Otsu, and (b)
  making `src/metrics.py`'s IoU matrix bounding-box-restricted so it stays fast
  regardless of how noisy a model's predictions are.
