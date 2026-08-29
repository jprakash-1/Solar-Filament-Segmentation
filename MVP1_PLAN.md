# Solar Filament Segmentation Challenge 2026 — MVP1 Implementation Plan

## 0. Framing

MVP1 is **not** about score. It is about proving every interface boundary in the pipeline is correct while the cost of being wrong is cheap:

```
raw JPEG + COCO JSON  →  Dataset/DataLoader  →  tiny/fast model  →  raw prediction
     →  postprocess (instance extraction, resize back to 2048×2048)
     →  RLE (pycocotools) encode  →  submission.csv  →  upload  →  leaderboard score
```

Every arrow above is a place where MVP1 will silently break in ways that only surface at submission time: wrong axis order in RLE, mismatched `filament_id` scheme, wrong image size, off-by-one in resize, forgetting `iscrowd`/Fortran-order requirements of `pycocotools`, etc. The goal for day 1–2 is a **correct, low-score submission**, not a good one. Everything below is organized to hit that milestone first, then leaves clear hooks for iterating toward the 0.3–0.7 PQ range other teams are already showing on the public leaderboard.

---

## 1. What is actually being scored

### 1.1 Task type
This is **class-agnostic instance segmentation**, not semantic segmentation. The submission format requires one row per predicted *filament instance* (`filament_id`, `segmentation_rle`), not one mask per image. A plain U-Net that outputs a single binary "filament vs. background" mask is **not sufficient by itself** — its output has to be split into distinct instances (e.g. via connected components, or via an inherently instance-aware architecture like Mask R-CNN / YOLO-seg) before it can be submitted.

The four annotation categories (`Left`, `Right`, `Unidentifiable`, `Ambiguous`) describe filament *chirality/type*, not something the scoring appears to require in the submission schema (only `filament_id` + `segmentation_rle` are submitted, no category). Treat classification as out of scope for MVP1; it may matter later only if the organizers weight the qualitative rubric by class-correctness (not evidenced in the schema).

### 1.2 Metrics
Two metrics are named:

- **Mean Dice score** — `torchmetrics.segmentation.DiceScore`, computed per matched pair (or per image, then averaged — replicate via the self-eval notebook to be sure of the exact reduction they use). Dice = `2|A∩B| / (|A|+|B|)`.
- **Panoptic Quality (PQ)** — the primary ranking metric, from Kirillov et al. 2019 (`10.1109/CVPR.2019.00963`):

```
PQ(Y, Ŷ) = [ Σ_{(y,ŷ)∈TP} IoU(y, ŷ) ] / ( |TP| + 0.5|FP| + 0.5|FN| )
```

where:
- `Y` = ground-truth instance set, `Ŷ` = predicted instance set (per image).
- A pair `(y, ŷ)` is a **TP** iff `IoU(y, ŷ) > 0.5` (standard PQ convention — the >0.5 threshold guarantees a unique matching, i.e. no GT segment can match more than one prediction and vice versa).
- **FN** = unmatched GT instances, **FP** = unmatched predicted instances.
- `IoU(y, ŷ) = Σ(y ⊙ ŷ) / Σ(y ⊕ ŷ ⊖ y⊙ŷ)` — standard pixel IoU.

PQ decomposes into `PQ = SQ × RQ` where `SQ` (segmentation quality) = mean IoU over TPs only, and `RQ` (recognition quality) = `|TP| / (|TP| + 0.5|FP| + 0.5|FN|)` — essentially an F1 score over instance detection. **This means fragmenting one filament into several predicted blobs, or merging several filaments into one blob, is directly penalized** even if pixel-level Dice looks fine — each extra/missing blob counts as a full FP/FN in the denominator. This is explicitly called out in the competition's evaluation write-up ("penalties related to fragmentation and over-merging, i.e., one-to-many and many-to-one correspondences").

- **Rubric weighting** (per Overview → Evaluation): 70% quantitative (mean Dice, PQ, Dice/IoU distributions, one-to-many/many-to-one distribution) + 30% qualitative (pipeline write-up, visual morphology quality, code quality/modularity/documentation) — the qualitative component only kicks in for the **final judged submission** (top teams), not the live leaderboard, which is PQ-only. Public leaderboard score is announced as "any PQ > 0.3 is of great value" (Aug 20 announcement) — so don't over-index on chasing marginal PQ gains during MVP stage; getting a *working, honestly-scored* pipeline matters more early on.

- **Public/private split**: leaderboard shows Dice on ~50% of test images; the other ~50% is private/organizer-only. Design local validation so public and private-style scores should track each other (i.e., don't overfit to quirks of whichever half happens to be public).

### 1.3 Submission file contract
```
filament_id,segmentation_rle
20150125172714Mh_1,"f8uSDds ... VQNC"
20150125172714Mh_2,"KHT%$HD ... 9>km"
```
- One row per **predicted filament instance**, not per image. Zero, one, or many rows per test image are all valid — driven entirely by how many instances your model finds.
- `filament_id` = `{image_stem}_{k}` where `k` is any tail that makes rows for the same image unique (doesn't have to be sequential/contiguous — "As long as the tail strings render the rows unique, and the image id remains unchanged, your filament id is acceptable"). Simplest: 1-indexed per image.
- `segmentation_rle` = **RLE *counts* only** (COCO RLE `counts` field, not the full `{size, counts}` dict) — **do not** wrap in quotes yourself (pandas/csv will quote as needed for the comma-containing string; don't double-quote). Size is fixed and implicit: 2048×2048 for every image — this is critical, because if your working resolution during training/inference is different, you **must** resize/upsample the binary mask back to 2048×2048 *before* running `pycocotools.mask.encode`, not resize the RLE after the fact.
- Matching between predicted and GT instances at scoring time is by **overlap**, not by row order or `filament_id` value: "the number of ground-truth segmentations may also be different from the number of predicted ones... matched based on their actual overlap, not their index." So `filament_id` naming is purely for uniqueness — no need to try to align IDs with anything in the training JSON.
- Use `pycocotools.mask` directly: `encodeMask`/`encode` for producing RLE from a binary mask (`fortran_order`-contiguous `uint8` array, i.e. `np.asfortranarray(mask)`), and `decodeMask`/`decode` for the round-trip sanity check. `annToMask` on an annotation dict converts polygon → binary mask, which is exactly what's needed to build GT masks for local metric computation from the training JSON.

### 1.4 Self-evaluation notebook
Organizers published `https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook` (Aug 9 update) with reference Dice/PQ implementations and plots. **Read/fork this before writing a custom local metric** — matching their exact TP/FP/FN bookkeeping (especially the IoU>0.5 tie-breaking rule and how empty predictions/empty GT images are scored) avoids a local-vs-leaderboard score mismatch that would otherwise burn submission budget (5/day) debugging.

---

## 2. Data

### 2.1 Layout (after unzip)
```
MAGFiLO_1.0_Kaggle_2026/
├── train/
│   ├── train_images/*.jpeg                                    # 2048×2048, 8-bit grayscale JPEG
│   └── MAGFiLO_1.0_Annotations_kaggle2026_train.json           # COCO-style
└── test/
    └── test_images/*.jpeg
```
888 files total, 750.97 MB, types `{jpeg, json}` — a genuinely small dataset (train+test images likely in the low hundreds), which is good news for MVP1 iteration speed on CPU/single-GPU.

### 2.2 Filename semantics
`YYYYMMDDHHMMSSII.jpeg` — capture timestamp + 2-letter instrument code (e.g. `Bh` = Big Bear). Not strictly needed for MVP1, but useful later for temporal train/val splitting or instrument-based stratification (different GONG stations may have different noise/PSF characteristics).

### 2.3 Annotation JSON (COCO-style)
```python
{
  "info": {...}, "licenses": [...], "categories": [...],
  "images": [ {"id": str, "width": int, "height": int, "file_name": str, ...} ],
  "annotations": [
    {
      "id": str,                       # uuid, unique per filament instance
      "image_id": str,                 # FK into images[].id
      "category_id": int,              # 1=Left,2=Right,3=Unidentifiable,4=Ambiguous
      "segmentation": [[x0,y0,...,xn,yn]],  # single polygon, closed (first==last point)
      "area": float,                   # polygon area via pycocotools, not bbox area
      "spine": [x0,y0,...],            # filament centerline, NOT used for MVP1
      "bbox": [x,y,w,h],
      "iscrowd": 0                     # always 0 here
    }
  ]
}
```

**Critical gotcha #1 — `image_id` is a string, not an int**, and encodes *both* an annotator/batch id *and* the filename: e.g. `010401-20160920230134Lh` (annotator batch `010401` for image `20160920230134Lh.jpg`). The same physical image can appear multiple times under different `image_id`s, one per independent annotator pass (e.g. `010101-...` and `010102-...`). **Treat each `(annotator_batch, image)` pair as a fully independent training sample** — do not deduplicate by `file_name`, and do not let two rows sharing the same underlying JPEG land on opposite sides of a train/val split (that's leakage: the model would see the same pixels in both train and val, just with a different annotator's polygons). Group-aware splitting key = `file_name` (or the timestamp+instrument suffix of `image_id`), not `image_id` itself.

**Critical gotcha #2** — polygon `segmentation` is a *single*-polygon list (`[[...]]` with exactly one inner list per annotation, never multiple disjoint pieces), and `iscrowd` is always 0, so `pycocotools.mask.frPyObjects` / `annToMask`-style conversion is straightforward (no crowd-RLE branch to worry about).

### 2.4 No official train/val split
Only `train/` and `test/` are provided — validation must be carved out of `train/` manually. Recommended MVP1 split: `GroupShuffleSplit` (scikit-learn) grouped by `file_name`, ~85/15 train/val, stratified loosely by instrument code if time allows. This directly prevents the annotator-duplication leakage above.

---

## 3. MVP1 architecture

### 3.1 Repo layout
```
filament-mvp1/
├── configs/mvp1.yaml
├── data/                      # gitignored — unzipped competition data lives here
├── src/
│   ├── dataset.py              # COCO parsing, polygon→mask, PyTorch Dataset
│   ├── model.py                 # tiny U-Net (or classical baseline)
│   ├── train.py                 # training loop
│   ├── infer.py                  # run model over test/, produce raw masks
│   ├── postprocess.py            # connected components → per-instance masks
│   ├── rle_utils.py              # mask ⇄ RLE helpers (pycocotools wrappers)
│   ├── submission.py             # assemble + validate submission.csv
│   └── metrics.py                # local Dice + PQ (mirrors self-eval notebook)
├── notebooks/
│   └── 00_eda.ipynb              # sanity-check image/mask overlays before training anything
├── outputs/
│   ├── checkpoints/
│   └── submissions/
├── requirements.txt
└── README.md
```

### 3.2 Step 0 — EDA sanity notebook (before any code that trains)
Load a handful of `(image, annotations)` pairs, rasterize polygons with `pycocotools.mask.frPyObjects(seg, h, w)` → `decode`, overlay on the JPEG, and eyeball it. This single step catches axis-order bugs (`(x,y)` vs `(row,col)`), off-by-one errors, and confirms whether the sun disk fills the frame or sits inside a black border (affects whether to crop/mask background before feeding the model). Also tabulate: instances per image (distribution), image count per split, area distribution of filaments (many are likely small/thin — relevant for choosing loss and eval thresholds).

### 3.3 Step 1 — Dataset / DataLoader (`src/dataset.py`)
```python
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from torch.utils.data import Dataset
import numpy as np, cv2, torch

class FilamentDataset(Dataset):
    def __init__(self, coco_json, img_dir, image_ids, img_size=256, train=True):
        self.coco = COCO(coco_json)          # pycocotools handles COCO-style JSON directly
        self.img_dir = img_dir
        self.ids = image_ids                  # pre-filtered/split list of image_ids
        self.img_size = img_size
        self.train = train

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image_id = self.ids[idx]
        info = self.coco.imgs[image_id]
        img = cv2.imread(f"{self.img_dir}/{info['file_name']}", cv2.IMREAD_GRAYSCALE)
        h, w = img.shape                      # should be 2048, 2048 — assert it

        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)
        semantic_mask = np.zeros((h, w), dtype=np.uint8)
        for ann in anns:
            m = self.coco.annToMask(ann)      # polygon -> binary mask at native (h,w)
            semantic_mask = np.maximum(semantic_mask, m)

        # MVP1: downsample aggressively for speed; remember the resize factor
        img_rs = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        mask_rs = cv2.resize(semantic_mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        img_t = torch.from_numpy(img_rs).float().unsqueeze(0) / 255.0
        mask_t = torch.from_numpy(mask_rs).float().unsqueeze(0)
        return img_t, mask_t, image_id, (h, w)   # keep original (h,w) for inverse-resize at inference
```
Notes:
- `self.coco.annToMask(ann)` is the load-bearing call — it wraps `frPyObjects` + `decode` correctly for single-polygon, `iscrowd=0` annotations, so don't hand-roll polygon rasterization.
- MVP1 collapses all instances into **one binary semantic mask** per image at train time (simplest possible target). Instance separation happens at *postprocessing* time (§3.6), not in the loss. This is the fastest path to a working submission; it is also exactly the approach several public notebooks are already using (U-Net + connected components / U-Net + crop-and-refine), so it's a legitimate longer-term direction too, not just scaffolding.
- Treat images with zero annotations as valid negatives (empty mask) — don't filter them out; the model needs to learn what "no filament" looks like, and the test set will contain images that produce zero predicted rows.

### 3.4 Step 2 — Tiny/fast model (`src/model.py`)
Two legitimate choices for "tiny/fast" — pick based on how many hours are budgeted for MVP1:

**Option A (fastest, ~10 min, no training): classical CV baseline.** Sun disk is roughly circular and bright; filaments are dark, elongated regions on the disk. Otsu or adaptive threshold on disk-masked pixels + morphological opening/closing + `cv2.connectedComponentsWithStats` directly gives instances with **zero training code**. This is the single fastest way to validate the *entire* postprocess → RLE → submission.csv → upload path today, decoupled from any model-training bugs. Strongly recommended as the literal first submission (Day 1, hour 1–2), even before the U-Net exists.

**Option B (still tiny, ~20–40 min training on GPU, better ceiling): 4-level U-Net, small channel width, from scratch or via `segmentation_models_pytorch`** (`smp.Unet(encoder_name="resnet18", encoder_weights="imagenet", in_channels=1, classes=1)` — trivial to swap encoder later once MVP1 is validated). At `img_size=256` and a few hundred images, a handful of epochs on a single GPU (or even CPU) finishes in minutes. This is "tiny" in both parameter count and wall-clock, by design — the point is pipeline validation, not accuracy.

Do **both**: ship Option A as literally the first submission (proves the format end-to-end fastest), then Option B as the second submission the same day/next day (proves the DataLoader → training loop → checkpoint → inference path). This directly satisfies the "get a submission uploaded on day one or two, even a bad one" instruction with two independent, cheap checkpoints instead of one all-or-nothing path.

### 3.5 Step 3 — Training loop (`src/train.py`)
- Loss: `BCEWithLogitsLoss + DiceLoss` (sum or weighted sum) — Dice component directly targets the competition's own metric family and helps a lot with the class imbalance (filaments are a small fraction of disk pixels).
- Optimizer: Adam, lr `1e-3`, few epochs (5–15) at `img_size=256` — this should run in single-digit minutes.
- Track: train/val loss, val Dice (via `torchmetrics.segmentation.DiceScore` — same class the competition itself cites, so numbers are directly comparable) every epoch.
- Checkpoint: save best-val-Dice weights to `outputs/checkpoints/mvp1_unet.pt`.
- **Explicitly not in scope for MVP1**: augmentation beyond maybe a flip, LR scheduling, multi-fold CV, TTA, mixed precision tuning — all of that is post-MVP1 iteration once the interfaces are trusted.

### 3.6 Step 4 — Inference + postprocessing (`src/infer.py`, `src/postprocess.py`)
```python
def predict_instances(model, img_2048_gray, img_size=256, prob_thresh=0.5, min_area_px=15):
    h, w = img_2048_gray.shape                       # = (2048, 2048)
    img_rs = cv2.resize(img_2048_gray, (img_size, img_size), interpolation=cv2.INTER_AREA)
    x = torch.from_numpy(img_rs).float().unsqueeze(0).unsqueeze(0) / 255.0
    with torch.no_grad():
        prob = torch.sigmoid(model(x))[0, 0].numpy()
    prob_full = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)   # upsample BEFORE thresholding
    binary_full = (prob_full > prob_thresh).astype(np.uint8)              # now at native 2048x2048

    n_labels, labels = cv2.connectedComponentsWithStats(binary_full, connectivity=8)[:2]
    instance_masks = []
    for lbl in range(1, n_labels):                    # 0 = background
        inst = (labels == lbl).astype(np.uint8)
        if inst.sum() >= min_area_px:                 # drop speckle noise
            instance_masks.append(inst)
    return instance_masks   # list of (2048,2048) uint8 binary masks, one per predicted filament
```
Key discipline: **upsample the probability map to 2048×2048 first, then threshold, then connected-component-label** — thresholding at low res and upsampling the *binary* mask instead introduces blocky artifacts and merges nearby filaments that shouldn't be merged, which directly hurts PQ's RQ term (over-merging).

`min_area_px` is a cheap first lever against the "background noise → tiny spurious components" failure mode called out in the competition description; tune it against local PQ rather than guessing.

### 3.7 Step 5 — RLE + submission assembly (`src/rle_utils.py`, `src/submission.py`)
```python
from pycocotools import mask as maskUtils
import numpy as np, pandas as pd

def mask_to_rle_counts(binary_mask: np.ndarray) -> str:
    fortran_mask = np.asfortranarray(binary_mask.astype(np.uint8))
    rle = maskUtils.encode(fortran_mask)          # {'size': [h,w], 'counts': bytes}
    return rle['counts'].decode('utf-8')           # submission wants counts only, as text

def build_submission(image_id_to_instances: dict, out_path="outputs/submissions/mvp1.csv"):
    rows = []
    for file_stem, instance_masks in image_id_to_instances.items():
        for k, inst_mask in enumerate(instance_masks, start=1):
            rows.append({
                "filament_id": f"{file_stem}_{k}",
                "segmentation_rle": mask_to_rle_counts(inst_mask),
            })
    df = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    df.to_csv(out_path, index=False)
    return df
```
`file_stem` = test image filename without extension (e.g. `20150125172714Mh`) — **not** the training set's `annotator_batch-filename` `image_id` scheme, since test images presumably don't carry a batch prefix (confirm against an actual `test_images/` filename during Step 0 EDA).

**Pre-submission validation checklist** (script this — `src/submission.py --validate`):
1. Every `filament_id` unique.
2. Every image in `test_images/` appears with ≥0 rows (fine to have 0 for genuinely filament-free frames — just don't *accidentally* drop images due to a globbing bug).
3. Round-trip each RLE: `maskUtils.decode({'size':[2048,2048],'counts': row.encode()})` and confirm shape `(2048,2048)` and non-empty (`.sum() > 0`) — catches encoding corruption before burning one of the 5 daily submissions.
4. No stray quote characters or newlines inside the `segmentation_rle` string (pandas' default CSV quoting handles the comma-containing RLE string correctly — don't hand-add quotes).
5. Spot-check 2–3 rows by re-decoding and overlaying on the source JPEG.

### 3.8 Step 6 — Local metrics (`src/metrics.py`)
Implement Dice + PQ against the held-out validation split (never against test — no local GT for test), matching the self-evaluation notebook's conventions as closely as possible:
```python
def panoptic_quality(gt_masks: list[np.ndarray], pred_masks: list[np.ndarray], iou_thresh=0.5):
    if not gt_masks and not pred_masks:
        return 1.0  # or exclude from mean per organizer convention — verify against self-eval notebook
    iou_matrix = np.zeros((len(gt_masks), len(pred_masks)))
    for i, g in enumerate(gt_masks):
        for j, p in enumerate(pred_masks):
            inter = np.logical_and(g, p).sum()
            union = np.logical_or(g, p).sum()
            iou_matrix[i, j] = inter / union if union > 0 else 0.0

    matched_gt, matched_pred, tp_ious = set(), set(), []
    # greedy matching is fine since IoU>0.5 already guarantees uniqueness in well-formed cases
    gi, pj = np.where(iou_matrix > iou_thresh)
    for i, j in sorted(zip(gi, pj), key=lambda t: -iou_matrix[t]):
        if i in matched_gt or j in matched_pred:
            continue
        matched_gt.add(i); matched_pred.add(j); tp_ious.append(iou_matrix[i, j])

    tp, fp, fn = len(tp_ious), len(pred_masks) - len(matched_pred), len(gt_masks) - len(matched_gt)
    denom = tp + 0.5 * fp + 0.5 * fn
    return (sum(tp_ious) / denom) if denom > 0 else 1.0
```
Run this over the validation split after every training run, **before** spending a daily submission slot. Treat leaderboard PQ as a confirmation of local PQ, not the primary feedback signal — 5 submissions/day is scarce relative to iteration speed on ~a few hundred local images.

---

## 4. Day-by-day MVP1 checklist

**Day 1**
1. Download + unzip data; run EDA notebook (polygon overlay sanity check; confirm image size, disk framing, filename schemes for train vs test).
2. Implement `dataset.py`; visually confirm a batch of `(image, mask)` pairs.
3. Ship **Option A (classical CV baseline)** end-to-end: threshold → connected components → RLE → `submission.csv` → validate → upload. This is the first leaderboard signal and, more importantly, the first proof the *format* is accepted.
4. Fork/read the organizers' self-evaluation notebook; port its Dice/PQ logic into `metrics.py`.

**Day 2**
5. Implement tiny U-Net (`model.py`) + `train.py`; train a few epochs at 256×256.
6. Implement `infer.py` + `postprocess.py` (resize-then-threshold-then-CC discipline from §3.6).
7. Regenerate `submission.csv` from the U-Net path; validate locally (Dice/PQ on held-out split) before uploading.
8. Upload; compare local PQ vs. leaderboard PQ — a large gap flags a bug in local metric computation or in the train/val split (leakage), not necessarily a bad model.
9. Write down in the repo README exactly what "MVP1 done" means (this checklist) so later iteration work is clearly scoped as post-MVP.

**Definition of done for MVP1**: two accepted submissions on the leaderboard (baseline CV + tiny U-Net), a reproducible local Dice/PQ number that's in the right ballpark relative to the leaderboard, and every module in `src/` runnable end-to-end via a single `python -m src.infer && python -m src.submission` style entrypoint.

---

## 5. Known risk list (things likely to bite specifically in this competition)

- **Annotator-duplicated images causing train/val leakage** (§2.3) — split by `file_name`, not `image_id`.
- **Thresholding before upsampling** instead of after — merges/distorts instances, directly hurts PQ (§3.6).
- **RLE size mismatch** — always encode at native 2048×2048; never submit RLE for a resized mask.
- **Empty-prediction / empty-GT images** — decide (and match the organizers' convention) how these contribute to PQ/Dice aggregation; don't let a naive implementation silently divide by zero or silently exclude them in a way that diverges from the leaderboard's computation.
- **Fragmentation vs. over-merging tradeoff** — this is the PQ-specific failure mode that a "just maximize per-pixel Dice" mindset misses entirely; `min_area_px` and morphological closing are cheap early levers, but the real fix (per top public notebooks) is moving from pure semantic-seg-then-CC toward inherently instance-aware detection (YOLO-seg / Mask R-CNN) once MVP1's interfaces are trusted.
- **Category labels are a distractor for MVP1** — don't spend time on 4-class classification until the core instance pipeline is solid; the submission schema doesn't ask for it.
- **Submission budget is scarce** (5/day, 2 selectable finals) — lean on the local Dice/PQ harness (§3.8) validated against the organizers' self-eval notebook so leaderboard uploads are confirmations, not debugging tools.

---

## 6. Path beyond MVP1 (not part of this deliverable, for context)

Once the interfaces above are trusted, the highest-leverage next steps — visible from what's already scoring 0.3–0.7 PQ on the public leaderboard — are: (1) replace semantic-seg+CC with a genuine instance method (YOLO11-seg or detect→crop→U-Net refine, or Mask R-CNN) to directly attack the fragmentation/over-merging penalty; (2) train at higher resolution or with tiled full-resolution windows (2048×2048 is large — a "blended-tile" approach appears among top public notebooks) rather than the aggressive 256×256 downsampling used for MVP1 speed; (3) proper k-fold CV grouped by `file_name`; (4) loss/augmentation tuned for thin, elongated structures (e.g. boundary-aware or Tversky loss, elastic/rotation augmentation) given the fine-scale "barb" structures called out as a core challenge; (5) using `spine` annotations as auxiliary supervision (not required by scoring, but may help shape-consistency).
