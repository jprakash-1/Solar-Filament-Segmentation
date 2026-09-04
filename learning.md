# CondInst — resources

Ranked best-supported of the five instance-segmentation candidates evaluated
in `PRETRAIN_PLAN.md` §5.1 given current evidence: it's the only one with
in-domain precedent (a real Hα solar filament paper used it successfully), and
it fixes the confirmed box-crop/fill-ratio problem (thin, non-compact
filaments make a poor fit for RoIAlign-cropped mask heads) without paying for
adjacency-resolving machinery that the data shows is rarely needed (69% of
instances have no neighbor within 60px). Still conditional on the pending
per-image error-analysis measurement in §5.1 — not a final decision.

**Core paper**
- [Conditional Convolutions for Instance Segmentation](https://arxiv.org/abs/2003.05664) (Tian, Shen, Chen — ECCV 2020) — the original paper. Read the abstract/intro for the "why not RoIAlign" motivation, then the method section for how per-instance dynamic filters are generated from a controller branch.
- [Instance and Panoptic Segmentation Using Conditional Convolutions](https://ieeexplore.ieee.org/document/9693155/) — extended journal (TPAMI) version, adds a panoptic-segmentation extension on top of the ECCV paper.

**Prerequisite — read first if FCOS is unfamiliar**

CondInst's instance-identification step (which pixel belongs to which
instance, before the dynamic mask filters run) is built directly on FCOS:
- [FCOS: Fully Convolutional One-Stage Object Detection](https://arxiv.org/abs/1904.01355) (Tian et al., ICCV 2019) — anchor-free, per-pixel detection.
- [Review — FCOS: Fully Convolutional One-Stage Object Detection](https://sh-tsang.medium.com/review-fcos-fully-convolutional-one-stage-object-detection-90d57274b19f) (Medium) — faster way in than the paper.

**Accessible summary**
- [CondInst — Learning-Deep-Learning paper notes](https://patrick-llgc.github.io/Learning-Deep-Learning/paper_notes/condinst.html) — concise, diagram-heavy, good as a second pass after the abstract.

**Code**
- [aim-uofa/AdelaiDet](https://github.com/aim-uofa/AdelaiDet) — official implementation (Detectron2-based), from the paper's own authors. Also has FCOS and SOLOv2 in the same repo.
- [mmdetection/configs/condinst](https://github.com/open-mmlab/mmdetection/tree/main/configs/condinst) — alternative if not already invested in Detectron2; more actively maintained config system.
- [Pretrained CondInst weights (ResNet-50-FPN, COCO)](https://huggingface.co/tianzhi/AdelaiDet-CondInst/tree/main) — real starting checkpoint rather than training from scratch, relevant given the small (707-image) labeled set here.

**Directly in-domain**
- [A universal method for solar filament detection from Hα observations using semi-supervised deep learning](https://www.aanda.org/articles/aa/full_html/2024/06/aa48314-23/aa48314-23.html) (A&A, 2024) — detects filaments in Hα full-disk images (same modality as this project) using CondInst. The paper that actually motivated ranking this approach first — worth re-reading closely for how they handled the detection front end's confidence calibration on faint filaments, since that's the recall risk this architecture carries here.

No dedicated explainer video exists for CondInst — it's a compact, single-idea
paper, so the paper plus the Learning-Deep-Learning notes above cover it
faster than a video would.
