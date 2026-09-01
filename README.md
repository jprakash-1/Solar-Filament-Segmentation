# Solar Filament Segmentation Challenge 2026

Kaggle competition: [Solar Filament Segmentation Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026)
Task: class-agnostic instance segmentation of solar filaments in GONG H-Alpha
full-disk imagery.

This branch (`jp-pretraining-data-prep`) is boilerplate plus a self-supervised
pretraining data pipeline — MVP1's pipeline (`src/`, `configs/`, training
notebooks) lives on `jp-mvp1` and is intentionally not carried over here. See
`PRETRAIN_PLAN.md` for the full design.

## Pretraining data pipeline

```
scripts/pretrain_data/
  gong_halpha.py       # Stage 1: manifest + download GONG H-Alpha FITS (gong2.nso.edu)
  preprocess_gong.py   # Stage 2: FITS -> JPEG, no other processing (native 2048x2048, 1ch)
pretrain_gong_kaggle.ipynb   # repo root -- upload this to Kaggle to run both stages
```

`pretrain_gong_kaggle.ipynb` clones this branch and drives both scripts; no GPU
needed. Its output (two directories) gets versioned as Kaggle Datasets
(`halpha-raw`, `halpha-preprocessed`) that later pretraining notebooks (not yet
built — see `PRETRAIN_PLAN.md` sections 4-5) mount as read-only inputs.
