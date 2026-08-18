# Solar Filament Segmentation Challenge 2026

Project scaffold for the Kaggle competition: [Solar Filament Segmentation Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026)

Task: automatic segmentation of solar filaments in GONG H-Alpha full-disk observational imagery.

The competition page's data/evaluation/timeline sections aren't accessible via automated fetch (JS-rendered), so `configs/config.yaml` and the modeling stack are left blank on purpose — fill them in after you open the competition's **Data** and **Evaluation** tabs directly and run the EDA notebook.

## Structure

```
.
├── configs/
│   └── config.yaml          # paths, seed, and placeholders for data/model params
├── data/
│   ├── raw/                 # untouched competition download (gitignored)
│   └── processed/           # derived/cached artifacts (gitignored)
├── notebooks/
│   └── 01_eda.ipynb         # starting point once data is downloaded
├── outputs/
│   ├── checkpoints/         # model weights
│   ├── logs/                # training logs
│   └── submissions/         # submission CSVs
├── scripts/
│   └── download_data.sh     # pulls competition data via the Kaggle API
├── src/
│   ├── data/                # dataset / dataloader code (to be added)
│   ├── models/               # model definitions (to be added)
│   └── utils/                # config loader, seeding, etc.
├── requirements.txt
└── .gitignore
```

## Setup

1. Create an environment and install dependencies:

   ```bash
   python -m venv .venv && source .venv/bin/activate   # or conda/mamba equivalent
   pip install -r requirements.txt
   ```

2. Get a Kaggle API token: [kaggle.com/settings](https://www.kaggle.com/settings) → API → "Create New Token" → downloads `kaggle.json`.

   ```bash
   mkdir -p ~/.kaggle
   mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
   chmod 600 ~/.kaggle/kaggle.json
   ```

3. Accept the competition rules on the [competition page](https://www.kaggle.com/competitions/filament-segmentation-2026/rules) — the API refuses downloads until you do.

4. Download the data:

   ```bash
   bash scripts/download_data.sh
   ```

5. Open `notebooks/01_eda.ipynb` and start exploring — the TODO list at the bottom covers what to figure out first (image/mask format, resolution, class balance, evaluation metric, submission format).

## Notes

- `configs/config.yaml` has `data:` and `model:` sections deliberately left as `null` — the competition's exact data schema (FITS vs. image files, mask encoding, image size) and the modeling stack (architecture/encoder) weren't confirmed yet. Fill these in after EDA.
- `src/utils/seed.py` and `src/utils/config.py` are the only utility code included so far; dataset/model/training code is intentionally not scaffolded yet — add it under `src/data`, `src/models` once the data format and modeling approach are settled.
- Evaluation metric for pixel-wise segmentation challenges is typically Dice or IoU — confirm on the competition's **Evaluation** tab and implement it in `src/utils` before building a validation loop.
