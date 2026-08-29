"""Assemble + validate submission.csv.

Pre-submission validation checklist (run before spending one of the 5 daily
submission slots) -- items 1/3/4/5 are errors (fatal), item 2 is a warning (a
genuinely filament-free test image legitimately produces zero rows, so it can't be
treated as fatal without also hiding a real globbing-bug regression; see
validate_submission's errors/warnings split):
  1. every filament_id unique
  2. every test image appears with >=0 rows (no image silently dropped by a globbing bug)
  3. every RLE round-trips: decode -> shape (2048,2048) -> non-empty
  4. no stray quote/newline characters inside segmentation_rle
  5. (manual) spot-check a few rows by re-decoding and overlaying on the source JPEG
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.rle_utils import SUBMISSION_SIZE, mask_to_rle_counts, rle_counts_to_mask


def build_submission(image_stem_to_instances: dict[str, list], out_path: str | Path) -> pd.DataFrame:
    rows = []
    for file_stem, instance_masks in image_stem_to_instances.items():
        for k, inst_mask in enumerate(instance_masks, start=1):
            rows.append({"filament_id": f"{file_stem}_{k}", "segmentation_rle": mask_to_rle_counts(inst_mask)})
    df = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df


def validate_submission(csv_path: str | Path, expected_image_stems: list[str] | None = None) -> dict[str, list[str]]:
    """Returns {"errors": [...], "warnings": [...]}. Errors mean something is
    actually broken (duplicate ids, corrupt RLE, unexpected image). Warnings flag
    things that are plausibly fine (e.g. a genuinely filament-free test image
    producing zero rows) but worth a human glance -- callers should fail on errors,
    not warnings.
    """
    errors: list[str] = []
    warnings: list[str] = []
    df = pd.read_csv(csv_path, dtype={"filament_id": str, "segmentation_rle": str}, keep_default_na=False)

    if df["filament_id"].duplicated().any():
        dupes = df.loc[df["filament_id"].duplicated(), "filament_id"].tolist()
        errors.append(f"{len(dupes)} duplicate filament_id values, e.g. {dupes[:5]}")

    if expected_image_stems is not None:
        covered = set(df["filament_id"].str.rsplit("_", n=1).str[0])
        missing = set(expected_image_stems) - covered
        if missing:
            warnings.append(f"{len(missing)} test images have zero predicted rows (fine if truly filament-free): {sorted(missing)[:5]}...")
        unexpected = covered - set(expected_image_stems)
        if unexpected:
            errors.append(f"{len(unexpected)} filament_id prefixes don't match any expected test image stem, e.g. {sorted(unexpected)[:5]}")

    for _, row in df.iterrows():
        rle = row["segmentation_rle"]
        if "\n" in rle or rle.startswith('"') or rle.endswith('"'):
            errors.append(f"{row['filament_id']}: stray quote/newline in segmentation_rle")
            continue
        try:
            m = rle_counts_to_mask(rle, size=SUBMISSION_SIZE)
        except Exception as e:
            errors.append(f"{row['filament_id']}: RLE failed to decode ({e})")
            continue
        if m.shape != SUBMISSION_SIZE:
            errors.append(f"{row['filament_id']}: decoded shape {m.shape} != {SUBMISSION_SIZE}")
        if m.sum() == 0:
            errors.append(f"{row['filament_id']}: decoded mask is empty")

    return {"errors": errors, "warnings": warnings}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--validate", type=Path, required=True, help="path to submission.csv to validate")
    p.add_argument(
        "--test-images-dir",
        type=Path,
        default=Path("data/raw/MAGFiLO_1.0_Kaggle_2026/test/test_images"),
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    expected_stems = [p.stem for p in sorted(args.test_images_dir.glob("*.jpeg"))]
    result = validate_submission(args.validate, expected_image_stems=expected_stems)
    for w in result["warnings"]:
        print(f"  WARNING: {w}")
    if not result["errors"]:
        print(f"OK: {args.validate} passed all checks ({len(expected_stems)} expected test images).")
    else:
        print(f"FOUND {len(result['errors'])} ERROR(S) in {args.validate}:")
        for e in result["errors"]:
            print(f"  - {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
