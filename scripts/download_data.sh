#!/usr/bin/env bash
# Download and unzip the Solar Filament Segmentation Challenge 2026 competition data.
#
# Prerequisites:
#   1. pip install -r requirements.txt   (installs the `kaggle` CLI)
#   2. Get your API token from https://www.kaggle.com/settings -> API -> "Create New Token"
#      This downloads kaggle.json.
#   3. Place it at ~/.kaggle/kaggle.json and run: chmod 600 ~/.kaggle/kaggle.json
#   4. Accept the competition rules on the Kaggle website (required before the API can download data):
#      https://www.kaggle.com/competitions/filament-segmentation-2026/rules
#
# Usage:
#   bash scripts/download_data.sh

set -euo pipefail

COMPETITION="filament-segmentation-2026"
DEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data/raw"

if ! command -v kaggle &> /dev/null; then
    echo "ERROR: kaggle CLI not found. Run: pip install -r requirements.txt" >&2
    exit 1
fi

if [ ! -f "${HOME}/.kaggle/kaggle.json" ] && [ -z "${KAGGLE_KEY:-}" ]; then
    echo "ERROR: No Kaggle credentials found." >&2
    echo "Place your token at ~/.kaggle/kaggle.json (see header of this script) or set KAGGLE_USERNAME / KAGGLE_KEY env vars." >&2
    exit 1
fi

mkdir -p "${DEST_DIR}"

echo "Downloading competition data for '${COMPETITION}' into ${DEST_DIR} ..."
kaggle competitions download -c "${COMPETITION}" -p "${DEST_DIR}"

echo "Unzipping..."
for f in "${DEST_DIR}"/*.zip; do
    [ -e "$f" ] || continue
    unzip -o "$f" -d "${DEST_DIR}"
done

echo "Done. Contents of ${DEST_DIR}:"
ls -la "${DEST_DIR}"
