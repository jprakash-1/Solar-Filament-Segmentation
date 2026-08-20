#!/usr/bin/env bash
# Refresh kaggle_package/ from the current src/, scripts/, configs/, requirements.txt
# so it's ready to upload as a Kaggle Dataset (the code half of the training setup;
# the data half is the competition dataset, attached separately in the notebook).
#
# Usage:
#   bash scripts/kaggle_sync.sh
#   cd kaggle_package && kaggle datasets create -p . --dir-mode zip   # first time
#   cd kaggle_package && kaggle datasets version -p . -m "update" --dir-mode zip   # updates

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${ROOT}/kaggle_package"

rsync -a --delete "${ROOT}/src/" "${DEST}/src/"
rsync -a --delete "${ROOT}/scripts/" "${DEST}/scripts/"
rsync -a --delete "${ROOT}/configs/" "${DEST}/configs/"
cp "${ROOT}/requirements.txt" "${DEST}/requirements.txt"

find "${DEST}" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

echo "Synced code into ${DEST}"
echo "Contents:"
find "${DEST}" -type f | sort
