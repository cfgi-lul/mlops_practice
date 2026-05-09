#!/usr/bin/env bash
set -euo pipefail

if ! command -v dvc >/dev/null 2>&1; then
  echo "dvc is not installed. Install from requirements.txt first."
  exit 1
fi

dvc init
dvc remote add -d localremote ../dvc-storage-final-task || true
dvc add data/sample_input.csv

echo "DVC metadata generated. Commit these files:"
echo "  data/sample_input.csv.dvc"
echo "  .dvc/config"
echo "  .gitignore"
