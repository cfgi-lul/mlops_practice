#!/usr/bin/env bash
set -euo pipefail

if ! command -v dvc >/dev/null 2>&1; then
  echo "dvc is not installed. Install from requirements.txt first."
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

dvc init -f
dvc remote add -d localremote ../dvc-storage-final-task --force
python "final task/scripts/generate_datasets.py"
dvc add "final task/data/train.csv" "final task/data/test.csv"

echo "DVC metadata generated. Commit these files:"
echo "  final task/data/train.csv.dvc"
echo "  final task/data/test.csv.dvc"
echo "  .dvc/config"
