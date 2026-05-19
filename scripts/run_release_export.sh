#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 FINAL_DATASET_DIR RELEASE_OUTPUT_DIR" >&2
  exit 2
fi

python -m heterqa.cli release export-hf \
  --input "$1" \
  --output "$2"
