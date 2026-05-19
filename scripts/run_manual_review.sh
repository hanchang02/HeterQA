#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 INPUT_DIR REVIEW_OUTPUT_DIR" >&2
  exit 2
fi

python -m heterqa.cli review export \
  --input "$1" \
  --output "$2"
