#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 CERTIFIED_DIR QUALITY_OUTPUT_DIR [RATINGS_CSV]" >&2
  exit 2
fi

python -m heterqa.cli quality query-metrics \
  --input "$1" \
  --output "$2"

if [[ $# -ge 3 && -n "$3" ]]; then
  python -m heterqa.cli quality human-summary \
    --ratings "$3" \
    --queries "$1" \
    --output "$2"
fi
