#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 INPUT_DIR OUTPUT_DIR [CONFIG_PATH]" >&2
  exit 2
fi

ARGS=(
  certify contradiction-detect
  --input "$1"
  --output "$2"
)

if [[ $# -ge 3 && -n "$3" ]]; then
  ARGS+=(--config "$3")
fi

python -m heterqa.cli "${ARGS[@]}"
