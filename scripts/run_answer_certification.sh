#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 INPUT_DIR OUTPUT_DIR [EXPECTED_COUNT]" >&2
  exit 2
fi

ARGS=(
  certify answer-set
  --input "$1"
  --output "$2"
)

if [[ $# -ge 3 && -n "$3" ]]; then
  ARGS+=(--expected-count "$3")
fi

python -m heterqa.cli "${ARGS[@]}"
