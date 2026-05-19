#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 CONFIG_PATH OUTPUT_DIR" >&2
  exit 2
fi

python -m heterqa.cli construct run \
  --config "$1" \
  --output "$2"
