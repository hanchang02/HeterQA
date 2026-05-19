#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 SOURCE_JSONL OUTPUT_DIR TEXT_FIELD [CONFIG_PATH]" >&2
  exit 2
fi

SOURCE_JSONL="$1"
OUTPUT_DIR="$2"
TEXT_FIELD="$3"
CONFIG_PATH="${4:-}"

mkdir -p "$OUTPUT_DIR"

EXTRACT_ARGS=(
  graph extract-features
  --input "$SOURCE_JSONL"
  --output "$OUTPUT_DIR/extracted_features.jsonl"
  --text-field "$TEXT_FIELD"
)

if [[ -n "$CONFIG_PATH" ]]; then
  EXTRACT_ARGS+=(--config "$CONFIG_PATH")
fi

python -m heterqa.cli "${EXTRACT_ARGS[@]}"

python -m heterqa.cli graph canonicalize \
  --input "$OUTPUT_DIR/extracted_features.jsonl" \
  --output "$OUTPUT_DIR/canonical_features.jsonl"

python -m heterqa.cli graph build \
  --input "$OUTPUT_DIR/canonical_features.jsonl" \
  --output-dir "$OUTPUT_DIR"
