#!/usr/bin/env bash
set -euo pipefail

WORK_DIR="${1:-data/yelp}"
TABLES="${HETERQA_YELP_TABLES:-business,review,user,checkin,tip,photo}"
BATCH_SIZE="${HETERQA_INGEST_BATCH_SIZE:-1000}"

if [[ "${HETERQA_ACCEPT_YELP_TERMS:-}" != "1" ]]; then
  echo "Set HETERQA_ACCEPT_YELP_TERMS=1 after reviewing Yelp Open Dataset terms." >&2
  echo "Official source: https://business.yelp.com/data/resources/open-dataset/" >&2
  exit 2
fi

ARGS=(
  data prepare-yelp-oceanbase
  --work-dir "${WORK_DIR}"
  --accept-yelp-terms
  --tables "${TABLES}"
  --batch-size "${BATCH_SIZE}"
)

if [[ "${HETERQA_INCLUDE_YELP_PHOTOS:-0}" == "1" ]]; then
  ARGS+=(--include-photos)
fi

if [[ "${HETERQA_RESET_YELP_TABLES:-0}" == "1" ]]; then
  ARGS+=(--reset-tables)
fi

if [[ -n "${HETERQA_INGEST_MAX_ROWS_PER_TABLE:-}" ]]; then
  ARGS+=(--max-rows-per-table "${HETERQA_INGEST_MAX_ROWS_PER_TABLE}")
fi

python -m heterqa.cli "${ARGS[@]}"
