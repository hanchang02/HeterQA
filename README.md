# HeterQA

This repository contains the public data-generation and release code for
HeterQA. The implementation follows an answer-driven dataset construction
workflow:

1. Relational initialization and missing-value recovery
2. Source-specific constraint instantiation
3. Candidate filtering
4. Question verbalization and contradiction detection
5. Quality certification

The repository does not redistribute Yelp reviews, photos, complete business
profiles, local indexes, model credentials, or non-public review notes.
Reconstruction requires separately obtained Yelp Open Dataset files plus local
model, vector-index, and graph-index configuration.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

For model-backed reconstruction, also install:

```bash
pip install -e ".[models]"
```

For SQL/OceanBase-backed reconstruction, also install:

```bash
pip install -e ".[db]"
```

## Quick Start

Prepare local graph feature artifacts when starting from review or tip rows:

```bash
heterqa graph extract-features --input local_reviews.jsonl --output runs/graph/extracted_features.jsonl
heterqa graph canonicalize --input runs/graph/extracted_features.jsonl --output runs/graph/canonical_features.jsonl
heterqa graph build --input runs/graph/canonical_features.jsonl --output-dir runs/graph
```

Run the data-generation workflow:

```bash
heterqa construct run \
  --config configs/construction.example.yaml \
  --output runs/construction

heterqa certify contradiction-detect \
  --config configs/audit.example.yaml \
  --input runs/construction \
  --output runs/contradiction

heterqa review export \
  --input runs/contradiction \
  --output runs/review_packets

heterqa review apply \
  --review-dir runs/review_packets \
  --input runs/contradiction \
  --output runs/review_applied

heterqa certify answer-set \
  --input runs/review_applied \
  --output runs/certified

heterqa quality query-metrics \
  --input runs/certified \
  --output runs/quality

heterqa release export-hf \
  --input runs/certified \
  --output heterqa_hf_upload
```

Human rating aggregation accepts a local rating CSV:

```bash
heterqa quality human-summary \
  --ratings ratings.csv \
  --queries runs/certified \
  --output runs/quality
```

The rating CSV schema is:

```text
qid,subset,annotator_id,naturalness,diversity,practicality
```

## Repository Contents

- `src/heterqa/core`: shared config, I/O, schema, provenance, and safety helpers.
- `src/heterqa/providers`: boundaries for Yelp records, review text, photos, graph evidence, vector indexes, and model clients.
- `src/heterqa/construction`: relational initialization, missing-value recovery, source-specific evidence, candidate filtering, and question verbalization.
- `src/heterqa/graph`: graph feature extraction and graph build utilities.
- `src/heterqa/audit`: contradiction detection and semantic consistency checks.
- `src/heterqa/review`: manual review packet export and decision application.
- `src/heterqa/finalize`: answer-set certification and dataset-instance materialization.
- `src/heterqa/quality`: query diversity and human-rating aggregation.
- `src/heterqa/release`: public dataset extraction, validation, and schemas.
- `tests/fixtures`: small local fixtures for unit tests only.

## Yelp Preparation

To load Yelp JSON into an already installed OceanBase/MySQL deployment:

```bash
export HETERQA_ACCEPT_YELP_TERMS=1
export HETERQA_DB_HOST=127.0.0.1
export HETERQA_DB_PORT=2881
export HETERQA_DB_USER=root
export HETERQA_DB_PASSWORD=change_me
export HETERQA_DB_NAME=heterqa_yelp
bash scripts/prepare_yelp_oceanbase.sh data/yelp
```

See `docs/DATA_PREPARATION.md` for table details and photo archive options.

## License

Code is released under the Apache License 2.0. The HeterQA annotation release and the
underlying Yelp source data have separate terms.
