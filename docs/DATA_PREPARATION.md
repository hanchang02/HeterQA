# Data Preparation

HeterQA uses Yelp businesses as the target-record instantiation. The public
release is annotation-only and does not redistribute Yelp raw content.

## Public Release

The Hugging Face release contains:

- `data/queries.jsonl`
- `data/answers.jsonl`
- `data/evidence.jsonl`
- `data/qrels/test.tsv`
- `data/source_manifest.json`
- `schemas/*.schema.json`

These files distribute generated annotations and ground-truth business
identifiers without redistributing Yelp raw content.

## Yelp Reconstruction Inputs

Full reconstruction requires separately obtained Yelp Open Dataset files:

- business records
- review records
- photo metadata
- photo files
- optional tip records for graph construction

Set local paths through `.env` or config files:

```dotenv
HETERQA_YELP_BUSINESS_JSON=
HETERQA_YELP_REVIEW_JSON=
HETERQA_YELP_PHOTO_JSON=
HETERQA_YELP_PHOTO_DIR=
HETERQA_DB_HOST=
HETERQA_DB_PORT=
HETERQA_DB_USER=
HETERQA_DB_PASSWORD=
HETERQA_DB_NAME=
```

Raw Yelp files, local indexes, model keys, and generated traces are excluded
from the public repository.

For direct local reconstruction from Yelp files, use the `yelp_open_dataset`
provider in `configs/construction.example.yaml`:

```yaml
data:
  provider: yelp_open_dataset
  # Preferred after `heterqa data download-yelp`: uses the official Yelp
  # extracted layout under this root.
  yelp_root: data/yelp/extracted

  # Alternatively, provide exact file paths.
  business_jsonl: path/to/yelp_academic_dataset_business.json
  review_jsonl: path/to/yelp_academic_dataset_review.json
  photos_json: path/to/photos.json
  photo_dir: path/to/Yelp_Photos/photos
  graph_features_jsonl: path/to/graph_features.jsonl
  review_embedding_jsonl: path/to/review_embedding_index.jsonl
  photo_embedding_jsonl: path/to/photo_embedding_index.jsonl
  feature_embedding_jsonl: path/to/feature_embedding_index.jsonl
  max_reviews_per_business: 50
```

`max_reviews_per_business` is optional. It is useful for smoke runs; full runs
can use larger local indexes or custom providers backed by a database/vector
service.

## Local Indexes

Large-scale runs typically materialize:

- serialized record-field indexes
- review-text indexes
- photo embedding indexes
- feature graph indexes

The local JSONL adapters expect precomputed embedding rows such as:

```json
{"business_id": "BUSINESS_ID", "review_id": "REVIEW_ID", "text": "support summary", "embedding": [0.1, 0.2]}
{"business_id": "BUSINESS_ID", "photo_id": "PHOTO_ID", "caption": "photo caption", "path": "local/photo.jpg", "embedding": [0.1, 0.2]}
{"feature_key": "quiet patio", "embedding": [0.1, 0.2]}
```

The optional graph feature file is JSONL with fields such as `business_id`,
`feature`, `sentiment`, and optionally `user_id`. It supports the KG stage's
feature/path operations without publishing raw graph edges and can be generated
locally from review/tip rows:

```bash
heterqa graph extract-features --input local_reviews.jsonl --output runs/graph/extracted_features.jsonl
heterqa graph canonicalize --input runs/graph/extracted_features.jsonl --output runs/graph/canonical_features.jsonl
heterqa graph build --input runs/graph/canonical_features.jsonl --output-dir runs/graph
```

The graph feature artifacts store `source_text_sha256` locators and hashed
reviewer nodes rather than raw review text.

For OceanBase/MySQL deployments, use `provider: oceanbase` and keep
credentials in a local config or environment variables. Optional SQL template
fields such as `review_embedding_search_sql`, `photo_embedding_search_sql`, and
`feature_embedding_search_sql` can preserve deployment-specific vector syntax
without hardcoding it in the public repository.

## Loading Yelp into OceanBase

If OceanBase is already installed, HeterQA provides a helper to download Yelp's
public Open Dataset archives and stream the JSON files into OceanBase-compatible
tables. The official Yelp source is:

```text
https://business.yelp.com/data/resources/open-dataset/
```

The script requires an explicit acknowledgement flag for the Yelp Open Dataset
terms:

```bash
export HETERQA_DB_HOST=127.0.0.1
export HETERQA_DB_PORT=2881
export HETERQA_DB_USER=root
export HETERQA_DB_PASSWORD=change_me
export HETERQA_DB_NAME=heterqa_yelp
export HETERQA_ACCEPT_YELP_TERMS=1

bash scripts/prepare_yelp_oceanbase.sh data/yelp
```

Equivalent CLI commands:

```bash
heterqa data download-yelp \
  --output-dir data/yelp \
  --accept-yelp-terms

heterqa data load-yelp-oceanbase \
  --input-dir data/yelp/extracted \
  --host "$HETERQA_DB_HOST" \
  --port "$HETERQA_DB_PORT" \
  --user "$HETERQA_DB_USER" \
  --password "$HETERQA_DB_PASSWORD" \
  --database "$HETERQA_DB_NAME" \
  --reset-tables
```

Use `--include-photos` or `HETERQA_INCLUDE_YELP_PHOTOS=1` to download and
load the large Yelp photo archive. The downloader extracts the
official zip payloads and their named tar archives into a fixed layout under
`data/yelp/extracted`.

The loader creates these tables when needed:

```text
business
review
user
checkin
tip
photo
```

`business` is flattened for HeterQA construction. Yelp nested attributes such as
`BusinessAcceptsCreditCards`, `OutdoorSeating`, and `Ambience.casual` are stored
as snake-case columns such as `business_accepts_credit_cards`,
`outdoor_seating`, and `ambience_casual`. The original `attributes` and `hours`
objects are kept locally in `attributes_json` and `hours_json`; they are not
included in the public HeterQA release.

The official Yelp archives do not contain HeterQA's review/photo/feature
embedding tables. Existing embeddings can be supplied through
`review_embedding_table`, `photo_embedding_table`, `feature_embedding_table`, or
the corresponding `*_embedding_search_sql` entries in
`configs/construction.example.yaml`.

Embedding artifacts remain local. Public releases should contain only derived
metadata allowed by the source data terms.
