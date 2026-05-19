# Data-Generation Pipeline

HeterQA uses an answer-driven construction protocol. Each case starts from
business records and source-specific constraints, then candidate businesses are
filtered and checked before the answer set is certified for public release.

## Workflow Steps

1. Relational initialization and missing-value recovery

   The pipeline selects seed businesses and query-relevant record-field
   predicates. Missing text, photo, spatial, or graph evidence is recovered from
   configured providers.

2. Source-specific constraint instantiation

   Text, image, geo, and graph constraints are instantiated from source
   evidence. Vector indexes, graph indexes, and model calls are accessed through
   configured providers.

3. Candidate filtering

   Candidate businesses are expanded, hydrated, deduplicated, and filtered by
   structured predicates, spatial constraints, and source-specific evidence
   checks.

4. Question verbalization and contradiction detection

   The construction stage verbalizes a user-facing question from the verified
   constraints. The certification stage re-checks retained candidates for
   textual, visual, structured, and semantic contradictions.

5. Quality certification

   Manual review packets can be exported for unresolved candidate decisions.
   The answer-set certification step keeps only cases with one to ten final
   answers. Query metrics and human-rating aggregation scripts support the
   naturalness, diversity, and practicality checks.

## Public Commands

```bash
heterqa graph extract-features --input local_reviews.jsonl --output runs/graph/extracted_features.jsonl
heterqa graph canonicalize --input runs/graph/extracted_features.jsonl --output runs/graph/canonical_features.jsonl
heterqa graph build --input runs/graph/canonical_features.jsonl --output-dir runs/graph

heterqa construct run --config configs/construction.example.yaml --output runs/construction
heterqa certify contradiction-detect --config configs/audit.example.yaml --input runs/construction --output runs/contradiction
heterqa review export --input runs/contradiction --output runs/review_packets
heterqa review apply --review-dir runs/review_packets --input runs/contradiction --output runs/review_applied
heterqa certify answer-set --input runs/review_applied --output runs/certified
heterqa quality query-metrics --input runs/certified --output runs/quality
heterqa release export-hf --input runs/certified --output heterqa_hf_upload
```

Model-backed commands can load local components from config. The public
repository defines the interface; endpoints, keys, and deployment
details stay outside the repository:

```yaml
model:
  factory: local_package.model_factory:create_model
  kwargs:
    config_path: path/to/local_model_config.yaml
```

## Public Evidence Families

- `record_field`
- `text`
- `image`
- `geo`
- `kg`
- `cross_modal`

Subset names use readable composition labels such as `Geo_Text` and
`Text_Image`.
