# Release Format

The public release follows a qrels plus structured-evidence layout.

## Files

```text
README.md
METHOD.md
data/
  queries.jsonl
  answers.jsonl
  evidence.jsonl
  qrels/test.tsv
  source_manifest.json
schemas/
  query.schema.json
  answer.schema.json
  evidence.schema.json
```

The release exporter writes the data tables, manifest, and schemas. Additional
hosting or submission metadata is maintained with the hosted dataset release.

## Core Tables

`queries.jsonl` contains:

- `qid`
- `query`
- `subset`
- `answer_count`

`answers.jsonl` contains:

- `qid`
- `answer_business_ids`
- `answer_business_names`
- `answer_count`
- `source_case_category`

`qrels/test.tsv` contains BEIR-style relevance rows:

- `query-id`
- `corpus-id`
- `score`

`evidence.jsonl` contains structured, non-raw support records:

- `qid`
- `business_id`
- `family`
- `support_status`
- `claim_summary`
- `source_locator_type`
- `source_locator`
- `verification_method`
- `confidence`
- `raw_content_released`
- `details`

`raw_content_released` must be `false` for every evidence row.
