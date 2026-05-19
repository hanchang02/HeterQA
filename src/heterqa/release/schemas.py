"""Write JSON schemas for the public HeterQA release."""

from __future__ import annotations

import json
from pathlib import Path


SCHEMAS = {
    "query.schema.json": {
        "type": "object",
        "required": ["qid", "query", "subset", "answer_count"],
        "properties": {
            "qid": {"type": "string"},
            "query": {"type": "string"},
            "subset": {"type": "string"},
            "answer_count": {"type": "integer"},
        },
    },
    "answer.schema.json": {
        "type": "object",
        "required": ["qid", "answer_business_ids", "answer_business_names", "answer_count"],
        "properties": {
            "qid": {"type": "string"},
            "answer_business_ids": {"type": "array", "items": {"type": "string"}},
            "answer_business_names": {"type": "array", "items": {"type": "string"}},
            "answer_count": {"type": "integer"},
            "source_case_category": {"type": ["string", "null"]},
        },
    },
    "evidence.schema.json": {
        "type": "object",
        "required": [
            "qid",
            "business_id",
            "family",
            "support_status",
            "claim_summary",
            "source_locator_type",
            "source_locator",
            "verification_method",
            "raw_content_released",
        ],
        "properties": {
            "qid": {"type": "string"},
            "business_id": {"type": "string"},
            "family": {"type": "string"},
            "support_status": {"type": "string"},
            "claim_summary": {"type": "string"},
            "source_locator_type": {"type": "string"},
            "source_locator": {"type": "string"},
            "verification_method": {"type": "string"},
            "confidence": {"type": ["number", "null"]},
            "raw_content_released": {"type": "boolean"},
            "details": {"type": "object"},
        },
    },
}


def write_release_schemas(output_dir: Path) -> None:
    schema_dir = output_dir / "schemas"
    schema_dir.mkdir(parents=True, exist_ok=True)
    for filename, payload in SCHEMAS.items():
        (schema_dir / filename).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

