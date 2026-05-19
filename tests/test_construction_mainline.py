from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

from heterqa.construction.contracts import CandidateState, GenerationCase, GenerationResult
from heterqa.construction.mainline import run_mainline_from_config


ROOT = Path(__file__).resolve().parents[1]


def test_mainline_construction_keeps_supported_answer(tmp_path: Path) -> None:
    output = run_mainline_from_config(
        ROOT / "tests" / "fixtures" / "mainline_construction" / "config.yaml",
        tmp_path,
    )

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

    assert len(rows) == 1
    assert rows[0]["final_answer_business_ids"] == ["synthetic_cafe_1"]
    stage_names = [stage["name"] for stage in rows[0]["stage_summaries"]]
    assert "record_field_initialization" in stage_names
    assert "heterogeneous_candidate_recall" in stage_names
    assert "text_verification" in stage_names


def test_generation_result_serializes_sql_decimal_values() -> None:
    result = GenerationResult(
        case=GenerationCase(qid="q1", subset="Record_Field"),
        candidates=[
            CandidateState(
                business_id="b1",
                scores={"geo_distance_km": Decimal("1.25")},
                metadata={"stars": Decimal("4.0"), "review_count": Decimal("12")},
            )
        ],
        final_answer_business_ids=["b1"],
    )

    payload = result.to_dict()

    assert payload["candidates"][0]["scores"]["geo_distance_km"] == 1.25
    assert payload["candidates"][0]["metadata"]["stars"] == 4
    assert payload["candidates"][0]["metadata"]["review_count"] == 12
    json.dumps(payload)
