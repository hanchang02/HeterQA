from __future__ import annotations

import json
from pathlib import Path

from heterqa.audit.contradiction_detection import (
    ContradictionDetector,
    SemanticConsistencyChecker,
    derive_consistency_query,
    run_contradiction_detection,
)
from heterqa.construction.contracts import BusinessRecord
from heterqa.core.io import read_jsonl, write_jsonl


class FakeProvider:
    def get_business(self, business_id: str):
        if business_id == "b1":
            return BusinessRecord(business_id="b1", fields={"business_id": "b1", "stars": 4.5})
        return None

    def search_reviews(self, *_args, **_kwargs):
        return [
            {"business_id": "b1", "text": "The place is extremely loud.", "source_locator": "r1"},
            {"business_id": "b1", "text": "Service was fine.", "source_locator": "r2"},
        ]

    def search_photos(self, *_args, **_kwargs):
        return []


class FakeJudge:
    def ask_json(self, *, prompt: str, temperature: float = 0.0, max_tokens: int = 2000, image: str | None = None):
        assert temperature == 0.0
        assert max_tokens == 2000
        assert image is None
        if "extremely loud" in prompt:
            return {"json_text": json.dumps({"label": "CONTRADICTION", "reason": "The review says it is loud."})}
        return {"json_text": json.dumps({"label": "COMPATIBLE", "reason": "No contradiction."})}


class FakeBundle:
    semantic_judge = FakeJudge()


def test_contradiction_detector_refetches_text_evidence_and_drops_conflict() -> None:
    case = {
        "final_query": "Find quiet businesses rated at least 4 stars.",
        "case": {"structured_filters": [{"field": "stars", "operator": ">=", "value": 4.0, "is_numeric": True}]},
        "candidates": [{"business_id": "b1", "is_active": True, "metadata": {"stars": 4.5}}],
    }

    ContradictionDetector(FakeProvider(), model=FakeBundle()).detect_case(case)  # type: ignore[arg-type]

    candidate = case["candidates"][0]
    assert candidate["verdict"] == "no"
    assert candidate["drop_reason"].startswith("HIGH_TEXT_CONTRADICTION")
    assert candidate["audit_metadata"]["contradiction_text_stats"] == {"COMPATIBLE": 1, "CONTRADICTION": 1}
    assert candidate["contradiction_detection"]["text_contradiction_ratio"] == 0.5


def test_semantic_consistency_query_uses_source_specific_semantics() -> None:
    case = {
        "final_query": "Find businesses near downtown with quiet patio seating.",
        "case": {
            "text_query": "quiet patio seating",
            "geo_query": "near downtown",
            "structured_filters": [{"field": "stars", "operator": ">=", "value": 4.0}],
        },
    }

    assert derive_consistency_query(case) == "quiet patio seating"


def test_semantic_consistency_checker_runs_three_role_decision() -> None:
    class TriJudge:
        def ask_json(self, *, prompt: str, temperature: float = 0.0, max_tokens: int = 2000, image: str | None = None):
            if "arbiter" in prompt.lower():
                return {
                    "json_text": json.dumps(
                        {
                            "verdict": "yes",
                            "confidence": 0.82,
                            "semantic_risk_score": 18,
                            "needs_manual_review": False,
                            "final_reason": "Semantic support is adequate.",
                            "why_unclear": "",
                            "decisive_evidence_ids": ["P2"],
                            "summary": "Keep.",
                        }
                    )
                }
            if "support reviewer" in prompt:
                return {"json_text": json.dumps({"position": "keep", "summary": "Support exists.", "cited_evidence_ids": ["P2"]})}
            return {"json_text": json.dumps({"position": "remove", "summary": "No decisive mismatch.", "cited_evidence_ids": []})}

    class TriBundle:
        semantic_judge = TriJudge()

    result = SemanticConsistencyChecker(TriBundle()).check_candidate(
        {"case": {"text_query": "quiet patio seating"}},
        {
            "business_id": "b1",
            "name": "Cafe",
            "origin": "text_vector_recall",
            "is_active": True,
            "verdict": "yes",
            "text_verify": {"judgement": "yes", "confidence": 0.9, "reason": "Reviews mention quiet patio seating."},
        },
    )

    assert result["verdict"] == "yes"
    assert result["semantic_risk_score"] == 18
    assert result["support_review"]["position"] == "keep"


def test_run_contradiction_detection_writes_public_outputs(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    write_jsonl(
        input_dir / "construction_cases.jsonl",
        [
            {
                "qid": "q1",
                "query": "Find quiet patios.",
                "final_query": "Find quiet patios.",
                "case": {"text_query": "quiet patios"},
                "candidates": [
                    {"business_id": "b1", "name": "Unresolved Cafe", "origin": "text_vector_recall", "is_active": True, "verdict": "yes"}
                ],
            }
        ],
    )

    run_contradiction_detection(input_dir, output_dir)

    rows = read_jsonl(output_dir / "contradiction_checked_cases.jsonl")
    queue = read_jsonl(output_dir / "manual_review_queue.jsonl")
    summary = json.loads((output_dir / "contradiction_detection_summary.json").read_text(encoding="utf-8"))
    candidate = rows[0]["candidates"][0]
    assert candidate["semantic_consistency"]["selected_for_check"] is True
    assert candidate["verdict"] in {"yes", "unclear"}
    assert isinstance(queue, list)
    assert summary["semantic_checked_candidates"] == 1
