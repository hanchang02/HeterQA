from __future__ import annotations

import json

from heterqa.construction.contracts import CandidateState, GenerationCase, GenerationResult, PipelineContext, StructuredFilter
from heterqa.construction.query_generation import QueryComposer
from heterqa.construction.query_validation import validate_generated_query, validate_query_components
from heterqa.construction.record_fields import _ask_json


class FakeComposerModel:
    def ask_json(self, *, prompt: str, temperature: float = 0.0, max_tokens: int = 2000, image: str | None = None):
        assert temperature == 0.0
        assert max_tokens == 2000
        assert image is None
        if '"structured_nl"' in prompt:
            return {"json_text": json.dumps({"structured_nl": "Find businesses rated at least 4 stars."})}
        if '"nl_query"' in prompt:
            return {
                "json_text": json.dumps(
                    {
                        "nl_query": (
                            "Find businesses rated at least 4 stars with quiet patio seating "
                            "that are known for relaxed neighborhood visits."
                        )
                    }
                )
            }
        return {
            "json_text": json.dumps(
                {
                    "subject_analysis": "The subject is businesses.",
                    "missing_constraints": [],
                    "hallucination_check": "",
                    "is_pass": True,
                }
            )
        }


class FakeComposerBundle:
    semantic_judge = FakeComposerModel()


def test_ask_json_parses_json_text_wrapper() -> None:
    payload = _ask_json(FakeComposerBundle(), "semantic validation")

    assert payload["is_pass"] is True


def test_query_composer_generates_and_validates_final_query() -> None:
    case = GenerationCase(
        qid="q1",
        subset="Text_KG",
        structured_filters=[StructuredFilter(field="stars", operator=">=", value=4.0, is_numeric=True)],
    )
    ctx = PipelineContext(
        task_id="q1",
        text_query="quiet patio seating",
        kg_query="known for relaxed neighborhood visits",
    )

    composer = QueryComposer(FakeComposerBundle())
    final_query = composer.compose(ctx, case)
    validation = validate_query_components(
        final_query,
        dict(ctx.metadata["query_components"]),
        model=FakeComposerBundle(),
    )

    assert "quiet patio seating" in final_query
    assert ctx.metadata["query_components"]["structured"] == "Find businesses rated at least 4 stars."
    assert validation["is_pass"] is True
    assert validation["mode"] == "semantic_model"


def test_deterministic_query_validation_rejects_missing_hard_component() -> None:
    validation = validate_query_components(
        "Find cafes within 2 km with quiet service.",
        {
            "structured": "Find cafes.",
            "geo": "within 2 km of City Hall",
            "text": "quiet service",
            "image": "blue mural visible in photos",
            "kg": "popular with local students",
        },
        model=None,
    )

    assert validation["is_pass"] is False
    assert "image" in validation["missing_constraints"]


def test_deterministic_query_validation_rejects_extra_constraint() -> None:
    validation = validate_query_components(
        "Find cafes within 2 km with quiet service and waterfront views.",
        {
            "structured": "Find cafes.",
            "geo": "within 2 km of City Hall",
            "text": "quiet service",
            "image": "",
            "kg": "",
        },
        model=None,
    )

    assert validation["is_pass"] is False
    assert "waterfront" in validation["hallucinated_terms"]


def test_validate_generated_query_checks_active_answer_consistency() -> None:
    result = GenerationResult(
        case=GenerationCase(qid="q1", subset="Text", target_answer_count_min=1, target_answer_count_max=10),
        candidates=[
            CandidateState(business_id="b1", is_active=True),
            CandidateState(business_id="b2", is_active=False, drop_reason="record_field_filter"),
        ],
        final_answer_business_ids=["b2"],
        final_query="Find businesses with quiet service.",
    )

    assert "final answer ids do not match active candidate ids" in validate_generated_query(result)
