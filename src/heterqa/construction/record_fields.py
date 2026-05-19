"""Record-field initialization and predicate verification."""

from __future__ import annotations

import json
import re
from typing import Any

from heterqa.construction.contracts import (
    EvidenceItem,
    LLMCallTrace,
    PipelineContext,
    StructuredFilter,
    VerificationResult,
)


SEMANTIC_ORTHOGONALITY_PROMPT = (
    "You are a semantic consistency judge. Check if the business categories match the user requirement.\n"
    "Requirement: {required}\n"
    "Actual Categories: {actual}\n"
    "Respond in JSON: {{\"status\": \"PASS\" or \"FAIL\", \"reason\": \"concise reason\"}}"
)


def _coerce_scalar(value: Any) -> Any:
    if isinstance(value, str):
        lower = value.strip().lower()
        if lower in {"true", "yes"}:
            return True
        if lower in {"false", "no"}:
            return False
        try:
            return float(value)
        except ValueError:
            return value.strip()
    return value


def _metadata_value(metadata: dict[str, Any], field: str) -> Any:
    for key, value in metadata.items():
        if key.lower() == field.lower():
            return value
    return None


def hard_constraint_passes(
    observed: Any,
    operator: str,
    expected: Any,
    *,
    is_numeric: bool | None = None,
    allow_missing: bool = True,
) -> bool:
    """Evaluate structured hard constraints.

    Candidate expansion treats missing database values as non-filtering for
    sparse Yelp record fields.
    """

    if observed is None:
        return allow_missing
    operator = operator.strip().upper()
    if operator == "LIKE":
        return str(expected).lower() in str(observed).lower()
    if operator in {"=", "=="}:
        if isinstance(observed, str) and isinstance(expected, list):
            return all(str(item).lower() in observed.lower() for item in expected)
        if isinstance(observed, list) and isinstance(expected, list):
            return set(str(item).lower() for item in expected).issubset({str(item).lower() for item in observed})
        return _coerce_scalar(observed) == _coerce_scalar(expected)

    try:
        observed_value = float(observed) if is_numeric or isinstance(expected, (int, float)) else str(observed).lower()
        expected_value = float(expected) if is_numeric or isinstance(expected, (int, float)) else str(expected).lower()
    except (TypeError, ValueError):
        return False
    if operator == ">=":
        return observed_value >= expected_value
    if operator == "<=":
        return observed_value <= expected_value
    if operator == ">":
        return observed_value > expected_value
    if operator == "<":
        return observed_value < expected_value
    return False


def _parse_json_response(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        if "json_text" in raw and isinstance(raw["json_text"], str):
            try:
                return json.loads(raw["json_text"])
            except json.JSONDecodeError:
                return {}
        return raw
    text = str(raw or "")
    if "```" in text:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            text = match.group(0)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def _ask_json(model: Any, prompt: str) -> dict[str, Any]:
    if model is None:
        return {}
    target = getattr(model, "semantic_judge", None)
    if target is None:
        raise ValueError("Configured model object must expose semantic_judge.ask_json(...).")
    if not hasattr(target, "ask_json"):
        raise ValueError("Configured semantic_judge must expose ask_json(prompt=..., temperature=..., max_tokens=...).")
    return _parse_json_response(target.ask_json(prompt=prompt, temperature=0.0, max_tokens=2000))


def semantic_category_passes(required: Any, actual: Any, model: Any = None) -> VerificationResult:
    prompt = SEMANTIC_ORTHOGONALITY_PROMPT.format(required=required, actual=actual or "NULL")
    trace = LLMCallTrace(stage="record_field_category_semantic_check", prompt=prompt)
    if model is None:
        required_terms = {str(item).lower() for item in required} if isinstance(required, list) else {str(required).lower()}
        actual_text = str(actual or "").lower()
        passed = bool(actual_text) and any(term in actual_text for term in required_terms)
        return VerificationResult(
            judgement="yes" if passed else "no",
            confidence=1.0 if passed else 0.0,
            reason="Category keyword overlap check used because no semantic judge was configured.",
            trace=trace,
            evidence_locator_type="business_field",
            evidence_locator="categories",
            evidence_summary=f"Actual categories: {actual}",
        )
    response = _ask_json(model, prompt)
    status = str(response.get("status", "")).upper()
    return VerificationResult(
        judgement="yes" if status == "PASS" else "no",
        confidence=1.0 if status in {"PASS", "FAIL"} else 0.0,
        reason=str(response.get("reason", "")),
        trace=trace,
        evidence_locator_type="business_field",
        evidence_locator="categories",
        evidence_summary=f"Actual categories: {actual}",
    )


def verify_record_fields(
    ctx: PipelineContext,
    *,
    filters: list[StructuredFilter],
    semantic_model: Any = None,
    allow_missing: bool = True,
) -> tuple[int, int]:
    """Apply hard constraints and category semantic checks to active candidates."""

    dropped = 0
    checked = 0
    for candidate in ctx.candidates:
        if not candidate.is_active or candidate.origin == "initial_seed":
            continue
        checked += 1
        predicate_logs: list[dict[str, Any]] = []
        hard_violations: list[str] = []
        semantic_failures: list[str] = []
        for predicate in filters:
            if predicate.operator == "semantic_category" or predicate.field == "categories":
                actual = _metadata_value(candidate.metadata, "categories")
                result = semantic_category_passes(predicate.value, actual, semantic_model)
                predicate_logs.append(
                    {
                        "field": "categories",
                        "expected": predicate.value,
                        "observed_summary": None if actual is None else str(actual)[:240],
                        "semantic_status": result.judgement,
                        "reason": result.reason,
                    }
                )
                if not result.is_passed(0.5):
                    semantic_failures.append(result.reason or "Category mismatch")
                continue
            observed = _metadata_value(candidate.metadata, predicate.field)
            passed = hard_constraint_passes(
                observed,
                predicate.operator,
                predicate.value,
                is_numeric=predicate.is_numeric,
                allow_missing=allow_missing,
            )
            predicate_logs.append(
                {
                    "field": predicate.field,
                    "operator": predicate.operator,
                    "expected": predicate.value,
                    "observed_summary": None if observed is None else str(observed)[:240],
                    "passed": passed,
                }
            )
            if not passed:
                hard_violations.append(f"{predicate.field} {predicate.operator} {predicate.value} (Act: {observed})")

        candidate.add_evidence(
            EvidenceItem(
                family="record_field",
                source_locator_type="query_predicates",
                source_locator=f"predicate_count={len(filters)}",
                summary="Structured predicates evaluated against the business record.",
                score=1.0 if not hard_violations and not semantic_failures else 0.0,
                supports=not hard_violations and not semantic_failures,
                metadata={"predicates": predicate_logs},
            )
        )
        if hard_violations:
            candidate.drop(f"HARD_CONSTRAINT_FAIL: {'; '.join(hard_violations)}")
            dropped += 1
        elif semantic_failures:
            candidate.drop(f"SEMANTIC_CONFLICT: {'; '.join(semantic_failures)}")
            dropped += 1
        else:
            candidate.stage_status["record_field"] = "passes"
    return checked, dropped
