"""Query-answer consistency checks for generated HeterQA cases."""

from __future__ import annotations

import re
from typing import Any

from heterqa.construction.contracts import GenerationResult
from heterqa.construction.record_fields import _ask_json


QUERY_VALIDATION_PROMPT = (
    "You are a strict semantic validator for a search engine.\n"
    "Verify whether a generated search query accurately reflects the source components without hallucination "
    "or subject drift.\n\n"
    "Target entity:\n{target_entity}\n\n"
    "Candidate query:\n{query}\n\n"
    "Source components:\n"
    "- Structured: {structured}\n"
    "- Geo: {geo}\n"
    "- Text: {text}\n"
    "- Image: {image}\n"
    "- KG insights: {kg}\n\n"
    "Validation steps:\n"
    "1) Subject check: the grammatical subject must be the target entity or a direct synonym.\n"
    "2) Constraint check: all strict constraints from Structured/Geo/Text/Image must be present.\n"
    "3) KG integration check: KG insights may be reflected when consistent, merged when redundant, or omitted if they "
    "contradict hard constraints.\n"
    "4) Hallucination check: no invented constraints.\n\n"
    "Return JSON only:\n"
    "{{"
    "\"subject_analysis\": \"short analysis\", "
    "\"missing_constraints\": [\"hard constraints missing from the query\"], "
    "\"hallucination_check\": \"invented details or empty\", "
    "\"is_pass\": true"
    "}}"
)


STOPWORDS = {
    "a",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "backed",
    "be",
    "business",
    "businesses",
    "by",
    "constraint",
    "constraints",
    "context",
    "evidence",
    "explicit",
    "find",
    "filter",
    "filters",
    "for",
    "from",
    "hard",
    "have",
    "has",
    "in",
    "insight",
    "insights",
    "is",
    "known",
    "line",
    "look",
    "looking",
    "matching",
    "me",
    "near",
    "nearby",
    "nl",
    "not",
    "numeric",
    "of",
    "offer",
    "offers",
    "offering",
    "or",
    "photo",
    "photos",
    "places",
    "provide",
    "provides",
    "query",
    "record",
    "requirement",
    "requirements",
    "review",
    "reviews",
    "show",
    "soft",
    "source",
    "sql",
    "that",
    "the",
    "to",
    "true",
    "false",
    "type",
    "user",
    "value",
    "venues",
    "view",
    "views",
    "visual",
    "with",
}

TARGET_ENTITY_TERMS = {
    "business",
    "businesses",
    "place",
    "places",
    "venue",
    "venues",
    "restaurant",
    "restaurants",
    "cafe",
    "cafes",
    "shop",
    "shops",
    "store",
    "stores",
    "hotel",
    "hotels",
    "service",
    "services",
}

STRICT_COMPONENTS = ("structured", "geo", "text", "image")
SOFT_COMPONENTS = ("kg",)
MIN_COMPONENT_COVERAGE = 0.45
MIN_SHORT_COMPONENT_MATCHED_TOKENS = 1


def _normalize_token(token: str) -> str:
    token = token.lower().strip(".-+&")
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and (
        token.endswith("ches") or token.endswith("shes") or token[-3] in {"s", "x", "z"}
    ):
        return token[:-2]
    if len(token) > 4 and token.endswith("s"):
        return token[:-1]
    return token


def _keywords(text: str) -> set[str]:
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9_&+.-]*", text.lower())
    keywords: set[str] = set()
    for token in tokens:
        for part in re.split(r"[_/&+.-]+", token):
            normalized = _normalize_token(part)
            if len(normalized) >= 3 and normalized not in STOPWORDS:
                keywords.add(normalized)
    return keywords


def _component_overlap(query: str, component: str) -> dict[str, Any]:
    component_tokens = _keywords(component)
    if not component_tokens:
        return {"required": [], "matched": [], "coverage": 1.0}
    query_tokens = _keywords(query)
    matched = sorted(component_tokens & query_tokens)
    return {
        "required": sorted(component_tokens),
        "matched": matched,
        "coverage": len(matched) / max(1, len(component_tokens)),
    }


def _subject_ok(final_query: str, target_entity: str) -> bool:
    query_tokens = _keywords(final_query) | {_normalize_token(token) for token in re.findall(r"[A-Za-z]+", final_query.lower())}
    target_tokens = _keywords(target_entity) | {_normalize_token(token) for token in re.findall(r"[A-Za-z]+", target_entity.lower())}
    return bool(query_tokens & (target_tokens | TARGET_ENTITY_TERMS))


def _coverage_missing(coverage: dict[str, dict[str, Any]]) -> list[str]:
    missing: list[str] = []
    for component_name in STRICT_COMPONENTS:
        item = coverage.get(component_name)
        if not item:
            continue
        required = item["required"]
        matched = item["matched"]
        if len(required) <= 2:
            if len(matched) < MIN_SHORT_COMPONENT_MATCHED_TOKENS:
                missing.append(component_name)
            continue
        if float(item["coverage"]) < MIN_COMPONENT_COVERAGE:
            missing.append(component_name)
    return missing


def _hallucinated_terms(final_query: str, normalized_components: dict[str, str], target_entity: str) -> list[str]:
    component_tokens: set[str] = set()
    for value in normalized_components.values():
        component_tokens.update(_keywords(value))
    component_tokens.update(_keywords(target_entity))
    component_tokens.update({_normalize_token(token) for token in TARGET_ENTITY_TERMS})
    query_tokens = _keywords(final_query)
    return sorted(token for token in query_tokens - component_tokens if token not in TARGET_ENTITY_TERMS)


def validate_query_components(
    final_query: str,
    components: dict[str, str],
    *,
    model: Any = None,
    target_entity: str = "businesses",
) -> dict[str, Any]:
    """Validate final query faithfulness without rewriting it.

    With a model, this performs semantic consistency checks over the composed
    query and its source components.
    Without a model, it enforces deterministic checks that are intentionally
    conservative: hard Structured/Geo/Text/Image components must be visibly
    represented, the subject must remain the target entity, and unsupported
    query terms are reported as hallucination risks.
    """

    normalized_components = {
        "structured": str(components.get("structured", "")).strip(),
        "geo": str(components.get("geo", "")).strip(),
        "text": str(components.get("text", "")).strip(),
        "image": str(components.get("image", "")).strip(),
        "kg": str(components.get("kg", "")).strip(),
    }
    coverage = {
        name: _component_overlap(final_query, value)
        for name, value in normalized_components.items()
        if value
    }
    if not final_query.strip():
        return {
            "is_pass": False,
            "mode": "deterministic_faithfulness_checks",
            "reason": "final query is empty",
            "coverage": coverage,
            "missing_constraints": ["final query"],
            "hallucinated_terms": [],
            "subject_ok": False,
        }
    if model is None:
        subject_ok = _subject_ok(final_query, target_entity)
        missing_constraints = _coverage_missing(coverage)
        hallucinated = _hallucinated_terms(final_query, normalized_components, target_entity)
        is_pass = subject_ok and not missing_constraints and not hallucinated
        reasons: list[str] = []
        if not subject_ok:
            reasons.append("query subject does not match the target entity")
        if missing_constraints:
            reasons.append("missing hard components: " + ", ".join(missing_constraints))
        if hallucinated:
            reasons.append("possible invented constraints: " + ", ".join(hallucinated))
        return {
            "is_pass": is_pass,
            "mode": "deterministic_faithfulness_checks",
            "reason": "; ".join(reasons) if reasons else "query covers hard components without unsupported deterministic terms",
            "coverage": coverage,
            "missing_constraints": missing_constraints,
            "hallucinated_terms": hallucinated,
            "subject_ok": subject_ok,
            "kg_coverage": {name: coverage.get(name) for name in SOFT_COMPONENTS if name in coverage},
        }
    payload = _ask_json(
        model,
        QUERY_VALIDATION_PROMPT.format(
            target_entity=target_entity,
            query=final_query,
            structured=normalized_components["structured"],
            geo=normalized_components["geo"],
            text=normalized_components["text"],
            image=normalized_components["image"],
            kg=normalized_components["kg"],
        ),
    )
    return {
        "is_pass": bool(payload.get("is_pass", False)),
        "mode": "semantic_model",
        "reason": " ".join(
            item
            for item in [
                str(payload.get("subject_analysis") or "").strip(),
                str(payload.get("hallucination_check") or "").strip(),
            ]
            if item
        ),
        "coverage": coverage,
        "missing_constraints": list(payload.get("missing_constraints") or []),
        "hallucinated_terms": _hallucinated_terms(final_query, normalized_components, target_entity),
        "subject_ok": str(payload.get("subject_analysis") or ""),
    }


def validate_generated_query(result: GenerationResult) -> list[str]:
    errors: list[str] = []
    if not result.final_query.strip():
        errors.append("final query is empty")
    answer_count = len(result.final_answer_business_ids)
    if not (result.case.target_answer_count_min <= answer_count <= result.case.target_answer_count_max):
        errors.append("final answer count is outside the configured target range")
    active_ids = sorted(candidate.business_id for candidate in result.candidates if candidate.is_active)
    final_ids = sorted(result.final_answer_business_ids)
    if active_ids != final_ids:
        errors.append("final answer ids do not match active candidate ids")
    validation = {}
    if result.context is not None:
        validation = dict(result.context.metadata.get("query_validation") or {})
    if validation and not validation.get("is_pass", False):
        errors.append("final query failed query-component validation")
    return errors
