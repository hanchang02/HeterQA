"""Contradiction detection and semantic consistency checks for HeterQA cases."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from heterqa.construction.contracts import StructuredFilter
from heterqa.construction.providers import ConstructionDataProvider, build_construction_provider
from heterqa.construction.record_fields import _ask_json, hard_constraint_passes, semantic_category_passes
from heterqa.core.config import load_yaml_config
from heterqa.core.io import read_jsonl, write_jsonl
from heterqa.providers.model_client import build_model_bundle


VERIFY_FIELDS = (
    "geo_verify",
    "text_verify",
    "image_verify",
    "kg_verify",
    "text_to_image_verify",
    "image_to_text_verify",
)
SEMANTIC_QUERY_FIELDS = ("text_query", "image_query", "kg_query")
SEMANTIC_VERIFY_FIELDS = (
    "text_verify",
    "image_verify",
    "text_to_image_verify",
    "image_to_text_verify",
    "kg_verify",
)

TEXT_CONTRADICTION_PROMPT = (
    "You are a textual contradiction detector for HeterQA.\n"
    "Classify the relationship between a user question and a review segment.\n\n"
    "Question: {query}\n"
    "Review Segment: {review}\n\n"
    "Labels:\n"
    "- CONTRADICTION: the segment explicitly negates a question requirement.\n"
    "- COMPATIBLE: the segment supports the question or is neutral/irrelevant.\n\n"
    "Output JSON only: {{\"label\": \"COMPATIBLE\", \"reason\": \"brief reason\"}}"
)

IMAGE_CONTRADICTION_PROMPT = (
    "You are a visual contradiction detector for HeterQA.\n"
    "Classify whether the image content contradicts the user question.\n\n"
    "Question: {query}\n\n"
    "Labels:\n"
    "- CONTRADICTION: the image clearly negates a question requirement.\n"
    "- COMPATIBLE: the image supports the question or is compatible/irrelevant.\n\n"
    "Output JSON only: {{\"label\": \"COMPATIBLE\", \"reason\": \"brief reason\"}}"
)

DEFENDER_PROMPT = (
    "You are the support reviewer in a HeterQA semantic consistency check.\n"
    "Argue why the candidate should remain in the answer set.\n\n"
    "Rules:\n"
    "1) Use only the provided evidence.\n"
    "2) Cite evidence IDs.\n"
    "3) Judge only the semantic consistency query.\n"
    "4) Structured fields and geography are outside this semantic check.\n\n"
    "Semantic Consistency Query:\n{query}\n\n"
    "Candidate:\n{candidate_block}\n\n"
    "Evidence Bundle:\n{evidence_block}\n\n"
    "Output JSON only:\n"
    "{{\"position\": \"keep\", \"summary\": \"short summary\", \"main_points\": [], "
    "\"cited_evidence_ids\": [], \"caveats\": []}}"
)

PROSECUTOR_PROMPT = (
    "You are the challenge reviewer in a HeterQA semantic consistency check.\n"
    "Argue why the candidate should be removed from the answer set.\n\n"
    "Rules:\n"
    "1) Use only the provided evidence.\n"
    "2) Cite evidence IDs.\n"
    "3) Judge only the semantic consistency query.\n"
    "4) Structured fields and geography are outside this semantic check.\n"
    "5) Focus on semantic mismatch, contradiction, or lack of support.\n\n"
    "Semantic Consistency Query:\n{query}\n\n"
    "Candidate:\n{candidate_block}\n\n"
    "Evidence Bundle:\n{evidence_block}\n\n"
    "Output JSON only:\n"
    "{{\"position\": \"remove\", \"summary\": \"short summary\", \"main_points\": [], "
    "\"cited_evidence_ids\": [], \"caveats\": []}}"
)

ARBITER_PROMPT = (
    "You are the arbiter in a HeterQA semantic consistency check.\n"
    "Read the evidence, support review, and challenge review, then decide whether the candidate belongs in the answer set.\n\n"
    "Verdict rules:\n"
    "- yes: keep the candidate.\n"
    "- no: remove the candidate.\n"
    "- unclear: evidence is insufficient and manual review is needed.\n\n"
    "Rules:\n"
    "1) Use only the provided evidence and arguments.\n"
    "2) Cite decisive evidence IDs.\n"
    "3) Judge only the semantic consistency query.\n"
    "4) Structured fields and geography are outside this semantic check.\n\n"
    "Semantic Consistency Query:\n{query}\n\n"
    "Candidate:\n{candidate_block}\n\n"
    "Evidence Bundle:\n{evidence_block}\n\n"
    "Support Review:\n{defender_block}\n\n"
    "Challenge Review:\n{prosecutor_block}\n\n"
    "Output JSON only:\n"
    "{{\"verdict\": \"yes\", \"confidence\": 0.0, \"semantic_risk_score\": 0, "
    "\"needs_manual_review\": false, \"final_reason\": \"short summary\", "
    "\"why_unclear\": \"\", \"decisive_evidence_ids\": [], \"summary\": \"one paragraph\"}}"
)


def _input_path(input_dir: Path) -> Path:
    for name in [
        "construction_cases.jsonl",
        "contradiction_checked_cases.jsonl",
    ]:
        path = input_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(f"No construction case file found in {input_dir}")


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _candidate_business_id(candidate: dict[str, Any]) -> str:
    return str(candidate.get("business_id") or candidate.get("id") or "")


def _candidate_fields(candidate: dict[str, Any]) -> dict[str, Any]:
    fields = candidate.get("metadata") or candidate.get("fields") or {}
    return fields if isinstance(fields, dict) else {}


def _candidate_verdict(candidate: dict[str, Any]) -> str:
    return str(candidate.get("verdict") or candidate.get("final_verdict") or "").strip().lower()


def _parse_filters(case: dict[str, Any]) -> list[StructuredFilter]:
    raw_case = _as_dict(case.get("case"))
    raw_filters = raw_case.get("structured_filters") or case.get("structured_filters") or case.get("all_filters") or []
    filters: list[StructuredFilter] = []
    for item in raw_filters:
        try:
            filters.append(StructuredFilter.from_raw(item))
        except ValueError:
            continue
    return filters


def _fetch_business_record(
    provider: ConstructionDataProvider | None,
    candidate: dict[str, Any],
) -> dict[str, Any] | None:
    business_id = _candidate_business_id(candidate)
    if provider is not None and business_id:
        record = provider.get_business(business_id)
        if record is not None:
            return dict(record.fields)
    fields = _candidate_fields(candidate)
    return dict(fields) if fields else None


def _lookup_field(record: dict[str, Any], field: str) -> Any:
    for key, value in record.items():
        if key.lower() == field.lower():
            return value
    return None


def _normalise_label(payload: dict[str, Any]) -> str:
    label = str(payload.get("label") or payload.get("judgement") or payload.get("status") or "").upper()
    if "CONTRADICTION" in label or "FAIL" in label:
        return "CONTRADICTION"
    return "COMPATIBLE"


def _parse_json_payload(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        if isinstance(raw.get("json_text"), str):
            return _parse_json_payload(raw["json_text"])
        return raw
    if isinstance(raw, (list, tuple)) and raw:
        return _parse_json_payload(raw[0])
    text = str(raw or "")
    if "```" in text:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            text = match.group(0)
    try:
        return json.loads(text)
    except Exception:
        return {}


def _ratio(stats: dict[str, Any]) -> float:
    total = sum(_as_float(value) for value in stats.values())
    if total <= 0:
        return 0.0
    return _as_float(stats.get("CONTRADICTION")) / total


def _target_candidates(case: dict[str, Any]) -> list[dict[str, Any]]:
    raw_candidates = [candidate for candidate in case.get("candidates", []) if isinstance(candidate, dict)]
    answer_ids = {str(item) for item in case.get("final_answer_business_ids", []) if str(item)}
    if answer_ids:
        return [candidate for candidate in raw_candidates if _candidate_business_id(candidate) in answer_ids]
    return [candidate for candidate in raw_candidates if candidate.get("is_active", True)]


def _slot_confidence(candidate: dict[str, Any]) -> float:
    total = 0.0
    for field in VERIFY_FIELDS:
        payload = _as_dict(candidate.get(field))
        total += _as_float(payload.get("confidence"))
    return total


def _verification_status(field_name: str, payload: dict[str, Any]) -> dict[str, Any]:
    judgement = str(payload.get("judgement") or payload.get("can_answer") or "").lower()
    confidence = _as_float(payload.get("confidence"))
    if judgement in {"yes", "true", "pass", "passed", "supports"}:
        label = "compatible"
    elif judgement in {"no", "false", "fail", "failed", "contradiction"}:
        label = "contradiction"
    else:
        label = "unknown"
    return {
        "field": field_name,
        "label": label,
        "confidence": confidence,
        "reason": str(payload.get("reason", "")),
        "evidence_locator_type": str(payload.get("evidence_locator_type", "")),
        "evidence_locator": str(payload.get("evidence_locator", "")),
    }


def _evidence_items(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    evidence = candidate.get("evidence", [])
    if isinstance(evidence, list):
        return [item for item in evidence if isinstance(item, dict)]
    if isinstance(evidence, dict):
        rows: list[dict[str, Any]] = []
        for family, payload in evidence.items():
            if isinstance(payload, dict):
                rows.append({"family": family, **payload})
        return rows
    return []


def build_verification_summary(candidate: dict[str, Any]) -> dict[str, Any]:
    slot_results = [
        _verification_status(field_name, payload)
        for field_name in VERIFY_FIELDS
        if (payload := _as_dict(candidate.get(field_name)))
    ]
    evidence_rows = _evidence_items(candidate)
    evidence_supports = [row for row in evidence_rows if row.get("supports") is True or row.get("passes") is True]
    evidence_conflicts = [row for row in evidence_rows if row.get("supports") is False or row.get("passes") is False]
    compatible = [row for row in slot_results if row["label"] == "compatible"]
    contradictions = [row for row in slot_results if row["label"] == "contradiction"]
    return {
        "verification_slots": slot_results,
        "compatible_count": len(compatible) + len(evidence_supports),
        "contradiction_count": len(contradictions) + len(evidence_conflicts),
        "compatible": compatible[:20],
        "conflicts": contradictions[:20],
        "evidence_support_count": len(evidence_supports),
        "evidence_conflict_count": len(evidence_conflicts),
    }


class ContradictionDetector:
    """Re-check retained candidates against structured, textual, and visual evidence."""

    def __init__(
        self,
        provider: ConstructionDataProvider | None = None,
        *,
        model: Any = None,
        text_contradiction_ratio: float = 0.15,
        visual_contradiction_ratio: float = 0.15,
        text_evidence_limit: int = 100,
        image_evidence_limit: int = 100,
    ) -> None:
        self.provider = provider
        self.model = model
        self.text_contradiction_ratio = text_contradiction_ratio
        self.visual_contradiction_ratio = visual_contradiction_ratio
        self.text_evidence_limit = text_evidence_limit
        self.image_evidence_limit = image_evidence_limit

    def detect_case(self, case: dict[str, Any]) -> None:
        final_query = str(case.get("final_query") or case.get("query") or "")
        filters = _parse_filters(case)
        targets = _target_candidates(case)
        summary = {"checked": 0, "yes": 0, "no": 0, "unclear": 0}
        unresolved: list[dict[str, Any]] = []
        if not final_query or not targets:
            case.setdefault("audit_summary", {})["contradiction_detection"] = {
                **summary,
                "unresolved_reason": "missing_final_query_or_no_retained_candidates",
            }
            case["contradiction_unresolved"] = [
                {
                    "business_id": _candidate_business_id(candidate),
                    "reason": "missing_final_query_or_no_retained_candidates",
                }
                for candidate in targets
            ]
            return
        for candidate in targets:
            self.detect_candidate(candidate, final_query=final_query, filters=filters)
            summary["checked"] += 1
            summary[candidate["verdict"]] = summary.get(candidate["verdict"], 0) + 1
            if candidate["verdict"] == "unclear":
                unresolved.append(
                    {
                        "business_id": _candidate_business_id(candidate),
                        "reason": "contradiction_detection_unclear",
                        "drop_reason": candidate.get("drop_reason"),
                    }
                )
        case.setdefault("audit_summary", {})["contradiction_detection"] = summary
        case["contradiction_unresolved"] = unresolved

    def detect_candidate(
        self,
        candidate: dict[str, Any],
        *,
        final_query: str,
        filters: list[StructuredFilter],
    ) -> None:
        metadata = candidate.setdefault("audit_metadata", {})
        structured_ok, structured_failures, db_record = self._check_structured(candidate, filters)
        if db_record is not None:
            metadata["db_ground_truth"] = db_record
        metadata["structured_check"] = {"pass": structured_ok, "failures": structured_failures}

        text_stats, text_conflicts, supporting_reviews = self._audit_text(_candidate_business_id(candidate), final_query)
        image_stats, image_conflicts, visual_evidence = self._audit_images(_candidate_business_id(candidate), final_query)
        text_ratio = _ratio(text_stats)
        image_ratio = _ratio(image_stats)

        metadata["contradiction_text_stats"] = text_stats
        metadata["contradiction_image_stats"] = image_stats
        metadata["supporting_text_evidence"] = supporting_reviews
        metadata["supporting_image_evidence"] = visual_evidence
        metadata["contradiction_conflicts"] = text_conflicts + image_conflicts

        if not structured_ok:
            candidate["is_active"] = False
            candidate["drop_reason"] = "HARD_CONSTRAINT_FAIL: " + "; ".join(structured_failures)
        elif text_ratio > self.text_contradiction_ratio:
            candidate["is_active"] = False
            candidate["drop_reason"] = f"HIGH_TEXT_CONTRADICTION ({text_ratio:.1%})"
        elif image_ratio > self.visual_contradiction_ratio:
            candidate["is_active"] = False
            candidate["drop_reason"] = f"HIGH_VISUAL_CONTRADICTION ({image_ratio:.1%})"

        candidate.setdefault("scores", {})
        candidate["scores"]["text_contradiction_stats"] = text_stats
        candidate["scores"]["visual_contradiction_stats"] = image_stats
        candidate["scores"]["composite_final"] = round(_slot_confidence(candidate) - (text_ratio + image_ratio), 4)
        candidate["contradiction_detection"] = self._build_full_verification(candidate)
        candidate["contradiction_verdict"] = self._candidate_verdict(candidate)
        candidate["verdict"] = candidate["contradiction_verdict"]

    def _check_structured(
        self,
        candidate: dict[str, Any],
        filters: list[StructuredFilter],
    ) -> tuple[bool, list[str], dict[str, Any] | None]:
        if not filters:
            return True, [], _fetch_business_record(self.provider, candidate)
        record = _fetch_business_record(self.provider, candidate)
        if not record:
            return False, ["DB_RECORD_MISSING"], None
        failures: list[str] = []
        for predicate in filters:
            actual = _lookup_field(record, predicate.field)
            if predicate.operator == "semantic_category":
                result = semantic_category_passes(predicate.value, _lookup_field(record, "categories"), self.model)
                if not result.is_passed(0.5):
                    failures.append(f"SEMANTIC_CONFLICT: {result.reason or predicate.value}")
                continue
            if not hard_constraint_passes(
                actual,
                predicate.operator,
                predicate.value,
                is_numeric=predicate.is_numeric,
                allow_missing=True,
            ):
                failures.append(f"{predicate.field} {predicate.operator} {predicate.value} (Act: {actual})")
        return not failures, failures, record

    def _audit_text(self, business_id: str, final_query: str) -> tuple[dict[str, int], list[dict[str, Any]], list[dict[str, Any]]]:
        stats = {"COMPATIBLE": 0, "CONTRADICTION": 0}
        conflicts: list[dict[str, Any]] = []
        supports: list[dict[str, Any]] = []
        rows = self._fetch_text_evidence(business_id, final_query)[: self.text_evidence_limit]
        judgements = self._judge_text_batch(final_query, rows)
        for row, payload in zip(rows, judgements, strict=False):
            text = str(row.get("text") or row.get("summary") or "")
            if not text:
                continue
            label = _normalise_label(payload)
            stats[label] += 1
            record = {
                "type": "text",
                "source_locator_type": row.get("source_locator_type", "review_id"),
                "source_locator": row.get("source_locator", row.get("review_id", "")),
                "reason": str(payload.get("reason", "")),
            }
            if label == "CONTRADICTION":
                record["evidence"] = text
                conflicts.append(record)
            else:
                supports.append(record)
        return stats, conflicts, supports

    def _audit_images(self, business_id: str, final_query: str) -> tuple[dict[str, int], list[dict[str, Any]], list[dict[str, Any]]]:
        stats = {"COMPATIBLE": 0, "CONTRADICTION": 0}
        conflicts: list[dict[str, Any]] = []
        supports: list[dict[str, Any]] = []
        rows = self._fetch_image_evidence(business_id, final_query)[: self.image_evidence_limit]
        judgements = self._judge_image_batch(final_query, rows)
        for row, payload in zip(rows, judgements, strict=False):
            locator = str(row.get("source_locator") or row.get("path") or row.get("photo_id") or "")
            label = _normalise_label(payload)
            stats[label] += 1
            record = {
                "type": "image",
                "source_locator_type": row.get("source_locator_type", "photo_id"),
                "source_locator": locator,
                "reason": str(payload.get("reason", "")),
            }
            if label == "CONTRADICTION":
                conflicts.append(record)
            else:
                supports.append(record)
        return stats, conflicts, supports

    def _fetch_text_evidence(self, business_id: str, final_query: str) -> list[dict[str, Any]]:
        if self.provider is not None and business_id and final_query:
            return list(self.provider.search_reviews(final_query, [business_id], self.text_evidence_limit))
        return []

    def _fetch_image_evidence(self, business_id: str, final_query: str) -> list[dict[str, Any]]:
        if self.provider is not None and business_id and final_query:
            return list(self.provider.search_photos(final_query, [business_id], self.image_evidence_limit))
        return []

    def _judge_text(self, query: str, review: str) -> dict[str, Any]:
        if self.model is None:
            return {"label": "COMPATIBLE", "reason": "No text contradiction judge configured."}
        return _ask_json(self.model, TEXT_CONTRADICTION_PROMPT.format(query=query, review=review))

    def _judge_text_batch(self, query: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        prompts = [
            TEXT_CONTRADICTION_PROMPT.format(query=query, review=str(row.get("text") or row.get("summary") or ""))
            for row in rows
        ]
        batcher = _resolve_parallel_json(getattr(self.model, "semantic_judge", None))
        if batcher is not None and prompts:
            tasks = [{"prompt": prompt, "temperature": 0.0, "max_tokens": 2000} for prompt in prompts]
            return [_parse_json_payload(item) or {"label": "COMPATIBLE", "reason": "empty judge response"} for item in batcher(tasks)]
        return [self._judge_text(query, str(row.get("text") or row.get("summary") or "")) for row in rows]

    def _judge_image(self, query: str, row: dict[str, Any]) -> dict[str, Any]:
        if self.model is None:
            supports = row.get("supports")
            if supports is False:
                return {"label": "CONTRADICTION", "reason": "Image evidence row is marked unsupported."}
            return {"label": "COMPATIBLE", "reason": "No visual contradiction judge configured."}
        target = getattr(self.model, "visual_judge", None)
        if target is None:
            raise ValueError("A visual_judge is required for image contradiction detection.")
        locator = str(row.get("path") or row.get("source_locator") or row.get("photo_id") or "")
        return _parse_json_payload(
            target.ask_json(prompt=IMAGE_CONTRADICTION_PROMPT.format(query=query), image=locator or None, temperature=0.0, max_tokens=2000)
        )

    def _judge_image_batch(self, query: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        batcher = _resolve_parallel_json(getattr(self.model, "visual_judge", None))
        if batcher is not None and rows:
            tasks = []
            for row in rows:
                locator = str(row.get("path") or row.get("source_locator") or row.get("photo_id") or "")
                tasks.append(
                    {
                        "prompt": IMAGE_CONTRADICTION_PROMPT.format(query=query),
                        "images": [locator] if locator else [],
                        "image": locator or None,
                        "temperature": 0.0,
                        "max_tokens": 2000,
                    }
                )
            return [_parse_json_payload(item) or {"label": "COMPATIBLE", "reason": "empty visual judge response"} for item in batcher(tasks)]
        return [self._judge_image(query, row) for row in rows]

    def _build_full_verification(self, candidate: dict[str, Any]) -> dict[str, Any]:
        base = build_verification_summary(candidate)
        metadata = _as_dict(candidate.get("audit_metadata"))
        text_stats = _as_dict(metadata.get("contradiction_text_stats"))
        image_stats = _as_dict(metadata.get("contradiction_image_stats"))
        conflicts = list(metadata.get("contradiction_conflicts") or [])
        base.update(
            {
                "supporting_reviews": list(metadata.get("supporting_text_evidence") or []),
                "visual_evidence": list(metadata.get("supporting_image_evidence") or []),
                "text_stats": text_stats,
                "image_stats": image_stats,
                "conflicts": base["conflicts"] + conflicts,
                "text_contradiction_ratio": _ratio(text_stats),
                "image_contradiction_ratio": _ratio(image_stats),
            }
        )
        return base

    def _candidate_verdict(self, candidate: dict[str, Any]) -> str:
        audit = _as_dict(candidate.get("contradiction_detection"))
        if candidate.get("drop_reason") or candidate.get("is_active") is False or candidate.get("verdict") == "drop":
            return "no"
        if _ratio(_as_dict(audit.get("text_stats"))) > self.text_contradiction_ratio:
            return "no"
        if _ratio(_as_dict(audit.get("image_stats"))) > self.visual_contradiction_ratio:
            return "no"
        if audit.get("compatible_count", 0) > 0 or candidate.get("is_active") is True:
            return "yes"
        return "unclear"


def _query_context(case: dict[str, Any]) -> dict[str, Any]:
    raw_case = case.get("case") if isinstance(case.get("case"), dict) else {}
    context = case.get("context") if isinstance(case.get("context"), dict) else {}
    query_components = context.get("metadata", {}).get("query_components") if isinstance(context.get("metadata"), dict) else {}
    if not isinstance(query_components, dict):
        query_components = {}
    return {
        "text_query": raw_case.get("text_query") or context.get("text_query") or query_components.get("text") or "",
        "image_query": raw_case.get("image_query") or context.get("image_query") or query_components.get("image") or "",
        "kg_query": raw_case.get("kg_query") or context.get("kg_query") or query_components.get("kg") or "",
    }


def derive_consistency_query(case: dict[str, Any]) -> str:
    context = _query_context(case)
    return "\n".join(str(context[field]).strip() for field in SEMANTIC_QUERY_FIELDS if str(context[field]).strip())


def _truncate(value: Any, limit: int = 240) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _coerce_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if value is None:
        return []
    return [str(value)] if str(value).strip() else []


def _append_item(items: list[dict[str, Any]], prefix: str, payload: dict[str, Any]) -> None:
    index = 1 + sum(1 for item in items if str(item.get("id", "")).startswith(prefix))
    items.append(
        {
            "id": f"{prefix}{index}",
            "type": {"S": "STRUCT", "P": "PIPELINE", "T": "TRACE", "C": "CONTRADICTION"}[prefix],
            "title": payload["title"],
            "summary": payload["summary"],
            "raw_ref": payload.get("raw_ref", {}),
            "raw_payload": payload.get("raw_payload"),
        }
    )


def _extract_structured_evidence(case: dict[str, Any]) -> list[dict[str, Any]]:
    query = derive_consistency_query(case)
    included = [{"field": field, "query": _query_context(case)[field]} for field in SEMANTIC_QUERY_FIELDS if str(_query_context(case)[field]).strip()]
    items: list[dict[str, Any]] = []
    _append_item(
        items,
        "S",
        {
            "title": "Semantic Consistency Query",
            "summary": f"semantic_query={_truncate(query)}." if query else "No semantic query remains after structured and geo constraints.",
            "raw_ref": {"source": "context", "field": "text_query/image_query/kg_query"},
            "raw_payload": query,
        },
    )
    _append_item(
        items,
        "S",
        {
            "title": "Scope Guardrail",
            "summary": f"included={_truncate([item['field'] for item in included])}; structured fields and geography are checked separately.",
            "raw_ref": {"source": "context", "field": "query_components"},
            "raw_payload": {"included": included, "checked_elsewhere": ["geo_query", "all_filters"]},
        },
    )
    return items


def _extract_construction_evidence(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    _append_item(
        items,
        "P",
        {
            "title": "Candidate Snapshot",
            "summary": f"origin={_truncate(candidate.get('origin'))}, score_summary={_truncate(candidate.get('scores'))}.",
            "raw_ref": {"source": "candidate", "field": "origin_and_score"},
            "raw_payload": {"origin": candidate.get("origin"), "scores": candidate.get("scores")},
        },
    )
    available = []
    for field_name in SEMANTIC_VERIFY_FIELDS:
        verify_obj = candidate.get(field_name)
        if not isinstance(verify_obj, dict):
            continue
        available.append(field_name)
        _append_item(
            items,
            "P",
            {
                "title": f"Verify: {field_name}",
                "summary": (
                    f"judgement={_truncate(verify_obj.get('judgement'))}, "
                    f"confidence={_truncate(verify_obj.get('confidence'))}, "
                    f"reason={_truncate(verify_obj.get('reason'))}, "
                    f"evidence={_truncate(verify_obj.get('evidence_summary'), 360)}."
                ),
                "raw_ref": {"source": "candidate", "field": field_name},
                "raw_payload": verify_obj,
            },
        )
    _append_item(
        items,
        "P",
        {
            "title": "Semantic Verify Coverage",
            "summary": f"available={_truncate(available)}; missing={_truncate([f for f in SEMANTIC_VERIFY_FIELDS if f not in available])}.",
            "raw_ref": {"source": "candidate", "field": "semantic_verify_fields"},
            "raw_payload": {"available": available, "missing": [field for field in SEMANTIC_VERIFY_FIELDS if field not in available]},
        },
    )
    return items


def _extract_trace_evidence(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for field_name in SEMANTIC_VERIFY_FIELDS:
        verify_obj = candidate.get(field_name)
        trace = verify_obj.get("trace") if isinstance(verify_obj, dict) else None
        if not isinstance(trace, dict):
            continue
        _append_item(
            items,
            "T",
            {
                "title": f"Trace: {field_name}",
                "summary": f"stage={_truncate(trace.get('stage'))}, image_count={len(trace.get('image_paths') or [])}.",
                "raw_ref": {"source": "candidate", "field": f"{field_name}.trace"},
                "raw_payload": {
                    "stage": trace.get("stage"),
                    "image_count": len(trace.get("image_paths") or []),
                },
            },
        )
    if not items:
        _append_item(
            items,
            "T",
            {
                "title": "Missing Semantic Verify Traces",
                "summary": "No semantic verify trace is available.",
                "raw_ref": {"source": "candidate", "field": "semantic_verify_traces"},
            },
        )
    return items


def _extract_contradiction_evidence(candidate: dict[str, Any], query: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    payload = candidate.get("contradiction_detection") if isinstance(candidate.get("contradiction_detection"), dict) else {}
    if not payload:
        _append_item(
            items,
            "C",
            {
                "title": "Missing Contradiction Detection Payload",
                "summary": "No contradiction_detection payload is present.",
                "raw_ref": {"source": "candidate", "field": "contradiction_detection"},
            },
        )
        return items
    for field_name in ("text_stats", "image_stats"):
        if field_name in payload:
            _append_item(
                items,
                "C",
                {
                    "title": f"Contradiction {field_name}",
                    "summary": _truncate(payload.get(field_name)),
                    "raw_ref": {"source": "candidate", "field": f"contradiction_detection.{field_name}"},
                    "raw_payload": payload.get(field_name),
                },
            )
    conflicts = payload.get("conflicts") or []
    _append_item(
        items,
        "C",
        {
            "title": "Contradiction Count",
            "summary": f"conflict_count={len(conflicts)}.",
            "raw_ref": {"source": "candidate", "field": "contradiction_detection.conflicts"},
            "raw_payload": {"count": len(conflicts)},
        },
    )
    keywords = set(re.findall(r"[a-zA-Z]{4,}", query.lower()))
    selected = []
    for conflict in conflicts:
        evidence_text = str(conflict.get("evidence") or conflict.get("reason") or "").lower() if isinstance(conflict, dict) else ""
        if keywords and keywords & set(re.findall(r"[a-zA-Z]{4,}", evidence_text)):
            selected.append(conflict)
    for index, conflict in enumerate((selected or conflicts)[:3], start=1):
        _append_item(
            items,
            "C",
            {
                "title": f"Contradiction Snippet {index}",
                "summary": _truncate(conflict, 360),
                "raw_ref": {"source": "candidate", "field": f"contradiction_detection.conflicts[{index - 1}]"},
                "raw_payload": conflict,
            },
        )
    return items


def build_evidence_bundle(case: dict[str, Any], candidate: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    query = derive_consistency_query(case)
    return {
        "structured": _extract_structured_evidence(case),
        "construction": _extract_construction_evidence(candidate),
        "traces": _extract_trace_evidence(candidate),
        "contradiction": _extract_contradiction_evidence(candidate, query),
    }


def _all_bundle_items(bundle: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for items in bundle.values():
        rows.extend(items)
    return rows


def _render_candidate(candidate: dict[str, Any]) -> str:
    payload = {
        "business_id": candidate.get("business_id"),
        "name": candidate.get("name"),
        "origin": candidate.get("origin"),
        "scores": candidate.get("scores"),
        "verdict": candidate.get("verdict"),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


def _render_bundle(bundle: dict[str, list[dict[str, Any]]]) -> str:
    lines: list[str] = []
    for section_name, items in (
        ("Structured Evidence", bundle["structured"]),
        ("Construction Evidence", bundle["construction"]),
        ("Trace Evidence", bundle["traces"]),
        ("Contradiction Evidence", bundle["contradiction"]),
    ):
        lines.append(f"{section_name}:")
        if not items:
            lines.append("- None")
            continue
        for item in items:
            lines.append(f"- [{item['id']}] {item['title']}: {item['summary']}")
    return "\n".join(lines)


def _normalize_side(payload: dict[str, Any], position: str) -> dict[str, Any]:
    return {
        "position": str(payload.get("position", position)),
        "summary": str(payload.get("summary", "")),
        "main_points": _coerce_list(payload.get("main_points")),
        "cited_evidence_ids": _coerce_list(payload.get("cited_evidence_ids")),
        "caveats": _coerce_list(payload.get("caveats")),
        "raw_json": payload,
    }


def _clamp_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, parsed))


def _clamp_int(value: Any, default: int = 50) -> int:
    try:
        parsed = int(round(float(value)))
    except (TypeError, ValueError):
        return default
    return max(0, min(100, parsed))


def _normalize_arbiter(payload: dict[str, Any]) -> dict[str, Any]:
    verdict = str(payload.get("verdict", "unclear")).lower().strip()
    if verdict not in {"yes", "no", "unclear"}:
        verdict = "unclear"
    why_unclear = payload.get("why_unclear")
    if verdict == "unclear" and not why_unclear:
        why_unclear = "Arbiter did not provide a decisive automatic judgement."
    return {
        "verdict": verdict,
        "confidence": _clamp_float(payload.get("confidence")),
        "semantic_risk_score": _clamp_int(payload.get("semantic_risk_score")),
        "needs_manual_review": bool(payload.get("needs_manual_review", verdict == "unclear")),
        "final_reason": str(payload.get("final_reason", "")),
        "why_unclear": str(why_unclear) if why_unclear is not None else None,
        "decisive_evidence_ids": _coerce_list(payload.get("decisive_evidence_ids")),
        "summary": str(payload.get("summary", "")),
        "raw_json": payload,
    }


class SemanticConsistencyChecker:
    """Defender/challenger/arbiter check over semantic evidence bundles."""

    def __init__(self, model: Any = None) -> None:
        self.model = model

    def check_candidate(self, case: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
        query = derive_consistency_query(case)
        bundle = build_evidence_bundle(case, candidate)
        if candidate.get("drop_reason") or candidate.get("is_active") is False:
            return self._policy_result("no", 1.0, 100, f"Candidate inactive before semantic consistency check: {candidate.get('drop_reason')}", bundle)
        if not query.strip():
            return self._policy_result("yes", 1.0, 0, "No text/image/kg semantic query remains; keep by policy.", bundle)
        if self.model is None:
            return self._no_model_result(bundle, query)

        candidate_block = _render_candidate(candidate)
        evidence_block = _render_bundle(bundle)
        defender = _normalize_side(
            _ask_json(self.model, DEFENDER_PROMPT.format(query=query, candidate_block=candidate_block, evidence_block=evidence_block)),
            "keep",
        )
        prosecutor = _normalize_side(
            _ask_json(self.model, PROSECUTOR_PROMPT.format(query=query, candidate_block=candidate_block, evidence_block=evidence_block)),
            "remove",
        )
        arbiter = _normalize_arbiter(
            _ask_json(
                self.model,
                ARBITER_PROMPT.format(
                    query=query,
                    candidate_block=candidate_block,
                    evidence_block=evidence_block,
                    defender_block=json.dumps(defender["raw_json"], ensure_ascii=False, indent=2),
                    prosecutor_block=json.dumps(prosecutor["raw_json"], ensure_ascii=False, indent=2),
                ),
            )
        )
        return {
            "verdict": arbiter["verdict"],
            "confidence": arbiter["confidence"],
            "semantic_risk_score": arbiter["semantic_risk_score"],
            "needs_manual_review": arbiter["needs_manual_review"],
            "final_reason": arbiter["final_reason"],
            "why_unclear": arbiter["why_unclear"],
            "decisive_evidence_ids": arbiter["decisive_evidence_ids"],
            "semantic_consistency_query": query,
            "support_review": defender,
            "challenge_review": prosecutor,
            "arbiter": arbiter,
            "evidence_bundle": bundle,
        }

    @staticmethod
    def _policy_result(
        verdict: str,
        confidence: float,
        risk: int,
        reason: str,
        bundle: dict[str, list[dict[str, Any]]],
    ) -> dict[str, Any]:
        return {
            "verdict": verdict,
            "confidence": confidence,
            "semantic_risk_score": risk,
            "needs_manual_review": verdict == "unclear",
            "final_reason": reason,
            "why_unclear": reason if verdict == "unclear" else None,
            "decisive_evidence_ids": [],
            "semantic_consistency_query": "",
            "support_review": {},
            "challenge_review": {},
            "arbiter": {
                "verdict": verdict,
                "confidence": confidence,
                "semantic_risk_score": risk,
                "needs_manual_review": verdict == "unclear",
                "final_reason": reason,
                "why_unclear": reason if verdict == "unclear" else None,
                "decisive_evidence_ids": [],
            },
            "evidence_bundle": bundle,
        }

    def _no_model_result(self, bundle: dict[str, list[dict[str, Any]]], query: str) -> dict[str, Any]:
        items = _all_bundle_items(bundle)
        supporting = [
            item for item in items if any(word in item["summary"].lower() for word in ("yes", "support", "compatible", "judgement=true"))
        ]
        conflicts = [
            item for item in items if any(word in item["summary"].lower() for word in ("contradiction", "conflict", "unsupported", "judgement=no"))
        ]
        if conflicts and not supporting:
            verdict, risk, reason = "no", 85, "No-model policy detected semantic conflict evidence and no supporting semantic evidence."
        elif supporting:
            verdict, risk, reason = "yes", 25, "No-model policy detected supporting semantic evidence."
        else:
            verdict, risk, reason = "unclear", 70, "No model configured and semantic evidence is insufficient."
        result = self._policy_result(verdict, 1.0 - risk / 100, risk, reason, bundle)
        result["semantic_consistency_query"] = query
        result["needs_manual_review"] = verdict == "unclear"
        return result


def _selection_status(candidate: dict[str, Any]) -> tuple[bool, str]:
    if candidate.get("is_active") is False:
        return False, "inactive_before_semantic_consistency"
    if candidate.get("drop_reason"):
        return False, "dropped_before_semantic_consistency"
    verdict = candidate.get("verdict")
    if verdict in {"yes", "unclear"} or verdict is None:
        return True, "selected"
    if verdict == "no":
        return False, "verdict_no_before_semantic_consistency"
    return False, f"unsupported_verdict:{verdict}"


def _skip_result(reason: str) -> dict[str, Any]:
    return {
        "selected_for_check": False,
        "skip_reason": reason,
        "verdict": "skipped",
        "confidence": None,
        "semantic_risk_score": None,
        "needs_manual_review": False,
        "final_reason": f"Candidate skipped before semantic consistency check: {reason}.",
        "why_unclear": None,
        "decisive_evidence_ids": [],
        "semantic_consistency_query": "",
        "support_review": {},
        "challenge_review": {},
        "arbiter": {},
        "evidence_bundle": {},
    }


def _apply_consistency_result(candidate: dict[str, Any], judgement: dict[str, Any]) -> None:
    candidate["semantic_consistency"] = {"selected_for_check": True, **judgement}
    candidate["semantic_consistency_verdict"] = judgement["verdict"]
    candidate["verdict"] = judgement["verdict"]
    stage_status = candidate.setdefault("stage_status", {})
    if not isinstance(stage_status, dict):
        stage_status = {}
        candidate["stage_status"] = stage_status
    if judgement["verdict"] == "no":
        candidate["is_active"] = False
        candidate["drop_reason"] = "semantic_consistency_no"
        stage_status["semantic_consistency"] = "removed"
    elif judgement["verdict"] == "unclear" or judgement.get("needs_manual_review"):
        candidate["needs_manual_review"] = True
        stage_status["semantic_consistency"] = "needs_manual_review"
    else:
        stage_status["semantic_consistency"] = "passes"


def _unresolved_record(case: dict[str, Any], candidate: dict[str, Any], judgement: dict[str, Any]) -> dict[str, Any]:
    return {
        "qid": case.get("qid") or case.get("query_id") or case.get("task_id"),
        "business_id": candidate.get("business_id"),
        "name": candidate.get("name"),
        "verdict": judgement.get("verdict"),
        "needs_manual_review": judgement.get("needs_manual_review"),
        "why_unclear": judgement.get("why_unclear") or judgement.get("final_reason"),
        "semantic_risk_score": judgement.get("semantic_risk_score"),
        "semantic_consistency_query": judgement.get("semantic_consistency_query"),
    }


def _provider_from_config(config_path: Path | None) -> ConstructionDataProvider | None:
    if config_path is None:
        return None
    config = load_yaml_config(config_path)
    if not config.get("data"):
        return None
    return build_construction_provider(config["data"])


def _resolve_parallel_json(model_component: Any) -> Any | None:
    if model_component is not None and hasattr(model_component, "parallel_ask_json"):
        return model_component.parallel_ask_json
    return None


def run_contradiction_detection(input_dir: Path, output_dir: Path, config_path: Path | None = None) -> Path:
    rows = read_jsonl(_input_path(input_dir))
    provider = _provider_from_config(config_path)
    config = load_yaml_config(config_path) if config_path is not None else {}
    model = build_model_bundle(config.get("model"))
    detector = ContradictionDetector(provider, model=model)
    checker = SemanticConsistencyChecker(model)
    run_summary = {
        "case_count": len(rows),
        "checked_candidates": 0,
        "semantic_checked_candidates": 0,
        "skipped_candidates": 0,
        "yes": 0,
        "no": 0,
        "unclear": 0,
        "unresolved": 0,
    }
    unresolved: list[dict[str, Any]] = []
    for case in rows:
        detector.detect_case(case)
        case_summary = {"yes": 0, "no": 0, "unclear": 0, "skipped": 0, "unresolved": 0}
        for candidate in case.get("candidates", []):
            selected, reason = _selection_status(candidate)
            if not selected:
                candidate["semantic_consistency"] = _skip_result(reason)
                candidate["semantic_consistency_verdict"] = "skipped"
                case_summary["skipped"] += 1
                run_summary["skipped_candidates"] += 1
                continue
            run_summary["semantic_checked_candidates"] += 1
            judgement = checker.check_candidate(case, candidate)
            _apply_consistency_result(candidate, judgement)
            case_summary[judgement["verdict"]] = case_summary.get(judgement["verdict"], 0) + 1
            run_summary[judgement["verdict"]] = run_summary.get(judgement["verdict"], 0) + 1
            if judgement["verdict"] == "unclear" or judgement.get("needs_manual_review"):
                case_summary["unresolved"] += 1
                run_summary["unresolved"] += 1
                unresolved.append(_unresolved_record(case, candidate, judgement))
        run_summary["checked_candidates"] += case.get("audit_summary", {}).get("contradiction_detection", {}).get("checked", 0)
        case.setdefault("audit_summary", {})["semantic_consistency"] = case_summary
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "contradiction_checked_cases.jsonl"
    write_jsonl(output_path, rows)
    write_jsonl(output_dir / "manual_review_queue.jsonl", unresolved)
    (output_dir / "contradiction_detection_summary.json").write_text(
        json.dumps(run_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path
