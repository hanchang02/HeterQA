"""Extract public Hugging Face-style HeterQA release files from final cases."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from heterqa.core.io import read_jsonl, write_jsonl, write_qrels
from heterqa.core.schema import candidate_name, case_qid, case_query, case_subset
from heterqa.release.schemas import write_release_schemas


VERIFY_TO_FAMILY = {
    "geo_verify": "geo",
    "text_verify": "text",
    "image_verify": "image",
    "kg_verify": "kg",
    "text_to_image_verify": "cross_modal",
    "image_to_text_verify": "cross_modal",
    "text2image_verify": "cross_modal",
    "image2text_verify": "cross_modal",
}


def _safe_text(value: object, limit: int = 480) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False, default=str)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"/(?:mnt|home)/\S+", "redacted_path", text)
    text = re.sub(r"\.(?:jpg|jpeg|png|webp)\b", "", text, flags=re.I)
    if len(text) > limit:
        text = text[: limit - 3].rstrip() + "..."
    return text


SENSITIVE_DETAIL_KEYS = {
    "raw_review",
    "raw_reviews",
    "review_text",
    "review_texts",
    "text",
    "raw_text",
    "prompt",
    "prompts",
    "trace",
    "debug_trace",
    "messages",
    "image_path",
    "photo_path",
    "local_path",
    "source_result_file",
    "source_file",
    "source_yaml",
    "materialized_yaml",
    "attributes_json",
    "hours_json",
}


def _safe_detail_value(value: Any, *, key: str = "") -> Any:
    lowered = key.lower()
    if lowered in SENSITIVE_DETAIL_KEYS or lowered.endswith("_path") or lowered.endswith("_file"):
        return "redacted"
    if isinstance(value, dict):
        return {
            str(item_key): _safe_detail_value(item_value, key=str(item_key))
            for item_key, item_value in value.items()
            if str(item_key).lower() not in SENSITIVE_DETAIL_KEYS
        }
    if isinstance(value, list):
        return [_safe_detail_value(item, key=key) for item in value[:20]]
    if isinstance(value, str):
        return _safe_text(value, 360)
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return _safe_text(value, 360)


def _candidate_verdict(candidate: dict[str, Any]) -> str:
    return str(candidate.get("verdict") or candidate.get("final_verdict") or "").strip().lower()


def _normalise_subset(value: str) -> str:
    text = str(value or "").strip()
    return re.sub(r"^[A-Z]{1,2}_", "", text) if "_" in text else text


def _verification_method(family: str, source_key: str | None = None) -> str:
    if family == "record_field":
        return "structured_field_check"
    if source_key:
        return source_key
    if family == "geo":
        return "geo_verify"
    return f"{family}_verify"


def _confidence(verify: dict[str, Any]) -> float | None:
    value = verify.get("confidence")
    if value is None and isinstance(verify.get("parsed_score"), dict):
        value = verify["parsed_score"].get("confidence")
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _judgement(verify: dict[str, Any]) -> str | None:
    value = verify.get("judgement")
    if value is None and isinstance(verify.get("parsed_score"), dict):
        value = verify["parsed_score"].get("judgement")
    return _safe_text(value, 80).lower() if value is not None else None


def _source_locator(family: str, verify: dict[str, Any]) -> tuple[str, str]:
    for key in ["source_locator", "review_id", "review_ids", "photo_id", "photo_ids", "photo_id_stem", "photo_id_stems"]:
        value = verify.get(key)
        if value:
            locator = "|".join(str(item) for item in value) if isinstance(value, list) else str(value)
            if "photo" in key:
                return "photo_id_stem", _safe_text(locator, 600)
            if "review" in key:
                return "yelp_review_id", _safe_text(locator, 600)
            return str(verify.get("source_locator_type") or family), _safe_text(locator, 600)
    if family == "geo":
        distance = verify.get("distance_km") or verify.get("computed_distance_km")
        bearing = verify.get("bearing_deg") or verify.get("computed_bearing_deg")
        direction = verify.get("direction") or verify.get("computed_direction")
        parts = []
        if distance is not None:
            parts.append(f"distance_km={distance}")
        if bearing is not None:
            parts.append(f"bearing_deg={bearing}")
        if direction:
            parts.append(f"direction={direction}")
        return "computed_geo", ";".join(parts) if parts else "computed_geo_summary"
    return "verification_summary", f"{family}_summary"


def _record_predicates(case: dict[str, Any], candidate: dict[str, Any]) -> list[dict[str, Any]]:
    predicates = []
    source = candidate.get("record_field_evidence") or candidate.get("structured_predicates")
    if isinstance(source, list):
        for item in source:
            if isinstance(item, dict):
                predicates.append(
                    {
                        "field": _safe_text(item.get("field"), 80),
                        "expected": item.get("expected"),
                        "observed_summary": _safe_text(item.get("observed_summary") or item.get("observed"), 180),
                        **({"operator": item.get("operator")} if item.get("operator") else {}),
                    }
                )
    filters = case.get("all_filters") or case.get("structured_filters") or {}
    if not predicates and isinstance(filters, list):
        for item in filters:
            predicates.append({"field": _safe_text(item, 120), "expected": None, "observed_summary": None})
    return predicates


def _record_field_row(qid: str, case: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    predicates = _record_predicates(case, candidate)
    return {
        "qid": qid,
        "business_id": candidate["business_id"],
        "family": "record_field",
        "support_status": "passes",
        "claim_summary": "The answer business satisfies the query-relevant structured predicates used by the released case.",
        "source_locator_type": "query_predicates",
        "source_locator": f"predicate_count={len(predicates)}",
        "verification_method": "structured_field_check",
        "confidence": 1.0,
        "raw_content_released": False,
        "details": {"predicates": predicates},
    }


def _verification_row(qid: str, candidate: dict[str, Any], source_key: str, verify: dict[str, Any]) -> dict[str, Any] | None:
    family = VERIFY_TO_FAMILY[source_key]
    judgement = _judgement(verify)
    confidence = _confidence(verify)
    if judgement in {"no", "false", "drop", "contradiction"}:
        return None
    locator_type, locator = _source_locator(family, verify)
    reason = verify.get("reason") or verify.get("evidence_summary") or verify.get("summary") or "Verifier retained this answer."
    return {
        "qid": qid,
        "business_id": candidate["business_id"],
        "family": family,
        "support_status": "passes" if family == "geo" else "supports",
        "claim_summary": _safe_text(reason),
        "source_locator_type": locator_type,
        "source_locator": locator,
        "verification_method": _verification_method(family, source_key),
        "confidence": confidence,
        "raw_content_released": False,
        "details": {
            "verifier_judgement": judgement,
            "verifier_reason": _safe_text(reason),
        },
    }


def _explicit_evidence_rows(qid: str, candidate: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    evidence = candidate.get("evidence", [])
    if isinstance(evidence, dict):
        evidence = [{"family": family, **payload} for family, payload in evidence.items() if isinstance(payload, dict)]
    for item in evidence if isinstance(evidence, list) else []:
        if not isinstance(item, dict):
            continue
        family = str(item.get("family", "record_field"))
        rows.append(
            {
                "qid": qid,
                "business_id": candidate["business_id"],
                "family": family,
                "support_status": str(item.get("support_status") or ("passes" if family == "record_field" else "supports")),
                "claim_summary": _safe_text(item.get("claim_summary") or item.get("summary") or "Evidence supports the retained answer."),
                "source_locator_type": str(item.get("source_locator_type") or family),
                "source_locator": _safe_text(item.get("source_locator") or f"{family}_summary", 600),
                "verification_method": str(item.get("verification_method") or _verification_method(family)),
                "confidence": item.get("confidence", item.get("score")),
                "raw_content_released": False,
                "details": _safe_detail_value(dict(item.get("details") or item.get("metadata") or {})),
            }
        )
    return rows


def _evidence_rows(qid: str, case: dict[str, Any], candidate: dict[str, Any]) -> list[dict[str, Any]]:
    rows = _explicit_evidence_rows(qid, candidate)
    if not any(row["family"] == "record_field" for row in rows):
        rows.append(_record_field_row(qid, case, candidate))
    for source_key in VERIFY_TO_FAMILY:
        verify = candidate.get(source_key)
        if isinstance(verify, dict):
            row = _verification_row(qid, candidate, source_key, verify)
            if row is not None:
                rows.append(row)
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for row in rows:
        key = (row["qid"], row["business_id"], row["family"], row["verification_method"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def _yes_candidates(case: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = [candidate for candidate in case.get("candidates", []) or [] if _candidate_verdict(candidate) == "yes"]
    if candidates:
        return candidates
    answer_ids = set(str(item) for item in case.get("final_answer_business_ids", []) or [])
    return [candidate for candidate in case.get("candidates", []) or [] if str(candidate.get("business_id") or "") in answer_ids]


def _source_manifest(cases: list[dict[str, Any]], *, answer_pair_count: int, evidence_count: int) -> dict[str, Any]:
    return {
        "dataset_name": "HeterQA",
        "version": "1.0.0",
        "query_count": len(cases),
        "answer_pair_count": answer_pair_count,
        "evidence_count": evidence_count,
        "raw_yelp_content_included": False,
        "source_dataset": "Yelp Open Dataset",
        "source_access": "Users must obtain the Yelp Open Dataset separately and comply with its terms.",
        "source_url": "https://business.yelp.com/data/resources/open-dataset/",
        "join_key": "business_id",
        "review_text_locator": "Yelp review_id for review-text evidence when recoverable",
        "photo_locator": "Yelp photo_id stem for image evidence when recoverable",
        "yelp_files_needed": [
            "yelp_academic_dataset_business.json",
            "yelp_academic_dataset_review.json",
            "photos.json",
            "Yelp photo files keyed by photo_id when reconstructing image evidence",
        ],
        "license_scope": "CC-BY-4.0 applies only to HeterQA annotations and metadata, not Yelp source content.",
        "released_files": [
            "data/queries.jsonl",
            "data/answers.jsonl",
            "data/evidence.jsonl",
            "data/qrels/test.tsv",
        ],
    }


def extract_release(input_dir: Path, output_dir: Path) -> Path:
    final_cases = input_dir / "final_cases.jsonl"
    if not final_cases.exists():
        raise FileNotFoundError(final_cases)
    cases = read_jsonl(final_cases)
    data_dir = output_dir / "data"
    (data_dir / "qrels").mkdir(parents=True, exist_ok=True)

    queries: list[dict[str, Any]] = []
    answers: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    qrels: dict[str, set[str]] = {}
    for case in cases:
        qid = case_qid(case)
        subset = _normalise_subset(case_subset(case))
        yes_candidates = _yes_candidates(case)
        answer_ids = [str(candidate["business_id"]) for candidate in yes_candidates]
        answer_names = [candidate_name(candidate) for candidate in yes_candidates]
        queries.append({"qid": qid, "query": case_query(case), "subset": subset, "answer_count": len(answer_ids)})
        answers.append(
            {
                "qid": qid,
                "answer_business_ids": answer_ids,
                "answer_business_names": answer_names,
                "answer_count": len(answer_ids),
                "source_case_category": subset,
            }
        )
        qrels[qid] = set(answer_ids)
        for candidate in yes_candidates:
            evidence.extend(_evidence_rows(qid, case, candidate))

    write_jsonl(data_dir / "queries.jsonl", queries)
    write_jsonl(data_dir / "answers.jsonl", answers)
    write_jsonl(data_dir / "evidence.jsonl", evidence)
    write_qrels(data_dir / "qrels" / "test.tsv", qrels)
    (data_dir / "source_manifest.json").write_text(
        json.dumps(
            _source_manifest(cases, answer_pair_count=sum(len(row["answer_business_ids"]) for row in answers), evidence_count=len(evidence)),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "README.md").write_text(
        "# HeterQA\n\n"
        "This repository contains HeterQA annotations, qrels, and structured evidence summaries. "
        "It does not redistribute raw Yelp reviews, photos, or full business records.\n",
        encoding="utf-8",
    )
    (output_dir / "METHOD.md").write_text(
        "# Method\n\n"
        "The release is generated from answer-driven HeterQA cases after relational initialization, "
        "missing-value recovery, source-specific constraint instantiation, candidate filtering, "
        "question verbalization, contradiction detection, manual review when needed, and "
        "answer-set quality certification.\n",
        encoding="utf-8",
    )
    write_release_schemas(output_dir)
    return output_dir
