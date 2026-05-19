"""Public data contracts for HeterQA release and workflow artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


EvidenceFamily = Literal["record_field", "text", "image", "geo", "kg", "cross_modal"]
SupportStatus = Literal["passes", "supports"]
CandidateVerdict = Literal["pending", "yes", "no", "unclear", "drop"]


@dataclass(frozen=True)
class QueryRecord:
    qid: str
    query: str
    subset: str
    answer_count: int


@dataclass(frozen=True)
class AnswerRecord:
    qid: str
    answer_business_ids: list[str]
    answer_business_names: list[str]
    answer_count: int
    source_case_category: str | None = None


@dataclass(frozen=True)
class EvidenceRecord:
    qid: str
    business_id: str
    family: EvidenceFamily
    support_status: SupportStatus
    claim_summary: str
    source_locator_type: str
    source_locator: str
    verification_method: str
    confidence: float | None
    raw_content_released: bool
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ReleaseBundle:
    queries: list[QueryRecord]
    answers: list[AnswerRecord]
    evidence: list[EvidenceRecord]
    qrels: dict[str, set[str]]


def case_qid(case: dict[str, Any]) -> str:
    nested = case.get("case")
    if isinstance(nested, dict):
        return str(nested.get("qid", case.get("qid", "")))
    return str(case.get("qid", ""))


def case_subset(case: dict[str, Any]) -> str:
    nested = case.get("case")
    if isinstance(nested, dict):
        return str(nested.get("subset", case.get("subset", "")))
    return str(case.get("subset", ""))


def case_query(case: dict[str, Any]) -> str:
    if case.get("final_query"):
        return str(case["final_query"])
    nested = case.get("case")
    if isinstance(nested, dict):
        return str(nested.get("final_query") or nested.get("query") or case.get("query", ""))
    return str(case.get("query", ""))


def candidate_name(candidate: dict[str, Any]) -> str:
    return str(candidate.get("name") or candidate.get("business_name") or "")

