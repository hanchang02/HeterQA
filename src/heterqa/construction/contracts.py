"""Data contracts for the HeterQA construction mainline."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal


EvidenceFamily = Literal["record_field", "geo", "text", "image", "kg", "cross_modal"]
CandidateVerdict = Literal["pending", "yes", "no", "unclear", "drop"]


@dataclass
class LLMCallTrace:
    """Sanitizable model-call summary."""

    stage: str = "unknown"
    prompt: str = ""
    response: str = ""
    image_paths: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    token_usage: dict[str, int] = field(default_factory=dict)
    is_json_valid: bool = True
    error_msg: str | None = None

    def public_dict(self) -> dict[str, Any]:
        """Return a summary that omits prompts, model responses, and image paths."""

        return {
            "stage": self.stage,
            "latency_ms": self.latency_ms,
            "token_usage": self.token_usage,
            "is_json_valid": self.is_json_valid,
            "error_msg": self.error_msg,
        }


@dataclass
class VerificationResult:
    """Judgement attached to a candidate verification slot."""

    judgement: str = "no"
    confidence: float = 0.0
    reason: str = ""
    trace: LLMCallTrace | None = None
    evidence_locator_type: str = ""
    evidence_locator: str = ""
    evidence_summary: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_passed(self, threshold: float) -> bool:
        return str(self.judgement).lower() in {"yes", "true", "pass", "passed"} and self.confidence >= threshold


# Public alias used by construction modules.
LLMResult = VerificationResult


@dataclass(frozen=True)
class StructuredFilter:
    """Structured predicate selected before heterogeneous evidence enrichment."""

    field: str
    operator: str
    value: Any
    is_numeric: bool | None = None
    raw: Any = None

    @classmethod
    def from_raw(cls, raw: Any) -> "StructuredFilter":
        if isinstance(raw, cls):
            return raw
        if isinstance(raw, dict):
            return cls(
                field=str(raw.get("field") or raw.get("column") or raw.get("name")),
                operator=str(raw.get("operator", "=")),
                value=raw.get("value"),
                is_numeric=raw.get("is_numeric"),
                raw=raw,
            )
        if isinstance(raw, (list, tuple)) and len(raw) >= 3:
            if raw[2] == "categories":
                return cls(field="categories", operator="semantic_category", value=raw[1], raw=list(raw))
            return cls.from_source_predicate(str(raw[0]), raw[1], bool(raw[2]), raw=list(raw))
        raise ValueError(f"Unsupported structured filter: {raw!r}")

    @classmethod
    def from_source_predicate(
        cls,
        predicate: str,
        value: Any,
        is_numeric: bool | None,
        *,
        raw: Any = None,
    ) -> "StructuredFilter":
        """Parse source SQL-like predicates emitted by structured selection.

        Some configured inputs represent predicates as tuples such as
        ("city = %s", "Philadelphia", False) or ("stars >= 4.0", 4.0, True).
        The generation logic needs the actual operator, not the DB parameter marker.
        """

        text = predicate.strip()
        match = re.match(r"^`?([A-Za-z0-9_]+)`?\s+(LIKE|IS\s+NOT\s+NULL|>=|<=|==|=|>|<)\s*(.*)$", text, re.I)
        if not match:
            parts = text.split(maxsplit=1)
            return cls(
                field=parts[0] if parts else text,
                operator=parts[1] if len(parts) > 1 else "=",
                value=value,
                is_numeric=is_numeric,
                raw=raw,
            )
        field_name = match.group(1)
        operator = re.sub(r"\s+", " ", match.group(2).upper())
        trailing = match.group(3).strip()
        parsed_value = value
        if operator == "LIKE" and isinstance(parsed_value, str):
            parsed_value = parsed_value.strip("%")
        if parsed_value is None and trailing and "%s" not in trailing:
            literal = trailing.strip().strip("'").strip('"')
            parsed_value = literal
        return cls(field=field_name, operator=operator, value=parsed_value, is_numeric=is_numeric, raw=raw)

    def to_source_tuple(self) -> list[Any]:
        if self.raw is not None:
            return list(self.raw) if isinstance(self.raw, (list, tuple)) else self.raw
        if self.operator == "semantic_category":
            return ["categories", self.value, "categories"]
        return [f"{self.field} {self.operator}".strip(), self.value, bool(self.is_numeric)]


@dataclass(frozen=True)
class GeoConstraint:
    reference_latitude: float
    reference_longitude: float
    radius_km: float | None = None
    direction: str | None = None
    top_k: int | None = None
    anchor_business_id: str | None = None
    anchor_name: str | None = None
    anchor_kind: str | None = None
    relation_type: str | None = None
    nl_text: str = ""
    payload: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_raw(cls, raw: dict[str, Any] | None) -> "GeoConstraint | None":
        if not raw:
            return None
        return cls(
            reference_latitude=float(raw["reference_latitude"]),
            reference_longitude=float(raw["reference_longitude"]),
            radius_km=float(raw["radius_km"]) if raw.get("radius_km") is not None else None,
            direction=str(raw["direction"]).upper() if raw.get("direction") else None,
            top_k=int(raw["top_k"]) if raw.get("top_k") is not None else None,
            anchor_business_id=raw.get("anchor_business_id"),
            anchor_name=raw.get("anchor_name"),
            anchor_kind=raw.get("anchor_kind"),
            relation_type=raw.get("relation_type"),
            nl_text=str(raw.get("nl_text", "")),
            payload=dict(raw.get("payload", {})),
        )


@dataclass(frozen=True)
class GenerationCase:
    """Input case specification for one generated query instance."""

    qid: str
    subset: str
    structured_filters: list[StructuredFilter] = field(default_factory=list)
    seed_business_ids: list[str] = field(default_factory=list)
    seed_records: list[dict[str, Any]] = field(default_factory=list)
    geo_constraint: GeoConstraint | None = None
    recall_query: str = ""
    text_query: str = ""
    image_query: str = ""
    kg_query: str = ""
    final_query: str = ""
    target_answer_count_min: int = 1
    target_answer_count_max: int = 10
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_raw(cls, raw: dict[str, Any]) -> "GenerationCase":
        return cls(
            qid=str(raw.get("qid", "")),
            subset=str(raw.get("subset", "")),
            structured_filters=[StructuredFilter.from_raw(item) for item in raw.get("structured_filters", [])],
            seed_business_ids=[str(item) for item in raw.get("seed_business_ids", [])],
            seed_records=[dict(item) for item in raw.get("seed_records", [])],
            geo_constraint=GeoConstraint.from_raw(raw.get("geo_constraint")),
            recall_query=str(raw.get("recall_query", "")),
            text_query=str(raw.get("text_query", "")),
            image_query=str(raw.get("image_query", "")),
            kg_query=str(raw.get("kg_query", "")),
            final_query=str(raw.get("final_query", "")),
            target_answer_count_min=int(raw.get("target_answer_count_min", 1)),
            target_answer_count_max=int(raw.get("target_answer_count_max", 10)),
            metadata=dict(raw.get("metadata", {})),
        )


@dataclass(frozen=True)
class BusinessRecord:
    business_id: str
    name: str = ""
    fields: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_raw(cls, raw: dict[str, Any]) -> "BusinessRecord":
        business_id = str(raw.get("business_id") or raw.get("id") or "")
        if not business_id:
            raise ValueError(f"Business record lacks business_id: {raw!r}")
        fields = dict(raw)
        fields.pop("id", None)
        return cls(business_id=business_id, name=str(raw.get("name", "")), fields=fields)


@dataclass
class EvidenceItem:
    family: EvidenceFamily
    source_locator_type: str
    source_locator: str
    summary: str
    score: float | None = None
    supports: bool | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_verification(cls, family: EvidenceFamily, result: VerificationResult) -> "EvidenceItem":
        return cls(
            family=family,
            source_locator_type=result.evidence_locator_type,
            source_locator=result.evidence_locator,
            summary=result.evidence_summary or result.reason,
            score=result.confidence,
            supports=result.is_passed(0.0),
            metadata=dict(result.metadata),
        )


@dataclass
class CandidateState:
    """Candidate lifecycle state for the construction pipeline."""

    business_id: str
    name: str | None = None
    is_active: bool = True
    origin: str = "unknown"
    is_final_hit: bool = False
    drop_reason: str | None = None
    text_verify: VerificationResult | None = None
    image_verify: VerificationResult | None = None
    geo_verify: VerificationResult | None = None
    text_to_image_verify: VerificationResult | None = None
    image_to_text_verify: VerificationResult | None = None
    kg_verify: VerificationResult | None = None
    scores: dict[str, float | str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    evidence: list[EvidenceItem] = field(default_factory=list)
    verdict: CandidateVerdict = "pending"
    stage_status: dict[str, str] = field(default_factory=dict)

    @property
    def fields(self) -> dict[str, Any]:
        return self.metadata

    def add_evidence(self, evidence: EvidenceItem) -> None:
        self.evidence.append(evidence)

    def set_verification(self, slot: str, result: VerificationResult, family: EvidenceFamily) -> None:
        setattr(self, slot, result)
        if result.evidence_locator or result.evidence_summary or result.reason:
            self.evidence.append(EvidenceItem.from_verification(family, result))

    def drop(self, reason: str) -> None:
        self.is_active = False
        self.verdict = "drop"
        self.drop_reason = reason


# Public alias used by construction modules.
SearchCandidate = CandidateState


@dataclass
class PipelineContext:
    """Case-level state for the construction pipeline."""

    task_id: str
    recall_query: str = ""
    geo_query: str = ""
    text_query: str = ""
    image_query: str = ""
    kg_query: str = ""
    final_query: str = ""
    all_filters: list[Any] = field(default_factory=list)
    seed_records: list[dict[str, Any]] = field(default_factory=list)
    candidates: list[CandidateState] = field(default_factory=list)
    refined_results: list[CandidateState] = field(default_factory=list)
    global_traces: list[LLMCallTrace] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=lambda: {"counts": {}, "errors": [], "kg_state": []})
    metadata: dict[str, Any] = field(default_factory=dict)

    def find_candidate(self, business_id: str) -> CandidateState | None:
        for candidate in self.candidates:
            if candidate.business_id == business_id:
                return candidate
        return None

    def active_candidates(self) -> list[CandidateState]:
        return [candidate for candidate in self.candidates if candidate.is_active]


@dataclass
class StageSummary:
    name: str
    input_candidates: int
    output_candidates: int
    dropped_candidates: int = 0
    unclear_candidates: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationResult:
    case: GenerationCase
    candidates: list[CandidateState]
    final_answer_business_ids: list[str]
    stage_summaries: list[StageSummary] = field(default_factory=list)
    final_query: str = ""
    context: PipelineContext | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = _strip_runtime_details(asdict(self))
        if payload.get("context"):
            payload["context"] = None
        return _json_safe(payload)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, set):
        return sorted(_json_safe(item) for item in value)
    if isinstance(value, Decimal):
        return int(value) if value == value.to_integral_value() else float(value)
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception as exc:
            raise TypeError(f"Object with item() could not be converted to JSON-safe value: {type(value).__name__}") from exc
    return value


def _strip_runtime_details(value: Any) -> Any:
    """Remove prompt/model-response details from serialized construction artifacts."""

    if isinstance(value, dict):
        trace_keys = {"stage", "prompt", "response", "image_paths", "latency_ms", "token_usage", "is_json_valid", "error_msg"}
        if trace_keys.issubset(set(value)):
            return {
                "stage": value.get("stage"),
                "latency_ms": value.get("latency_ms", 0.0),
                "token_usage": value.get("token_usage", {}),
                "is_json_valid": value.get("is_json_valid", True),
                "error_msg": value.get("error_msg"),
            }
        return {key: _strip_runtime_details(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_strip_runtime_details(item) for item in value]
    return value


@dataclass(frozen=True)
class ConstructionSettings:
    enabled_geo: bool = False
    enabled_text: bool = False
    enabled_image: bool = False
    enabled_kg: bool = False
    enable_cross_modal: bool = True
    top_k: int = 300
    text_rerank_thres: float = 0.6
    image_coarse_thres: float = 0.65
    image_reranker_thres: float = 0.25
    llm_judge_threshold: float = 0.7
    evidence_limit: int = 5
    support_threshold: float = 0.7
    allow_missing_structured_values: bool = True
    target_answer_count_min: int = 1
    target_answer_count_max: int = 10
    max_image_query_attempts: int = 5
    kg_max_retries: int = 5
    kg_min_survivors: int = 1
    db_name: str = "yelpdb"
    structured_selection_compact: bool = True
    structured_selection_seeded: bool = False
    field_selection_mode: str = "random"
    structured_selection_max_retries: int = 3
    geo_seed: int | None = 2233
    geo_anchor_choice_probs: dict[str, float] = field(default_factory=lambda: {"user": 0.5, "poi": 0.5})
    geo_relation_choice_probs: dict[str, float] = field(
        default_factory=lambda: {"within_radius": 1 / 3, "direction": 1 / 3, "nearest": 1 / 3}
    )
    geo_direction_choice_probs: dict[str, float] = field(
        default_factory=lambda: {"N": 1 / 8, "NE": 1 / 8, "E": 1 / 8, "SE": 1 / 8, "S": 1 / 8, "SW": 1 / 8, "W": 1 / 8, "NW": 1 / 8}
    )
    geo_radius_km_range: tuple[float, float] = (0.5, 10.0)
    geo_user_offset_km_range: tuple[float, float] = (0.5, 10.0)
    geo_nearest_topk_max: int = 1
    geo_enable_direction: bool = True
    geo_max_anchor_scans: int = 1024
    geo_determinism: str = "stateless"

    @classmethod
    def from_raw(cls, raw: dict[str, Any] | None) -> "ConstructionSettings":
        raw = raw or {}
        return cls(
            enabled_geo=bool(raw.get("enabled_geo", False)),
            enabled_text=bool(raw.get("enabled_text", False)),
            enabled_image=bool(raw.get("enabled_image", False)),
            enabled_kg=bool(raw.get("enabled_kg", False)),
            enable_cross_modal=bool(raw.get("enable_cross_modal", True)),
            top_k=int(raw.get("top_k", 300)),
            text_rerank_thres=float(raw.get("text_rerank_thres", 0.6)),
            image_coarse_thres=float(raw.get("image_coarse_thres", 0.65)),
            image_reranker_thres=float(raw.get("image_reranker_thres", 0.25)),
            llm_judge_threshold=float(raw.get("llm_judge_threshold", raw.get("support_threshold", 0.7))),
            evidence_limit=int(raw.get("evidence_limit", 5)),
            support_threshold=float(raw.get("support_threshold", 0.7)),
            allow_missing_structured_values=bool(raw.get("allow_missing_structured_values", True)),
            target_answer_count_min=int(raw.get("target_answer_count_min", 1)),
            target_answer_count_max=int(raw.get("target_answer_count_max", 10)),
            max_image_query_attempts=int(raw.get("max_image_query_attempts", 5)),
            kg_max_retries=int(raw.get("kg_max_retries", 5)),
            kg_min_survivors=int(raw.get("kg_min_survivors", 1)),
            db_name=str(raw.get("db_name", "yelpdb")),
            structured_selection_compact=bool(raw.get("structured_selection_compact", True)),
            structured_selection_seeded=bool(raw.get("structured_selection_seeded", False)),
            field_selection_mode=str(raw.get("field_selection_mode", "random")),
            structured_selection_max_retries=int(raw.get("structured_selection_max_retries", 3)),
            geo_seed=int(raw["geo_seed"]) if raw.get("geo_seed") is not None else None,
            geo_anchor_choice_probs=dict(raw.get("geo_anchor_choice_probs", {"user": 0.5, "poi": 0.5})),
            geo_relation_choice_probs=dict(
                raw.get("geo_relation_choice_probs", {"within_radius": 1 / 3, "direction": 1 / 3, "nearest": 1 / 3})
            ),
            geo_direction_choice_probs=dict(
                raw.get(
                    "geo_direction_choice_probs",
                    {"N": 1 / 8, "NE": 1 / 8, "E": 1 / 8, "SE": 1 / 8, "S": 1 / 8, "SW": 1 / 8, "W": 1 / 8, "NW": 1 / 8},
                )
            ),
            geo_radius_km_range=tuple(float(x) for x in raw.get("geo_radius_km_range", [0.5, 10.0])),  # type: ignore[arg-type]
            geo_user_offset_km_range=tuple(float(x) for x in raw.get("geo_user_offset_km_range", [0.5, 10.0])),  # type: ignore[arg-type]
            geo_nearest_topk_max=int(raw.get("geo_nearest_topk_max", 1)),
            geo_enable_direction=bool(raw.get("geo_enable_direction", True)),
            geo_max_anchor_scans=int(raw.get("geo_max_anchor_scans", 1024)),
            geo_determinism=str(raw.get("geo_determinism", "stateless")),
        )
