"""Validation checks for public HeterQA release directories."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from heterqa.core.io import load_release_bundle, read_jsonl

REQUIRED_EVIDENCE_FAMILIES = {"record_field"}
ALLOWED_FAMILIES = {"record_field", "text", "image", "geo", "kg", "cross_modal"}
PHOTO_LOCATOR_RE = re.compile(r"^[A-Za-z0-9_-]{8,128}$")
LOCAL_PATH_RE = re.compile(r"/(?:m(?:nt)|h(?:ome))/")
OBSOLETE_SUBSET_PREFIXES = tuple(
    left + right + "_"
    for left, right in [
        ("A", "B"),
        ("A", "C"),
        ("A", "D"),
        ("B", "C"),
        ("B", "D"),
        ("C", "D"),
    ]
)
DEPRECATED_RELEASE_TERMS = [
    "_".join(parts)
    for parts in [
        ("source", "case", "id"),
        ("certification", "status"),
        ("not", "available"),
        ("review", "text", "match", "method"),
        ("source", "evidence", "type"),
        ("cross", "modal", "key"),
    ]
]

EXPECTED_JSONL_FIELDS = {
    "data/queries.jsonl": {"qid", "query", "subset", "answer_count"},
    "data/answers.jsonl": {"qid", "answer_business_ids", "answer_business_names", "answer_count", "source_case_category"},
    "data/evidence.jsonl": {
        "qid",
        "business_id",
        "family",
        "support_status",
        "claim_summary",
        "source_locator_type",
        "source_locator",
        "verification_method",
        "confidence",
        "raw_content_released",
        "details",
    },
}

FORBIDDEN_PATTERNS = [
    LOCAL_PATH_RE,
    re.compile(r"20\d{6}_\d{6}"),
    re.compile(r"\bevidence_id\b"),
    re.compile(r"\b" + "V" + "5" + r"\b"),
    re.compile(r"attributes_json|hours_json", re.I),
    re.compile(r"\.(?:jpg|jpeg|png|webp)\b", re.I),
    re.compile(r"Input-Review|Review Segment|prompt\s+trace|model input prompt|debug[_\s]+trace", re.I),
]
FORBIDDEN_PATTERNS.extend(re.compile(rf"\b{re.escape(prefix)}") for prefix in OBSOLETE_SUBSET_PREFIXES)
FORBIDDEN_PATTERNS.extend(re.compile(rf"\b{re.escape(term)}\b", re.I) for term in DEPRECATED_RELEASE_TERMS)


@dataclass
class ValidationReport:
    status: str
    counts: dict[str, int]
    family_counts: dict[str, int]
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_markdown(self) -> str:
        lines = [f"# HeterQA Release Validation: {self.status}", "", "## Counts"]
        for key, value in self.counts.items():
            lines.append(f"- {key}: {value}")
        lines.extend(["", "## Evidence Families"])
        for key, value in sorted(self.family_counts.items()):
            lines.append(f"- {key}: {value}")
        lines.extend(["", "## Warnings"])
        lines.extend([f"- {item}" for item in self.warnings] or ["- None"])
        lines.extend(["", "## Errors"])
        lines.extend([f"- {item}" for item in self.errors] or ["- None"])
        return "\n".join(lines) + "\n"


def _iter_strings(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_strings(item)


def _validate_table_fields(dataset_dir: Path) -> list[str]:
    errors: list[str] = []
    for relpath, expected_fields in EXPECTED_JSONL_FIELDS.items():
        path = dataset_dir / relpath
        if not path.exists():
            errors.append(f"{relpath} is missing")
            continue
        try:
            rows = read_jsonl(path)
        except Exception as exc:
            errors.append(f"{relpath} is not valid JSONL: {exc}")
            continue
        for index, row in enumerate(rows, start=1):
            actual = set(row)
            missing = expected_fields - actual
            extra = actual - expected_fields
            if missing:
                errors.append(f"{relpath} row {index} missing fields {sorted(missing)}")
                break
            if extra:
                errors.append(f"{relpath} row {index} has non-public fields {sorted(extra)}")
                break
    return errors


def _validate_public_file_leakage(dataset_dir: Path) -> list[str]:
    errors: list[str] = []
    public_files = [
        path
        for path in dataset_dir.rglob("*")
        if path.is_file()
        and ".git" not in path.parts
        and path.suffix.lower() in {".jsonl", ".json", ".tsv", ".yaml", ".yml"}
    ]
    for path in public_files:
        text = path.read_text(encoding="utf-8", errors="ignore")
        relpath = str(path.relative_to(dataset_dir))
        for pattern in FORBIDDEN_PATTERNS:
            if pattern.search(text):
                errors.append(f"{relpath} contains forbidden pattern: {pattern.pattern}")
                break
    return errors


def _validate_sidecar_files(dataset_dir: Path, counts: dict[str, int]) -> list[str]:
    errors: list[str] = []
    for relpath in [
        "schemas/query.schema.json",
        "schemas/answer.schema.json",
        "schemas/evidence.schema.json",
        "data/source_manifest.json",
    ]:
        if not (dataset_dir / relpath).exists():
            errors.append(f"{relpath} is missing")
    manifest_path = dataset_dir / "data" / "source_manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            errors.append(f"source_manifest.json is invalid JSON: {exc}")
        else:
            raw_flag = manifest.get("raw_yelp_content_included", manifest.get("raw_content_released", False))
            if raw_flag is not False:
                errors.append("source_manifest must state raw Yelp/source content is not included")
            if manifest.get("query_count") is not None and int(manifest.get("query_count")) != counts.get("queries", -1):
                errors.append("source_manifest query_count does not match queries.jsonl")
            if manifest.get("answer_pair_count") is not None and int(manifest.get("answer_pair_count")) != counts.get("qrels", -1):
                errors.append("source_manifest answer_pair_count does not match qrels")
            if "source_url" not in manifest:
                errors.append("source_manifest must include source_url")
    return errors


def validate_release(dataset_dir: Path) -> ValidationReport:
    errors: list[str] = []
    warnings: list[str] = []
    errors.extend(_validate_table_fields(dataset_dir))
    try:
        bundle = load_release_bundle(dataset_dir)
    except Exception as exc:
        errors.append(f"release tables could not be loaded: {exc}")
        errors.extend(_validate_sidecar_files(dataset_dir, {"queries": 0, "answers": 0, "qrels": 0, "evidence": 0}))
        errors.extend(_validate_public_file_leakage(dataset_dir))
        return ValidationReport(status="FAIL", counts={}, family_counts={}, warnings=warnings, errors=errors)

    counts = {
        "queries": len(bundle.queries),
        "answers": len(bundle.answers),
        "qrels": sum(len(v) for v in bundle.qrels.values()),
        "evidence": len(bundle.evidence),
    }

    query_ids = {row.qid for row in bundle.queries}
    answer_ids = {row.qid for row in bundle.answers}
    if query_ids != answer_ids:
        errors.append("query IDs and answer IDs differ")

    answer_pairs = {
        (row.qid, business_id)
        for row in bundle.answers
        for business_id in row.answer_business_ids
    }
    qrel_pairs = {
        (qid, business_id)
        for qid, business_ids in bundle.qrels.items()
        for business_id in business_ids
    }
    if answer_pairs != qrel_pairs:
        errors.append("qrels do not exactly match answers.answer_business_ids")

    for row in bundle.queries:
        raw = row.__dict__
        for forbidden_field in ["level", "split", "evidence_families"]:
            if forbidden_field in raw:
                errors.append(f"queries.jsonl must not expose {forbidden_field}")
    for row in bundle.answers:
        raw = row.__dict__
        for forbidden_field in ["source_case_id", "certification_status"]:
            if forbidden_field in raw:
                errors.append(f"answers.jsonl must not expose {forbidden_field}")

    family_counts: Counter[str] = Counter()
    evidence_pairs: dict[tuple[str, str], set[str]] = {}
    for idx, row in enumerate(bundle.evidence, start=1):
        family_counts[row.family] += 1
        if row.family not in ALLOWED_FAMILIES:
            errors.append(f"evidence row {idx} has unknown family: {row.family}")
        if row.raw_content_released is not False:
            errors.append(f"evidence row {idx} must set raw_content_released=false")
        pair = (row.qid, row.business_id)
        evidence_pairs.setdefault(pair, set()).add(row.family)
        if row.source_locator_type == "photo_id_stem":
            locators = row.source_locator.split("|")
            if any(not PHOTO_LOCATOR_RE.fullmatch(locator) for locator in locators):
                errors.append(f"evidence row {idx} has invalid photo_id_stem locator")
        for text in _iter_strings(row.__dict__):
            if len(text) > 1400:
                errors.append(f"evidence row {idx} contains an overlong string")
                break
            for pattern in FORBIDDEN_PATTERNS:
                if pattern.search(text):
                    errors.append(f"evidence row {idx} contains forbidden pattern: {pattern.pattern}")
                    break

    for pair in answer_pairs:
        missing = REQUIRED_EVIDENCE_FAMILIES - evidence_pairs.get(pair, set())
        if missing:
            errors.append(f"answer pair {pair} lacks required evidence families {sorted(missing)}")
            break

    extra_pairs = set(evidence_pairs) - answer_pairs
    if extra_pairs:
        errors.append(f"{len(extra_pairs)} evidence pairs are not answer/qrel pairs")

    errors.extend(_validate_sidecar_files(dataset_dir, counts))
    errors.extend(_validate_public_file_leakage(dataset_dir))

    return ValidationReport(
        status="PASS" if not errors else "FAIL",
        counts=counts,
        family_counts=dict(family_counts),
        warnings=warnings,
        errors=errors,
    )


def write_validation_report(output_path: Path, report: ValidationReport) -> Path:
    path = output_path if output_path.suffix else output_path / "reports" / "validation_report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.to_markdown(), encoding="utf-8")
    return path
