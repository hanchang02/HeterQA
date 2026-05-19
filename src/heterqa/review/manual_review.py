"""Manual review packet export and decision application for HeterQA."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from heterqa.audit.contradiction_detection import build_evidence_bundle, derive_consistency_query
from heterqa.core.io import read_jsonl, write_jsonl
from heterqa.core.schema import candidate_name, case_qid, case_query


ALLOWED_DECISIONS = {"yes", "no", "unclear", "rerun", "skip"}
OVERRIDE_DECISIONS = {"yes", "no", "unclear"}
START_RE = re.compile(r"<!--\s*MANUAL_REVIEW_ITEM_START(?P<attrs>.*?)-->", re.DOTALL)
END_TOKEN = "<!-- MANUAL_REVIEW_ITEM_END -->"
ATTR_RE = re.compile(r'([a-zA-Z0-9_]+)="(.*?)"')

REVIEW_FIELDNAMES = [
    "queue_id",
    "qid",
    "business_id",
    "business_name",
    "query",
    "current_verdict",
    "reason_class",
    "suggested_disposition",
    "why_unclear",
    "semantic_consistency_query",
    "confidence",
    "semantic_risk_score",
    "needs_manual_review",
    "review_status",
    "manual_decision",
    "manual_confidence",
    "manual_reason",
    "reviewer",
    "reviewed_at",
    "manual_notes",
    "review_packet_relpath",
]


def _input_path(input_dir: Path) -> Path:
    for name in [
        "contradiction_checked_cases.jsonl",
        "review_applied_cases.jsonl",
        "construction_cases.jsonl",
    ]:
        path = input_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(f"No reviewed case file found in {input_dir}")


def _queue_path(input_dir: Path) -> Path | None:
    path = input_dir / "manual_review_queue.jsonl"
    return path if path.exists() else None


def _safe_slug(value: object) -> str:
    text = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(value or "").strip())
    return text.strip("._") or "item"


def _truncate(value: object, limit: int = 1000) -> str:
    if value is None:
        return ""
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _candidate_verdict(candidate: dict[str, Any]) -> str:
    return str(candidate.get("verdict") or candidate.get("final_verdict") or "").strip().lower()


def _reason_class(candidate: dict[str, Any]) -> str:
    if bool(candidate.get("needs_manual_review")):
        return "needs_manual_review"
    verdict = _candidate_verdict(candidate)
    if verdict == "unclear":
        return "semantic_consistency_unclear"
    return "manual_review_candidate"


def _case_lookup(cases: list[dict[str, Any]]) -> dict[tuple[str, str], tuple[dict[str, Any], dict[str, Any]]]:
    lookup: dict[tuple[str, str], tuple[dict[str, Any], dict[str, Any]]] = {}
    for case in cases:
        qid = case_qid(case)
        for candidate in case.get("candidates", []) or []:
            business_id = str(candidate.get("business_id") or "")
            if business_id:
                lookup[(qid, business_id)] = (case, candidate)
    return lookup


def _queue_rows(input_dir: Path, cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    path = _queue_path(input_dir)
    if path is not None:
        return read_jsonl(path)
    rows: list[dict[str, Any]] = []
    for case in cases:
        for candidate in case.get("candidates", []) or []:
            if _candidate_verdict(candidate) != "unclear" and not bool(candidate.get("needs_manual_review")):
                continue
            qid = case_qid(case)
            business_id = str(candidate.get("business_id") or "")
            consistency = candidate.get("semantic_consistency") if isinstance(candidate.get("semantic_consistency"), dict) else {}
            rows.append(
                {
                    "queue_id": f"{qid}::{business_id}",
                    "qid": qid,
                    "query": case_query(case),
                    "business_id": business_id,
                    "business_name": candidate_name(candidate),
                    "current_verdict": _candidate_verdict(candidate) or "unclear",
                    "reason_class": _reason_class(candidate),
                    "why_unclear": str(candidate.get("why_unclear") or consistency.get("why_unclear") or consistency.get("final_reason") or ""),
                    "semantic_consistency_query": derive_consistency_query(case),
                    "semantic_risk_score": consistency.get("semantic_risk_score"),
                    "needs_manual_review": True,
                }
            )
    return rows


def _normalise_queue_row(
    row: dict[str, Any],
    case: dict[str, Any] | None,
    candidate: dict[str, Any] | None,
    packet_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    qid = str(row.get("qid") or (case_qid(case) if case else ""))
    business_id = str(row.get("business_id") or (candidate or {}).get("business_id") or "")
    consistency = (candidate or {}).get("semantic_consistency") if isinstance((candidate or {}).get("semantic_consistency"), dict) else {}
    reason_class = str(row.get("reason_class") or (_reason_class(candidate) if candidate else "manual_review_candidate"))
    return {
        "queue_id": str(row.get("queue_id") or f"{qid}::{business_id}"),
        "qid": qid,
        "business_id": business_id,
        "business_name": str(row.get("business_name") or candidate_name(candidate or {})),
        "query": str(row.get("query") or (case_query(case) if case else "")),
        "current_verdict": str(row.get("current_verdict") or _candidate_verdict(candidate or {}) or "unclear"),
        "reason_class": reason_class,
        "suggested_disposition": str(row.get("suggested_disposition") or "manual_review"),
        "why_unclear": str(row.get("why_unclear") or (candidate or {}).get("why_unclear") or consistency.get("why_unclear") or consistency.get("final_reason") or ""),
        "semantic_consistency_query": str(row.get("semantic_consistency_query") or (derive_consistency_query(case) if case else "")),
        "confidence": str(row.get("confidence") or (candidate or {}).get("confidence") or ""),
        "semantic_risk_score": str(row.get("semantic_risk_score") or consistency.get("semantic_risk_score") or ""),
        "needs_manual_review": str(row.get("needs_manual_review") or (candidate or {}).get("needs_manual_review") or True),
        "review_status": str(row.get("review_status") or "todo"),
        "manual_decision": str(row.get("manual_decision") or ""),
        "manual_confidence": str(row.get("manual_confidence") or ""),
        "manual_reason": str(row.get("manual_reason") or ""),
        "reviewer": str(row.get("reviewer") or ""),
        "reviewed_at": str(row.get("reviewed_at") or ""),
        "manual_notes": str(row.get("manual_notes") or ""),
        "review_packet_relpath": packet_path.relative_to(output_dir).as_posix(),
    }


def _render_evidence_section(title: str, items: list[dict[str, Any]], max_items: int = 8) -> list[str]:
    lines = [f"### {title}", ""]
    if not items:
        return lines + ["- None", ""]
    for item in items[:max_items]:
        lines.extend(
            [
                f"#### {_truncate(item.get('id'), 80)} {_truncate(item.get('title'), 140)}".strip(),
                "",
                _truncate(item.get("summary"), 1400) or "<empty>",
                "",
            ]
        )
    return lines


def _write_packet(path: Path, row: dict[str, Any], case: dict[str, Any] | None, candidate: dict[str, Any] | None) -> None:
    candidate = candidate or {}
    bundle = build_evidence_bundle(case or {}, candidate) if case is not None else {}
    contradiction = candidate.get("contradiction_detection") if isinstance(candidate.get("contradiction_detection"), dict) else {}
    consistency = candidate.get("semantic_consistency") if isinstance(candidate.get("semantic_consistency"), dict) else {}
    lines = [
        "# HeterQA Manual Review Packet",
        "",
        "## Review Task",
        "",
        f"- queue_id: {row['queue_id']}",
        f"- qid: {row['qid']}",
        f"- business_id: {row['business_id']}",
        f"- business_name: {_truncate(row['business_name'], 240)}",
        f"- current_verdict: {row['current_verdict']}",
        f"- reason_class: {row['reason_class']}",
        f"- suggested_disposition: {row['suggested_disposition']}",
        f"- why_unclear: {_truncate(row['why_unclear'], 1200)}",
        "",
        "## Query",
        "",
        _truncate(row["query"], 1600) or "<empty>",
        "",
        "## Semantic Consistency Query",
        "",
        _truncate(row["semantic_consistency_query"], 1600) or "<empty>",
        "",
        "## Candidate State",
        "",
        f"- origin: {_truncate(candidate.get('origin'), 240)}",
        f"- is_active: {candidate.get('is_active')}",
        f"- drop_reason: {_truncate(candidate.get('drop_reason'), 500)}",
        f"- scores: {_truncate(candidate.get('scores'), 900)}",
        f"- text_verify: {_truncate(candidate.get('text_verify'), 900)}",
        f"- image_verify: {_truncate(candidate.get('image_verify'), 900)}",
        f"- kg_verify: {_truncate(candidate.get('kg_verify'), 900)}",
        f"- text_to_image_verify: {_truncate(candidate.get('text_to_image_verify'), 900)}",
        f"- image_to_text_verify: {_truncate(candidate.get('image_to_text_verify'), 900)}",
        "",
        "## Certification Summaries",
        "",
        f"- contradiction_detection: {_truncate(contradiction, 1400)}",
        f"- semantic_consistency: {_truncate(consistency, 1400)}",
        "",
    ]
    lines.extend(_render_evidence_section("Structured Evidence", bundle.get("structured") or []))
    lines.extend(_render_evidence_section("Construction Evidence", bundle.get("construction") or []))
    lines.extend(_render_evidence_section("Trace Evidence", bundle.get("traces") or []))
    lines.extend(_render_evidence_section("Contradiction Evidence", bundle.get("contradiction") or []))
    lines.extend(
        [
            "## Manual Decision Block",
            "",
            f'<!-- MANUAL_REVIEW_ITEM_START queue_id="{row["queue_id"]}" qid="{row["qid"]}" business_id="{row["business_id"]}" -->',
            f"- queue_id: {row['queue_id']}",
            f"- review_status: {row['review_status']}",
            f"- manual_decision: {row['manual_decision']}",
            f"- manual_confidence: {row['manual_confidence']}",
            f"- manual_reason: {row['manual_reason']}",
            f"- reviewer: {row['reviewer']}",
            f"- reviewed_at: {row['reviewed_at']}",
            f"- manual_notes: {row['manual_notes']}",
            END_TOKEN,
            "",
            "Valid manual_decision values: yes, no, unclear, rerun, skip.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def _write_case_yaml(case_dir: Path, rows: list[dict[str, Any]], case: dict[str, Any] | None) -> None:
    payload = {
        "qid": rows[0]["qid"] if rows else "",
        "query": case_query(case) if case else "",
        "instructions": {
            "editable_fields": [
                "review_status",
                "manual_decision",
                "manual_confidence",
                "manual_reason",
                "reviewer",
                "reviewed_at",
                "manual_notes",
            ],
            "valid_manual_decision": sorted(ALLOWED_DECISIONS),
            "review_status_values": ["todo", "doing", "done"],
        },
        "items": rows,
    }
    with (case_dir / "MANUAL_REVIEW.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, allow_unicode=True, sort_keys=False)


def _write_review_csv(rows: list[dict[str, Any]], review_csv: Path) -> None:
    with review_csv.open("w", encoding="utf-8", newline="") as dst:
        writer = csv.DictWriter(dst, fieldnames=REVIEW_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def export_manual_review(input_dir: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    packet_root = output_dir / "review_packets"
    cases = read_jsonl(_input_path(input_dir))
    lookup = _case_lookup(cases)
    exported: list[dict[str, Any]] = []
    grouped_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    grouped_cases: dict[str, dict[str, Any] | None] = {}

    for raw_row in _queue_rows(input_dir, cases):
        qid = str(raw_row.get("qid") or "")
        business_id = str(raw_row.get("business_id") or "")
        case, candidate = lookup.get((qid, business_id), (None, None))
        case_dir = packet_root / f"q{_safe_slug(qid)}"
        packet_path = case_dir / f"{_safe_slug(business_id)}.md"
        row = _normalise_queue_row(raw_row, case, candidate, packet_path, output_dir)
        _write_packet(packet_path, row, case, candidate)
        exported.append(row)
        grouped_rows[row["qid"]].append(row)
        grouped_cases[row["qid"]] = case

    for qid, rows in grouped_rows.items():
        _write_case_yaml(packet_root / f"q{_safe_slug(qid)}", rows, grouped_cases.get(qid))

    review_csv = output_dir / "manual_review.csv"
    _write_review_csv(exported, review_csv)
    summary = {
        "input_case_count": len(cases),
        "review_item_count": len(exported),
        "reason_class_counts": dict(Counter(row["reason_class"] for row in exported)),
        "valid_manual_decisions": sorted(ALLOWED_DECISIONS),
    }
    (output_dir / "manual_review_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return review_csv


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _as_float(value: object) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = float(text)
    except ValueError:
        return None
    return max(0.0, min(1.0, parsed))


def _read_decisions(review_dir: Path) -> list[dict[str, str]]:
    for name in ["manual_review_collected.csv", "manual_review.csv"]:
        path = review_dir / name
        if path.exists():
            with path.open("r", encoding="utf-8", newline="") as handle:
                return list(csv.DictReader(handle))
    raise FileNotFoundError(f"No manual review CSV found in {review_dir}")


def _queue_id(qid: str, business_id: str) -> str:
    return f"{qid}::{business_id}"


def _normalise_decision_row(row: dict[str, str]) -> dict[str, Any]:
    qid = str(row.get("qid") or "")
    business_id = str(row.get("business_id") or "")
    queue_id = str(row.get("queue_id") or _queue_id(qid, business_id))
    if not qid and "::" in queue_id:
        qid = queue_id.split("::", 1)[0]
    if not business_id and "::" in queue_id:
        business_id = queue_id.split("::", 1)[1]
    decision = str(row.get("manual_decision") or "").strip().lower()
    status = str(row.get("review_status") or "").strip().lower()
    return {
        "queue_id": queue_id,
        "qid": qid,
        "business_id": business_id,
        "manual_decision": decision,
        "review_status": status or ("done" if decision in ALLOWED_DECISIONS else ""),
        "manual_confidence": _as_float(row.get("manual_confidence")),
        "manual_reason": str(row.get("manual_reason") or "").strip(),
        "reviewer": str(row.get("reviewer") or "").strip(),
        "reviewed_at": str(row.get("reviewed_at") or "").strip() or _now_iso(),
        "manual_notes": str(row.get("manual_notes") or "").strip(),
        "raw_row": row,
    }


def _decision_map(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, Any]]:
    decisions: dict[tuple[str, str], dict[str, Any]] = {}
    for raw_row in rows:
        row = _normalise_decision_row(raw_row)
        if row["manual_decision"] not in ALLOWED_DECISIONS:
            continue
        if not row["qid"] or not row["business_id"]:
            continue
        decisions[(row["qid"], row["business_id"])] = row
    return decisions


def _apply_override(candidate: dict[str, Any], decision: dict[str, Any]) -> None:
    manual_decision = decision["manual_decision"]
    previous_verdict = _candidate_verdict(candidate)
    previous_reason = str(candidate.get("final_reason") or candidate.get("why_unclear") or "")
    confidence = decision["manual_confidence"]
    candidate["manual_review"] = {
        "source": "manual_review",
        "queue_id": decision["queue_id"],
        "review_status": decision["review_status"] or "done",
        "manual_decision": manual_decision,
        "manual_confidence": confidence,
        "manual_reason": decision["manual_reason"],
        "manual_notes": decision["manual_notes"],
        "reviewer": decision["reviewer"],
        "reviewed_at": decision["reviewed_at"],
        "previous_verdict": previous_verdict,
        "previous_reason": previous_reason,
    }
    candidate["manual_reviewed"] = True
    candidate["verdict"] = manual_decision
    candidate["final_verdict"] = manual_decision
    candidate["confidence"] = confidence if confidence is not None else (1.0 if manual_decision in {"yes", "no"} else candidate.get("confidence"))
    stage_status = candidate.setdefault("stage_status", {})
    if not isinstance(stage_status, dict):
        stage_status = {}
        candidate["stage_status"] = stage_status
    if manual_decision == "unclear":
        candidate["is_active"] = bool(candidate.get("is_active", True))
        candidate["needs_manual_review"] = True
        candidate["why_unclear"] = decision["manual_reason"] or previous_reason or "Manual review kept this candidate unresolved."
        candidate["final_reason"] = candidate["why_unclear"]
        stage_status["manual_review"] = "manual_unclear"
    elif manual_decision == "no":
        candidate["is_active"] = False
        candidate["drop_reason"] = "manual_review_no"
        candidate["needs_manual_review"] = False
        candidate["why_unclear"] = None
        candidate["final_reason"] = decision["manual_reason"] or "Manual review override: no."
        stage_status["manual_review"] = "manual_no"
    else:
        candidate["is_active"] = True
        candidate.pop("drop_reason", None)
        candidate["needs_manual_review"] = False
        candidate["why_unclear"] = None
        candidate["final_reason"] = decision["manual_reason"] or "Manual review override: yes."
        stage_status["manual_review"] = "manual_yes"


def _record_non_override(candidate: dict[str, Any], decision: dict[str, Any]) -> None:
    stage_status = candidate.setdefault("stage_status", {})
    if not isinstance(stage_status, dict):
        stage_status = {}
        candidate["stage_status"] = stage_status
    candidate["manual_review"] = {
        "source": "manual_review",
        "queue_id": decision["queue_id"],
        "review_status": decision["review_status"] or "done",
        "manual_decision": decision["manual_decision"],
        "manual_confidence": decision["manual_confidence"],
        "manual_reason": decision["manual_reason"],
        "manual_notes": decision["manual_notes"],
        "reviewer": decision["reviewer"],
        "reviewed_at": decision["reviewed_at"],
        "previous_verdict": _candidate_verdict(candidate),
        "action_applied_to_verdict": False,
    }
    candidate["manual_reviewed"] = False
    candidate["needs_manual_review"] = True
    stage_status["manual_review"] = f"manual_{decision['manual_decision']}_recorded"


def _recompute_answer_set(case: dict[str, Any]) -> dict[str, Any]:
    yes_candidates = [
        candidate
        for candidate in case.get("candidates", []) or []
        if _candidate_verdict(candidate) == "yes" and candidate.get("business_id")
    ]
    answer_ids = [str(candidate["business_id"]) for candidate in yes_candidates]
    answer_names = [candidate_name(candidate) for candidate in yes_candidates]
    verdict_counts = Counter(_candidate_verdict(candidate) or "<empty>" for candidate in case.get("candidates", []) or [])
    case["final_answer_business_ids"] = answer_ids
    case["final_answer_business_names"] = answer_names
    case["answer_count"] = len(answer_ids)
    case.setdefault("audit_summary", {})["manual_review"] = {
        "answer_count": len(answer_ids),
        "verdict_counts": dict(sorted(verdict_counts.items())),
    }
    return {"answer_count": len(answer_ids), "verdict_counts": dict(sorted(verdict_counts.items()))}


def apply_manual_review(review_dir: Path, input_dir: Path, output_dir: Path) -> Path:
    decisions = _decision_map(_read_decisions(review_dir))
    rows = read_jsonl(_input_path(input_dir))
    applied_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    decision_counts = Counter()
    status_counts = Counter()
    applied_case_count = 0

    for case in rows:
        qid = case_qid(case)
        case_touched = False
        for candidate in case.get("candidates", []) or []:
            business_id = str(candidate.get("business_id") or "")
            decision = decisions.get((qid, business_id))
            if decision is None:
                continue
            manual_decision = decision["manual_decision"]
            status_counts[decision["review_status"] or "<empty>"] += 1
            decision_counts[manual_decision] += 1
            if manual_decision in OVERRIDE_DECISIONS:
                _apply_override(candidate, decision)
                case_touched = True
                applied_rows.append({"queue_id": decision["queue_id"], "qid": qid, "business_id": business_id, "action": manual_decision, "applied": True})
            else:
                _record_non_override(candidate, decision)
                applied_rows.append({"queue_id": decision["queue_id"], "qid": qid, "business_id": business_id, "action": manual_decision, "applied": False})
        summary = _recompute_answer_set(case)
        if case_touched:
            applied_case_count += 1
        case.setdefault("audit_summary", {})["manual_review"]["case_touched"] = case_touched
        case.setdefault("audit_summary", {})["manual_review"]["summary_after_apply"] = summary

    seen_keys = {
        (str(row["qid"]), str(row["business_id"]))
        for row in applied_rows
        if row.get("qid") and row.get("business_id")
    }
    for (qid, business_id), decision in decisions.items():
        if (qid, business_id) not in seen_keys:
            skipped_rows.append({"queue_id": decision["queue_id"], "qid": qid, "business_id": business_id, "reason": "candidate_not_found"})

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "review_applied_cases.jsonl"
    write_jsonl(output_path, rows)
    write_jsonl(output_dir / "manual_review_applied.jsonl", applied_rows)
    write_jsonl(output_dir / "manual_review_skipped.jsonl", skipped_rows)
    summary = {
        "input_case_count": len(rows),
        "manual_decision_row_count": len(decisions),
        "applied_case_count": applied_case_count,
        "applied_decision_count": len(applied_rows),
        "skipped_decision_count": len(skipped_rows),
        "manual_decision_counts": dict(sorted(decision_counts.items())),
        "review_status_counts": dict(sorted(status_counts.items())),
    }
    (output_dir / "manual_review_apply_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_path


def collect_manual_decisions(review_dir: Path, output_csv: Path | None = None) -> Path:
    rows: dict[str, dict[str, str]] = {}
    csv_path = review_dir / "manual_review.csv"
    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                queue_id = str(row.get("queue_id") or "")
                if queue_id:
                    rows[queue_id] = {key: str(value or "") for key, value in row.items()}

    for md_path in sorted((review_dir / "review_packets").rglob("*.md")):
        text = md_path.read_text(encoding="utf-8")
        for match in START_RE.finditer(text):
            end = text.find(END_TOKEN, match.end())
            if end == -1:
                continue
            attrs = dict(ATTR_RE.findall(match.group("attrs")))
            block = text[match.end() : end]
            values = {}
            for line in block.splitlines():
                if not line.strip().startswith("-") or ":" not in line:
                    continue
                key, value = line.strip()[1:].split(":", 1)
                values[key.strip()] = value.strip()
            queue_id = attrs.get("queue_id") or values.get("queue_id") or ""
            if not queue_id:
                continue
            row = rows.setdefault(queue_id, {})
            row.update(values)
            row.setdefault("queue_id", queue_id)
            row.setdefault("qid", attrs.get("qid", ""))
            row.setdefault("business_id", attrs.get("business_id", ""))

    output = output_csv or (review_dir / "manual_review_collected.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        for row in rows.values():
            writer.writerow(row)
    return output
