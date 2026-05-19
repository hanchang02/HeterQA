"""Answer-set certification for HeterQA dataset instances."""

from __future__ import annotations

import json
import re
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

from heterqa.core.io import read_jsonl, write_jsonl
from heterqa.core.schema import candidate_name, case_qid, case_query, case_subset


PATH_LIKE_FIELDS = {
    "source_result_file",
    "source_artifact",
    "resolved_context_file",
    "declared_source_context_file",
    "source_file",
    "source_yaml",
    "result_yaml",
    "result_yaml_relpath",
    "review_case_dir",
}


def _input_path(input_dir: Path) -> Path:
    for name in [
        "review_applied_cases.jsonl",
        "contradiction_checked_cases.jsonl",
        "construction_cases.jsonl",
    ]:
        path = input_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(f"No candidate case file found in {input_dir}")


def _candidate_verdict(candidate: dict[str, Any]) -> str:
    return str(candidate.get("verdict") or candidate.get("final_verdict") or "").strip().lower()


def _candidate_business_id(candidate: dict[str, Any]) -> str:
    return str(candidate.get("business_id") or "").strip()


def _yes_candidates(case: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        candidate
        for candidate in case.get("candidates", []) or []
        if _candidate_verdict(candidate) == "yes" and _candidate_business_id(candidate)
    ]


def _has_unresolved_candidate(case: dict[str, Any]) -> bool:
    unresolved = {"pending", "unclear", "rerun", "skip", ""}
    return any(_candidate_verdict(candidate) in unresolved for candidate in case.get("candidates", []) or [])


def _skip_reason(case: dict[str, Any], answer_count: int) -> str | None:
    if answer_count == 0:
        return "zero_answers"
    if answer_count > 10:
        return "more_than_ten_answers"
    if _has_unresolved_candidate(case):
        return "contains_unresolved_candidate"
    return None


def _public_safe_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _public_safe_payload(item)
            for key, item in value.items()
            if str(key) not in PATH_LIKE_FIELDS
        }
    if isinstance(value, list):
        return [_public_safe_payload(item) for item in value]
    return value


def _safe_filename(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return text or "case"


def _answer_ids(case: dict[str, Any]) -> list[str]:
    declared = case.get("final_answer_business_ids")
    if isinstance(declared, list) and declared:
        return [str(item) for item in declared]
    return [_candidate_business_id(candidate) for candidate in _yes_candidates(case)]


def _answer_names(case: dict[str, Any]) -> list[str]:
    declared = case.get("final_answer_business_names")
    if isinstance(declared, list) and declared:
        return [str(item) for item in declared]
    return [candidate_name(candidate) for candidate in _yes_candidates(case)]


def _case_keys(case: dict[str, Any]) -> set[str]:
    keys = {case_qid(case), str(case.get("qid") or ""), str(case.get("case_id") or ""), str(case.get("task_id") or "")}
    nested = case.get("case")
    if isinstance(nested, dict):
        keys.update({str(nested.get("qid") or ""), str(nested.get("case_id") or ""), str(nested.get("task_id") or "")})
    return {key for key in keys if key}


def _case_lookup(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for case in rows:
        for key in _case_keys(case):
            lookup.setdefault(key, case)
    return lookup


def _read_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        payload = json.loads(text)
        if not isinstance(payload, list):
            raise ValueError(f"Instance index must be a list or JSONL: {path}")
        return [row for row in payload if isinstance(row, dict)]
    return read_jsonl(path)


def _instance_match_key(row: dict[str, Any]) -> str:
    for key in ["source_qid", "source_case_id", "case_key", "case_id", "qid"]:
        value = str(row.get(key) or "").strip()
        if value:
            return value
    raise ValueError(f"Instance index row has no match key: {row}")


def _instance_answer_ids(row: dict[str, Any]) -> list[str]:
    for key in ["answer_business_ids", "ground_truths", "answer_ids"]:
        value = row.get(key)
        if isinstance(value, list):
            return [str(item) for item in value]
    return []


def _apply_instance_row(case: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    copied = deepcopy(case)
    output_qid = str(row.get("qid") or case_qid(copied)).strip()
    if output_qid:
        copied["qid"] = output_qid
        if isinstance(copied.get("case"), dict):
            copied["case"]["qid"] = output_qid
    output_query = str(row.get("query") or row.get("final_query") or "").strip()
    current_query = case_query(copied)
    if output_query and current_query and output_query != current_query:
        raise ValueError(f"qid {output_qid or case_qid(copied)} query mismatch between instance index and case")
    if output_query:
        copied["query"] = output_query
        copied["final_query"] = output_query
        if isinstance(copied.get("case"), dict):
            copied["case"]["query"] = output_query
            copied["case"]["final_query"] = output_query
    if row.get("subset"):
        copied["subset"] = str(row["subset"])
        if isinstance(copied.get("case"), dict):
            copied["case"]["subset"] = str(row["subset"])
    expected_answers = _instance_answer_ids(row)
    if expected_answers and sorted(expected_answers) != sorted(_answer_ids(copied)):
        raise ValueError(f"qid {output_qid or case_qid(copied)} answer ids mismatch between instance index and case")
    copied.setdefault("answer_certification", {})["instance_index"] = {
        "applied": True,
        "match_key": _instance_match_key(row),
    }
    return copied


def _select_instance_rows(rows: list[dict[str, Any]], instance_index: Path | None) -> list[dict[str, Any]]:
    if instance_index is None:
        return [deepcopy(case) for case in rows]
    index_rows = _read_json_or_jsonl(instance_index)
    lookup = _case_lookup(rows)
    selected: list[dict[str, Any]] = []
    missing: list[str] = []
    for index_row in index_rows:
        match_key = _instance_match_key(index_row)
        case = lookup.get(match_key)
        if case is None:
            missing.append(match_key)
            continue
        selected.append(_apply_instance_row(case, index_row))
    if missing:
        raise ValueError("Instance index rows missing from candidate input: " + ", ".join(missing[:20]))
    return selected


def _finalize_case(case: dict[str, Any], yes_candidates: list[dict[str, Any]]) -> dict[str, Any]:
    copied = deepcopy(case)
    answers = [deepcopy(candidate) for candidate in yes_candidates]
    answer_ids = [_candidate_business_id(candidate) for candidate in answers]
    answer_names = [candidate_name(candidate) for candidate in answers]
    copied["candidates"] = answers
    copied["survivor_count"] = len(answers)
    copied["final_answer_business_ids"] = answer_ids
    copied["final_answer_business_names"] = answer_names
    copied["answer_count"] = len(answer_ids)
    copied.setdefault("answer_certification", {})["certified"] = True
    copied["answer_certification"]["answer_count"] = len(answer_ids)
    return copied


def _index_row(case: dict[str, Any]) -> dict[str, Any]:
    answers = case.get("candidates", []) or []
    return {
        "qid": case_qid(case),
        "query": case_query(case),
        "subset": case_subset(case),
        "answer_business_ids": [_candidate_business_id(candidate) for candidate in answers],
        "answer_business_names": [candidate_name(candidate) for candidate in answers],
        "answer_count": len(answers),
    }


def _validate_before_write(rows: list[dict[str, Any]], expected_count: int | None) -> list[str]:
    errors: list[str] = []
    if expected_count is not None and len(rows) != expected_count:
        errors.append(f"case count {len(rows)} != {expected_count}")
    seen_qids: set[str] = set()
    for index, case in enumerate(rows, start=1):
        qid = case_qid(case)
        query = case_query(case)
        answer_ids = _answer_ids(case)
        candidate_answer_ids = [_candidate_business_id(candidate) for candidate in _yes_candidates(case)]
        if not qid:
            errors.append(f"case {index} has empty qid")
        elif qid in seen_qids:
            errors.append(f"duplicate qid {qid}")
        seen_qids.add(qid)
        if not query:
            errors.append(f"qid {qid or index} has empty query")
        if not (1 <= len(answer_ids) <= 10):
            errors.append(f"qid {qid or index} has invalid answer count {len(answer_ids)}")
        if sorted(answer_ids) != sorted(candidate_answer_ids):
            errors.append(f"qid {qid or index} answer ids do not match yes candidates")
        if len(answer_ids) != len(set(answer_ids)):
            errors.append(f"qid {qid or index} has duplicate answer ids")
        unresolved = [
            _candidate_business_id(candidate)
            for candidate in case.get("candidates", []) or []
            if _candidate_verdict(candidate) in {"pending", "unclear", "rerun", "skip", ""}
        ]
        if unresolved:
            errors.append(f"qid {qid or index} still has unresolved candidates: {unresolved[:5]}")
    return errors


def _write_materialized_files(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    materialized_dir = output_dir / "materialized"
    materialized_dir.mkdir(parents=True, exist_ok=True)
    for case in rows:
        qid = case_qid(case)
        (materialized_dir / f"{_safe_filename(qid)}.json").write_text(
            json.dumps(_public_safe_payload(case), ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )


def certify_answer_set(
    input_dir: Path,
    output_dir: Path,
    *,
    instance_index: Path | None = None,
    expected_count: int | None = None,
) -> Path:
    source_rows = read_jsonl(_input_path(input_dir))
    selected_source_rows = _select_instance_rows(source_rows, instance_index)
    certified_rows: list[dict[str, Any]] = []
    index_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    verdict_counts = Counter()
    answer_hist = Counter()
    subset_counts = Counter()

    for case in selected_source_rows:
        for candidate in case.get("candidates", []) or []:
            verdict_counts[_candidate_verdict(candidate) or "<empty>"] += 1
        yes = _yes_candidates(case)
        reason = _skip_reason(case, len(yes))
        if reason is not None:
            skipped.append(
                {
                    "qid": case_qid(case),
                    "reason": reason,
                    "yes_count": len(yes),
                    "verdict_counts": dict(
                        Counter(_candidate_verdict(candidate) or "<empty>" for candidate in case.get("candidates", []) or [])
                    ),
                }
            )
            continue
        finalized = _finalize_case(case, yes)
        certified_rows.append(finalized)
        row = _index_row(finalized)
        index_rows.append(row)
        answer_hist[row["answer_count"]] += 1
        subset_counts[case_subset(finalized) or "<empty>"] += 1

    errors = _validate_before_write(certified_rows, expected_count)
    if errors:
        raise ValueError("Cannot certify invalid answer set:\n" + "\n".join(errors))

    output_dir.mkdir(parents=True, exist_ok=True)
    certified_path = output_dir / "answer_set_certified_cases.jsonl"
    final_path = output_dir / "final_cases.jsonl"
    safe_rows = [_public_safe_payload(case) for case in certified_rows]
    write_jsonl(certified_path, safe_rows)
    write_jsonl(final_path, safe_rows)
    write_jsonl(output_dir / "certified_index.jsonl", index_rows)
    write_jsonl(output_dir / "final_index.jsonl", index_rows)
    _write_materialized_files(output_dir, certified_rows)
    summary = {
        "input_case_count": len(source_rows),
        "selected_input_case_count": len(selected_source_rows),
        "certified_case_count": len(certified_rows),
        "certified_answer_pair_count": sum(row["answer_count"] for row in index_rows),
        "skipped_case_count": len(skipped),
        "input_candidate_verdict_counts": dict(sorted(verdict_counts.items())),
        "answer_count_histogram": dict(sorted(answer_hist.items())),
        "subset_counts": dict(sorted(subset_counts.items())),
        "skipped_reason_counts": dict(sorted(Counter(item["reason"] for item in skipped).items())),
        "skipped_samples": skipped[:50],
        "expected_count": expected_count,
        "instance_index_used": instance_index is not None,
        "files": {
            "answer_set_certified_cases": "answer_set_certified_cases.jsonl",
            "final_cases": "final_cases.jsonl",
            "certified_index": "certified_index.jsonl",
            "final_index": "final_index.jsonl",
            "materialized_dir": "materialized/",
        },
    }
    (output_dir / "answer_certification_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    (output_dir / "final_manifest.json").write_text(
        json.dumps(
            {
                "case_count": len(certified_rows),
                "expected_count": expected_count,
                "answer_pair_count": sum(row["answer_count"] for row in index_rows),
                "answer_count_histogram": dict(sorted(answer_hist.items())),
                "subsets": sorted({str(row.get("subset") or "") for row in index_rows}),
                "instance_index_used": instance_index is not None,
                "files": summary["files"],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return certified_path
