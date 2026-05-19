"""Query and human-rating quality certification helpers."""

from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from heterqa.core.io import read_jsonl
from heterqa.core.schema import case_qid, case_query, case_subset


def _input_path(input_dir: Path) -> Path:
    for name in ["answer_set_certified_cases.jsonl", "final_cases.jsonl", "construction_cases.jsonl"]:
        path = input_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(f"No HeterQA case file found in {input_dir}")


def _tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+", text.lower())


def word_entropy(queries: list[str]) -> float:
    counter = Counter(token for query in queries for token in _tokens(query))
    total = sum(counter.values())
    if total == 0:
        return 0.0
    return -sum((count / total) * math.log(count / total, 2) for count in counter.values())


def type_token_ratio(queries: list[str]) -> float:
    tokens = [token for query in queries for token in _tokens(query)]
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def query_metrics(input_dir: Path, output_dir: Path) -> Path:
    rows = read_jsonl(_input_path(input_dir))
    queries = [case_query(row) for row in rows]
    token_counts = [len(_tokens(query)) for query in queries]
    answer_counts = [int(row.get("answer_count") or len(row.get("final_answer_business_ids") or [])) for row in rows]
    subset_counts = Counter(case_subset(row) or "<empty>" for row in rows)
    payload = {
        "case_count": len(rows),
        "word_entropy": word_entropy(queries),
        "type_token_ratio": type_token_ratio(queries),
        "average_query_length_tokens": (sum(token_counts) / len(token_counts)) if token_counts else 0.0,
        "average_answer_set_size": (sum(answer_counts) / len(answer_counts)) if answer_counts else 0.0,
        "subset_counts": dict(sorted(subset_counts.items())),
        "qid_count": len({case_qid(row) for row in rows}),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "query_quality_metrics.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _rating_value(row: dict[str, str], key: str) -> float | None:
    text = str(row.get(key) or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def human_rating_summary(ratings_csv: Path, queries_dir: Path, output_dir: Path) -> Path:
    cases = read_jsonl(_input_path(queries_dir))
    subset_by_qid = {case_qid(row): case_subset(row) for row in cases}
    with ratings_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    metrics = ["naturalness", "diversity", "practicality"]
    values_by_metric: dict[str, list[float]] = {metric: [] for metric in metrics}
    values_by_subset: dict[str, dict[str, list[float]]] = defaultdict(lambda: {metric: [] for metric in metrics})
    annotators: set[str] = set()
    rated_qids: set[str] = set()
    for row in rows:
        qid = str(row.get("qid") or "").strip()
        subset = str(row.get("subset") or subset_by_qid.get(qid) or "<empty>")
        if qid:
            rated_qids.add(qid)
        annotator = str(row.get("annotator_id") or "").strip()
        if annotator:
            annotators.add(annotator)
        for metric in metrics:
            value = _rating_value(row, metric)
            if value is None:
                continue
            values_by_metric[metric].append(value)
            values_by_subset[subset][metric].append(value)

    def _summary(values: list[float]) -> dict[str, float]:
        if not values:
            return {"mean": 0.0, "positive_rate": 0.0, "non_negative_rate": 0.0}
        return {
            "mean": sum(values) / len(values),
            "positive_rate": sum(1 for value in values if value > 0) / len(values),
            "non_negative_rate": sum(1 for value in values if value >= 0) / len(values),
        }

    metric_summary = {metric: _summary(values) for metric, values in values_by_metric.items()}
    subset_summary: dict[str, Any] = {}
    for subset, metric_values in sorted(values_by_subset.items()):
        subset_summary[subset] = {metric: _summary(values) for metric, values in metric_values.items()}
    subset_weighted_overall = {
        metric: (
            sum(subset_summary[subset][metric]["mean"] for subset in subset_summary) / len(subset_summary)
            if subset_summary
            else 0.0
        )
        for metric in metrics
    }
    payload = {
        "rating_row_count": len(rows),
        "rated_query_count": len(rated_qids),
        "annotator_count": len(annotators),
        "metrics": metric_summary,
        "subset_summary": subset_summary,
        "subset_weighted_overall": subset_weighted_overall,
        "rating_schema": ["qid", "subset", "annotator_id", *metrics],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "human_rating_summary.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path
