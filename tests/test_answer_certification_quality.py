from __future__ import annotations

import csv
import json
from pathlib import Path

from heterqa.finalize.certify_answer_set import certify_answer_set
from heterqa.quality.metrics import human_rating_summary, query_metrics, type_token_ratio, word_entropy


def _write_case(path: Path) -> None:
    path.mkdir()
    (path / "review_applied_cases.jsonl").write_text(
        json.dumps(
            {
                "qid": "q1",
                "query": "Find quiet cafes with patio seating.",
                "final_query": "Find quiet cafes with patio seating.",
                "subset": "Text_Image",
                "candidates": [
                    {"business_id": "b1", "name": "Cafe A", "verdict": "yes"},
                    {"business_id": "b2", "name": "Cafe B", "verdict": "no"},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_certify_answer_set_writes_release_ready_files(tmp_path: Path) -> None:
    src = tmp_path / "src"
    out = tmp_path / "certified"
    _write_case(src)

    path = certify_answer_set(src, out)

    assert path.name == "answer_set_certified_cases.jsonl"
    assert (out / "final_cases.jsonl").exists()
    assert (out / "final_index.jsonl").exists()
    summary = json.loads((out / "answer_certification_summary.json").read_text(encoding="utf-8"))
    assert summary["certified_case_count"] == 1
    assert summary["certified_answer_pair_count"] == 1


def test_query_metrics_and_human_summary(tmp_path: Path) -> None:
    src = tmp_path / "src"
    certified = tmp_path / "certified"
    quality = tmp_path / "quality"
    _write_case(src)
    certify_answer_set(src, certified)

    metrics_path = query_metrics(certified, quality)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["case_count"] == 1
    assert metrics["word_entropy"] > 0
    assert metrics["type_token_ratio"] > 0

    ratings = tmp_path / "ratings.csv"
    with ratings.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["qid", "subset", "annotator_id", "naturalness", "diversity", "practicality"])
        writer.writeheader()
        writer.writerow({"qid": "q1", "subset": "Text_Image", "annotator_id": "a1", "naturalness": "1", "diversity": "0", "practicality": "1"})

    summary_path = human_rating_summary(ratings, certified, quality)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["rating_row_count"] == 1
    assert summary["annotator_count"] == 1
    assert summary["metrics"]["naturalness"]["positive_rate"] == 1.0


def test_quality_metric_helpers_are_stable() -> None:
    queries = ["quiet cafe patio", "quiet bakery patio"]
    assert word_entropy(queries) > 0
    assert 0 < type_token_ratio(queries) <= 1
