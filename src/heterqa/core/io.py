"""File readers and writers for HeterQA JSONL/TSV assets."""

from __future__ import annotations

import csv
import json
import shutil
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable

from heterqa.core.schema import AnswerRecord, EvidenceRecord, QueryRecord, ReleaseBundle


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: Iterable[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            payload = asdict(row) if is_dataclass(row) else row
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def load_qrels(path: Path) -> dict[str, set[str]]:
    qrels: dict[str, set[str]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        required = {"query-id", "corpus-id", "score"}
        if set(reader.fieldnames or []) != required:
            raise ValueError(f"qrels header must be {sorted(required)}")
        for row in reader:
            if int(row["score"]) > 0:
                qrels.setdefault(row["query-id"], set()).add(row["corpus-id"])
    return qrels


def write_qrels(path: Path, qrels: dict[str, set[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["query-id", "corpus-id", "score"], delimiter="\t")
        writer.writeheader()
        for qid in sorted(qrels, key=lambda value: int(value) if value.isdigit() else value):
            for business_id in sorted(qrels[qid]):
                writer.writerow({"query-id": qid, "corpus-id": business_id, "score": 1})


def load_release_bundle(dataset_dir: Path) -> ReleaseBundle:
    data_dir = dataset_dir / "data"
    queries = [QueryRecord(**row) for row in read_jsonl(data_dir / "queries.jsonl")]
    answers = [AnswerRecord(**row) for row in read_jsonl(data_dir / "answers.jsonl")]
    evidence = [EvidenceRecord(**row) for row in read_jsonl(data_dir / "evidence.jsonl")]
    qrels = load_qrels(data_dir / "qrels" / "test.tsv")
    return ReleaseBundle(queries=queries, answers=answers, evidence=evidence, qrels=qrels)


def copy_release_tree(input_dir: Path, output_dir: Path) -> None:
    include_dirs = ["data", "metadata", "schemas"]
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ["README.md", "METHOD.md"]:
        src = input_dir / name
        if src.exists():
            shutil.copy2(src, output_dir / name)
    for name in include_dirs:
        src = input_dir / name
        dst = output_dir / name
        if not src.exists():
            continue
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)

