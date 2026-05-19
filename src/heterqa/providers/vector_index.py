"""Vector-index provider boundary and local JSONL adapter."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Protocol


class VectorIndex(Protocol):
    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        ...


def _parse_vector(value: Any) -> list[float]:
    if value is None:
        return []
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, list):
        return []
    return [float(item) for item in value]


def _cosine(query: list[float], document: list[float]) -> float:
    if not query or not document or len(query) != len(document):
        return 0.0
    numerator = sum(left * right for left, right in zip(query, document, strict=False))
    query_norm = math.sqrt(sum(item * item for item in query))
    document_norm = math.sqrt(sum(item * item for item in document))
    if query_norm == 0 or document_norm == 0:
        return 0.0
    return numerator / (query_norm * document_norm)


class VectorDocumentIndex:
    """Small local vector index for review/photo/feature embedding artifacts.

    This is an adapter for precomputed embedding files. It deliberately does
    not implement HeterQA recall logic; construction modules decide which
    embeddings to query, how to rerank hits, and how hits affect candidates.
    """

    def __init__(self, rows: Iterable[dict[str, Any]], *, vector_field: str = "embedding"):
        self.rows: list[dict[str, Any]] = []
        for row in rows:
            vector = _parse_vector(row.get(vector_field, row.get("vector")))
            if not vector:
                continue
            payload = {key: value for key, value in row.items() if key not in {vector_field, "vector"}}
            payload["_vector"] = vector
            self.rows.append(payload)

    @classmethod
    def from_jsonl(cls, path: Path | None, *, vector_field: str = "embedding") -> "VectorDocumentIndex | None":
        if path is None:
            return None
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid vector JSONL record at {path}:{line_number}: {exc}") from exc
                if isinstance(payload, dict):
                    rows.append(payload)
        return cls(rows, vector_field=vector_field)

    def search(
        self,
        query_vector: list[float],
        *,
        business_ids: Iterable[str] | None = None,
        top_k: int = 100,
    ) -> list[dict[str, Any]]:
        allowed = {str(item) for item in business_ids} if business_ids is not None else None
        hits = []
        for row in self.rows:
            business_id = str(row.get("business_id", ""))
            if allowed is not None and business_id not in allowed:
                continue
            score = _cosine(query_vector, row["_vector"])
            item = {key: value for key, value in row.items() if key != "_vector"}
            item["_score"] = score
            item.setdefault("score", score)
            hits.append(item)
        hits.sort(key=lambda item: float(item.get("_score") or 0), reverse=True)
        return hits[:top_k]
