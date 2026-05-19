"""Feature canonicalization helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from heterqa.core.io import read_jsonl, write_jsonl
from heterqa.graph.build import canonicalize_feature


def canonicalize_feature_rows(
    rows: list[dict[str, Any]],
    *,
    alias_map: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Normalize extracted feature rows into graph-index rows."""

    aliases = {canonicalize_feature(key): canonicalize_feature(value) for key, value in (alias_map or {}).items()}
    output: list[dict[str, Any]] = []
    for row in rows:
        feature = canonicalize_feature(str(row.get("feature") or row.get("feature_name") or ""))
        if not feature:
            continue
        canonical = aliases.get(feature, feature)
        output.append(
            {
                **row,
                "feature": canonical,
                "raw_feature": row.get("feature"),
                "sentiment": _sentiment_from_polarity(row.get("sentiment", row.get("polarity"))),
            }
        )
    return output


def canonicalize_feature_file(
    input_jsonl: Path,
    output_jsonl: Path,
    *,
    alias_map: dict[str, str] | None = None,
) -> Path:
    rows = canonicalize_feature_rows(read_jsonl(input_jsonl), alias_map=alias_map)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_jsonl, rows)
    return output_jsonl


def _sentiment_from_polarity(value: Any) -> str:
    lowered = str(value or "").strip().lower()
    if lowered in {"negative", "neg"}:
        return "neg"
    if lowered in {"neutral", "mixed"}:
        return "neutral"
    return "pos"

__all__ = ["canonicalize_feature", "canonicalize_feature_file", "canonicalize_feature_rows"]
