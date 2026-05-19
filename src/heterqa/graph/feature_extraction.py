"""Review/tip feature extraction for KG graph construction."""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from heterqa.construction.record_fields import _ask_json
from heterqa.core.io import read_jsonl, write_jsonl
from heterqa.core.safety import tokenize


FEATURE_EXTRACTION_PROMPT = """
Extract concise business features from one review or tip.

Rules:
- Output valid JSON only.
- Each feature should be a short noun phrase or service/attribute phrase.
- Do not copy full sentences.
- polarity must be one of positive, negative, neutral.
- Keep only features grounded in the text.

Input text:
{text}

Output schema:
{{"features": [{{"feature": "friendly staff", "polarity": "positive", "confidence": 0.9}}]}}
"""


@dataclass(frozen=True)
class ExtractedFeature:
    business_id: str
    reviewer_id: str
    feature: str
    polarity: str = "positive"
    confidence: float = 1.0
    source_locator_type: str = "source_id"
    source_locator: str = ""
    source_text_sha256: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_text_for_hash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def sha256_text(text: str) -> str:
    return hashlib.sha256(normalize_text_for_hash(text).encode("utf-8")).hexdigest()


def _normalise_polarity(value: Any) -> str:
    lowered = str(value or "").strip().lower()
    if lowered in {"negative", "neg", "bad", "complaint"}:
        return "negative"
    if lowered in {"neutral", "mixed"}:
        return "neutral"
    return "positive"


def _deterministic_features(text: str, *, limit: int) -> list[dict[str, Any]]:
    tokens = [token for token in tokenize(text) if len(token) > 2]
    stop = {
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "was",
        "were",
        "are",
        "but",
        "from",
        "have",
        "had",
        "they",
        "you",
        "our",
        "their",
    }
    filtered = [token for token in tokens if token not in stop]
    phrases: list[str] = []
    for width in [2, 3]:
        for index in range(0, max(0, len(filtered) - width + 1)):
            phrase = " ".join(filtered[index : index + width])
            if phrase not in phrases:
                phrases.append(phrase)
            if len(phrases) >= limit:
                break
        if len(phrases) >= limit:
            break
    return [{"feature": phrase, "polarity": "neutral", "confidence": 0.25} for phrase in phrases[:limit]]


def extract_features_from_text(text: str, *, model: Any = None, limit: int = 5) -> list[dict[str, Any]]:
    if not text.strip():
        return []
    if model is None:
        return _deterministic_features(text, limit=limit)
    payload = _ask_json(model, FEATURE_EXTRACTION_PROMPT.format(text=text[:4000]))
    rows = payload.get("features", []) if isinstance(payload, dict) else []
    output = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        feature = str(row.get("feature") or "").strip()
        if not feature:
            continue
        try:
            confidence = float(row.get("confidence", 1.0))
        except (TypeError, ValueError):
            confidence = 1.0
        output.append(
            {
                "feature": feature,
                "polarity": _normalise_polarity(row.get("polarity")),
                "confidence": max(0.0, min(1.0, confidence)),
            }
        )
        if len(output) >= limit:
            break
    return output


def extract_feature_rows(
    source_rows: list[dict[str, Any]],
    *,
    model: Any = None,
    text_field: str = "text",
    source_locator_type: str = "yelp_review_id",
    max_features_per_text: int = 5,
    hash_user_ids: bool = True,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in source_rows:
        business_id = str(row.get("business_id") or "")
        text = str(row.get(text_field) or row.get("text") or row.get("tip") or "")
        if not business_id or not text.strip():
            continue
        raw_user_id = str(row.get("user_id") or row.get("reviewer_id") or "")
        reviewer_id = (
            hashlib.sha256(raw_user_id.encode("utf-8")).hexdigest()[:16]
            if hash_user_ids and raw_user_id
            else raw_user_id
        )
        locator = str(row.get("review_id") or row.get("tip_id") or row.get("source_locator") or "")
        for feature in extract_features_from_text(text, model=model, limit=max_features_per_text):
            output.append(
                ExtractedFeature(
                    business_id=business_id,
                    reviewer_id=reviewer_id,
                    feature=str(feature["feature"]),
                    polarity=_normalise_polarity(feature.get("polarity")),
                    confidence=float(feature.get("confidence", 1.0)),
                    source_locator_type=source_locator_type,
                    source_locator=locator,
                    source_text_sha256=sha256_text(text),
                ).to_dict()
            )
    return output


def extract_feature_file(
    input_jsonl: Path,
    output_jsonl: Path,
    *,
    model: Any = None,
    text_field: str = "text",
    source_locator_type: str = "yelp_review_id",
    max_features_per_text: int = 5,
) -> Path:
    rows = read_jsonl(input_jsonl)
    output = extract_feature_rows(
        rows,
        model=model,
        text_field=text_field,
        source_locator_type=source_locator_type,
        max_features_per_text=max_features_per_text,
    )
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_jsonl, output)
    return output_jsonl
