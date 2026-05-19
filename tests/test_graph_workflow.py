from __future__ import annotations

import json
from pathlib import Path

from heterqa.graph.build import build_graph_feature_rows, write_graph_feature_index
from heterqa.graph.canonicalization import canonicalize_feature_rows
from heterqa.graph.feature_extraction import extract_feature_rows


def test_graph_feature_extraction_does_not_emit_raw_text() -> None:
    rows = [
        {
            "business_id": "b1",
            "review_id": "r1",
            "user_id": "u1",
            "text": "Friendly staff and quiet patio seating.",
        }
    ]

    features = extract_feature_rows(rows, max_features_per_text=2)

    assert features
    assert "text" not in features[0]
    assert features[0]["business_id"] == "b1"
    assert features[0]["source_locator"] == "r1"
    assert len(features[0]["source_text_sha256"]) == 64
    assert features[0]["reviewer_id"] != "u1"


def test_canonicalize_and_build_graph_feature_index(tmp_path: Path) -> None:
    rows = [
        {"business_id": "b1", "reviewer_id": "u1", "feature": "Quiet Patio!", "polarity": "positive"},
        {"business_id": "b2", "reviewer_id": "u2", "feature": "quiet patio", "polarity": "negative"},
    ]
    canonical = canonicalize_feature_rows(rows)
    graph_rows, stats = build_graph_feature_rows(canonical)

    assert graph_rows[0]["feature"] == "quiet patio"
    assert graph_rows[0]["sentiment"] == "pos"
    assert graph_rows[1]["sentiment"] == "neg"
    assert stats.business_feature_rows == 2

    input_path = tmp_path / "features.jsonl"
    input_path.write_text("\n".join(json.dumps(row) for row in canonical) + "\n", encoding="utf-8")
    outputs = write_graph_feature_index(input_path, tmp_path / "graph")

    assert outputs["graph_features"].exists()
    assert outputs["stats"].exists()
