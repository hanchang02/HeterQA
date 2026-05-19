"""Feature-centric graph construction utilities."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from heterqa.core.io import read_jsonl, write_jsonl
from heterqa.core.safety import tokenize


@dataclass(frozen=True)
class GraphStats:
    reviewer_nodes: int
    businesses: int
    features: int
    edges: int
    business_feature_rows: int = 0


def canonicalize_feature(text: str) -> str:
    return " ".join(tokenize(text))[:120]


def anonymized_reviewer_node(raw_id: str) -> str:
    digest = hashlib.sha256(raw_id.encode("utf-8")).hexdigest()[:16]
    return f"reviewer:{digest}"


def build_feature_graph(review_rows: list[dict[str, str]]) -> tuple[dict[str, list[dict[str, str]]], GraphStats]:
    """Build a small feature graph from already extracted review feature rows.

    Expected row fields: reviewer_id, business_id, feature, polarity. The
    reviewer identifier is hashed before graph materialization.
    """
    graph: dict[str, list[dict[str, str]]] = defaultdict(list)
    reviewer_nodes: set[str] = set()
    businesses: set[str] = set()
    features: Counter[str] = Counter()
    for row in review_rows:
        reviewer_node = anonymized_reviewer_node(str(row["reviewer_id"]))
        business_id = str(row["business_id"])
        feature = canonicalize_feature(str(row["feature"]))
        polarity = str(row.get("polarity", "positive"))
        reviewer_nodes.add(reviewer_node)
        businesses.add(business_id)
        features[feature] += 1
        graph[reviewer_node].append({"target": feature, "relation": f"prefers_{polarity}"})
        graph[business_id].append({"target": feature, "relation": f"supports_{polarity}"})
    stats = GraphStats(
        reviewer_nodes=len(reviewer_nodes),
        businesses=len(businesses),
        features=len(features),
        edges=sum(len(edges) for edges in graph.values()),
        business_feature_rows=len(review_rows),
    )
    return dict(graph), stats


def build_graph_feature_rows(feature_rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], GraphStats]:
    """Build the `graph_features_jsonl` index consumed by KG providers."""

    output: list[dict[str, str]] = []
    reviewer_nodes: set[str] = set()
    businesses: set[str] = set()
    features: Counter[str] = Counter()
    edge_count = 0
    for row in feature_rows:
        business_id = str(row.get("business_id") or "")
        feature = canonicalize_feature(str(row.get("feature") or ""))
        if not business_id or not feature:
            continue
        raw_reviewer = str(row.get("reviewer_id") or row.get("user_id") or "")
        reviewer_id = anonymized_reviewer_node(raw_reviewer) if raw_reviewer and not raw_reviewer.startswith("reviewer:") else raw_reviewer
        sentiment = str(row.get("sentiment") or row.get("polarity") or "pos").lower()
        if sentiment == "positive":
            sentiment = "pos"
        elif sentiment == "negative":
            sentiment = "neg"
        businesses.add(business_id)
        features[feature] += 1
        if reviewer_id:
            reviewer_nodes.add(reviewer_id)
            edge_count += 2
        else:
            edge_count += 1
        output.append(
            {
                "business_id": business_id,
                "feature": feature,
                "sentiment": sentiment,
                "user_id": reviewer_id,
                "source_locator_type": str(row.get("source_locator_type") or ""),
                "source_locator": str(row.get("source_locator") or ""),
                "source_text_sha256": str(row.get("source_text_sha256") or ""),
            }
        )
    stats = GraphStats(
        reviewer_nodes=len(reviewer_nodes),
        businesses=len(businesses),
        features=len(features),
        edges=edge_count,
        business_feature_rows=len(output),
    )
    return output, stats


def write_graph_artifacts(rows_path: Path, output_dir: Path) -> dict[str, Path]:
    rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines() if line]
    graph, stats = build_feature_graph(rows)
    graph_features, feature_stats = build_graph_feature_rows(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    graph_path = output_dir / "feature_graph.json"
    graph_features_path = output_dir / "graph_features.jsonl"
    stats_path = output_dir / "graph_stats.json"
    graph_path.write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
    write_jsonl(graph_features_path, graph_features)
    merged_stats = {**stats.__dict__, "business_feature_index": feature_stats.__dict__}
    stats_path.write_text(json.dumps(merged_stats, indent=2), encoding="utf-8")
    return {"graph": graph_path, "graph_features": graph_features_path, "stats": stats_path}


def write_graph_feature_index(rows_path: Path, output_dir: Path) -> dict[str, Path]:
    rows = read_jsonl(rows_path)
    graph_features, stats = build_graph_feature_rows(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    graph_features_path = output_dir / "graph_features.jsonl"
    stats_path = output_dir / "graph_stats.json"
    write_jsonl(graph_features_path, graph_features)
    stats_path.write_text(json.dumps(stats.__dict__, indent=2), encoding="utf-8")
    return {"graph_features": graph_features_path, "stats": stats_path}
