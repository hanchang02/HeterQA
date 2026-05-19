from __future__ import annotations

import json
from pathlib import Path

import yaml

from heterqa.construction.mainline import run_mainline_from_config


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_offline_construction_runs_all_evidence_families(tmp_path: Path) -> None:
    business_path = tmp_path / "business.jsonl"
    text_path = tmp_path / "text.jsonl"
    image_path = tmp_path / "image.jsonl"
    graph_path = tmp_path / "graph.jsonl"
    config_path = tmp_path / "config.yaml"

    _write_jsonl(
        business_path,
        [
            {
                "business_id": "b1",
                "name": "Cafe One",
                "categories": "Cafe, Coffee & Tea",
                "outdoor_seating": True,
                "latitude": 39.9527,
                "longitude": -75.1653,
            },
            {
                "business_id": "b2",
                "name": "Cafe Two",
                "categories": "Cafe, Coffee & Tea",
                "outdoor_seating": True,
                "latitude": 40.5,
                "longitude": -75.9,
            },
        ],
    )
    _write_jsonl(
        text_path,
        [
            {
                "business_id": "b1",
                "source_locator": "r1",
                "summary": "friendly service outdoor seating quiet patio graph feature",
                "score": 1.0,
                "supports": True,
            },
            {
                "business_id": "b2",
                "source_locator": "r2",
                "summary": "friendly service outdoor seating quiet patio graph feature",
                "score": 0.95,
                "supports": True,
            },
        ],
    )
    _write_jsonl(
        image_path,
        [
            {
                "business_id": "b1",
                "source_locator": "p1",
                "caption": "friendly outdoor patio seating",
                "summary": "friendly outdoor patio seating",
                "score": 1.0,
                "supports": True,
            },
            {
                "business_id": "b2",
                "source_locator": "p2",
                "caption": "friendly outdoor patio seating",
                "summary": "friendly outdoor patio seating",
                "score": 0.95,
                "supports": True,
            },
        ],
    )
    _write_jsonl(
        graph_path,
        [
            {"business_id": "b1", "feature": "quiet patio", "sentiment": "pos", "user_id": "u1"},
            {"business_id": "b1", "feature": "friendly service", "sentiment": "pos", "user_id": "u1"},
            {"business_id": "b2", "feature": "quiet patio", "sentiment": "pos", "user_id": "u2"},
        ],
    )

    config_path.write_text(
        yaml.safe_dump(
            {
                "mode": "mainline",
                "data": {
                    "business_records_jsonl": str(business_path),
                    "text_evidence_jsonl": str(text_path),
                    "image_evidence_jsonl": str(image_path),
                    "graph_features_jsonl": str(graph_path),
                },
                "settings": {
                    "enabled_geo": True,
                    "enabled_text": True,
                    "enabled_image": True,
                    "enabled_kg": True,
                    "top_k": 10,
                    "text_rerank_thres": 0.0,
                    "image_coarse_thres": 0.0,
                    "image_reranker_thres": 0.0,
                    "llm_judge_threshold": 0.1,
                    "allow_missing_structured_values": False,
                    "kg_max_retries": 1,
                    "kg_min_survivors": 1,
                },
                "case": {
                    "qid": "offline-construction",
                    "subset": "Geo_Text_Image_KG",
                    "seed_records": [
                        {
                            "business_id": "b1",
                            "name": "Cafe One",
                            "categories": "Cafe, Coffee & Tea",
                            "outdoor_seating": True,
                            "latitude": 39.9527,
                            "longitude": -75.1653,
                        }
                    ],
                    "structured_filters": [
                        {"field": "categories", "operator": "LIKE", "value": "Cafe"},
                        {"field": "outdoor_seating", "operator": "=", "value": True},
                    ],
                    "geo_constraint": {
                        "reference_latitude": 39.9526,
                        "reference_longitude": -75.1652,
                        "radius_km": 2,
                        "relation_type": "within_radius",
                    },
                    "text_query": "friendly service outdoor seating",
                    "image_query": "friendly outdoor patio seating",
                    "kg_query": "quiet patio friendly service",
                    "final_query": "Find nearby cafes with outdoor seating, friendly service, patio photos, and graph-backed quiet patio evidence.",
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    output = run_mainline_from_config(config_path, tmp_path / "out")
    row = json.loads(output.read_text(encoding="utf-8").splitlines()[0])

    candidates = {item["business_id"]: item for item in row["candidates"]}
    assert row["final_answer_business_ids"] == ["b1"]
    assert candidates["b1"]["origin"] == "initial_seed"
    assert candidates["b1"]["text_verify"]["judgement"] == "yes"
    assert candidates["b1"]["image_verify"]["judgement"] == "yes"
    assert candidates["b1"]["kg_verify"]["judgement"] == "yes"
    assert candidates["b2"]["origin"] in {"text_vector_recall", "image_vector_recall"}
    assert candidates["b2"]["drop_reason"] == "geo_boundary_filter"
    stage_names = [stage["name"] for stage in row["stage_summaries"]]
    assert stage_names == [
        "record_field_initialization",
        "heterogeneous_candidate_recall",
        "geo_verification",
        "text_verification",
        "image_verification",
        "kg_verification",
        "final_query_generation",
        "retained_candidate_set",
    ]
