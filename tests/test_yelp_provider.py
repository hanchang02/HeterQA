from __future__ import annotations

import json
from pathlib import Path

from heterqa.construction.providers import SQLConstructionProvider, YelpOpenDatasetProvider, build_construction_provider


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_yelp_open_dataset_provider_loads_raw_business_reviews_and_photos(tmp_path: Path) -> None:
    business_path = tmp_path / "yelp_academic_dataset_business.json"
    review_path = tmp_path / "yelp_academic_dataset_review.json"
    photos_path = tmp_path / "photos.json"
    photo_dir = tmp_path / "photos"
    photo_dir.mkdir()

    _write_jsonl(
        business_path,
        [
            {
                "business_id": "b1",
                "name": "Cafe One",
                "city": "Philadelphia",
                "state": "PA",
                "latitude": 39.95,
                "longitude": -75.16,
                "stars": 4.5,
                "review_count": 10,
                "categories": "Cafes, Coffee & Tea",
                "attributes": {"OutdoorSeating": "True", "RestaurantsTakeOut": "True"},
            },
            {
                "business_id": "b2",
                "name": "Repair Shop",
                "city": "Philadelphia",
                "state": "PA",
                "latitude": 39.9,
                "longitude": -75.1,
                "stars": 4.0,
                "review_count": 5,
                "categories": "Auto Repair",
                "attributes": {"OutdoorSeating": "False"},
            },
        ],
    )
    _write_jsonl(
        review_path,
        [
            {
                "review_id": "r1",
                "business_id": "b1",
                "user_id": "u1",
                "stars": 5,
                "text": "Friendly staff and a quiet patio for coffee.",
            },
            {
                "review_id": "r2",
                "business_id": "b2",
                "user_id": "u2",
                "stars": 4,
                "text": "Fast brake repair.",
            },
        ],
    )
    _write_jsonl(
        photos_path,
        [
            {"photo_id": "p1", "business_id": "b1", "caption": "Outdoor patio seating", "label": "outside"},
            {"photo_id": "p2", "business_id": "b2", "caption": "Garage bay", "label": "inside"},
        ],
    )

    provider = build_construction_provider(
        {
            "provider": "yelp_open_dataset",
            "business_jsonl": str(business_path),
            "review_jsonl": str(review_path),
            "photos_json": str(photos_path),
            "photo_dir": str(photo_dir),
        }
    )

    assert isinstance(provider, YelpOpenDatasetProvider)
    business = provider.get_business("b1")
    assert business is not None
    assert business.fields["outdoor_seating"] is True
    assert business.fields["restaurants_take_out"] is True

    review_hits = provider.search_reviews("quiet patio coffee", ["b1", "b2"], top_k=2)
    assert review_hits[0]["business_id"] == "b1"
    assert review_hits[0]["source_locator"] == "r1"

    photo_hits = provider.search_photos("outdoor patio", ["b1", "b2"], top_k=2)
    assert photo_hits[0]["business_id"] == "b1"
    assert photo_hits[0]["source_locator"] == "p1"
    assert photo_hits[0]["path"].endswith("p1.jpg")


def test_yelp_open_dataset_provider_resolves_files_from_yelp_root(tmp_path: Path) -> None:
    yelp_root = tmp_path / "extracted" / "json" / "Yelp JSON"
    yelp_root.mkdir(parents=True)
    photo_dir = tmp_path / "extracted" / "photos" / "Yelp Photos" / "photos"
    photo_dir.mkdir(parents=True)
    (photo_dir / "p1.jpg").write_bytes(b"jpeg")
    _write_jsonl(
        yelp_root / "yelp_academic_dataset_business.json",
        [
            {
                "business_id": "b1",
                "name": "Cafe One",
                "categories": "Cafes",
                "attributes": {"OutdoorSeating": "True"},
            }
        ],
    )
    _write_jsonl(
        yelp_root / "yelp_academic_dataset_review.json",
        [{"review_id": "r1", "business_id": "b1", "text": "Quiet patio."}],
    )
    _write_jsonl(
        yelp_root / "photos.json",
        [{"photo_id": "p1", "business_id": "b1", "caption": "Patio"}],
    )

    provider = build_construction_provider({"provider": "yelp_open_dataset", "yelp_root": str(tmp_path)})

    assert isinstance(provider, YelpOpenDatasetProvider)
    business = provider.get_business("b1")
    assert business is not None
    assert business.fields["outdoor_seating"] is True
    assert provider.search_reviews("patio", ["b1"], top_k=1)[0]["source_locator"] == "r1"
    photo_hit = provider.search_photos("patio", ["b1"], top_k=1)[0]
    assert photo_hit["source_locator"] == "p1"
    assert photo_hit["path"].endswith("p1.jpg")


def test_file_provider_exposes_feature_graph_methods(tmp_path: Path) -> None:
    business_path = tmp_path / "business.jsonl"
    graph_path = tmp_path / "graph.jsonl"
    _write_jsonl(
        business_path,
        [
            {"business_id": "b1", "name": "Cafe One", "categories": "Cafe"},
            {"business_id": "b2", "name": "Cafe Two", "categories": "Cafe"},
        ],
    )
    _write_jsonl(
        graph_path,
        [
            {"business_id": "b1", "feature": "quiet patio", "sentiment": "pos", "user_id": "u1"},
            {"business_id": "b2", "feature": "quiet patio", "sentiment": "pos", "user_id": "u2"},
            {"business_id": "b2", "feature": "slow service", "sentiment": "neg", "user_id": "u3"},
        ],
    )

    provider = build_construction_provider(
        {
            "provider": "file",
            "business_records_jsonl": str(business_path),
            "graph_features_jsonl": str(graph_path),
        }
    )

    assert provider.get_features_of_business("b1", "pos") == ["quiet patio"]  # type: ignore[attr-defined]
    assert provider.get_businesses_by_feature("quiet patio", "pos") == ["b1", "b2"]  # type: ignore[attr-defined]
    assert provider.get_users_connected_via_feature("b1", "quiet patio", "pos") == ["u1"]  # type: ignore[attr-defined]
    assert provider.get_users_by_feature("slow service", "neg") == ["u3"]  # type: ignore[attr-defined]


def test_file_provider_exposes_local_vector_embedding_indexes(tmp_path: Path) -> None:
    business_path = tmp_path / "business.jsonl"
    review_index_path = tmp_path / "review_vectors.jsonl"
    photo_index_path = tmp_path / "photo_vectors.jsonl"
    feature_index_path = tmp_path / "feature_vectors.jsonl"
    _write_jsonl(
        business_path,
        [
            {"business_id": "b1", "name": "Cafe One", "categories": "Cafe"},
            {"business_id": "b2", "name": "Repair Shop", "categories": "Auto Repair"},
        ],
    )
    _write_jsonl(
        review_index_path,
        [
            {"business_id": "b1", "review_id": "r1", "text": "quiet patio", "embedding": [1.0, 0.0]},
            {"business_id": "b2", "review_id": "r2", "text": "brake repair", "embedding": [0.0, 1.0]},
        ],
    )
    _write_jsonl(
        photo_index_path,
        [
            {"business_id": "b1", "photo_id": "p1", "caption": "patio seating", "path": "p1.jpg", "embedding": [1.0, 0.0]},
            {"business_id": "b2", "photo_id": "p2", "caption": "garage bay", "path": "p2.jpg", "embedding": [0.0, 1.0]},
        ],
    )
    _write_jsonl(
        feature_index_path,
        [
            {"feature_key": "quiet patio", "embedding": [1.0, 0.0]},
            {"feature_key": "garage bay", "embedding": [0.0, 1.0]},
        ],
    )

    provider = build_construction_provider(
        {
            "provider": "file",
            "business_records_jsonl": str(business_path),
            "review_embedding_jsonl": str(review_index_path),
            "photo_embedding_jsonl": str(photo_index_path),
            "feature_embedding_jsonl": str(feature_index_path),
        }
    )

    review_hits = provider.search_review_embeddings([1.0, 0.0], top_k=1)  # type: ignore[attr-defined]
    photo_hits = provider.search_photo_embeddings([1.0, 0.0], top_k=1)  # type: ignore[attr-defined]
    feature_hits = provider.search_feature_embeddings([1.0, 0.0], limit=1)  # type: ignore[attr-defined]

    assert review_hits[0]["business_id"] == "b1"
    assert photo_hits[0]["photo_id"] == "p1"
    assert feature_hits[0]["feature_key"] == "quiet patio"


def test_sql_provider_exposes_db_anchor_neighbor_lookup_without_real_db() -> None:
    provider = _fake_sql_provider()

    def fake_query(_sql, _params=()):
        return [
            {"business_id": "far", "name": "Far", "latitude": 35.0, "longitude": 139.2},
            {"business_id": "near", "name": "Near", "latitude": 35.0, "longitude": 139.001},
        ]

    provider._query = fake_query  # type: ignore[method-assign]

    row = provider.fetch_one_near_seeded(35.0, 139.0, 2.0, seed=7)

    assert row is not None
    assert row["business_id"] == "near"


def test_sql_provider_builds_default_source_vector_sql_without_real_db() -> None:
    provider = _fake_sql_provider()
    seen_sql = []

    def fake_query(sql, _params=()):
        seen_sql.append(sql)
        return [{"business_id": "b1", "text": "quiet patio", "_score": 0.9}]

    provider._query = fake_query  # type: ignore[method-assign]

    review_hits = provider.search_review_embeddings([0.1, 0.2], business_ids=["b1"], top_k=5)
    photo_hits = provider.search_photo_embeddings([0.1, 0.2], business_ids=["b1"], top_k=5)
    feature_hits = provider.search_feature_embeddings([0.1, 0.2], limit=5)
    hybrid_hits = provider.hybrid_search_reviews("quiet patio", [0.1, 0.2], ["b1"], top_k=5)

    joined = "\n".join(seen_sql)
    assert "cosine_distance" in joined
    assert "FROM content_combined" in seen_sql[0]
    assert "APPROXIMATE" not in seen_sql[0]
    assert "photo_embedding" in joined
    assert "feature_embedding" in joined
    assert "FULL JOIN" in joined
    assert review_hits[0]["business_id"] == "b1"
    assert photo_hits[0]["business_id"] == "b1"
    assert feature_hits[0]["business_id"] == "b1"
    assert hybrid_hits[0]["coarse_score"] == 0.9


def _fake_sql_provider() -> SQLConstructionProvider:
    provider = SQLConstructionProvider.__new__(SQLConstructionProvider)
    provider.config = {}
    provider.business_table = "business"
    provider.review_table = "content_combined"
    provider.photo_table = "photos"
    provider.business_id_column = "business_id"
    provider.business_name_column = "name"
    provider.latitude_column = "latitude"
    provider.longitude_column = "longitude"
    provider.geom_column = None
    provider.review_text_column = "text"
    provider.photo_caption_column = "caption"
    return provider
