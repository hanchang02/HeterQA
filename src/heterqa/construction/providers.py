"""Data provider interfaces for HeterQA construction."""

from __future__ import annotations

import ast
import gzip
import json
import math
import re
import zlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Protocol, TextIO

from heterqa.core.io import read_jsonl
from heterqa.core.safety import tokenize
from heterqa.providers.vector_index import VectorDocumentIndex

from .contracts import BusinessRecord, EvidenceFamily, EvidenceItem

YELP_DATASET_FILENAMES = {
    "business_jsonl": "yelp_academic_dataset_business.json",
    "review_jsonl": "yelp_academic_dataset_review.json",
    "photos_json": "photos.json",
}

YELP_JSON_ROOT_NAME = "Yelp JSON"
YELP_PHOTOS_ROOT_NAME = "Yelp Photos"


def _ordered_yelp_provider_roots(root: Path) -> list[Path]:
    roots = [
        root / "extracted" / "json" / YELP_JSON_ROOT_NAME,
        root / "extracted" / "photos" / YELP_PHOTOS_ROOT_NAME,
        root / "json" / YELP_JSON_ROOT_NAME,
        root / "photos" / YELP_PHOTOS_ROOT_NAME,
        root / YELP_JSON_ROOT_NAME,
        root / YELP_PHOTOS_ROOT_NAME,
        root,
    ]
    deduped: list[Path] = []
    seen: set[Path] = set()
    for candidate in roots:
        resolved = candidate.resolve() if candidate.exists() else candidate.absolute()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(candidate)
    return deduped


def _resolve_yelp_provider_files(root: Path) -> dict[str, str]:
    resolved_files: dict[str, str] = {}
    for source_root in _ordered_yelp_provider_roots(root):
        for config_key, filename in YELP_DATASET_FILENAMES.items():
            path = source_root / filename
            if config_key not in resolved_files and path.is_file():
                resolved_files[config_key] = str(path)
    return resolved_files


def _resolve_yelp_provider_photo_dir(root: Path) -> str | None:
    for source_root in _ordered_yelp_provider_roots(root):
        candidate = source_root / "photos"
        if candidate.is_dir():
            return str(candidate)
    return None


class FeatureGraphStore:
    """File-backed feature graph used by KG construction logic."""

    def __init__(self, rows: Iterable[dict[str, Any]]):
        self._business_features: dict[tuple[str, str], set[str]] = defaultdict(set)
        self._feature_businesses: dict[tuple[str, str], set[str]] = defaultdict(set)
        self._business_feature_users: dict[tuple[str, str, str], set[str]] = defaultdict(set)
        self._business_users: dict[str, set[str]] = defaultdict(set)
        self._user_features: dict[tuple[str, str], set[str]] = defaultdict(set)
        self._feature_users: dict[tuple[str, str], set[str]] = defaultdict(set)
        for row in rows:
            business_id = str(row.get("business_id") or "")
            feature = str(row.get("feature") or row.get("feature_name") or row.get("summary") or "").strip()
            sentiment = str(row.get("sentiment") or "pos").strip().lower()
            user_id = str(row.get("user_id") or "")
            if not business_id or not feature:
                continue
            self._business_features[(business_id, sentiment)].add(feature)
            self._feature_businesses[(feature, sentiment)].add(business_id)
            if user_id:
                self._business_feature_users[(business_id, feature, sentiment)].add(user_id)
                self._business_users[business_id].add(user_id)
                self._user_features[(user_id, sentiment)].add(feature)
                self._feature_users[(feature, sentiment)].add(user_id)

    @classmethod
    def from_jsonl(cls, path: Path | None) -> "FeatureGraphStore | None":
        if path is None:
            return None
        return cls(_iter_json_records(path))

    def get_features_of_business(self, business_id: str, sentiment: str = "pos") -> list[str]:
        return sorted(self._business_features.get((business_id, sentiment), set()))

    def get_businesses_by_feature(self, feature: str, sentiment: str = "pos") -> list[str]:
        return sorted(self._feature_businesses.get((feature, sentiment), set()))

    def sample_features_by_distribution(self, pool: Iterable[str], top_k: int = 1) -> list[str]:
        counts = Counter(str(item) for item in pool if str(item).strip())
        return [feature for feature, _ in counts.most_common(top_k)]

    def get_users_connected_via_feature(self, business_id: str, feature: str, sentiment: str = "pos") -> list[str]:
        direct = set(self._business_feature_users.get((business_id, feature, sentiment), set()))
        if direct:
            return sorted(direct)
        return sorted(self._business_users.get(business_id, set()) & self._feature_users.get((feature, sentiment), set()))

    def get_features_of_user(self, user_id: str, sentiment: str = "pos") -> list[str]:
        return sorted(self._user_features.get((user_id, sentiment), set()))

    def get_users_by_feature(self, feature: str, sentiment: str = "pos") -> list[str]:
        return sorted(self._feature_users.get((feature, sentiment), set()))


class ConstructionDataProvider(Protocol):
    def iter_businesses(self) -> list[BusinessRecord]:
        ...

    def get_business(self, business_id: str) -> BusinessRecord | None:
        ...

    def recall(self, family: EvidenceFamily, query: str, top_k: int) -> list[tuple[str, float]]:
        ...

    def get_evidence(
        self,
        family: EvidenceFamily,
        business_id: str,
        query: str,
        limit: int,
    ) -> list[EvidenceItem]:
        ...

    # Production providers should expose these narrower operations.  They keep
    # the construction algorithms in construction/* instead of hiding them in a
    # single opaque recall() call.
    def get_reviews(self, business_id: str, limit: int | None = None) -> list[dict[str, Any]]:
        ...

    def search_reviews(
        self,
        query: str,
        business_ids: list[str],
        top_k: int,
    ) -> list[dict[str, Any]]:
        ...

    def get_photos(self, business_id: str, limit: int | None = None) -> list[dict[str, Any]]:
        ...

    def search_photos(
        self,
        query: str,
        business_ids: list[str],
        top_k: int,
    ) -> list[dict[str, Any]]:
        ...

    def search_review_embeddings(
        self,
        query_vector: list[float],
        *,
        business_ids: list[str] | None = None,
        top_k: int = 100,
    ) -> list[dict[str, Any]]:
        ...

    def hybrid_search_reviews(
        self,
        search_text: str,
        query_vector: list[float] | None,
        business_ids: list[str],
        *,
        top_k: int = 300,
        fulltext_limit: int | None = None,
        vector_limit: int | None = None,
        rank_smoothing: int = 60,
    ) -> list[dict[str, Any]]:
        ...

    def search_photo_embeddings(
        self,
        query_vector: list[float],
        *,
        business_ids: list[str] | None = None,
        top_k: int = 100,
    ) -> list[dict[str, Any]]:
        ...

    def search_feature_embeddings(self, query_vector: list[float], *, limit: int = 100) -> list[dict[str, Any]]:
        ...

    def get_businesses_by_feature(self, feature: str, sentiment: str = "pos") -> list[str]:
        ...

    def sample_features_by_distribution(self, pool: Iterable[str], top_k: int = 1) -> list[str]:
        ...

    def build_geo_constraint(self, seed_rows: list[dict[str, Any]]) -> Any:
        ...

    def order_records_by_seed(self, records: list[dict[str, Any]], seed: int, id_key: str = "business_id") -> list[dict[str, Any]]:
        ...

    def fetch_one_near_seeded(
        self,
        center_lat: float,
        center_lon: float,
        radius_km: float,
        seed: int,
        *,
        exclude_id: str | None = None,
    ) -> dict[str, Any] | None:
        ...


def _overlap_score(query: str, text: str) -> float:
    query_tokens = set(tokenize(query))
    if not query_tokens:
        return 0.0
    text_tokens = set(tokenize(text))
    return len(query_tokens & text_tokens) / len(query_tokens)


def _graph_from_config(data_config: dict[str, Any]) -> FeatureGraphStore | None:
    if data_config.get("graph_features_jsonl"):
        return FeatureGraphStore.from_jsonl(Path(data_config["graph_features_jsonl"]))
    return None


def _provider_crc32(primary: str, seed: int) -> int:
    return zlib.crc32(f"{primary}:{int(seed)}".encode("utf-8", errors="ignore")) & 0xFFFFFFFF


def _provider_haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0088
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


def _provider_deg_bbox(center_lat: float, center_lon: float, radius_km: float) -> tuple[float, float, float, float]:
    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * max(1e-6, abs(math.cos(math.radians(center_lat))))
    dlat = radius_km / km_per_deg_lat
    dlon = radius_km / km_per_deg_lon
    return center_lat - dlat, center_lat + dlat, center_lon - dlon, center_lon + dlon


def _provider_has_valid_coordinates(row: dict[str, Any]) -> bool:
    try:
        lat = float(row["latitude"])
        lon = float(row["longitude"])
    except (KeyError, TypeError, ValueError):
        return False
    return math.isfinite(lat) and math.isfinite(lon) and -90 <= lat <= 90 and -180 <= lon <= 180


def _sql_literal(value: str) -> str:
    return "'" + str(value).replace("\\", "\\\\").replace("'", "''") + "'"


def _sql_id_list(values: Iterable[str]) -> str:
    return ", ".join(_sql_literal(str(value)) for value in values)


def _sql_vector(vector: list[float]) -> str:
    return "[" + ", ".join(str(float(value)) for value in vector) + "]"


class FileBackedConstructionProvider:
    """Simple provider backed by JSONL files.

    This implementation is meant for reproducible public runs. Large-scale runs
    may replace it with database, vector-index, or graph-backed providers that
    implement the same protocol.
    """

    def __init__(
        self,
        business_records: list[BusinessRecord],
        evidence_by_family: dict[EvidenceFamily, list[tuple[str, EvidenceItem]]],
        *,
        graph: FeatureGraphStore | None = None,
        review_index: VectorDocumentIndex | None = None,
        photo_index: VectorDocumentIndex | None = None,
        feature_index: VectorDocumentIndex | None = None,
    ):
        self._businesses = {record.business_id: record for record in business_records}
        self._evidence: dict[EvidenceFamily, dict[str, list[EvidenceItem]]] = defaultdict(lambda: defaultdict(list))
        self.graph = graph
        self.review_index = review_index
        self.photo_index = photo_index
        self.feature_index = feature_index
        for family, rows in evidence_by_family.items():
            for business_id, item in rows:
                self._evidence[family][business_id].append(item)

    @classmethod
    def from_config(cls, data_config: dict[str, Any]) -> "FileBackedConstructionProvider":
        business_path = Path(data_config["business_records_jsonl"])
        business_records = [BusinessRecord.from_raw(row) for row in read_jsonl(business_path)]
        evidence_by_family: dict[EvidenceFamily, list[tuple[str, EvidenceItem]]] = {}
        for family, config_key in [
            ("text", "text_evidence_jsonl"),
            ("image", "image_evidence_jsonl"),
            ("kg", "kg_evidence_jsonl"),
            ("cross_modal", "cross_modal_evidence_jsonl"),
        ]:
            path_text = data_config.get(config_key)
            if path_text:
                evidence_by_family[family] = cls._load_evidence_file(Path(path_text), family)  # type: ignore[index]
        graph = _graph_from_config(data_config)
        review_index = VectorDocumentIndex.from_jsonl(Path(data_config["review_embedding_jsonl"])) if data_config.get("review_embedding_jsonl") else None
        photo_index = VectorDocumentIndex.from_jsonl(Path(data_config["photo_embedding_jsonl"])) if data_config.get("photo_embedding_jsonl") else None
        feature_index = VectorDocumentIndex.from_jsonl(Path(data_config["feature_embedding_jsonl"])) if data_config.get("feature_embedding_jsonl") else None
        return cls(
            business_records,
            evidence_by_family,
            graph=graph,
            review_index=review_index,
            photo_index=photo_index,
            feature_index=feature_index,
        )

    @staticmethod
    def _load_evidence_file(path: Path, family: EvidenceFamily) -> list[tuple[str, EvidenceItem]]:
        rows: list[tuple[str, EvidenceItem]] = []
        for raw in read_jsonl(path):
            business_id = str(raw["business_id"])
            item = EvidenceItem(
                family=family,
                source_locator_type=str(raw.get("source_locator_type", family)),
                source_locator=str(raw.get("source_locator", raw.get("id", ""))),
                summary=str(raw.get("summary", raw.get("text", raw.get("caption", "")))),
                score=float(raw["score"]) if raw.get("score") is not None else None,
                supports=raw.get("supports"),
                metadata={key: value for key, value in raw.items() if key not in {"business_id", "summary", "text", "caption", "score", "supports"}},
            )
            rows.append((business_id, item))
        return rows

    def iter_businesses(self) -> list[BusinessRecord]:
        return list(self._businesses.values())

    def get_business(self, business_id: str) -> BusinessRecord | None:
        return self._businesses.get(business_id)

    def build_geo_constraint(self, seed_rows: list[dict[str, Any]]) -> Any:
        return None

    def order_records_by_seed(self, records: list[dict[str, Any]], seed: int, id_key: str = "business_id") -> list[dict[str, Any]]:
        output = [dict(row) for row in records if isinstance(row, dict) and row.get(id_key) is not None]
        output.sort(key=lambda row: (_provider_crc32(str(row[id_key]), seed), str(row[id_key])))
        return output

    def fetch_one_near_seeded(
        self,
        center_lat: float,
        center_lon: float,
        radius_km: float,
        seed: int,
        *,
        exclude_id: str | None = None,
    ) -> dict[str, Any] | None:
        candidates: list[tuple[int, str, dict[str, Any]]] = []
        for record in self.iter_businesses():
            row = dict(record.fields)
            if not _provider_has_valid_coordinates(row):
                continue
            business_id = str(row.get("business_id", ""))
            if exclude_id is not None and business_id == exclude_id:
                continue
            distance = _provider_haversine_km(center_lat, center_lon, float(row["latitude"]), float(row["longitude"]))
            if distance <= radius_km:
                candidates.append((_provider_crc32(business_id, seed), business_id, row))
        if not candidates:
            return None
        candidates.sort(key=lambda item: (item[0], item[1]))
        return candidates[0][2]

    def recall(self, family: EvidenceFamily, query: str, top_k: int) -> list[tuple[str, float]]:
        scored: dict[str, float] = defaultdict(float)
        for business_id, evidence_items in self._evidence.get(family, {}).items():
            for item in evidence_items:
                score = item.score if item.score is not None else _overlap_score(query, item.summary)
                scored[business_id] = max(scored[business_id], score)
        ordered = sorted(scored.items(), key=lambda item: item[1], reverse=True)
        return ordered[:top_k]

    def get_evidence(
        self,
        family: EvidenceFamily,
        business_id: str,
        query: str,
        limit: int,
    ) -> list[EvidenceItem]:
        rows = list(self._evidence.get(family, {}).get(business_id, []))
        rows.sort(
            key=lambda item: item.score if item.score is not None else _overlap_score(query, item.summary),
            reverse=True,
        )
        return rows[:limit]

    def get_reviews(self, business_id: str, limit: int | None = None) -> list[dict[str, Any]]:
        rows = [
            {
                "business_id": business_id,
                "text": item.summary,
                "source_locator_type": item.source_locator_type,
                "source_locator": item.source_locator,
                "score": item.score,
                "supports": item.supports,
            }
            for item in self._evidence.get("text", {}).get(business_id, [])
        ]
        return rows if limit is None else rows[:limit]

    def search_reviews(self, query: str, business_ids: list[str], top_k: int) -> list[dict[str, Any]]:
        hits: list[dict[str, Any]] = []
        allowed = set(business_ids)
        for business_id, rows in self._evidence.get("text", {}).items():
            if business_id not in allowed:
                continue
            for item in rows:
                score = item.score if item.score is not None else _overlap_score(query, item.summary)
                hits.append(
                    {
                        "business_id": business_id,
                        "text": item.summary,
                        "coarse_score": score,
                        "source_locator_type": item.source_locator_type,
                        "source_locator": item.source_locator,
                        "supports": item.supports,
                    }
                )
        hits.sort(key=lambda row: float(row.get("coarse_score") or 0), reverse=True)
        return hits[:top_k]

    def search_review_embeddings(
        self,
        query_vector: list[float],
        *,
        business_ids: list[str] | None = None,
        top_k: int = 100,
    ) -> list[dict[str, Any]]:
        if self.review_index is None:
            return []
        return self.review_index.search(query_vector, business_ids=business_ids, top_k=top_k)

    def hybrid_search_reviews(
        self,
        search_text: str,
        query_vector: list[float] | None,
        business_ids: list[str],
        *,
        top_k: int = 300,
        fulltext_limit: int | None = None,
        vector_limit: int | None = None,
        rank_smoothing: int = 60,
    ) -> list[dict[str, Any]]:
        text_rows = self.search_reviews(search_text, business_ids, fulltext_limit or top_k)
        vector_rows = (
            self.search_review_embeddings(query_vector, business_ids=business_ids, top_k=vector_limit or top_k)
            if query_vector
            else []
        )
        merged: dict[str, dict[str, Any]] = {}
        for rank, row in enumerate(text_rows, start=1):
            key = str(row.get("source_locator") or row.get("review_id") or f"text:{rank}:{row.get('business_id')}")
            item = dict(row)
            item["_keyword_rank"] = rank
            item["_keyword_score"] = float(row.get("coarse_score", row.get("score", 0)) or 0)
            item["_score"] = 1.0 / (rank + rank_smoothing)
            merged[key] = item
        for rank, row in enumerate(vector_rows, start=1):
            key = str(row.get("source_locator") or row.get("review_id") or f"vector:{rank}:{row.get('business_id')}")
            item = merged.get(key, dict(row))
            item["_semantic_rank"] = rank
            item["_semantic_score"] = float(row.get("_score", row.get("score", 0)) or 0)
            item["_score"] = float(item.get("_score", 0)) + 1.0 / (rank + rank_smoothing)
            item.setdefault("business_id", row.get("business_id"))
            item.setdefault("text", row.get("text") or row.get("summary", ""))
            merged[key] = item
        rows = list(merged.values())
        rows.sort(key=lambda row: float(row.get("_score") or 0), reverse=True)
        return rows[:top_k]

    def get_photos(self, business_id: str, limit: int | None = None) -> list[dict[str, Any]]:
        rows = [
            {
                "business_id": business_id,
                "path": item.source_locator,
                "caption": item.summary,
                "source_locator_type": item.source_locator_type,
                "source_locator": item.source_locator,
                "score": item.score,
                "supports": item.supports,
            }
            for item in self._evidence.get("image", {}).get(business_id, [])
        ]
        return rows if limit is None else rows[:limit]

    def search_photos(self, query: str, business_ids: list[str], top_k: int) -> list[dict[str, Any]]:
        hits: list[dict[str, Any]] = []
        allowed = set(business_ids)
        for business_id, rows in self._evidence.get("image", {}).items():
            if business_id not in allowed:
                continue
            for item in rows:
                score = item.score if item.score is not None else _overlap_score(query, item.summary)
                hits.append(
                    {
                        "business_id": business_id,
                        "path": item.source_locator,
                        "caption": item.summary,
                        "score": score,
                        "source_locator_type": item.source_locator_type,
                        "source_locator": item.source_locator,
                        "supports": item.supports,
                    }
                )
        hits.sort(key=lambda row: float(row.get("score") or 0), reverse=True)
        return hits[:top_k]

    def search_photo_embeddings(
        self,
        query_vector: list[float],
        *,
        business_ids: list[str] | None = None,
        top_k: int = 100,
    ) -> list[dict[str, Any]]:
        if self.photo_index is None:
            return []
        return self.photo_index.search(query_vector, business_ids=business_ids, top_k=top_k)

    def search_feature_embeddings(self, query_vector: list[float], *, limit: int = 100) -> list[dict[str, Any]]:
        if self.feature_index is None:
            return []
        return self.feature_index.search(query_vector, top_k=limit)

    def get_features_of_business(self, business_id: str, sentiment: str = "pos") -> list[str]:
        return self.graph.get_features_of_business(business_id, sentiment) if self.graph else []

    def get_businesses_by_feature(self, feature: str, sentiment: str = "pos") -> list[str]:
        return self.graph.get_businesses_by_feature(feature, sentiment) if self.graph else []

    def sample_features_by_distribution(self, pool: Iterable[str], top_k: int = 1) -> list[str]:
        return self.graph.sample_features_by_distribution(pool, top_k) if self.graph else []

    def get_users_connected_via_feature(self, business_id: str, feature: str, sentiment: str = "pos") -> list[str]:
        return self.graph.get_users_connected_via_feature(business_id, feature, sentiment) if self.graph else []

    def get_features_of_user(self, user_id: str, sentiment: str = "pos") -> list[str]:
        return self.graph.get_features_of_user(user_id, sentiment) if self.graph else []

    def get_users_by_feature(self, feature: str, sentiment: str = "pos") -> list[str]:
        return self.graph.get_users_by_feature(feature, sentiment) if self.graph else []


def _open_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")  # type: ignore[return-value]
    return path.open("r", encoding="utf-8")


def _iter_json_records(path: Path) -> Iterable[dict[str, Any]]:
    with _open_text(path) as handle:
        cursor = handle.tell()
        first = ""
        while True:
            char = handle.read(1)
            if not char:
                return
            if not char.isspace():
                first = char
                break
        handle.seek(cursor)
        if first == "[":
            payload = json.load(handle)
            if isinstance(payload, list):
                for item in payload:
                    if isinstance(item, dict):
                        yield item
            return
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON record at {path}:{line_number}: {exc}") from exc
            if isinstance(payload, dict):
                yield payload


def _pascal_or_camel_to_snake(value: str) -> str:
    value = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", value)
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value)
    return value.replace(" ", "_").replace("-", "_").lower()


def _parse_attribute_value(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.lower() in {"true", "false"}:
            return stripped.lower() == "true"
        if stripped.lower() in {"none", "null"}:
            return None
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                return ast.literal_eval(stripped)
            except (ValueError, SyntaxError):
                return stripped
    return value


def _flatten_yelp_business(raw: dict[str, Any]) -> dict[str, Any]:
    row = dict(raw)
    attributes = raw.get("attributes") if isinstance(raw.get("attributes"), dict) else {}
    row["attributes_json"] = json.dumps(attributes, ensure_ascii=False, sort_keys=True) if attributes else ""
    hours = raw.get("hours") if isinstance(raw.get("hours"), dict) else {}
    row["hours_json"] = json.dumps(hours, ensure_ascii=False, sort_keys=True) if hours else ""
    for key, value in attributes.items():
        parsed = _parse_attribute_value(value)
        field_name = _pascal_or_camel_to_snake(str(key))
        if isinstance(parsed, dict):
            for child_key, child_value in parsed.items():
                row[f"{field_name}_{_pascal_or_camel_to_snake(str(child_key))}"] = _parse_attribute_value(child_value)
        else:
            row[field_name] = parsed
    return row


class YelpOpenDatasetProvider(FileBackedConstructionProvider):
    """Provider backed by locally downloaded Yelp Open Dataset files.

    This provider reads the user's local Yelp business, review, and photo
    metadata files. It does not redistribute source content; it only exposes the
    same construction-provider interface used by the generation pipeline.
    """

    def __init__(
        self,
        business_records: list[BusinessRecord],
        *,
        reviews_by_business: dict[str, list[dict[str, Any]]] | None = None,
        photos_by_business: dict[str, list[dict[str, Any]]] | None = None,
        kg_evidence: dict[str, list[EvidenceItem]] | None = None,
        graph: FeatureGraphStore | None = None,
        review_index: VectorDocumentIndex | None = None,
        photo_index: VectorDocumentIndex | None = None,
        feature_index: VectorDocumentIndex | None = None,
        photo_dir: Path | None = None,
    ):
        super().__init__(
            business_records,
            {"kg": [(bid, item) for bid, rows in (kg_evidence or {}).items() for item in rows]},
            graph=graph,
            review_index=review_index,
            photo_index=photo_index,
            feature_index=feature_index,
        )
        self._reviews_by_business = reviews_by_business or {}
        self._photos_by_business = photos_by_business or {}
        self.photo_dir = photo_dir

    @classmethod
    def from_config(cls, data_config: dict[str, Any]) -> "YelpOpenDatasetProvider":
        resolved_config = dict(data_config)
        if resolved_config.get("yelp_root"):
            yelp_root = Path(resolved_config["yelp_root"])
            resolved_files = _resolve_yelp_provider_files(yelp_root)
            for key, value in resolved_files.items():
                resolved_config.setdefault(key, value)
            photo_dir = _resolve_yelp_provider_photo_dir(yelp_root)
            if photo_dir is not None:
                resolved_config.setdefault("photo_dir", photo_dir)
        business_path = Path(resolved_config["business_jsonl"])
        business_records = [BusinessRecord.from_raw(_flatten_yelp_business(row)) for row in _iter_json_records(business_path)]
        business_ids = {record.business_id for record in business_records}
        reviews = cls._load_reviews(
            Path(resolved_config["review_jsonl"]) if resolved_config.get("review_jsonl") else None,
            business_ids,
            max_reviews_per_business=resolved_config.get("max_reviews_per_business"),
            max_total_reviews=resolved_config.get("max_total_reviews"),
        )
        photos = cls._load_photos(
            Path(resolved_config["photos_json"]) if resolved_config.get("photos_json") else None,
            business_ids,
            photo_dir=Path(resolved_config["photo_dir"]) if resolved_config.get("photo_dir") else None,
        )
        kg = cls._load_kg_evidence(Path(resolved_config["kg_evidence_jsonl"]) if resolved_config.get("kg_evidence_jsonl") else None)
        graph = _graph_from_config(resolved_config)
        review_index = VectorDocumentIndex.from_jsonl(Path(resolved_config["review_embedding_jsonl"])) if resolved_config.get("review_embedding_jsonl") else None
        photo_index = VectorDocumentIndex.from_jsonl(Path(resolved_config["photo_embedding_jsonl"])) if resolved_config.get("photo_embedding_jsonl") else None
        feature_index = VectorDocumentIndex.from_jsonl(Path(resolved_config["feature_embedding_jsonl"])) if resolved_config.get("feature_embedding_jsonl") else None
        return cls(
            business_records,
            reviews_by_business=reviews,
            photos_by_business=photos,
            kg_evidence=kg,
            graph=graph,
            review_index=review_index,
            photo_index=photo_index,
            feature_index=feature_index,
            photo_dir=Path(resolved_config["photo_dir"]) if resolved_config.get("photo_dir") else None,
        )

    @staticmethod
    def _load_reviews(
        path: Path | None,
        business_ids: set[str],
        *,
        max_reviews_per_business: Any = None,
        max_total_reviews: Any = None,
    ) -> dict[str, list[dict[str, Any]]]:
        if path is None:
            return {}
        per_business_limit = int(max_reviews_per_business) if max_reviews_per_business else None
        total_limit = int(max_total_reviews) if max_total_reviews else None
        rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
        total = 0
        for raw in _iter_json_records(path):
            business_id = str(raw.get("business_id") or "")
            if business_id not in business_ids:
                continue
            if per_business_limit is not None and len(rows[business_id]) >= per_business_limit:
                continue
            rows[business_id].append(
                {
                    "business_id": business_id,
                    "review_id": str(raw.get("review_id") or ""),
                    "user_id": str(raw.get("user_id") or ""),
                    "text": str(raw.get("text") or ""),
                    "stars": raw.get("stars"),
                    "date": raw.get("date"),
                    "source_locator_type": "yelp_review_id",
                    "source_locator": str(raw.get("review_id") or ""),
                }
            )
            total += 1
            if total_limit is not None and total >= total_limit:
                break
        return dict(rows)

    @staticmethod
    def _load_photos(
        path: Path | None,
        business_ids: set[str],
        *,
        photo_dir: Path | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        if path is None:
            return {}
        rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for raw in _iter_json_records(path):
            business_id = str(raw.get("business_id") or "")
            if business_id not in business_ids:
                continue
            photo_id = str(raw.get("photo_id") or raw.get("id") or "")
            photo_path = ""
            if photo_dir is not None and photo_id:
                photo_path = str(photo_dir / f"{photo_id}.jpg")
            rows[business_id].append(
                {
                    "business_id": business_id,
                    "photo_id": photo_id,
                    "caption": str(raw.get("caption") or raw.get("label") or ""),
                    "label": str(raw.get("label") or ""),
                    "path": photo_path,
                    "source_locator_type": "photo_id_stem",
                    "source_locator": photo_id,
                }
            )
        return dict(rows)

    @staticmethod
    def _load_kg_evidence(path: Path | None) -> dict[str, list[EvidenceItem]]:
        if path is None:
            return {}
        rows: dict[str, list[EvidenceItem]] = defaultdict(list)
        for raw in _iter_json_records(path):
            business_id = str(raw.get("business_id") or "")
            if not business_id:
                continue
            rows[business_id].append(
                EvidenceItem(
                    family="kg",
                    source_locator_type=str(raw.get("source_locator_type") or "kg_feature"),
                    source_locator=str(raw.get("source_locator") or raw.get("feature") or raw.get("id") or ""),
                    summary=str(raw.get("summary") or raw.get("feature") or raw.get("text") or ""),
                    score=float(raw["score"]) if raw.get("score") is not None else None,
                    supports=raw.get("supports"),
                    metadata={key: value for key, value in raw.items() if key not in {"business_id", "summary", "feature", "text", "score", "supports"}},
                )
            )
        return dict(rows)

    def get_reviews(self, business_id: str, limit: int | None = None) -> list[dict[str, Any]]:
        rows = list(self._reviews_by_business.get(business_id, []))
        return rows if limit is None else rows[:limit]

    def search_reviews(self, query: str, business_ids: list[str], top_k: int) -> list[dict[str, Any]]:
        hits: list[dict[str, Any]] = []
        allowed = set(business_ids)
        for business_id, rows in self._reviews_by_business.items():
            if business_id not in allowed:
                continue
            for row in rows:
                score = _overlap_score(query, str(row.get("text") or ""))
                hits.append({**row, "coarse_score": score, "score": score})
        hits.sort(key=lambda row: float(row.get("coarse_score") or 0), reverse=True)
        return hits[:top_k]

    def get_photos(self, business_id: str, limit: int | None = None) -> list[dict[str, Any]]:
        rows = list(self._photos_by_business.get(business_id, []))
        return rows if limit is None else rows[:limit]

    def search_photos(self, query: str, business_ids: list[str], top_k: int) -> list[dict[str, Any]]:
        hits: list[dict[str, Any]] = []
        allowed = set(business_ids)
        for business_id, rows in self._photos_by_business.items():
            if business_id not in allowed:
                continue
            business = self.get_business(business_id)
            business_text = " ".join(
                str((business.fields if business else {}).get(key) or "") for key in ["name", "categories", "city"]
            )
            for row in rows:
                text = " ".join([str(row.get("caption") or ""), str(row.get("label") or ""), business_text])
                score = _overlap_score(query, text)
                hits.append({**row, "score": score, "coarse_score": score})
        hits.sort(key=lambda row: float(row.get("score") or 0), reverse=True)
        return hits[:top_k]

    def recall(self, family: EvidenceFamily, query: str, top_k: int) -> list[tuple[str, float]]:
        if family == "text":
            return [
                (row["business_id"], float(row.get("coarse_score") or 0.0))
                for row in self.search_reviews(query, list(self._businesses), top_k)
            ]
        if family == "image":
            return [
                (row["business_id"], float(row.get("score") or 0.0))
                for row in self.search_photos(query, list(self._businesses), top_k)
            ]
        return super().recall(family, query, top_k)

    def get_evidence(
        self,
        family: EvidenceFamily,
        business_id: str,
        query: str,
        limit: int,
    ) -> list[EvidenceItem]:
        if family == "text":
            return [
                EvidenceItem(
                    family="text",
                    source_locator_type="yelp_review_id",
                    source_locator=str(row.get("review_id") or ""),
                    summary=str(row.get("text") or ""),
                    score=_overlap_score(query, str(row.get("text") or "")),
                    metadata={"stars": row.get("stars"), "date": row.get("date")},
                )
                for row in self.search_reviews(query, [business_id], limit)
            ]
        if family == "image":
            return [
                EvidenceItem(
                    family="image",
                    source_locator_type="photo_id_stem",
                    source_locator=str(row.get("photo_id") or ""),
                    summary=str(row.get("caption") or row.get("label") or ""),
                    score=float(row.get("score") or 0.0),
                    metadata={"label": row.get("label")},
                )
                for row in self.search_photos(query, [business_id], limit)
            ]
        return super().get_evidence(family, business_id, query, limit)


class SQLConstructionProvider(FileBackedConstructionProvider):
    """Configurable SQL/OceanBase provider.

    This adapter keeps database access behind config. It can use simple table
    defaults for local reconstruction or SQL templates matching an existing
    deployment. Construction modules still perform expansion, rerank, judge,
    and candidate-state transitions.
    """

    def __init__(self, data_config: dict[str, Any]):
        try:
            import pymysql  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError("SQLConstructionProvider requires installing the optional db dependencies.") from exc
        self.config = data_config
        self.business_table = str(data_config.get("business_table", "business"))
        self.review_table = str(data_config.get("review_table", "review"))
        self.photo_table = str(data_config.get("photo_table", "photo"))
        self.kg_table = str(data_config.get("kg_table", "kg_evidence"))
        self.business_id_column = str(data_config.get("business_id_column", "business_id"))
        self.business_name_column = str(data_config.get("business_name_column", "name"))
        self.latitude_column = str(data_config.get("latitude_column", "latitude"))
        self.longitude_column = str(data_config.get("longitude_column", "longitude"))
        self.geom_column = data_config.get("geom_column")
        self.review_text_column = str(data_config.get("review_text_column", "text"))
        self.photo_caption_column = str(data_config.get("photo_caption_column", "caption"))
        self.connection = pymysql.connect(
            host=str(data_config["host"]),
            port=int(data_config.get("port", 3306)),
            user=str(data_config["user"]),
            password=str(data_config.get("password", "")),
            charset=str(data_config.get("charset", "utf8mb4")),
            database=str(data_config["database"]),
            cursorclass=pymysql.cursors.DictCursor,
        )
        super().__init__([], {}, graph=_graph_from_config(data_config))

    def _query(self, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        with self.connection.cursor() as cursor:
            cursor.execute(sql, params)
            return list(cursor.fetchall())

    def _configured_query(self, key: str, params: tuple[Any, ...]) -> list[dict[str, Any]] | None:
        sql = self.config.get(key)
        if not sql:
            return None
        return self._query(str(sql), params)

    def order_records_by_seed(self, records: list[dict[str, Any]], seed: int, id_key: str = "business_id") -> list[dict[str, Any]]:
        output = [dict(row) for row in records if isinstance(row, dict) and row.get(id_key) is not None]
        output.sort(key=lambda row: (_provider_crc32(str(row[id_key]), seed), str(row[id_key])))
        return output

    def fetch_one_near_seeded(
        self,
        center_lat: float,
        center_lon: float,
        radius_km: float,
        seed: int,
        *,
        exclude_id: str | None = None,
    ) -> dict[str, Any] | None:
        """DBAnchorProvider-style seeded neighbor lookup.

        A config can force a spatial query with `geo_neighbor_mode: spatial`
        and `geom_column`. The default uses a bounding-box query followed by
        Python Haversine distance ordering.
        """

        if self.config.get("fetch_one_near_seeded_sql"):
            rows = self._query(
                str(self.config["fetch_one_near_seeded_sql"]),
                (center_lat, center_lon, radius_km, seed, exclude_id),
            )
            return dict(rows[0]) if rows else None
        mode = str(self.config.get("geo_neighbor_mode") or "traditional").lower()
        if mode == "spatial" and self.geom_column:
            row = self._fetch_one_near_seeded_spatial(center_lat, center_lon, radius_km, seed, exclude_id)
            if row is not None:
                return row
        return self._fetch_one_near_seeded_traditional(center_lat, center_lon, radius_km, seed, exclude_id)

    def _fetch_one_near_seeded_traditional(
        self,
        center_lat: float,
        center_lon: float,
        radius_km: float,
        seed: int,
        exclude_id: str | None,
    ) -> dict[str, Any] | None:
        min_lat, max_lat, min_lon, max_lon = _provider_deg_bbox(center_lat, center_lon, radius_km)
        where = [
            f"{self.latitude_column} BETWEEN %s AND %s",
            f"{self.longitude_column} BETWEEN %s AND %s",
        ]
        params: list[Any] = [min_lat, max_lat, min_lon, max_lon]
        if exclude_id is not None:
            where.append(f"{self.business_id_column} <> %s")
            params.append(str(exclude_id))
        if self.config.get("geo_extra_where"):
            where.append(f"({self.config['geo_extra_where']})")
            params.extend(list(self.config.get("geo_extra_params") or []))
        sql = f"""
        SELECT {self.business_id_column} AS business_id,
               {self.business_name_column} AS name,
               {self.latitude_column} AS latitude,
               {self.longitude_column} AS longitude
        FROM {self.business_table}
        WHERE {" AND ".join(where)}
        """
        candidates: list[tuple[int, str, dict[str, Any]]] = []
        for row in self._query(sql, tuple(params)):
            if not _provider_has_valid_coordinates(row):
                continue
            distance = _provider_haversine_km(
                center_lat,
                center_lon,
                float(row["latitude"]),
                float(row["longitude"]),
            )
            if distance <= radius_km:
                business_id = str(row["business_id"])
                candidates.append((_provider_crc32(business_id, seed), business_id, dict(row)))
        if not candidates:
            return None
        candidates.sort(key=lambda item: (item[0], item[1]))
        return candidates[0][2]

    def _fetch_one_near_seeded_spatial(
        self,
        center_lat: float,
        center_lon: float,
        radius_km: float,
        seed: int,
        exclude_id: str | None,
    ) -> dict[str, Any] | None:
        geom_col = str(self.geom_column)
        where_extra = []
        params_tail: list[Any] = []
        if exclude_id is not None:
            where_extra.append(f"{self.business_id_column} <> %s")
            params_tail.append(str(exclude_id))
        if self.config.get("geo_extra_where"):
            where_extra.append(f"({self.config['geo_extra_where']})")
            params_tail.extend(list(self.config.get("geo_extra_params") or []))
        where_tail = " AND " + " AND ".join(where_extra) if where_extra else ""
        sql = f"""
        SELECT {self.business_id_column} AS business_id,
               {self.business_name_column} AS name,
               ST_Y({geom_col}) AS latitude,
               ST_X({geom_col}) AS longitude
        FROM {self.business_table}
        WHERE {geom_col} IS NOT NULL
          AND ST_Distance_Sphere({geom_col}, ST_SRID(POINT(%s, %s), 4326)) <= %s
        {where_tail}
        ORDER BY CRC32(CONCAT({self.business_id_column}, ':', %s)) ASC, {self.business_id_column} ASC
        LIMIT 1
        """
        params = [center_lon, center_lat, float(radius_km) * 1000.0] + params_tail + [int(seed)]
        rows = self._query(sql, tuple(params))
        if not rows:
            return None
        row = dict(rows[0])
        return row if _provider_has_valid_coordinates(row) else None

    def iter_businesses(self) -> list[BusinessRecord]:
        configured = self._configured_query("business_select_sql", ())
        if configured is not None:
            return [BusinessRecord.from_raw(row) for row in configured]
        limit = int(self.config.get("business_limit", 0) or 0)
        sql = f"SELECT * FROM {self.business_table}"
        params: tuple[Any, ...] = ()
        if limit > 0:
            sql += " LIMIT %s"
            params = (limit,)
        return [BusinessRecord.from_raw(row) for row in self._query(sql, params)]

    def get_business(self, business_id: str) -> BusinessRecord | None:
        configured = self._configured_query("business_by_id_sql", (business_id,))
        rows = configured if configured is not None else self._query(
            f"SELECT * FROM {self.business_table} WHERE {self.business_id_column}=%s LIMIT 1",
            (business_id,),
        )
        if not rows:
            return None
        return BusinessRecord.from_raw(rows[0])

    def get_reviews(self, business_id: str, limit: int | None = None) -> list[dict[str, Any]]:
        rows = self._configured_query("reviews_by_business_sql", (business_id, limit or self.config.get("review_limit", 50)))
        if rows is not None:
            return rows
        return self._query(
            f"SELECT * FROM {self.review_table} WHERE {self.business_id_column}=%s LIMIT %s",
            (business_id, int(limit or self.config.get("review_limit", 50))),
        )

    def search_reviews(self, query: str, business_ids: list[str], top_k: int) -> list[dict[str, Any]]:
        rows = self._configured_query("review_search_sql", (query, top_k))
        if rows is None:
            rows = []
            for business_id in business_ids:
                rows.extend(self.get_reviews(business_id, limit=int(self.config.get("review_limit", 50))))
        hits = []
        allowed = set(business_ids)
        for row in rows:
            business_id = str(row.get("business_id") or row.get(self.business_id_column) or "")
            if business_id not in allowed:
                continue
            text = str(row.get("text") or row.get(self.review_text_column) or row.get("summary") or "")
            score = float(row.get("_score", row.get("score", _overlap_score(query, text))) or 0.0)
            hits.append({**row, "business_id": business_id, "text": text, "coarse_score": score})
        hits.sort(key=lambda item: float(item.get("coarse_score") or 0), reverse=True)
        return hits[:top_k]

    def hybrid_search_reviews(
        self,
        search_text: str,
        query_vector: list[float] | None,
        business_ids: list[str],
        *,
        top_k: int = 300,
        fulltext_limit: int | None = None,
        vector_limit: int | None = None,
        rank_smoothing: int = 60,
    ) -> list[dict[str, Any]]:
        configured = self.config.get("review_hybrid_search_sql")
        if configured:
            return self._query(str(configured), (search_text, json.dumps(query_vector or []), top_k))
        if not query_vector:
            return self.search_reviews(search_text, business_ids, top_k)
        table = str(self.config.get("review_embedding_table") or self.review_table)
        text_col = self.review_text_column
        vector_col = str(self.config.get("review_embedding_column", "embedding"))
        pk_col = str(self.config.get("review_pk_column", "__pk_increment"))
        ids_sql = _sql_id_list(business_ids)
        vector_sql = _sql_vector(query_vector)
        escaped_text = str(search_text).replace("\\", "\\\\").replace("'", "''")
        fulltext_limit = int(fulltext_limit or 3 * top_k)
        vector_limit = int(vector_limit or 3 * top_k)
        sql = f"""
        SELECT
            ifnull(_fts.business_id, _vs.business_id) as business_id,
            ifnull(_fts.{text_col}, _vs.{text_col}) as text,
            ifnull(_fts.{vector_col}, _vs.{vector_col}) as embedding,
            _keyword_score, _semantic_score,
            (ifnull(1 / (_fts._keyword_rank + {rank_smoothing}), 0)
             + ifnull(1 / (_vs._semantic_rank + {rank_smoothing}), 0)) as _score
        FROM (
            (SELECT *, RANK() over(ORDER BY _keyword_score DESC) as _keyword_rank
             FROM (
                SELECT /*+ opt_param('hidden_column_visible', 'true') */
                       {pk_col}, business_id, {text_col}, {vector_col},
                       match({text_col}) against('{escaped_text}' in natural language mode) as _keyword_score
                FROM {table}
                WHERE match({text_col}) against('{escaped_text}' in natural language mode)
                AND business_id IN ({ids_sql})
                ORDER BY _keyword_score DESC LIMIT {fulltext_limit}
             )
            ) _fts
            FULL JOIN
            (SELECT *, RANK() over(ORDER BY _semantic_score DESC) as _semantic_rank
             FROM (
                SELECT /*+ opt_param('hidden_column_visible', 'true') */
                       cosine_distance({vector_col}, '{vector_sql}') as _distance,
                       {pk_col}, business_id, {text_col}, {vector_col},
                       round(1 - cosine_distance({vector_col}, '{vector_sql}') / 2, 8) as _semantic_score
                FROM {table}
                WHERE business_id IN ({ids_sql})
                ORDER BY _distance APPROXIMATE LIMIT {vector_limit}
             )
            ) _vs
            ON _fts.{pk_col} = _vs.{pk_col}
        )
        ORDER BY _score DESC LIMIT {int(top_k)}
        """
        rows = self._query(sql)
        return [
            {
                **row,
                "business_id": str(row.get("business_id") or ""),
                "text": str(row.get("text") or row.get(text_col) or ""),
                "coarse_score": float(row.get("_score", row.get("score", 0)) or 0.0),
            }
            for row in rows
        ]

    def get_photos(self, business_id: str, limit: int | None = None) -> list[dict[str, Any]]:
        rows = self._configured_query("photos_by_business_sql", (business_id, limit or self.config.get("photo_limit", 50)))
        if rows is not None:
            return rows
        return self._query(
            f"SELECT * FROM {self.photo_table} WHERE {self.business_id_column}=%s LIMIT %s",
            (business_id, int(limit or self.config.get("photo_limit", 50))),
        )

    def search_photos(self, query: str, business_ids: list[str], top_k: int) -> list[dict[str, Any]]:
        rows = self._configured_query("photo_search_sql", (query, top_k))
        if rows is None:
            rows = []
            for business_id in business_ids:
                rows.extend(self.get_photos(business_id, limit=int(self.config.get("photo_limit", 50))))
        hits = []
        allowed = set(business_ids)
        for row in rows:
            business_id = str(row.get("business_id") or row.get(self.business_id_column) or "")
            if business_id not in allowed:
                continue
            caption = str(row.get("caption") or row.get(self.photo_caption_column) or row.get("summary") or "")
            score = float(row.get("_score", row.get("score", _overlap_score(query, caption))) or 0.0)
            hits.append({**row, "business_id": business_id, "caption": caption, "score": score, "coarse_score": score})
        hits.sort(key=lambda item: float(item.get("score") or 0), reverse=True)
        return hits[:top_k]

    def search_review_embeddings(
        self,
        query_vector: list[float],
        *,
        business_ids: list[str] | None = None,
        top_k: int = 100,
    ) -> list[dict[str, Any]]:
        sql = self.config.get("review_embedding_search_sql")
        if sql:
            rows = self._query(str(sql), (json.dumps(query_vector), top_k))
        else:
            table = str(self.config.get("review_embedding_table") or self.review_table)
            text_col = self.review_text_column
            vector_col = str(self.config.get("review_embedding_column", "embedding"))
            ids_clause = ""
            if business_ids:
                ids_clause = f"WHERE business_id IN ({_sql_id_list(business_ids)})"
            vector_sql = _sql_vector(query_vector)
            rows = self._query(
                f"""
                SELECT business_id, {text_col} AS text,
                       round(1 - cosine_distance({vector_col}, '{vector_sql}') / 2, 8) as _score
                FROM {table}
                {ids_clause}
                ORDER BY cosine_distance({vector_col}, '{vector_sql}')
                LIMIT {int(top_k)}
                """
            )
        if business_ids is None:
            return rows[:top_k]
        allowed = set(business_ids)
        return [row for row in rows if str(row.get("business_id") or row.get(self.business_id_column) or "") in allowed][:top_k]

    def search_photo_embeddings(
        self,
        query_vector: list[float],
        *,
        business_ids: list[str] | None = None,
        top_k: int = 100,
    ) -> list[dict[str, Any]]:
        sql = self.config.get("photo_embedding_search_sql")
        if sql:
            rows = self._query(str(sql), (json.dumps(query_vector), top_k))
        else:
            table = str(self.config.get("photo_embedding_table", "photo_embedding"))
            vector_col = str(self.config.get("photo_embedding_column", "embedding"))
            path_col = str(self.config.get("photo_path_column", "path"))
            ids_clause = ""
            if business_ids:
                ids_clause = f"WHERE business_id IN ({_sql_id_list(business_ids)})"
            vector_sql = _sql_vector(query_vector)
            rows = self._query(
                f"""
                SELECT {path_col} AS path, business_id,
                       round(1 - cosine_distance({vector_col}, '{vector_sql}') / 2, 8) as score
                FROM {table}
                {ids_clause}
                ORDER BY cosine_distance({vector_col}, '{vector_sql}') APPROXIMATE
                LIMIT {int(top_k)}
                """
            )
        if business_ids is None:
            return rows[:top_k]
        allowed = set(business_ids)
        return [row for row in rows if str(row.get("business_id") or row.get(self.business_id_column) or "") in allowed][:top_k]

    def search_feature_embeddings(self, query_vector: list[float], *, limit: int = 100) -> list[dict[str, Any]]:
        sql = self.config.get("feature_embedding_search_sql")
        if sql:
            return self._query(str(sql), (json.dumps(query_vector), limit))[:limit]
        table = str(self.config.get("feature_embedding_table", "feature_embedding"))
        feature_col = str(self.config.get("feature_key_column", "feature_key"))
        vector_col = str(self.config.get("feature_embedding_column", "embedding"))
        vector_sql = _sql_vector(query_vector)
        return self._query(
            f"""
            SELECT {feature_col} AS feature_key
            FROM {table}
            ORDER BY cosine_distance({vector_col}, '{vector_sql}') APPROXIMATE
            LIMIT {int(limit)}
            """
        )[:limit]


def build_construction_provider(data_config: dict[str, Any]) -> ConstructionDataProvider:
    provider_type = str(data_config.get("provider") or data_config.get("type") or "file").strip().lower()
    if provider_type in {"file", "file_backed", "jsonl"}:
        return FileBackedConstructionProvider.from_config(data_config)
    if provider_type in {"yelp", "yelp_open_dataset", "yelp_raw"}:
        return YelpOpenDatasetProvider.from_config(data_config)
    if provider_type in {"sql", "mysql", "oceanbase", "ob"}:
        return SQLConstructionProvider(data_config)
    raise ValueError(f"Unsupported construction provider: {provider_type}")
