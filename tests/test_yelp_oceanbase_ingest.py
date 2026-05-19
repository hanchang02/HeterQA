from __future__ import annotations

import argparse
import json
import tarfile
import zipfile
from pathlib import Path

import pytest

from heterqa.data.yelp_oceanbase import (
    _business_rows,
    _photo_rows,
    create_yelp_tables,
    download_yelp_open_dataset,
    extract_archive,
    oceanbase_config_from_args,
    resolve_yelp_files,
    resolve_yelp_photo_dir,
)


class FakeCursor:
    def __init__(self, connection: FakeConnection):
        self.connection = connection

    def __enter__(self) -> "FakeCursor":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def execute(self, sql: str, params: tuple | None = None) -> None:
        self.connection.statements.append((sql, params))


class FakeConnection:
    def __init__(self) -> None:
        self.statements: list[tuple[str, tuple | None]] = []
        self.commits = 0

    def cursor(self) -> FakeCursor:
        return FakeCursor(self)

    def commit(self) -> None:
        self.commits += 1


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_business_rows_flatten_yelp_attributes(tmp_path: Path) -> None:
    path = tmp_path / "yelp_academic_dataset_business.json"
    _write_jsonl(
        path,
        [
            {
                "business_id": "b1",
                "name": "Cafe",
                "categories": "Cafe, Restaurants",
                "stars": 4.5,
                "attributes": {
                    "BusinessAcceptsCreditCards": "True",
                    "OutdoorSeating": "False",
                    "Ambience": "{'casual': True, 'romantic': False}",
                },
                "hours": {"Monday": "9:00-17:00"},
            }
        ],
    )

    columns, rows = _business_rows(path, max_rows=1)
    row = next(rows)
    payload = dict(zip(columns, row, strict=True))

    assert payload["business_id"] == "b1"
    assert payload["business_accepts_credit_cards"] == 1
    assert payload["outdoor_seating"] == 0
    assert payload["ambience_casual"] == 1
    assert "BusinessAcceptsCreditCards" in payload["attributes_json"]
    assert "Monday" in payload["hours_json"]


def test_photo_rows_add_photo_path(tmp_path: Path) -> None:
    path = tmp_path / "photos.json"
    _write_jsonl(path, [{"photo_id": "p1", "business_id": "b1", "caption": "front", "label": "outside"}])

    columns, rows = _photo_rows(path, max_rows=1, photo_dir=tmp_path / "photos")
    payload = dict(zip(columns, next(rows), strict=True))

    assert payload["photo_id"] == "p1"
    assert payload["photo_path"].endswith("photos/p1.jpg")


def test_create_yelp_tables_contains_flat_business_columns() -> None:
    connection = FakeConnection()

    create_yelp_tables(connection, database="heterqa_yelp", reset_tables=True)
    sql = "\n".join(statement for statement, _ in connection.statements)

    assert "CREATE DATABASE IF NOT EXISTS `heterqa_yelp`" in sql
    assert "`business_accepts_credit_cards` TINYINT" in sql
    assert "`restaurants_price_range2` VARCHAR(32)" in sql
    assert "CREATE TABLE IF NOT EXISTS `photo`" in sql
    assert connection.commits == 1


def test_resolve_and_extract_yelp_files(tmp_path: Path) -> None:
    archive = tmp_path / "Yelp-JSON.zip"
    with zipfile.ZipFile(archive, "w") as zip_file:
        zip_file.writestr("Yelp JSON/yelp_academic_dataset_business.json", "{}\n")
        zip_file.writestr("Yelp JSON/yelp_academic_dataset_review.json", "{}\n")
    extracted = extract_archive(archive, tmp_path / "extracted")
    files = resolve_yelp_files(extracted)

    assert files["business"].name == "yelp_academic_dataset_business.json"
    assert files["review"].name == "yelp_academic_dataset_review.json"


def test_extract_yelp_json_zip_expands_nested_tar(tmp_path: Path) -> None:
    tar_path = tmp_path / "yelp_dataset.tar"
    business = tmp_path / "yelp_academic_dataset_business.json"
    review = tmp_path / "yelp_academic_dataset_review.json"
    business.write_text("{}\n", encoding="utf-8")
    review.write_text("{}\n", encoding="utf-8")
    with tarfile.open(tar_path, "w") as archive:
        archive.add(business, arcname="yelp_academic_dataset_business.json")
        archive.add(review, arcname="yelp_academic_dataset_review.json")

    zip_path = tmp_path / "Yelp-JSON.zip"
    with zipfile.ZipFile(zip_path, "w") as zip_file:
        zip_file.write(tar_path, arcname="Yelp JSON/yelp_dataset.tar")

    extracted = extract_archive(zip_path, tmp_path / "extracted")
    files = resolve_yelp_files(extracted)

    assert files["business"].read_text(encoding="utf-8") == "{}\n"
    assert files["review"].read_text(encoding="utf-8") == "{}\n"


def test_extract_yelp_photos_zip_expands_nested_tar_and_resolves_photo_dir(tmp_path: Path) -> None:
    photo_dir = tmp_path / "photos"
    photo_dir.mkdir()
    (photo_dir / "p1.jpg").write_bytes(b"jpeg")
    photos_json = tmp_path / "photos.json"
    photos_json.write_text('{"photo_id":"p1","business_id":"b1"}\n', encoding="utf-8")
    tar_path = tmp_path / "yelp_photos.tar"
    with tarfile.open(tar_path, "w") as archive:
        archive.add(photo_dir, arcname="photos")
        archive.add(photos_json, arcname="photos.json")

    zip_path = tmp_path / "Yelp-Photos.zip"
    with zipfile.ZipFile(zip_path, "w") as zip_file:
        zip_file.write(tar_path, arcname="Yelp Photos/yelp_photos.tar")

    extracted = extract_archive(zip_path, tmp_path / "extracted")
    files = resolve_yelp_files(extracted)

    assert files["photo"].name == "photos.json"
    assert resolve_yelp_photo_dir(extracted) == extracted / "Yelp Photos" / "photos"


def test_resolver_accepts_canonical_symlinked_roots(tmp_path: Path) -> None:
    source_json = tmp_path / "source-json"
    source_json.mkdir()
    source_photos = tmp_path / "source-photos"
    source_photos.mkdir()
    json_root = source_json / "Yelp JSON"
    json_root.mkdir()
    photo_root = source_photos / "Yelp Photos"
    photo_root.mkdir()
    _write_jsonl(json_root / "yelp_academic_dataset_business.json", [{"business_id": "b1"}])
    photo_dir = photo_root / "photos"
    photo_dir.mkdir(parents=True)
    (photo_dir / "p1.jpg").write_bytes(b"jpeg")
    _write_jsonl(photo_root / "photos.json", [{"photo_id": "p1", "business_id": "b1"}])

    root = tmp_path / "root" / "extracted"
    root.mkdir(parents=True)
    (root / "json").symlink_to(source_json, target_is_directory=True)
    (root / "photos").symlink_to(source_photos, target_is_directory=True)

    files = resolve_yelp_files(root)

    assert files["business"].name == "yelp_academic_dataset_business.json"
    assert files["photo"].name == "photos.json"
    assert resolve_yelp_photo_dir(root) == root / "photos" / "Yelp Photos" / "photos"


def test_download_requires_terms(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="accept_terms"):
        download_yelp_open_dataset(tmp_path, accept_terms=False)


def test_oceanbase_config_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HETERQA_DB_HOST", "127.0.0.1")
    monkeypatch.setenv("HETERQA_DB_PORT", "2881")
    monkeypatch.setenv("HETERQA_DB_USER", "root")
    monkeypatch.setenv("HETERQA_DB_PASSWORD", "pw")
    monkeypatch.setenv("HETERQA_DB_NAME", "heterqa")
    args = argparse.Namespace(host=None, port=None, user=None, password=None, database=None)

    config = oceanbase_config_from_args(args)

    assert config.host == "127.0.0.1"
    assert config.port == 2881
    assert config.database == "heterqa"
