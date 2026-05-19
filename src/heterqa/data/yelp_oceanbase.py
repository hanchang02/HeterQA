"""Download Yelp Open Dataset archives and load them into OceanBase/MySQL.

The downloader only fetches archives from Yelp's public Open Dataset page after
the caller explicitly confirms that they accept Yelp's dataset terms. The loader
streams JSON records into a local OceanBase-compatible MySQL database and
creates the flattened business columns expected by HeterQA construction.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tarfile
import urllib.request
import zipfile
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm

from heterqa.construction.providers import _flatten_yelp_business
from heterqa.construction.structured_selection import BOOL_RULES, ENUM_RULES, FIELD_WEIGHTS, NUMERIC_RULES


YELP_OPEN_DATASET_PAGE = "https://business.yelp.com/data/resources/open-dataset/"
YELP_JSON_URL = "https://business.yelp.com/external-assets/files/Yelp-JSON.zip"
YELP_PHOTOS_URL = "https://business.yelp.com/external-assets/files/Yelp-Photos.zip"

DATASET_FILES = {
    "business": "yelp_academic_dataset_business.json",
    "review": "yelp_academic_dataset_review.json",
    "user": "yelp_academic_dataset_user.json",
    "checkin": "yelp_academic_dataset_checkin.json",
    "tip": "yelp_academic_dataset_tip.json",
    "photo": "photos.json",
}

YELP_JSON_ROOT_NAME = "Yelp JSON"
YELP_PHOTOS_ROOT_NAME = "Yelp Photos"
YELP_NESTED_ARCHIVES = (
    Path(YELP_JSON_ROOT_NAME) / "yelp_dataset.tar",
    Path(YELP_PHOTOS_ROOT_NAME) / "yelp_photos.tar",
    Path("yelp_dataset.tar"),
    Path("yelp_photos.tar"),
)

BASE_BUSINESS_COLUMNS: list[tuple[str, str]] = [
    ("business_id", "VARCHAR(128) NOT NULL"),
    ("name", "TEXT"),
    ("address", "TEXT"),
    ("city", "VARCHAR(255)"),
    ("state", "VARCHAR(64)"),
    ("postal_code", "VARCHAR(32)"),
    ("latitude", "DOUBLE"),
    ("longitude", "DOUBLE"),
    ("stars", "DOUBLE"),
    ("review_count", "INT"),
    ("is_open", "TINYINT"),
    ("categories", "TEXT"),
    ("attributes_json", "LONGTEXT"),
    ("hours_json", "LONGTEXT"),
    ("raw_json", "LONGTEXT"),
]


@dataclass(frozen=True)
class OceanBaseConfig:
    host: str
    port: int
    user: str
    password: str
    database: str
    charset: str = "utf8mb4"


@dataclass
class LoadSummary:
    database: str
    tables: dict[str, int]
    source_files: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "database": self.database,
            "tables": dict(self.tables),
            "source_files": dict(self.source_files),
        }


class _DownloadProgress:
    def __init__(self, path: Path):
        self.path = path
        self.bar: tqdm | None = None

    def __call__(self, block_count: int, block_size: int, total_size: int) -> None:
        if self.bar is None:
            self.bar = tqdm(
                total=total_size if total_size > 0 else None,
                unit="B",
                unit_scale=True,
                desc=self.path.name,
            )
        downloaded = block_count * block_size
        delta = downloaded - int(self.bar.n)
        if delta > 0:
            self.bar.update(delta)
        if total_size > 0 and downloaded >= total_size:
            self.bar.close()


def download_yelp_open_dataset(
    output_dir: Path,
    *,
    include_json: bool = True,
    include_photos: bool = False,
    json_url: str = YELP_JSON_URL,
    photos_url: str = YELP_PHOTOS_URL,
    accept_terms: bool = False,
    extract: bool = True,
    force: bool = False,
) -> dict[str, str]:
    """Download Yelp Open Dataset archives and optionally extract them."""

    if not accept_terms:
        raise ValueError(
            "Refusing to download Yelp Open Dataset without accept_terms=True. "
            f"Review Yelp's terms from {YELP_OPEN_DATASET_PAGE} first."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_dir = output_dir / "archives"
    extract_dir = output_dir / "extracted"
    archive_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, str] = {"source_page": YELP_OPEN_DATASET_PAGE}
    jobs = []
    if include_json:
        jobs.append(("json", json_url, archive_dir / Path(json_url).name))
    if include_photos:
        jobs.append(("photos", photos_url, archive_dir / Path(photos_url).name))

    for label, url, archive_path in jobs:
        if force or not archive_path.exists():
            request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(request) as response, archive_path.open("wb") as handle:
                total = int(response.headers.get("Content-Length", "0") or 0)
                with tqdm(total=total if total > 0 else None, unit="B", unit_scale=True, desc=archive_path.name) as bar:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        handle.write(chunk)
                        bar.update(len(chunk))
        manifest[f"{label}_archive"] = str(archive_path)
        if extract:
            target = extract_dir / label
            extract_archive(archive_path, target, force=force)
            manifest[f"{label}_extracted"] = str(target)

    (output_dir / "download_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _safe_extract_zip(archive: zipfile.ZipFile, output_dir: Path) -> None:
    for member in archive.infolist():
        target = output_dir / member.filename
        if not _is_relative_to(target, output_dir):
            raise ValueError(f"Refusing unsafe zip member path: {member.filename}")
    archive.extractall(output_dir)


def _safe_extract_tar(archive: tarfile.TarFile, output_dir: Path) -> None:
    for member in archive.getmembers():
        target = output_dir / member.name
        if not _is_relative_to(target, output_dir):
            raise ValueError(f"Refusing unsafe tar member path: {member.name}")
    try:
        archive.extractall(output_dir, filter="data")
    except TypeError:  # pragma: no cover - Python < 3.12 compatibility
        archive.extractall(output_dir)


def _looks_like_archive(path: Path) -> bool:
    lowered = path.name.lower()
    return lowered.endswith((".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz"))


def _extract_nested_archives(output_dir: Path) -> None:
    """Extract archives nested inside Yelp's downloaded archives.

    Yelp publishes a zip whose payload is a named root directory containing one
    tar archive.  Keep this deterministic instead of scanning arbitrary nested
    archives from user-provided directories.
    """

    for relative in YELP_NESTED_ARCHIVES:
        nested = output_dir / relative
        if not nested.is_file():
            continue
        target_dir = nested.parent
        if zipfile.is_zipfile(nested):
            with zipfile.ZipFile(nested) as archive:
                _safe_extract_zip(archive, target_dir)
        elif tarfile.is_tarfile(nested):
            with tarfile.open(nested) as archive:
                _safe_extract_tar(archive, target_dir)


def extract_archive(archive_path: Path, output_dir: Path, *, force: bool = False, recursive: bool = True) -> Path:
    """Extract zip/tar archives used by Yelp Open Dataset."""

    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        if recursive:
            _extract_nested_archives(output_dir)
        return output_dir
    if output_dir.exists() and force:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path) as archive:
            _safe_extract_zip(archive, output_dir)
    elif tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path) as archive:
            _safe_extract_tar(archive, output_dir)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path}")
    if recursive:
        _extract_nested_archives(output_dir)
    return output_dir


def _load_download_manifest(input_dir: Path) -> dict[str, Any]:
    for candidate in [input_dir / "download_manifest.json", input_dir.parent / "download_manifest.json", input_dir.parent.parent / "download_manifest.json"]:
        if candidate.exists():
            return json.loads(candidate.read_text(encoding="utf-8"))
    return {}


def _ordered_yelp_source_roots(input_dir: Path) -> list[Path]:
    manifest = _load_download_manifest(input_dir)
    roots: list[Path] = []
    if manifest.get("json_extracted"):
        roots.append(Path(manifest["json_extracted"]) / YELP_JSON_ROOT_NAME)
    if manifest.get("photos_extracted"):
        roots.append(Path(manifest["photos_extracted"]) / YELP_PHOTOS_ROOT_NAME)
    roots.extend(
        [
            input_dir / "extracted" / "json" / YELP_JSON_ROOT_NAME,
            input_dir / "extracted" / "photos" / YELP_PHOTOS_ROOT_NAME,
            input_dir / "json" / YELP_JSON_ROOT_NAME,
            input_dir / "photos" / YELP_PHOTOS_ROOT_NAME,
            input_dir / YELP_JSON_ROOT_NAME,
            input_dir / YELP_PHOTOS_ROOT_NAME,
            input_dir,
        ]
    )
    deduped: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        resolved = root.resolve() if root.exists() else root.absolute()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(root)
    return deduped


def resolve_yelp_files(input_dir: Path) -> dict[str, Path]:
    """Resolve Yelp files from the documented downloader/extractor layouts."""

    files: dict[str, Path] = {}
    for root in _ordered_yelp_source_roots(input_dir):
        for key, filename in DATASET_FILES.items():
            path = root / filename
            if key not in files and path.is_file():
                files[key] = path
    return files


def resolve_yelp_photo_dir(input_dir: Path) -> Path | None:
    """Resolve the official extracted Yelp photo image directory when present."""

    for root in _ordered_yelp_source_roots(input_dir):
        candidate = root / "photos"
        if candidate.is_dir():
            return candidate
    return None


def connect_oceanbase(config: OceanBaseConfig, *, database: str | None = None):
    try:
        import pymysql  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Install database dependencies with `pip install -e '.[db]'`.") from exc
    return pymysql.connect(
        host=config.host,
        port=config.port,
        user=config.user,
        password=config.password,
        database=database,
        charset=config.charset,
        autocommit=False,
        cursorclass=pymysql.cursors.DictCursor,
    )


def _business_columns() -> list[tuple[str, str]]:
    seen = {name for name, _ in BASE_BUSINESS_COLUMNS}
    columns = list(BASE_BUSINESS_COLUMNS)
    for field_name in sorted(FIELD_WEIGHTS):
        if field_name in seen:
            continue
        seen.add(field_name)
        if field_name in BOOL_RULES:
            sql_type = "TINYINT"
        elif field_name == "restaurants_price_range2":
            sql_type = "VARCHAR(32)"
        elif field_name in NUMERIC_RULES:
            sql_type = "DOUBLE"
        elif field_name in ENUM_RULES:
            sql_type = "VARCHAR(128)"
        else:
            sql_type = "LONGTEXT"
        columns.append((field_name, sql_type))
    return columns


def create_yelp_tables(connection, *, database: str, reset_tables: bool = False) -> None:
    """Create HeterQA-compatible Yelp tables in OceanBase/MySQL."""

    with connection.cursor() as cursor:
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{database}` DEFAULT CHARACTER SET utf8mb4")
        cursor.execute(f"USE `{database}`")
        if reset_tables:
            for table in ["photo", "tip", "checkin", "review", "user", "business"]:
                cursor.execute(f"DROP TABLE IF EXISTS `{table}`")
        business_cols = ",\n  ".join(f"`{name}` {sql_type}" for name, sql_type in _business_columns())
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS `business` (
              {business_cols},
              PRIMARY KEY (`business_id`),
              KEY `idx_business_state_city` (`state`, `city`),
              KEY `idx_business_stars` (`stars`)
            ) DEFAULT CHARSET=utf8mb4
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS `review` (
              `review_id` VARCHAR(128) NOT NULL,
              `user_id` VARCHAR(128),
              `business_id` VARCHAR(128),
              `stars` DOUBLE,
              `useful` INT,
              `funny` INT,
              `cool` INT,
              `date` VARCHAR(64),
              `text` LONGTEXT,
              `raw_json` LONGTEXT,
              PRIMARY KEY (`review_id`),
              KEY `idx_review_business` (`business_id`),
              KEY `idx_review_user` (`user_id`)
            ) DEFAULT CHARSET=utf8mb4
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS `user` (
              `user_id` VARCHAR(128) NOT NULL,
              `name` TEXT,
              `review_count` INT,
              `yelping_since` VARCHAR(64),
              `useful` INT,
              `funny` INT,
              `cool` INT,
              `fans` INT,
              `average_stars` DOUBLE,
              `raw_json` LONGTEXT,
              PRIMARY KEY (`user_id`)
            ) DEFAULT CHARSET=utf8mb4
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS `checkin` (
              `business_id` VARCHAR(128) NOT NULL,
              `date` LONGTEXT,
              `raw_json` LONGTEXT,
              PRIMARY KEY (`business_id`)
            ) DEFAULT CHARSET=utf8mb4
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS `tip` (
              `id` BIGINT NOT NULL AUTO_INCREMENT,
              `user_id` VARCHAR(128),
              `business_id` VARCHAR(128),
              `text` LONGTEXT,
              `date` VARCHAR(64),
              `compliment_count` INT,
              `raw_json` LONGTEXT,
              PRIMARY KEY (`id`),
              KEY `idx_tip_business` (`business_id`),
              KEY `idx_tip_user` (`user_id`)
            ) DEFAULT CHARSET=utf8mb4
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS `photo` (
              `photo_id` VARCHAR(128) NOT NULL,
              `business_id` VARCHAR(128),
              `caption` TEXT,
              `label` VARCHAR(64),
              `photo_path` TEXT,
              `raw_json` LONGTEXT,
              PRIMARY KEY (`photo_id`),
              KEY `idx_photo_business` (`business_id`)
            ) DEFAULT CHARSET=utf8mb4
            """
        )
    connection.commit()


def iter_json_records(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if isinstance(payload, dict):
                yield payload


def _db_value(value: Any) -> Any:
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def _insert_many(connection, table: str, columns: Sequence[str], rows: list[Sequence[Any]]) -> None:
    if not rows:
        return
    value_markers = ", ".join(["%s"] * len(columns))
    column_sql = ", ".join(f"`{column}`" for column in columns)
    updates = ", ".join(f"`{column}`=VALUES(`{column}`)" for column in columns if column != "id")
    sql = f"INSERT INTO `{table}` ({column_sql}) VALUES ({value_markers}) ON DUPLICATE KEY UPDATE {updates}"
    with connection.cursor() as cursor:
        cursor.executemany(sql, rows)
    connection.commit()


def _load_stream(
    connection,
    table: str,
    columns: list[str],
    records: Iterable[Sequence[Any]],
    *,
    batch_size: int,
    desc: str,
) -> int:
    count = 0
    batch: list[Sequence[Any]] = []
    for row in tqdm(records, desc=desc, unit="row"):
        batch.append(row)
        if len(batch) >= batch_size:
            _insert_many(connection, table, columns, batch)
            count += len(batch)
            batch = []
    if batch:
        _insert_many(connection, table, columns, batch)
        count += len(batch)
    return count


def _limit(records: Iterable[dict[str, Any]], max_rows: int | None) -> Iterator[dict[str, Any]]:
    for index, row in enumerate(records):
        if max_rows is not None and index >= max_rows:
            return
        yield row


def _business_rows(path: Path, max_rows: int | None) -> tuple[list[str], Iterator[list[Any]]]:
    columns = [name for name, _ in _business_columns()]

    def rows() -> Iterator[list[Any]]:
        for raw in _limit(iter_json_records(path), max_rows):
            flattened = _flatten_yelp_business(raw)
            flattened["raw_json"] = json.dumps(raw, ensure_ascii=False, sort_keys=True)
            yield [_db_value(flattened.get(column)) for column in columns]

    return columns, rows()


def _review_rows(path: Path, max_rows: int | None) -> tuple[list[str], Iterator[list[Any]]]:
    columns = ["review_id", "user_id", "business_id", "stars", "useful", "funny", "cool", "date", "text", "raw_json"]

    def rows() -> Iterator[list[Any]]:
        for raw in _limit(iter_json_records(path), max_rows):
            yield [
                raw.get("review_id"),
                raw.get("user_id"),
                raw.get("business_id"),
                raw.get("stars"),
                raw.get("useful"),
                raw.get("funny"),
                raw.get("cool"),
                raw.get("date"),
                raw.get("text"),
                json.dumps(raw, ensure_ascii=False, sort_keys=True),
            ]

    return columns, rows()


def _user_rows(path: Path, max_rows: int | None) -> tuple[list[str], Iterator[list[Any]]]:
    columns = ["user_id", "name", "review_count", "yelping_since", "useful", "funny", "cool", "fans", "average_stars", "raw_json"]

    def rows() -> Iterator[list[Any]]:
        for raw in _limit(iter_json_records(path), max_rows):
            yield [
                raw.get("user_id"),
                raw.get("name"),
                raw.get("review_count"),
                raw.get("yelping_since"),
                raw.get("useful"),
                raw.get("funny"),
                raw.get("cool"),
                raw.get("fans"),
                raw.get("average_stars"),
                json.dumps(raw, ensure_ascii=False, sort_keys=True),
            ]

    return columns, rows()


def _checkin_rows(path: Path, max_rows: int | None) -> tuple[list[str], Iterator[list[Any]]]:
    columns = ["business_id", "date", "raw_json"]

    def rows() -> Iterator[list[Any]]:
        for raw in _limit(iter_json_records(path), max_rows):
            yield [raw.get("business_id"), raw.get("date"), json.dumps(raw, ensure_ascii=False, sort_keys=True)]

    return columns, rows()


def _tip_rows(path: Path, max_rows: int | None) -> tuple[list[str], Iterator[list[Any]]]:
    columns = ["user_id", "business_id", "text", "date", "compliment_count", "raw_json"]

    def rows() -> Iterator[list[Any]]:
        for raw in _limit(iter_json_records(path), max_rows):
            yield [
                raw.get("user_id"),
                raw.get("business_id"),
                raw.get("text"),
                raw.get("date"),
                raw.get("compliment_count"),
                json.dumps(raw, ensure_ascii=False, sort_keys=True),
            ]

    return columns, rows()


def _photo_rows(path: Path, max_rows: int | None, *, photo_dir: Path | None) -> tuple[list[str], Iterator[list[Any]]]:
    columns = ["photo_id", "business_id", "caption", "label", "photo_path", "raw_json"]

    def rows() -> Iterator[list[Any]]:
        for raw in _limit(iter_json_records(path), max_rows):
            photo_id = str(raw.get("photo_id") or raw.get("id") or "")
            photo_path = str(photo_dir / f"{photo_id}.jpg") if photo_dir is not None and photo_id else ""
            yield [
                photo_id,
                raw.get("business_id"),
                raw.get("caption"),
                raw.get("label"),
                photo_path,
                json.dumps(raw, ensure_ascii=False, sort_keys=True),
            ]

    return columns, rows()


def load_yelp_to_oceanbase(
    input_dir: Path,
    config: OceanBaseConfig,
    *,
    tables: Sequence[str] = ("business", "review", "user", "checkin", "tip", "photo"),
    create_database: bool = True,
    reset_tables: bool = False,
    batch_size: int = 1000,
    max_rows_per_table: int | None = None,
    photo_dir: Path | None = None,
) -> LoadSummary:
    """Stream Yelp Open Dataset JSON files into OceanBase/MySQL tables."""

    _extract_nested_archives(input_dir)
    files = resolve_yelp_files(input_dir)
    requested = {table.strip().lower() for table in tables}
    if "business" in requested and "business" not in files:
        raise FileNotFoundError(f"Could not find {DATASET_FILES['business']} under {input_dir}")
    missing_requested = sorted(table for table in requested if table in DATASET_FILES and table not in files)
    if len(missing_requested) == len([table for table in requested if table in DATASET_FILES]):
        raise FileNotFoundError(f"Could not find requested Yelp files {missing_requested} under {input_dir}")
    if photo_dir is None and "photo" in files:
        photo_dir = resolve_yelp_photo_dir(input_dir)
    connection = connect_oceanbase(config, database=None if create_database else config.database)
    try:
        if create_database:
            create_yelp_tables(connection, database=config.database, reset_tables=reset_tables)
        else:
            with connection.cursor() as cursor:
                cursor.execute(f"USE `{config.database}`")
            if reset_tables:
                create_yelp_tables(connection, database=config.database, reset_tables=True)
        table_counts: dict[str, int] = {}
        loaders = {
            "business": lambda path: _business_rows(path, max_rows_per_table),
            "review": lambda path: _review_rows(path, max_rows_per_table),
            "user": lambda path: _user_rows(path, max_rows_per_table),
            "checkin": lambda path: _checkin_rows(path, max_rows_per_table),
            "tip": lambda path: _tip_rows(path, max_rows_per_table),
            "photo": lambda path: _photo_rows(path, max_rows_per_table, photo_dir=photo_dir),
        }
        for table in ["business", "review", "user", "checkin", "tip", "photo"]:
            if table not in requested or table not in files:
                continue
            columns, rows = loaders[table](files[table])
            table_counts[table] = _load_stream(connection, table, columns, rows, batch_size=batch_size, desc=f"load {table}")
        return LoadSummary(
            database=config.database,
            tables=table_counts,
            source_files={name: str(path) for name, path in files.items()},
        )
    finally:
        connection.close()


def _env_or(value: str | None, env_name: str, default: str | None = None) -> str:
    resolved = value if value is not None else os.environ.get(env_name, default)
    if resolved is None or resolved == "":
        raise ValueError(f"Missing required value. Provide CLI argument or set {env_name}.")
    return resolved


def oceanbase_config_from_args(args: argparse.Namespace) -> OceanBaseConfig:
    return OceanBaseConfig(
        host=_env_or(args.host, "HETERQA_DB_HOST"),
        port=int(_env_or(str(args.port) if args.port is not None else None, "HETERQA_DB_PORT", "2881")),
        user=_env_or(args.user, "HETERQA_DB_USER"),
        password=_env_or(args.password, "HETERQA_DB_PASSWORD", ""),
        database=_env_or(args.database, "HETERQA_DB_NAME"),
    )


def add_data_cli(subparsers: argparse._SubParsersAction) -> None:
    data_parser = subparsers.add_parser("data")
    data_sub = data_parser.add_subparsers(dest="data_command", required=True)

    p_download = data_sub.add_parser("download-yelp")
    p_download.add_argument("--output-dir", required=True, type=Path)
    p_download.add_argument("--include-photos", action="store_true")
    p_download.add_argument("--json-url", default=YELP_JSON_URL)
    p_download.add_argument("--photos-url", default=YELP_PHOTOS_URL)
    p_download.add_argument("--no-extract", action="store_true")
    p_download.add_argument("--force", action="store_true")
    p_download.add_argument("--accept-yelp-terms", action="store_true")

    p_load = data_sub.add_parser("load-yelp-oceanbase")
    p_load.add_argument("--input-dir", required=True, type=Path)
    p_load.add_argument("--host")
    p_load.add_argument("--port", type=int)
    p_load.add_argument("--user")
    p_load.add_argument("--password")
    p_load.add_argument("--database")
    p_load.add_argument("--tables", default="business,review,user,checkin,tip,photo")
    p_load.add_argument("--photo-dir", type=Path)
    p_load.add_argument("--batch-size", type=int, default=1000)
    p_load.add_argument("--max-rows-per-table", type=int)
    p_load.set_defaults(create_database=True)
    p_load.add_argument("--no-create-database", dest="create_database", action="store_false")
    p_load.add_argument("--reset-tables", action="store_true")

    p_prepare = data_sub.add_parser("prepare-yelp-oceanbase")
    p_prepare.add_argument("--work-dir", required=True, type=Path)
    p_prepare.add_argument("--include-photos", action="store_true")
    p_prepare.add_argument("--json-url", default=YELP_JSON_URL)
    p_prepare.add_argument("--photos-url", default=YELP_PHOTOS_URL)
    p_prepare.add_argument("--accept-yelp-terms", action="store_true")
    p_prepare.add_argument("--host")
    p_prepare.add_argument("--port", type=int)
    p_prepare.add_argument("--user")
    p_prepare.add_argument("--password")
    p_prepare.add_argument("--database")
    p_prepare.add_argument("--tables", default="business,review,user,checkin,tip,photo")
    p_prepare.add_argument("--batch-size", type=int, default=1000)
    p_prepare.add_argument("--max-rows-per-table", type=int)
    p_prepare.add_argument("--reset-tables", action="store_true")


def run_data_cli(args: argparse.Namespace) -> dict[str, Any]:
    if args.data_command == "download-yelp":
        return {
            "download": download_yelp_open_dataset(
                args.output_dir,
                include_json=True,
                include_photos=args.include_photos,
                json_url=args.json_url,
                photos_url=args.photos_url,
                accept_terms=args.accept_yelp_terms,
                extract=not args.no_extract,
                force=args.force,
            )
        }
    if args.data_command == "load-yelp-oceanbase":
        summary = load_yelp_to_oceanbase(
            args.input_dir,
            oceanbase_config_from_args(args),
            tables=[item.strip() for item in args.tables.split(",") if item.strip()],
            create_database=args.create_database,
            reset_tables=args.reset_tables,
            batch_size=args.batch_size,
            max_rows_per_table=args.max_rows_per_table,
            photo_dir=args.photo_dir,
        )
        return {"load": summary.to_dict()}
    if args.data_command == "prepare-yelp-oceanbase":
        manifest = download_yelp_open_dataset(
            args.work_dir,
            include_json=True,
            include_photos=args.include_photos,
            json_url=args.json_url,
            photos_url=args.photos_url,
            accept_terms=args.accept_yelp_terms,
            extract=True,
            force=False,
        )
        extracted_roots = [Path(value) for key, value in manifest.items() if key.endswith("_extracted")]
        input_dir = args.work_dir / "extracted"
        if not input_dir.exists():
            input_dir = extracted_roots[0] if extracted_roots else args.work_dir
        summary = load_yelp_to_oceanbase(
            input_dir,
            oceanbase_config_from_args(args),
            tables=[item.strip() for item in args.tables.split(",") if item.strip()],
            create_database=True,
            reset_tables=args.reset_tables,
            batch_size=args.batch_size,
            max_rows_per_table=args.max_rows_per_table,
        )
        return {"download": manifest, "load": summary.to_dict()}
    raise AssertionError("unknown data command")
