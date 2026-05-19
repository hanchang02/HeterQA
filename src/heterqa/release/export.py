"""Release export helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path

from heterqa.core.io import copy_release_tree
from heterqa.release.extract import extract_release
from heterqa.release.validate import validate_release


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_hf_release(
    input_dir: Path,
    output_dir: Path,
    validate: bool = True,
) -> dict[str, str]:
    """Export a public release tree and optionally validate it."""
    if (input_dir / "data" / "queries.jsonl").exists():
        copy_release_tree(input_dir, output_dir)
    else:
        extract_release(input_dir, output_dir)
    files = [
        output_dir / "data" / "queries.jsonl",
        output_dir / "data" / "answers.jsonl",
        output_dir / "data" / "evidence.jsonl",
        output_dir / "data" / "qrels" / "test.tsv",
        output_dir / "data" / "source_manifest.json",
    ]
    hashes = {str(path.relative_to(output_dir)): sha256_file(path) for path in files if path.exists()}
    if validate:
        report = validate_release(output_dir)
        if not report.ok:
            raise ValueError(report.to_markdown())
    return hashes
