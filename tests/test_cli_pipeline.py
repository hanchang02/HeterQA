from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    return subprocess.run(
        [sys.executable, "-m", "heterqa.cli", *args],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )


def parse_json(stdout: str) -> dict:
    return json.loads(stdout)


def test_release_validate_cli_fixture() -> None:
    result = run_cli(
        "release",
        "validate",
        "--dataset-dir",
        "tests/fixtures/synthetic_release",
    )

    assert "HeterQA Release Validation: PASS" in result.stdout


def test_generation_pipeline_cli_fixture(tmp_path: Path) -> None:
    construction = tmp_path / "construction"
    contradiction = tmp_path / "contradiction"
    review = tmp_path / "review"
    applied = tmp_path / "review_applied"
    certified = tmp_path / "certified"
    quality = tmp_path / "quality"
    release = tmp_path / "release"

    out = run_cli(
        "construct",
        "run",
        "--config",
        "tests/fixtures/mainline_construction/config.yaml",
        "--output",
        str(construction),
    )
    assert Path(parse_json(out.stdout)["output"]).exists()

    out = run_cli("certify", "contradiction-detect", "--input", str(construction), "--output", str(contradiction))
    assert Path(parse_json(out.stdout)["output"]).exists()
    out = run_cli("review", "export", "--input", str(contradiction), "--output", str(review))
    assert Path(parse_json(out.stdout)["output"]).exists()
    out = run_cli(
        "review",
        "apply",
        "--review-dir",
        str(review),
        "--input",
        str(contradiction),
        "--output",
        str(applied),
    )
    assert Path(parse_json(out.stdout)["output"]).exists()
    out = run_cli("certify", "answer-set", "--input", str(applied), "--output", str(certified))
    assert Path(parse_json(out.stdout)["output"]).exists()
    out = run_cli("quality", "query-metrics", "--input", str(certified), "--output", str(quality))
    assert Path(parse_json(out.stdout)["output"]).exists()
    out = run_cli("release", "export-hf", "--input", str(certified), "--output", str(release))
    assert (release / "data" / "queries.jsonl").exists()
    assert "data/evidence.jsonl" in parse_json(out.stdout)["files"]
