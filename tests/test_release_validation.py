from __future__ import annotations

from pathlib import Path

from heterqa.core.io import load_release_bundle
from heterqa.release.export import export_hf_release
from heterqa.release.validate import validate_release


ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_RELEASE = ROOT / "tests" / "fixtures" / "synthetic_release"


def test_synthetic_release_contract_passes_without_full_count_check() -> None:
    report = validate_release(SYNTHETIC_RELEASE)

    assert report.ok, report.errors
    assert report.counts == {"queries": 2, "answers": 2, "qrels": 2, "evidence": 5}
    assert report.family_counts["record_field"] == 2


def test_qrels_match_answer_business_ids() -> None:
    bundle = load_release_bundle(SYNTHETIC_RELEASE)
    answer_pairs = {
        (answer.qid, business_id)
        for answer in bundle.answers
        for business_id in answer.answer_business_ids
    }
    qrel_pairs = {
        (qid, business_id)
        for qid, business_ids in bundle.qrels.items()
        for business_id in business_ids
    }

    assert answer_pairs == qrel_pairs


def test_release_export_writes_public_tables_and_sanitizes_evidence(tmp_path: Path) -> None:
    final_dir = tmp_path / "final"
    final_dir.mkdir()
    (final_dir / "final_cases.jsonl").write_text(
        (
            '{"qid":"1","query":"Find a cafe.","subset":"' + "AB" + '_Geo_Text",'
            '"candidates":[{"business_id":"b1","name":"Cafe One","verdict":"yes",'
            '"evidence":[{"family":"text","claim_summary":"Friendly service evidence.",'
            '"source_locator_type":"yelp_review_id","source_locator":"review_1",'
            '"verification_method":"text_verify","confidence":0.9,'
            '"details":{"review_text":"raw review should not leave","prompt":"hidden prompt","kept":"short summary"}}]}],'
            '"final_answer_business_ids":["b1"],"final_answer_business_names":["Cafe One"]}'
        )
        + "\n",
        encoding="utf-8",
    )

    output_dir = tmp_path / "release"
    hashes = export_hf_release(final_dir, output_dir, validate=True)
    report = validate_release(output_dir)
    evidence_text = (output_dir / "data" / "evidence.jsonl").read_text(encoding="utf-8")

    assert report.ok, report.errors
    assert "data/queries.jsonl" in hashes
    assert "data/evidence.jsonl" in hashes
    assert "raw review should not leave" not in evidence_text
    assert "hidden prompt" not in evidence_text
    assert '"source_case_category": "Geo_Text"' in (output_dir / "data" / "answers.jsonl").read_text(encoding="utf-8")


def test_release_validate_rejects_non_public_fields_and_leakage(tmp_path: Path) -> None:
    bad = tmp_path / "bad_release"
    data = bad / "data"
    (data / "qrels").mkdir(parents=True)
    (bad / "schemas").mkdir()
    (data / "queries.jsonl").write_text(
        '{"qid":"1","query":"Find a cafe.","subset":"Text_Only","answer_count":1,"split":"test"}\n',
        encoding="utf-8",
    )
    (data / "answers.jsonl").write_text(
        '{"qid":"1","answer_business_ids":["b1"],"answer_business_names":["Cafe"],"answer_count":1,"source_case_category":"Text_Only"}\n',
        encoding="utf-8",
    )
    forbidden_path = "/" + "/".join(["mnt", "data", "local", "file.yaml"])
    (data / "evidence.jsonl").write_text(
        '{"qid":"1","business_id":"b1","family":"record_field","support_status":"passes",'
        f'"claim_summary":"The source was {forbidden_path}",'
        '"source_locator_type":"query_predicates","source_locator":"predicate_count=1",'
        '"verification_method":"structured_field_check","confidence":1.0,"raw_content_released":false,"details":{}}\n',
        encoding="utf-8",
    )
    (data / "qrels" / "test.tsv").write_text("query-id\tcorpus-id\tscore\n1\tb1\t1\n", encoding="utf-8")
    (data / "source_manifest.json").write_text(
        '{"dataset_name":"Bad","raw_yelp_content_included":false,"source_url":"https://business.yelp.com/data/resources/open-dataset/"}\n',
        encoding="utf-8",
    )
    for name in ["query.schema.json", "answer.schema.json", "evidence.schema.json"]:
        (bad / "schemas" / name).write_text("{}\n", encoding="utf-8")

    report = validate_release(bad)

    assert not report.ok
    assert any("non-public fields" in error for error in report.errors)
    assert any("forbidden pattern" in error for error in report.errors)
