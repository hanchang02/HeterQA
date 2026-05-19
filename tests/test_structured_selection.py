from __future__ import annotations

import json
from pathlib import Path

import yaml

from heterqa.construction.contracts import BusinessRecord, StructuredFilter
from heterqa.construction.mainline import run_mainline_from_config
from heterqa.construction.providers import FileBackedConstructionProvider
from heterqa.construction.structured_selection import (
    StructuredQuerySelector,
    StructuredSelectionSettings,
)
from heterqa.construction.record_fields import _ask_json


def _provider() -> FileBackedConstructionProvider:
    records = [
        BusinessRecord.from_raw(
            {
                "business_id": "b1",
                "name": "Cafe One",
                "categories": "Cafe, Coffee & Tea",
                "city": "Philadelphia",
                "state": "PA",
                "stars": 4.5,
                "is_open": 1,
                "outdoor_seating": 1,
            }
        ),
        BusinessRecord.from_raw(
            {
                "business_id": "b2",
                "name": "Cafe Two",
                "categories": "Cafe, Coffee & Tea",
                "city": "Philadelphia",
                "state": "PA",
                "stars": 4.0,
                "is_open": 1,
                "outdoor_seating": 1,
            }
        ),
        BusinessRecord.from_raw(
            {
                "business_id": "b3",
                "name": "Bar One",
                "categories": "Bars",
                "city": "Philadelphia",
                "state": "PA",
                "stars": 3.0,
                "is_open": 1,
                "outdoor_seating": 0,
            }
        ),
    ]
    return FileBackedConstructionProvider(records, {})


def test_sql_filter_parser_handles_parameter_markers() -> None:
    city = StructuredFilter.from_raw(("city = %s", "Philadelphia", False))
    stars = StructuredFilter.from_raw(("stars >= 4.0", 4.0, True))
    category = StructuredFilter.from_raw(("categories LIKE %s", "%Cafe%", False))

    assert city.field == "city"
    assert city.operator == "="
    assert city.value == "Philadelphia"
    assert stars.operator == ">="
    assert stars.value == 4.0
    assert category.operator == "LIKE"
    assert category.value == "Cafe"


def test_structured_selector_generates_category_seed_records() -> None:
    selector = StructuredQuerySelector(
        _provider(),
        StructuredSelectionSettings(compact=True, seeded=True, field_selection_mode="never"),
    )

    selected = selector.select_fields(seed=7)

    assert selected.filters
    assert selected.filters[0].field == "categories"
    assert selected.source_filters[0][2] == "categories"
    assert selected.seed_records
    terms = selected.filters[0].value
    for row in selected.seed_records:
        assert all(term.lower() in row["categories"].lower() for term in terms)


def test_mainline_can_generate_case_from_structured_selection(tmp_path: Path) -> None:
    business_path = tmp_path / "business.jsonl"
    business_path.write_text(
        "\n".join(
            [
                json.dumps(record.fields)
                for record in _provider().iter_businesses()
                if "Cafe" in record.fields["categories"]
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "mode": "mainline",
                "data": {"business_records_jsonl": str(business_path)},
                "settings": {
                    "enabled_geo": False,
                    "enabled_text": False,
                    "enabled_image": False,
                    "enabled_kg": False,
                    "structured_selection_seeded": True,
                    "field_selection_mode": "never",
                },
                "case_generation": {"count": 1, "start_seed": 11, "subset": "Record_Field"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    output = run_mainline_from_config(config_path, tmp_path / "out")
    row = json.loads(output.read_text(encoding="utf-8").splitlines()[0])

    assert row["case"]["structured_filters"][0]["field"] == "categories"
    assert row["final_answer_business_ids"] == ["b1", "b2"]


def test_model_json_helper_uses_public_semantic_judge_interface() -> None:
    class SemanticJudge:
        def ask_json(self, *, prompt, temperature=0.0, max_tokens=0, image=None):
            assert prompt == "prompt"
            assert temperature == 0.0
            assert max_tokens == 2000
            assert image is None
            return {"json_text": json.dumps({"status": "PASS", "reason": "ok"})}

    class Bundle:
        semantic_judge = SemanticJudge()

    assert _ask_json(Bundle(), "prompt") == {"status": "PASS", "reason": "ok"}


def test_model_json_helper_uses_direct_public_model_interface() -> None:
    class SemanticJudge:
        def ask_json(self, *, prompt, temperature=0.0, max_tokens=0, image=None):
            assert prompt == "prompt"
            assert temperature == 0.0
            assert max_tokens == 2000
            assert image is None
            return {"json_text": json.dumps({"status": "PASS", "reason": "direct"})}

    class Bundle:
        semantic_judge = SemanticJudge()

    assert _ask_json(Bundle(), "prompt") == {"status": "PASS", "reason": "direct"}


def test_generation_case_preserves_source_recall_query() -> None:
    from heterqa.construction.contracts import GenerationCase

    case = GenerationCase.from_raw(
        {
            "qid": "source-case",
            "subset": "Text",
            "structured_filters": [("city = %s", "Philadelphia", False)],
            "recall_query": "Find Philadelphia places.",
        }
    )

    assert case.recall_query == "Find Philadelphia places."
