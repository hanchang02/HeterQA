from __future__ import annotations

import pytest

from heterqa.construction.candidate_generation import seed_candidates
from heterqa.construction.contracts import BusinessRecord, GenerationCase


class SeedProvider:
    def __init__(self) -> None:
        self.records = {
            "b1": BusinessRecord.from_raw({"business_id": "b1", "name": "Seed One"}),
            "b2": BusinessRecord.from_raw({"business_id": "b2", "name": "Seed Two"}),
        }

    def get_business(self, business_id: str) -> BusinessRecord | None:
        return self.records.get(business_id)

    def iter_businesses(self) -> list[BusinessRecord]:
        raise AssertionError("seed_candidates must not use the full corpus as initial seeds")


def test_seed_candidates_uses_only_explicit_seed_ids() -> None:
    candidates = seed_candidates(GenerationCase(qid="q1", subset="Text", seed_business_ids=["b1"]), SeedProvider())  # type: ignore[arg-type]

    assert list(candidates) == ["b1"]
    assert candidates["b1"].origin == "initial_seed"


def test_seed_candidates_raises_without_structured_selection_seed() -> None:
    with pytest.raises(ValueError, match="seed_records or seed_business_ids"):
        seed_candidates(GenerationCase(qid="q1", subset="Text"), SeedProvider())  # type: ignore[arg-type]
