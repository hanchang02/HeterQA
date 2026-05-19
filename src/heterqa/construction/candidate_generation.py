"""Candidate initialization helpers for construction.

This module only performs the initial-seed materialization step. Heterogeneous
recall and evidence verification live in `expansion.py`, `text_engine.py`,
`image_engine.py`, `geo_engine.py`, and `kg_engine.py`.
"""

from __future__ import annotations

from heterqa.construction.contracts import BusinessRecord, CandidateState, GenerationCase
from heterqa.construction.providers import ConstructionDataProvider


def seed_candidates(case: GenerationCase, provider: ConstructionDataProvider) -> dict[str, CandidateState]:
    """Create initial-seed candidates for one generation case.

    Old construction logic always started from selected seed businesses. It did
    not treat the entire business corpus as initial seeds. If a caller has not
    provided seed records or seed ids, the case has not completed structured
    selection and must fail fast.
    """

    candidates: dict[str, CandidateState] = {}
    records = [record for row in case.seed_records if (record := _record_from_seed_row(row))]
    for business_id in case.seed_business_ids:
        record = provider.get_business(business_id)
        if record is not None:
            records.append(record)
    if not records:
        raise ValueError("Initial candidate generation requires seed_records or seed_business_ids.")
    for record in records:
        if record.business_id in candidates:
            continue
        candidates[record.business_id] = CandidateState(
            business_id=record.business_id,
            name=record.name,
            origin="initial_seed",
            metadata=record.fields,
        )
    return candidates


def _record_from_seed_row(row: dict[str, object]) -> BusinessRecord:
    return BusinessRecord.from_raw(dict(row))
