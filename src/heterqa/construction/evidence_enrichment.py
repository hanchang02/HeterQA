"""Evidence-family enrichment helpers."""

from __future__ import annotations

from heterqa.construction.contracts import CandidateState, EvidenceFamily
from heterqa.construction.providers import ConstructionDataProvider


def attach_family_evidence(
    provider: ConstructionDataProvider,
    candidate: CandidateState,
    family: EvidenceFamily,
    query: str,
    limit: int,
) -> None:
    for evidence in provider.get_evidence(family, candidate.business_id, query, limit):
        candidate.add_evidence(evidence)

