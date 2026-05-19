"""Review-text evidence provider boundary."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReviewEvidenceLocator:
    business_id: str
    review_id: str
    support_summary: str

