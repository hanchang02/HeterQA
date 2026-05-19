"""Photo evidence provider boundary."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PhotoEvidenceLocator:
    business_id: str
    photo_id: str
    support_summary: str

