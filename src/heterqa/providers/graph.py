"""Graph evidence provider boundary."""

from __future__ import annotations

from dataclasses import dataclass

from heterqa.construction.providers import FeatureGraphStore


@dataclass(frozen=True)
class GraphEvidenceLocator:
    business_id: str
    feature_name: str
    support_summary: str


__all__ = ["FeatureGraphStore", "GraphEvidenceLocator"]
