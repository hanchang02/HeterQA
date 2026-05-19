from __future__ import annotations

from heterqa.construction.contracts import CandidateState, ConstructionSettings, PipelineContext
from heterqa.construction.kg_engine import KgSearchTask


class FakeGraph:
    def __init__(self) -> None:
        self.features = {
            "b1": {"pos": {"wifi", "patio"}, "neg": set()},
            "b2": {"pos": {"wifi"}, "neg": set()},
        }

    def get_features_of_business(self, business_id: str, sentiment: str = "all") -> set[str]:
        if sentiment == "all":
            return self.features[business_id]["pos"] | self.features[business_id]["neg"]
        return set(self.features[business_id][sentiment])

    def get_businesses_by_feature(self, feature: str, sentiment: str = "all") -> set[str]:
        return {
            business_id
            for business_id, buckets in self.features.items()
            if feature in buckets.get(sentiment, set())
        }

    def sample_features_by_distribution(self, candidate_features: list[str], top_k: int = 1) -> list[str]:
        ordered = []
        for feature in candidate_features:
            if feature not in ordered:
                ordered.append(feature)
        ordered.sort(key=lambda item: {"wifi": 0, "patio": 1}.get(item, 10))
        return ordered[:top_k]


class FakeProvider:
    def __init__(self) -> None:
        self.graph = FakeGraph()

    def get_evidence(self, *_args, **_kwargs):
        return []

    def search_reviews(self, *_args, **_kwargs):
        return []

    def search_photos(self, *_args, **_kwargs):
        return []


def test_kg_attribute_intersection_filters_by_topology(monkeypatch) -> None:
    import heterqa.construction.kg_engine as kg_module

    monkeypatch.setattr(kg_module.random, "choice", lambda seq: seq[0])
    ctx = PipelineContext(task_id="kg")
    ctx.candidates = [
        CandidateState(business_id="b1", origin="initial_seed", metadata={"business_id": "b1"}),
        CandidateState(business_id="b2", origin="initial_seed", metadata={"business_id": "b2"}),
    ]

    KgSearchTask(
        ctx,
        FakeProvider(),  # type: ignore[arg-type]
        ConstructionSettings(enabled_kg=True, kg_max_retries=1, kg_min_survivors=1),
    ).execute()

    assert [candidate.business_id for candidate in ctx.active_candidates()] == ["b1"]
    assert ctx.find_candidate("b1").kg_verify is not None  # type: ignore[union-attr]
    assert ctx.find_candidate("b2").drop_reason == "KG_Union_Fail: No Topology/Text/Image Evidence"  # type: ignore[union-attr]
