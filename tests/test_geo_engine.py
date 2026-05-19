from __future__ import annotations

from heterqa.construction.contracts import CandidateState, ConstructionSettings, GeoConstraint, PipelineContext
from heterqa.construction.geo_engine import GeoSearchTask, bearing_direction, haversine_km
from heterqa.construction.providers import FileBackedConstructionProvider


def _candidate(business_id: str, lat: float, lon: float) -> CandidateState:
    return CandidateState(
        business_id=business_id,
        name=business_id,
        origin="initial_seed",
        metadata={"business_id": business_id, "name": business_id, "latitude": lat, "longitude": lon},
    )


def test_geo_direction_filter_uses_eight_way_projected_direction() -> None:
    ctx = PipelineContext(task_id="geo-direction")
    ctx.candidates = [
        _candidate("east", 35.0000, 139.0090),
        _candidate("north", 35.0090, 139.0000),
        _candidate("far-east", 35.0000, 139.0900),
    ]
    provider = FileBackedConstructionProvider([], {})
    task = GeoSearchTask(ctx, provider, ConstructionSettings(enabled_geo=True))
    constraint = GeoConstraint(
        reference_latitude=35.0,
        reference_longitude=139.0,
        radius_km=2.0,
        direction="E",
        relation_type="direction",
    )

    task.verify_candidates(constraint)

    active_ids = {candidate.business_id for candidate in ctx.active_candidates()}
    assert active_ids == {"east"}
    assert ctx.find_candidate("north").drop_reason == "geo_boundary_filter"  # type: ignore[union-attr]


def test_geo_nearest_filter_is_global_top_k_not_per_candidate_radius() -> None:
    ctx = PipelineContext(task_id="geo-nearest")
    ctx.candidates = [
        _candidate("near", 35.0000, 139.0010),
        _candidate("far", 35.0000, 139.0200),
    ]
    provider = FileBackedConstructionProvider([], {})
    task = GeoSearchTask(ctx, provider, ConstructionSettings(enabled_geo=True))
    constraint = GeoConstraint(
        reference_latitude=35.0,
        reference_longitude=139.0,
        top_k=1,
        relation_type="nearest",
    )

    task.verify_candidates(constraint)

    assert [candidate.business_id for candidate in ctx.active_candidates()] == ["near"]
    assert ctx.find_candidate("far").drop_reason == "geo_boundary_filter"  # type: ignore[union-attr]


def test_geo_bearing_direction_matches_source_compass_buckets() -> None:
    _distance, bearing = haversine_km(35.0, 139.0, 35.0, 139.01)

    assert bearing_direction(bearing) == "E"
