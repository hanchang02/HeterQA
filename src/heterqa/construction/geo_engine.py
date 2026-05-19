"""Geo construction stage with guaranteed-anchor waterfall filtering."""

from __future__ import annotations

import math
import random
import zlib
from typing import Any

from heterqa.construction.contracts import (
    ConstructionSettings,
    GeoConstraint,
    LLMCallTrace,
    PipelineContext,
    VerificationResult,
)
from heterqa.construction.providers import ConstructionDataProvider


Direction = str


def has_valid_coordinates(row: dict[str, Any]) -> bool:
    if row is None or "latitude" not in row or "longitude" not in row:
        return False
    try:
        lat = float(row["latitude"])
        lon = float(row["longitude"])
    except (TypeError, ValueError):
        return False
    return math.isfinite(lat) and math.isfinite(lon) and -90 <= lat <= 90 and -180 <= lon <= 180


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> tuple[float, float]:
    radius = 6371.0088
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    distance = 2 * radius * math.asin(math.sqrt(a))
    y = math.sin(dlambda) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlambda)
    bearing = (math.degrees(math.atan2(y, x)) + 360) % 360
    return distance, bearing


def bearing_direction(bearing: float) -> str:
    dirs = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
    idx = int(((bearing + 22.5) % 360.0) // 45.0)
    return dirs[idx]


def crc32_stable_key(primary: str, seed: int) -> int:
    return zlib.crc32(f"{primary}:{int(seed)}".encode("utf-8", errors="ignore")) & 0xFFFFFFFF


def sample_point_around(lat_deg: float, lon_deg: float, distance_km: float, bearing_rad: float) -> tuple[float, float]:
    radius = 6371.0088
    phi1 = math.radians(lat_deg)
    lam1 = math.radians(lon_deg)
    delta = distance_km / radius
    theta = bearing_rad
    sin_phi2 = math.sin(phi1) * math.cos(delta) + math.cos(phi1) * math.sin(delta) * math.cos(theta)
    sin_phi2 = max(-1.0, min(1.0, sin_phi2))
    phi2 = math.asin(sin_phi2)
    y = math.sin(theta) * math.sin(delta) * math.cos(phi1)
    x = math.cos(delta) - math.sin(phi1) * math.sin(phi2)
    lam2 = lam1 + math.atan2(y, x)
    lon2 = (math.degrees(lam2) + 540.0) % 360.0 - 180.0
    lat2 = max(-90.0, min(90.0, math.degrees(phi2)))
    return lat2, lon2


def _normalize_probs(probs: dict[str, float], *, required: tuple[str, ...]) -> dict[str, float]:
    for key in required:
        if key not in probs:
            raise ValueError(f"Geo probability config missing required key: {key}")
    clean = {key: max(0.0, float(value)) for key, value in probs.items()}
    total = sum(clean.values())
    if total <= 0:
        raise ValueError("Geo probability config must have positive total weight.")
    return {key: value / total for key, value in clean.items()}


def _choice_from_probs(probs: dict[str, float], rng: random.Random) -> str:
    draw = rng.random()
    acc = 0.0
    items = list(probs.items())
    for key, prob in items:
        acc += prob
        if draw <= acc:
            return key
    return items[-1][0]


class GeoSearchTask:
    """Generate or apply a geo constraint, then hard-filter active candidates."""

    def __init__(
        self,
        ctx: PipelineContext,
        provider: ConstructionDataProvider,
        settings: ConstructionSettings,
        *,
        constraint: GeoConstraint | None = None,
    ):
        self.ctx = ctx
        self.provider = provider
        self.settings = settings
        self.constraint = constraint

    def execute(self) -> None:
        if not self.settings.enabled_geo:
            return
        constraint = self.constraint or self._build_constraint_from_provider()
        if constraint is None:
            self.ctx.stats["geo_skipped"] = "no_geo_constraint_generated"
            return
        self.constraint = constraint
        self.ctx.geo_query = constraint.nl_text or self._format_geo_query(constraint)
        self.ctx.stats["geo_info"] = {
            "anchor_name": constraint.anchor_name,
            "anchor_bid": constraint.anchor_business_id,
            "logic": {
                "direction": constraint.direction,
                "radius_km": constraint.radius_km,
                "relation_type": constraint.relation_type,
            },
        }
        self.verify_candidates(constraint)

    def _build_constraint_from_provider(self) -> GeoConstraint | None:
        seed_rows = [candidate.metadata for candidate in self.ctx.candidates if candidate.origin == "initial_seed" and candidate.is_active]
        built = self.provider.build_geo_constraint(seed_rows)
        if built is not None:
            return built
        return self._build_guaranteed_constraint(seed_rows)

    def _build_guaranteed_constraint(self, seed_rows: list[dict[str, Any]]) -> GeoConstraint | None:
        valid_rows = [dict(row) for row in seed_rows if has_valid_coordinates(row)]
        if not valid_rows:
            return None

        seed = int(self.settings.geo_seed or 0)
        ordered = self._order_rows_stable(valid_rows, seed)
        relation_probs = dict(self.settings.geo_relation_choice_probs)
        if not self.settings.geo_enable_direction:
            direction_weight = relation_probs.pop("direction", 0.0)
            relation_probs["within_radius"] = relation_probs.get("within_radius", 0.0) + direction_weight / 2.0
            relation_probs["nearest"] = relation_probs.get("nearest", 0.0) + direction_weight / 2.0
        relation_probs = _normalize_probs(relation_probs, required=("within_radius", "nearest"))
        anchor_probs = _normalize_probs(self.settings.geo_anchor_choice_probs, required=("user", "poi"))

        if self.settings.geo_determinism == "stateless":
            signature = "|".join(sorted(str(row.get("business_id", "")) for row in valid_rows))
            relation_rng = self._rng_with_salt(f"relation|{signature}|{seed}")
            anchor_rng = self._rng_with_salt(f"anchor|{signature}|{seed}")
        else:
            relation_rng = random.Random(seed) if self.settings.geo_seed is not None else random.Random()
            anchor_rng = relation_rng
            signature = "session"

        relation_kind = _choice_from_probs(relation_probs, relation_rng)
        anchor_kind = _choice_from_probs(anchor_probs, anchor_rng)

        if anchor_kind == "poi":
            constraint = self._build_poi_anchor_constraint(ordered, relation_kind, signature, seed)
            if constraint is not None:
                return constraint
        return self._build_user_anchor_constraint(ordered[0], relation_kind, signature, seed, mode="user")

    def _build_user_anchor_constraint(
        self,
        row: dict[str, Any],
        relation_kind: str,
        signature: str,
        seed: int,
        *,
        mode: str,
    ) -> GeoConstraint:
        rng_user = self._rng_with_salt(f"user_offset|{signature}|{seed}|{row.get('business_id', '')}|{mode}")
        anchor_lat, anchor_lon, distance = self._sample_user_around_row(row, rng_user)
        if relation_kind == "nearest":
            relation_type = "nearest"
            radius = None
            direction = None
        else:
            radius_rng = self._rng_with_salt(f"radius|{signature}|{seed}|{mode}|{relation_kind}|{row.get('business_id', '')}")
            radius = self._pick_radius_at_least(distance, radius_rng)
            relation_type = relation_kind
            direction = None
            if relation_kind == "direction":
                _, bearing = haversine_km(anchor_lat, anchor_lon, float(row["latitude"]), float(row["longitude"]))
                direction = bearing_direction(bearing)
        constraint = GeoConstraint(
            reference_latitude=anchor_lat,
            reference_longitude=anchor_lon,
            radius_km=radius,
            direction=direction,
            top_k=1 if relation_type == "nearest" else None,
            anchor_kind="user",
            relation_type=relation_type,
            payload={"mode": mode, "x_km": distance, "seed": seed, "relation": relation_kind},
        )
        return self._with_geo_nl(constraint)

    def _build_poi_anchor_constraint(
        self,
        ordered_rows: list[dict[str, Any]],
        relation_kind: str,
        signature: str,
        seed: int,
    ) -> GeoConstraint | None:
        radius_max = float(self.settings.geo_radius_km_range[1])
        anchor = None
        target = None
        for row in ordered_rows[: self.settings.geo_max_anchor_scans]:
            anchor = self._fetch_one_near_seeded(
                float(row["latitude"]),
                float(row["longitude"]),
                radius_max,
                seed,
                exclude_id=str(row.get("business_id", "")),
            )
            if anchor is not None:
                target = row
                break
        if anchor is None or target is None:
            return self._build_user_anchor_constraint(ordered_rows[0], relation_kind, signature, seed, mode="poi_no_neighbor_anchor")

        distance, bearing = haversine_km(
            float(anchor["latitude"]),
            float(anchor["longitude"]),
            float(target["latitude"]),
            float(target["longitude"]),
        )
        if relation_kind == "nearest":
            relation_type = "nearest"
            radius = None
            direction = None
        else:
            rng = self._rng_with_salt(
                f"radius|{signature}|{seed}|poi|{relation_kind}|{target.get('business_id', '')}|{anchor.get('business_id', '')}"
            )
            radius = self._pick_radius_at_least(distance, rng)
            relation_type = relation_kind
            direction = bearing_direction(bearing) if relation_kind == "direction" else None

        constraint = GeoConstraint(
            reference_latitude=float(anchor["latitude"]),
            reference_longitude=float(anchor["longitude"]),
            radius_km=radius,
            direction=direction,
            top_k=1 if relation_type == "nearest" else None,
            anchor_business_id=str(anchor.get("business_id", "")) or None,
            anchor_name=anchor.get("name"),
            anchor_kind="poi",
            relation_type=relation_type,
            payload={
                "mode": "poi",
                "dist_star_km": distance,
                "seed": seed,
                "relation": relation_kind,
                "r_star_id": str(target.get("business_id", "")),
            },
        )
        return self._with_geo_nl(constraint)

    def verify_candidates(self, constraint: GeoConstraint) -> None:
        passed_rows = self._filter_rows([candidate.metadata for candidate in self.ctx.candidates if candidate.is_active], constraint)
        passed_bids = {str(row["business_id"]) for row in passed_rows}
        passed = 0
        dropped = 0
        for candidate in self.ctx.candidates:
            if not candidate.is_active:
                continue
            lat = candidate.metadata.get("latitude")
            lon = candidate.metadata.get("longitude")
            if lat is None or lon is None or not has_valid_coordinates(candidate.metadata):
                result = VerificationResult(
                    judgement="no",
                    confidence=1.0,
                    reason="Missing coordinates",
                    trace=LLMCallTrace(stage="geo_verify"),
                    evidence_locator_type="computed_geo",
                    evidence_locator="missing_coordinates",
                )
                candidate.set_verification("geo_verify", result, "geo")
                candidate.drop("geo_missing_coordinates")
                dropped += 1
                continue
            distance, bearing = haversine_km(
                constraint.reference_latitude,
                constraint.reference_longitude,
                float(lat),
                float(lon),
            )
            direction = bearing_direction(bearing)
            ok = candidate.business_id in passed_bids
            result = VerificationResult(
                judgement="yes" if ok else "no",
                confidence=1.0,
                reason="Within geo boundary" if ok else "Outside geo boundary",
                trace=LLMCallTrace(stage="geo_verify"),
                evidence_locator_type="computed_geo",
                evidence_locator=f"distance_km={distance:.3f};bearing_deg={bearing:.1f};direction={direction}",
                evidence_summary="Spatial relation computed from business coordinates.",
                metadata={
                    "computed_distance_km": round(distance, 3),
                    "computed_bearing_deg": round(bearing, 1),
                    "computed_direction": direction,
                    "radius_km": constraint.radius_km,
                    "query_direction": constraint.direction,
                    "relation_type": constraint.relation_type,
                    "top_k": constraint.top_k,
                },
            )
            candidate.set_verification("geo_verify", result, "geo")
            if ok:
                candidate.stage_status["geo"] = "passes"
                passed += 1
            else:
                candidate.drop("geo_boundary_filter")
                dropped += 1
        self.ctx.stats["geo_task"] = {"passed": passed, "dropped": dropped}

    def _filter_rows(self, rows: list[dict[str, Any]], constraint: GeoConstraint) -> list[dict[str, Any]]:
        valid_rows = [dict(row) for row in rows if has_valid_coordinates(row)]
        relation = constraint.relation_type or ("direction" if constraint.direction else "within_radius")
        anchor_lat = constraint.reference_latitude
        anchor_lon = constraint.reference_longitude
        if relation == "nearest":
            scored = []
            for row in valid_rows:
                distance, _bearing = haversine_km(anchor_lat, anchor_lon, float(row["latitude"]), float(row["longitude"]))
                scored.append((distance, row))
            scored.sort(key=lambda item: item[0])
            return [row for _distance, row in scored[: max(1, int(constraint.top_k or 1))]]
        output = []
        for row in valid_rows:
            distance, bearing = haversine_km(anchor_lat, anchor_lon, float(row["latitude"]), float(row["longitude"]))
            if constraint.radius_km is not None and distance > constraint.radius_km:
                continue
            if relation == "direction" and constraint.direction and bearing_direction(bearing) != constraint.direction.upper():
                continue
            output.append(row)
        output.sort(
            key=lambda row: haversine_km(anchor_lat, anchor_lon, float(row["latitude"]), float(row["longitude"]))[0]
        )
        return output

    @staticmethod
    def _format_geo_query(constraint: GeoConstraint) -> str:
        return GeoSearchTask._with_geo_nl(constraint).nl_text

    @staticmethod
    def _with_geo_nl(constraint: GeoConstraint) -> GeoConstraint:
        if constraint.nl_text:
            return constraint
        relation = constraint.relation_type or (
            "direction" if constraint.direction else "within_radius" if constraint.radius_km is not None else "nearest"
        )
        if constraint.anchor_kind == "poi":
            name = f"POI name: ({constraint.anchor_name}). " if constraint.anchor_name else ""
            loc = f"POI location: (lat {constraint.reference_latitude:.6f}, lng {constraint.reference_longitude:.6f}). {name}"
            if relation == "within_radius":
                text = f"{loc}The businesses returned must be within {float(constraint.radius_km):.1f} km of this POI."
            elif relation == "direction":
                text = f"{loc}Return businesses to the {constraint.direction} of this POI within {float(constraint.radius_km):.1f} km."
            else:
                text = f"{loc}Return the nearest business to this POI."
        else:
            loc = f"User location: (lat {constraint.reference_latitude:.6f}, lng {constraint.reference_longitude:.6f})."
            if relation == "within_radius":
                text = f"{loc} The businesses returned must be within {float(constraint.radius_km):.1f} km of this location."
            elif relation == "direction":
                text = f"{loc} Return businesses to the {constraint.direction} of this location within {float(constraint.radius_km):.1f} km."
            else:
                text = f"{loc} Return the nearest business to this location."
        return GeoConstraint(
            reference_latitude=constraint.reference_latitude,
            reference_longitude=constraint.reference_longitude,
            radius_km=constraint.radius_km,
            direction=constraint.direction,
            top_k=constraint.top_k,
            anchor_business_id=constraint.anchor_business_id,
            anchor_name=constraint.anchor_name,
            anchor_kind=constraint.anchor_kind,
            relation_type=constraint.relation_type,
            nl_text=text,
            payload=dict(constraint.payload),
        )

    def _order_rows_stable(self, rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
        return list(self.provider.order_records_by_seed(rows, seed, id_key="business_id"))

    def _fetch_one_near_seeded(
        self,
        center_lat: float,
        center_lon: float,
        radius_km: float,
        seed: int,
        *,
        exclude_id: str | None = None,
    ) -> dict[str, Any] | None:
        return self.provider.fetch_one_near_seeded(center_lat, center_lon, radius_km, seed, exclude_id=exclude_id)

    def _rng_with_salt(self, salt: str) -> random.Random:
        seed = int(self.settings.geo_seed or 0)
        return random.Random(crc32_stable_key(f"{seed}:{salt}", 0))

    def _sample_user_around_row(self, row: dict[str, Any], rng: random.Random) -> tuple[float, float, float]:
        min_dist, max_dist = self.settings.geo_user_offset_km_range
        if max_dist < min_dist:
            min_dist, max_dist = max_dist, min_dist
        distance = min_dist + (max_dist - min_dist) * rng.random()
        theta = 2.0 * math.pi * rng.random()
        lat, lon = sample_point_around(float(row["latitude"]), float(row["longitude"]), distance, theta)
        return lat, lon, distance

    def _pick_radius_at_least(self, at_least_km: float, rng: random.Random) -> int:
        min_radius, max_radius = self.settings.geo_radius_km_range
        lower = max(float(at_least_km), float(min_radius))
        if lower <= max_radius:
            return self._sample_integer_range((lower, max_radius), rng)
        return int(math.ceil(float(at_least_km)))

    @staticmethod
    def _sample_integer_range(bounds: tuple[float, float], rng: random.Random) -> int:
        lower, upper = bounds
        if upper < lower:
            lower, upper = upper, lower
        lower_int = math.ceil(lower)
        upper_int = math.floor(upper)
        if lower_int > upper_int:
            return lower_int
        return int(lower_int + (upper_int - lower_int) * rng.random())
