"""Structured seed and record-field predicate selection.

This module selects category keywords, incrementally applies record-field
predicates, and returns the seed records that pass the selected filters. Data
access is kept in the provider; the selection and filtering algorithm lives
here.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from heterqa.construction.contracts import StructuredFilter
from heterqa.construction.providers import ConstructionDataProvider


NUMERIC_RULES: dict[str, dict[str, Any]] = {
    "stars": {
        "thresholds": [3.5, 4.0, 4.5],
        "probs": [0.30, 0.45, 0.25],
        "op": ">=",
    },
    "restaurants_price_range2": {
        "thresholds": ["<=2", "=3", "=4"],
        "probs": [0.70, 0.20, 0.10],
        "op": None,
    },
}

BOOL_RULES: dict[str, tuple[float, float]] = {
    "business_accepts_credit_cards": (1.0, 0.0),
    "restaurants_take_out": (1.0, 0.0),
    "restaurants_delivery": (1.0, 0.0),
    "good_for_kids": (1.0, 0.0),
    "wheelchair_accessible": (1.0, 0.0),
    "outdoor_seating": (1.0, 0.0),
    "restaurants_reservations": (1.0, 0.0),
    "restaurants_counter_service": (1.0, 0.2),
    "open_24_hours": (1.0, 0.0),
    "ambience_casual": (1.0, 0.0),
    "ambience_romantic": (1.0, 0.0),
    "ambience_classy": (1.0, 0.0),
    "ambience_trendy": (1.0, 0.0),
    "ambience_intimate": (1.0, 0.0),
    "ambience_touristy": (1.0, 0.0),
    "ambience_hipster": (1.0, 0.0),
    "ambience_divey": (1.0, 0.0),
    "ambience_upscale": (1.0, 0.0),
    "has_tv": (1.0, 0.2),
    "restaurants_good_for_groups": (1.0, 0.0),
    "bike_parking": (1.0, 0.0),
    "dogs_allowed": (1.0, 0.0),
    "drive_thru": (1.0, 0.0),
    "caters": (1.0, 0.0),
    "by_appointment_only": (1.0, 0.0),
    "happy_hour": (1.0, 0.0),
    "restaurants_table_service": (1.0, 0.0),
    "music_background_music": (1.0, 0.0),
    "music_live": (1.0, 0.0),
    "music_dj": (1.0, 0.0),
}

ENUM_RULES: dict[str, dict[str, list[Any]]] = {
    "wifi": {"values": ["free", "no", "paid"], "probs": [0.80, 0.18, 0.02]},
    "alcohol": {"values": ["none", "full_bar", "beer_and_wine"], "probs": [0.25, 0.45, 0.30]},
    "restaurants_attire": {"values": ["casual", "dressy", "formal"], "probs": [0.80, 0.18, 0.02]},
    "noise_level": {"values": ["quiet", "average"], "probs": [0.65, 0.35]},
    "smoking": {"values": ["no", "outdoor", "yes"], "probs": [0.50, 0.40, 0.10]},
    "byob_corkage": {"values": ["yes_free", "yes_corkage"], "probs": [0.5, 0.5]},
    "ages_allowed": {"values": ["allages", "21plus"], "probs": [0.7, 0.3]},
}

STOP_WORDS: set[str] = set()


FIELD_WEIGHTS: dict[str, float] = {
    "categories": 2.5,
    "is_open": 1.8,
    "stars": 1.8,
    "city": 1.7,
    "state": 1.6,
    "business_accepts_credit_cards": 1.2,
    "restaurants_take_out": 1.1,
    "restaurants_delivery": 1.1,
    "good_for_kids": 1.0,
    "wheelchair_accessible": 1.0,
    "restaurants_price_range2": 0.9,
    "outdoor_seating": 0.9,
    "wifi": 0.9,
    "alcohol": 0.8,
    "restaurants_reservations": 0.8,
    "restaurants_counter_service": 0.3,
    "open_24_hours": 0.3,
    "ambience_casual": 0.7,
    "has_tv": 0.7,
    "restaurants_good_for_groups": 0.7,
    "bike_parking": 0.6,
    "dogs_allowed": 0.6,
    "drive_thru": 0.6,
    "caters": 0.6,
    "ambience_romantic": 0.6,
    "by_appointment_only": 0.5,
    "happy_hour": 0.5,
    "restaurants_attire": 0.5,
    "noise_level": 0.5,
    "restaurants_table_service": 0.5,
    "ambience_classy": 0.5,
    "ambience_trendy": 0.5,
    "music_background_music": 0.5,
    "ambience_intimate": 0.4,
    "ambience_touristy": 0.4,
    "ambience_hipster": 0.4,
    "ambience_divey": 0.4,
    "ambience_upscale": 0.4,
    "music_live": 0.4,
    "music_dj": 0.4,
    "smoking": 0.4,
    "hair_specializes_in_coloring": 0.4,
    "coat_check": 0.3,
    "good_for_dancing": 0.3,
    "corkage": 0.3,
    "byob": 0.3,
    "business_accepts_bitcoin": 0.3,
    "accepts_insurance": 0.3,
    "byob_corkage": 0.3,
    "ages_allowed": 0.3,
    "music_jukebox": 0.3,
    "music_karaoke": 0.3,
    "music_no_music": 0.3,
    "music_video": 0.3,
    "hair_specializes_in_extensions": 0.3,
    "hair_specializes_in_kids": 0.3,
    "hair_specializes_in_perms": 0.3,
    "hair_specializes_in_straightperms": 0.3,
    "hair_specializes_in_africanamerican": 0.3,
    "hair_specializes_in_asian": 0.3,
    "hair_specializes_in_curly": 0.3,
    "best_nights_friday": 0.2,
    "best_nights_saturday": 0.2,
    "best_nights_thursday": 0.2,
    "best_nights_wednesday": 0.2,
    "best_nights_tuesday": 0.2,
    "best_nights_sunday": 0.2,
    "best_nights_monday": 0.2,
    "business_parking_garage": 0.1,
    "business_parking_lot": 0.1,
    "business_parking_street": 0.1,
    "business_parking_valet": 0.1,
    "business_parking_validated": 0.1,
    "good_for_meal_breakfast": 0.1,
    "good_for_meal_brunch": 0.1,
    "good_for_meal_lunch": 0.1,
    "good_for_meal_dinner": 0.1,
    "good_for_meal_dessert": 0.1,
    "good_for_meal_latenight": 0.1,
    "dietary_restrictions_dairy_free": 0.1,
    "dietary_restrictions_gluten_free": 0.1,
    "dietary_restrictions_vegan": 0.1,
    "dietary_restrictions_kosher": 0.1,
    "dietary_restrictions_halal": 0.1,
    "dietary_restrictions_soy_free": 0.1,
    "dietary_restrictions_vegetarian": 0.1,
}

EXCLUDED_FIELDS: set[str] = {
    "business_id",
    "attributes_json",
    "hours_json",
    "latitude",
    "longitude",
    "address",
    "postal_code",
    "review_count",
    "name",
}

TIER1_CORE = ["city", "state", "stars", "is_open"]
TIER2_HIGH_VALUE = [
    "business_accepts_credit_cards",
    "restaurants_take_out",
    "restaurants_delivery",
    "good_for_kids",
    "wheelchair_accessible",
    "restaurants_price_range2",
    "outdoor_seating",
    "wifi",
    "alcohol",
    "restaurants_reservations",
]


@dataclass(frozen=True)
class StructuredSelectionSettings:
    compact: bool = True
    seeded: bool = False
    field_selection_mode: str = "random"
    max_retries: int = 3


@dataclass
class StructuredSelectionResult:
    filters: list[StructuredFilter]
    seed_records: list[dict[str, Any]]
    source_filters: list[Any] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class StructuredQuerySelector:
    """Equivalent construction-stage selector for record-field seed constraints."""

    def __init__(
        self,
        provider: ConstructionDataProvider,
        settings: StructuredSelectionSettings | None = None,
    ):
        self.provider = provider
        self.settings = settings or StructuredSelectionSettings()
        self.business_rows = [dict(record.fields) for record in provider.iter_businesses()]
        if not self.business_rows:
            raise ValueError("StructuredQuerySelector requires at least one business record.")
        self.field_metadata = self._profile_rows(self.business_rows)
        self.individual_category_dict = self._get_categories(self.field_metadata)
        self.individual_category_key = list(self.individual_category_dict.keys())
        if not self.individual_category_key:
            raise ValueError("StructuredQuerySelector could not find usable category values.")
        self.fields_metadata = self._generate_fields_metadata()

    def select_fields(self, seed: int | None = None, compact: bool | None = None) -> StructuredSelectionResult:
        use_compact = self.settings.compact if compact is None else compact
        rng: random.Random | random.Random = random.Random(seed) if self.settings.seeded and seed is not None else random
        if use_compact:
            return self._select_fields_compact(rng)
        return self._select_fields_stratified(rng)

    def _select_fields_stratified(self, rng: random.Random) -> StructuredSelectionResult:
        filters, rows, source_filters = self._select_category_filter(rng)
        if not rows:
            return StructuredSelectionResult(filters, rows, source_filters)

        for tier_fields in [TIER1_CORE, TIER2_HIGH_VALUE, list(self._tier3_fields())]:
            available = self._available_fields(rows)
            candidates = [available[field] for field in tier_fields if field in available]
            if not candidates:
                if tier_fields == TIER1_CORE:
                    return StructuredSelectionResult(filters, rows, source_filters)
                continue
            chosen_meta = rng.sample(candidates, 1)[0]
            predicate, source = self._generate_sub_query(chosen_meta, rows, rng)
            next_rows = self._apply_filter(rows, predicate)
            if next_rows:
                filters.append(predicate)
                source_filters.append(source)
                rows = next_rows
        return StructuredSelectionResult(filters, rows, source_filters, metadata={"mode": "stratified"})

    def _select_fields_compact(self, rng: random.Random) -> StructuredSelectionResult:
        last_error: Exception | None = None
        for _attempt in range(1, self.settings.max_retries + 1):
            try:
                filters, rows, source_filters = self._select_category_filter(rng)
                if not rows:
                    return StructuredSelectionResult(filters, rows, source_filters)

                mode = self.settings.field_selection_mode
                if mode == "never":
                    extra_count = 0
                elif mode == "always":
                    extra_count = 1
                else:
                    extra_count = rng.randint(0, 1)
                if extra_count == 0:
                    return StructuredSelectionResult(filters, rows, source_filters, metadata={"mode": "compact"})

                tier_choice = rng.choices(["core", "high_value", "atmosphere"], weights=[0.6, 0.3, 0.1], k=1)[0]
                if tier_choice == "core":
                    tier_set = set(TIER1_CORE)
                elif tier_choice == "high_value":
                    tier_set = set(TIER2_HIGH_VALUE)
                else:
                    tier_set = self._tier3_fields()

                available = self._available_fields(rows)
                tier_candidates = [available[field] for field in available if field in tier_set]
                if not tier_candidates:
                    return StructuredSelectionResult(filters, rows, source_filters, metadata={"mode": "compact"})

                weights = [FIELD_WEIGHTS.get(meta["field_name"], 0.5) for meta in tier_candidates]
                chosen_meta = rng.choices(tier_candidates, weights=weights, k=1)[0]
                predicate, source = self._generate_sub_query(chosen_meta, rows, rng)
                next_rows = self._apply_filter(rows, predicate)
                if not next_rows:
                    raise ValueError(
                        f"Field filter {chosen_meta['field_name']!r} resulted in 0 records, retrying."
                    )
                filters.append(predicate)
                source_filters.append(source)
                return StructuredSelectionResult(filters, next_rows, source_filters, metadata={"mode": "compact"})
            except Exception as exc:
                last_error = exc
                continue
        if last_error:
            raise last_error
        raise RuntimeError("Structured selection failed without an explicit error.")

    def _select_category_filter(
        self,
        rng: random.Random,
    ) -> tuple[list[StructuredFilter], list[dict[str, Any]], list[Any]]:
        random_key = rng.choice(self.individual_category_key)
        category_terms = list(self.individual_category_dict[random_key])
        keyword_nums = rng.randint(1, min(2, len(category_terms)))
        random_keywords = rng.sample(category_terms, keyword_nums)
        rows = self._filter_rows_by_categories(self.business_rows, random_keywords)
        raw = (random_key, random_keywords, "categories")
        predicate = StructuredFilter(field="categories", operator="semantic_category", value=random_keywords, raw=raw)
        return [predicate], rows, [raw]

    def _generate_sub_query(
        self,
        field_meta: dict[str, Any],
        rows: list[dict[str, Any]],
        rng: random.Random,
    ) -> tuple[StructuredFilter, Any]:
        name = field_meta["field_name"]
        dtype = str(field_meta["data_type"]).lower()
        metadata = self._profile_rows(rows)

        if name == "categories":
            top_values = [item["value"] for item in metadata["fields"][name].get("top_values", [])]
            fine_terms: list[str] = []
            for value in top_values:
                fine_terms.extend([part.strip() for part in str(value).split(",") if part.strip() not in STOP_WORDS])
            term = rng.choice(fine_terms) if fine_terms else "Coffee & Tea"
            return (
                StructuredFilter(field=name, operator="LIKE", value=term, is_numeric=False, raw=[f"{name} LIKE %s", f"%{term}%", False]),
                (f"{name} LIKE %s", f"%{term}%", False),
            )

        if name in NUMERIC_RULES:
            rule = NUMERIC_RULES[name]
            choice = rng.choices(rule["thresholds"], rule["probs"], k=1)[0]
            if rule["op"]:
                operator = str(rule["op"])
                sql = f"{name} {operator} {choice}"
                value = float(choice)
            else:
                choice_text = str(choice)
                if choice_text.startswith("<="):
                    operator = "<="
                elif choice_text.startswith(">="):
                    operator = ">="
                elif choice_text.startswith("="):
                    operator = "="
                elif choice_text.startswith(">"):
                    operator = ">"
                elif choice_text.startswith("<"):
                    operator = "<"
                else:
                    operator = "="
                sql = f"{name} {choice_text}"
                value = float(choice_text.lstrip("<=>"))
            return (
                StructuredFilter(field=name, operator=operator, value=value, is_numeric=True, raw=[sql, value, True]),
                (sql, value, True),
            )

        if name in BOOL_RULES or dtype == "tinyint":
            _pos_prob, neg_prob = BOOL_RULES.get(name, (1.0, 0.0))
            value = 0 if neg_prob > 0 and rng.random() < neg_prob else 1
            sql = f"{name} = {value}"
            return (
                StructuredFilter(field=name, operator="=", value=value, is_numeric=True, raw=[sql, value, True]),
                (sql, value, True),
            )

        if name in ENUM_RULES:
            rule = ENUM_RULES[name]
            value = rng.choices(rule["values"], rule["probs"], k=1)[0]
            return (
                StructuredFilter(field=name, operator="=", value=value, is_numeric=False, raw=[f"{name} = %s", value, False]),
                (f"{name} = %s", value, False),
            )

        if name in {"city", "state"}:
            top_values = [item["value"] for item in metadata["fields"][name].get("top_values", [])]
            value = rng.choice(top_values)
            return (
                StructuredFilter(field=name, operator="=", value=value, is_numeric=False, raw=[f"{name} = %s", value, False]),
                (f"{name} = %s", value, False),
            )

        if dtype in {"varchar", "char", "text", "mediumtext", "longtext", "enum"}:
            values = metadata["fields"][name].get("unique_values") or []
            if values:
                value = rng.choice(values)
                return (
                    StructuredFilter(field=name, operator="=", value=value, is_numeric=False, raw=[f"{name} = %s", value, False]),
                    (f"{name} = %s", value, False),
                )
            return (
                StructuredFilter(field=name, operator="LIKE", value="Cafe", is_numeric=False, raw=[f"{name} LIKE %s", "%Cafe%", False]),
                (f"{name} LIKE %s", "%Cafe%", False),
            )

        stats = metadata["fields"][name]
        if dtype in {"int", "bigint", "decimal", "float", "double", "numeric"} and stats.get("min") is not None:
            min_val = float(stats["min"])
            max_val = float(stats["max"])
            value = int(rng.randint(int(min_val), int(max_val))) if dtype in {"int", "bigint"} else round(rng.uniform(min_val, max_val), 1)
            operator = rng.choice([">", "<", ">=", "<="])
            sql = f"{name} {operator} {value}"
            return (
                StructuredFilter(field=name, operator=operator, value=value, is_numeric=True, raw=[sql, value, True]),
                (sql, value, True),
            )

        return (
            StructuredFilter(field=name, operator="IS NOT NULL", value=None, is_numeric=False, raw=[f"{name} IS NOT NULL", None, False]),
            (f"{name} IS NOT NULL", None, False),
        )

    def _apply_filter(self, rows: list[dict[str, Any]], predicate: StructuredFilter) -> list[dict[str, Any]]:
        output = []
        for row in rows:
            value = _case_insensitive_get(row, predicate.field)
            if _predicate_passes(value, predicate):
                output.append(row)
        return output

    @staticmethod
    def _filter_rows_by_categories(rows: list[dict[str, Any]], terms: list[str]) -> list[dict[str, Any]]:
        output = []
        for row in rows:
            category_text = str(_case_insensitive_get(row, "categories") or "")
            if all(term.lower() in category_text.lower() for term in terms):
                output.append(row)
        return output

    def _available_fields(self, rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        metadata = self._profile_rows(rows)
        available = {}
        for field_name in metadata["fields"]:
            if field_name not in EXCLUDED_FIELDS and FIELD_WEIGHTS.get(field_name, 0) > 0:
                available[field_name] = self.fields_metadata[field_name]
        return available

    def _generate_fields_metadata(self) -> dict[str, dict[str, Any]]:
        fields_metadata = {}
        for field_name, field_info in self.field_metadata["fields"].items():
            if field_name in EXCLUDED_FIELDS:
                continue
            fields_metadata[field_name] = {
                "field_name": field_name,
                "data_type": field_info["type"],
                "role": self._determine_field_role(field_name),
                "weight": FIELD_WEIGHTS.get(field_name, 0.5),
            }
        return fields_metadata

    def _determine_field_role(self, field_name: str) -> str:
        if field_name in TIER1_CORE:
            return "core"
        if field_name in TIER2_HIGH_VALUE:
            return "high_value"
        if field_name in self._tier3_fields():
            return "atmosphere"
        return "other"

    def _tier3_fields(self) -> set[str]:
        return set(self.field_metadata["fields"]) - set(TIER1_CORE) - set(TIER2_HIGH_VALUE) - EXCLUDED_FIELDS

    @staticmethod
    def _get_categories(metadata: dict[str, Any]) -> dict[str, set[str]]:
        category_map: dict[str, set[str]] = defaultdict(set)
        for item in metadata.get("fields", {}).get("categories", {}).get("unique_values", []):
            for token in str(item).split(","):
                cleaned = token.strip().strip('"').strip("'")
                if cleaned:
                    category_map[str(item)].add(cleaned)
        return category_map

    @staticmethod
    def _profile_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
        fields: dict[str, dict[str, Any]] = {}
        keys: list[str] = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen and key not in EXCLUDED_FIELDS:
                    seen.add(key)
                    keys.append(key)

        for key in keys:
            values = [_case_insensitive_get(row, key) for row in rows]
            values = [value for value in values if value is not None]
            dtype = _infer_type(values)
            finfo: dict[str, Any] = {"type": dtype, "description": ""}
            if dtype in {"varchar", "char", "text", "mediumtext", "longtext", "enum"}:
                counter = Counter(str(value) for value in values)
                finfo["top_values"] = [{"value": value, "count": count} for value, count in counter.most_common(50)]
                finfo["unique_values"] = list(counter.keys())
                finfo["unique_count"] = len(counter)
                if key == "categories":
                    finfo["token_stats"] = _category_token_stats(counter)
            elif dtype in {"int", "bigint", "decimal", "float", "double", "numeric"}:
                numeric_values = [float(value) for value in values if _is_number(value)]
                if numeric_values:
                    avg = sum(numeric_values) / len(numeric_values)
                    finfo.update(
                        {
                            "min": min(numeric_values),
                            "max": max(numeric_values),
                            "avg": avg,
                            "stddev": 0.0,
                            "non_null_count": len(numeric_values),
                        }
                    )
            elif dtype == "tinyint":
                counter = Counter(str(int(_coerce_bool_int(value))) for value in values if _coerce_bool_int(value) is not None)
                finfo["value_counts"] = dict(counter)
            fields[key] = finfo
        return {"fields": fields, "total_records": len(rows)}


def select_structured_seed_case(
    provider: ConstructionDataProvider,
    settings: StructuredSelectionSettings,
    *,
    seed: int | None = None,
) -> StructuredSelectionResult:
    return StructuredQuerySelector(provider, settings).select_fields(seed=seed)


def _category_token_stats(category_counter: Counter[str]) -> dict[str, Any]:
    token_counter = Counter()
    cooccur: dict[str, Counter[str]] = defaultdict(Counter)
    for category_text, count in category_counter.items():
        tokens = list({part.strip().strip('"').strip("'") for part in str(category_text).split(",") if part.strip()})
        for token in tokens:
            token_counter[token] += count
        for i, token_a in enumerate(tokens):
            for token_b in tokens[i + 1 :]:
                cooccur[token_a][token_b] += count
                cooccur[token_b][token_a] += count
    return {
        "tokens_top": [{"token": token, "count": count} for token, count in token_counter.most_common(80)],
        "co_occurrence": {
            token: [{"token": other, "count": count} for other, count in counter.most_common(20)]
            for token, counter in cooccur.items()
        },
    }


def _case_insensitive_get(row: dict[str, Any], field: str) -> Any:
    for key, value in row.items():
        if key.lower() == field.lower():
            return value
    return None


def _predicate_passes(value: Any, predicate: StructuredFilter) -> bool:
    if predicate.operator == "semantic_category" or predicate.field == "categories":
        category_text = str(value or "").lower()
        terms = predicate.value if isinstance(predicate.value, list) else [predicate.value]
        return all(str(term).lower() in category_text for term in terms)
    if value is None:
        return False
    operator = predicate.operator.upper()
    expected = predicate.value
    if operator == "IS NOT NULL":
        return value is not None
    if operator == "LIKE":
        return str(expected).strip("%").lower() in str(value).lower()
    if operator in {"=", "=="}:
        return _normalize_value(value) == _normalize_value(expected)
    try:
        observed_num = float(value)
        expected_num = float(expected)
    except (TypeError, ValueError):
        return False
    if operator == ">=":
        return observed_num >= expected_num
    if operator == "<=":
        return observed_num <= expected_num
    if operator == ">":
        return observed_num > expected_num
    if operator == "<":
        return observed_num < expected_num
    return False


def _normalize_value(value: Any) -> Any:
    bool_value = _coerce_bool_int(value)
    if bool_value is not None:
        return bool_value
    if _is_number(value):
        return float(value)
    return str(value).strip().lower()


def _coerce_bool_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int) and value in {0, 1}:
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1"}:
            return 1
        if lowered in {"false", "no", "0"}:
            return 0
    return None


def _infer_type(values: list[Any]) -> str:
    bool_values = [_coerce_bool_int(value) for value in values]
    if values and all(value is not None for value in bool_values):
        return "tinyint"
    if values and all(_is_int_like(value) for value in values):
        return "int"
    if values and all(_is_number(value) for value in values):
        return "decimal"
    return "varchar"


def _is_int_like(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    try:
        return float(value).is_integer()
    except (TypeError, ValueError):
        return False


def _is_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False
