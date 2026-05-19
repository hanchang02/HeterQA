"""Final query generation for HeterQA construction cases.

This module converts structured predicates to natural language, merges
structured/geo/text/image/KG intents into one user-facing query, and leaves
validation to ``query_validation``.
"""

from __future__ import annotations

from typing import Any

from heterqa.construction.contracts import GenerationCase, PipelineContext, StructuredFilter
from heterqa.construction.providers import ConstructionDataProvider
from heterqa.construction.record_fields import _ask_json


STRUCTURED_NL_PROMPT = (
    "You are a query composer. Convert the following structured constraints into a single, concise, "
    "natural-language user query that sounds like what a person would type when searching.\n\n"
    "Rules:\n"
    "1) Be faithful: do not add, remove, or infer constraints.\n"
    "2) The output must sound like a search query, starting with an action phrase such as "
    "Find, Show me, or Look for.\n"
    "3) Use plain natural language; never include SQL or field names.\n"
    "4) Keep it concise: one sentence, ideally under 20 words.\n"
    "5) Output JSON only.\n\n"
    "Database schema for context only:\n{schema_text}\n\n"
    "Structured constraints:\n{structured_constraints}\n\n"
    "Example outputs:\n"
    "- Input: cuisine_type = 'Italian', stars >= 4.5, outdoor_seating = 1\n"
    "  Output JSON: {{\"structured_nl\": \"Find Italian restaurants with at least 4.5 stars and outdoor seating.\"}}\n"
    "- Input: category = 'Hotels', pet_friendly = 1, free_breakfast = 1\n"
    "  Output JSON: {{\"structured_nl\": \"Find hotels that are pet-friendly and offer free breakfast.\"}}\n\n"
    "Now return only JSON in the same format:\n"
    "{{\"structured_nl\": \"<one concise user query>\"}}"
)

COMPOSE_QUERY_PROMPT = (
    "Task:\n"
    "Merge up to five components into ONE concise, single-line natural-language query. "
    "The goal is to find TARGET_ENTITY; the grammatical subject must be TARGET_ENTITY.\n\n"
    "Components, any may be empty:\n"
    "- Structured: {structured_nl} (explicit record filters such as category, rating, price, city)\n"
    "- Geo: {geo_nl} (location context)\n"
    "- Text: {text_nl} (requirements supported by review text)\n"
    "- Image: {image_nl} (visual requirements supported by photos)\n"
    "- KG Insights: {kg_nl} (graph-derived behavioral or attribute patterns)\n\n"
    "Rules:\n"
    "1) Subject: must be TARGET_ENTITY, e.g. \"{target_entity}\".\n"
    "2) Priority: Structured/Geo/Text/Image are hard constraints. KG insights are soft context.\n"
    "3) Integrate KG naturally using phrases such as known for, popular with, or appealing to. "
    "If KG duplicates other constraints, merge it. If KG contradicts hard constraints, ignore KG.\n"
    "4) Do not add assumptions, locations, categories, attributes, or visual details not present above.\n"
    "5) Use natural connectors and avoid run-on sentences.\n"
    "6) Output JSON only.\n\n"
    "TARGET_ENTITY = \"{target_entity}\"\n\n"
    "Examples:\n"
    "[Good - Blending KG]\n"
    "- Structured: Coffee shops with free wifi.\n"
    "- Geo: in Seattle\n"
    "- Text: quiet atmosphere\n"
    "- Image: \n"
    "- KG Insights: popular with students who like indie music\n"
    "Output JSON: {{\"nl_query\": \"Find quiet coffee shops in Seattle with free wifi that are popular with students who like indie music.\"}}\n\n"
    "[Good - Handling Redundancy]\n"
    "- Structured: Italian restaurants\n"
    "- Geo: \n"
    "- Text: romantic vibe\n"
    "- Image: \n"
    "- KG Insights: places couples visit for dinner\n"
    "Output JSON: {{\"nl_query\": \"Find romantic Italian restaurants perfect for couples' dinners.\"}}\n\n"
    "[Good - Standard]\n"
    "- Structured: Gyms open 24/7\n"
    "- Geo: near me\n"
    "- Text: \n"
    "- Image: \n"
    "- KG Insights: \n"
    "Output JSON: {{\"nl_query\": \"Find 24/7 gyms near me.\"}}\n\n"
    "Return only JSON in this format:\n"
    "{{\"nl_query\": \"<one concise user query>\"}}"
)


def compose_final_query(case: GenerationCase) -> str:
    if case.final_query:
        return case.final_query
    parts = [case.text_query, case.image_query, case.kg_query]
    if case.geo_constraint:
        geo = case.geo_constraint
        if geo.radius_km is not None:
            parts.append(f"within {geo.radius_km:g} km of the reference point")
        else:
            parts.append("near the reference point")
    return " ".join(part for part in parts if part).strip()


def _field_to_words(field: str) -> str:
    return field.replace("_", " ").replace(".", " ").strip()


def _value_to_words(value: Any) -> str:
    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(item) for item in value)
    return str(value)


def _deterministic_structured_nl(filters: list[StructuredFilter], target_entity: str) -> str:
    """Deterministic public query text when no query-composer model is configured."""

    clauses: list[str] = []
    for predicate in filters:
        field = _field_to_words(predicate.field)
        operator = predicate.operator.upper()
        value = _value_to_words(predicate.value)
        if predicate.operator == "semantic_category":
            clauses.append(f"matching {value}")
        elif operator == "LIKE":
            clauses.append(f"matching {value}")
        elif isinstance(predicate.value, bool):
            clauses.append(f"with {field}" if predicate.value else f"without {field}")
        elif operator in {">", ">=", "<", "<=", "="}:
            comparator = {
                ">": "more than",
                ">=": "at least",
                "<": "less than",
                "<=": "at most",
                "=": "with",
            }[operator]
            if comparator == "with":
                clauses.append(f"with {field} {value}")
            else:
                clauses.append(f"with {field} {comparator} {value}")
        else:
            clauses.append(f"with {field} {operator.lower()} {value}")
    if not clauses:
        return ""
    phrase = clauses[0]
    for clause in clauses[1:]:
        if phrase.startswith("matching ") and clause.startswith("with "):
            phrase += f" {clause}"
        else:
            phrase += f" and {clause}"
    return f"Find {target_entity} {phrase}."


def _compose_without_model(components: dict[str, str], target_entity: str) -> str:
    structured = components.get("structured", "").strip().rstrip(".")
    if structured:
        query = structured
    else:
        query = f"Find {target_entity}"
    if components.get("geo"):
        query += f" {components['geo'].strip()}"
    if components.get("text"):
        query += f" with {components['text'].strip()}"
    if components.get("image"):
        query += f" with visual evidence of {components['image'].strip()}"
    if components.get("kg"):
        kg = components["kg"].strip()
        kg_lower = kg.lower()
        if kg_lower.startswith(("known for", "popular with", "appealing to")):
            query += f" and {kg}"
        else:
            query += f" known for {kg}"
    return query.strip().rstrip(".") + "."


class QueryComposer:
    """Question verbalization stage for construction cases."""

    def __init__(
        self,
        model: Any = None,
        *,
        provider: ConstructionDataProvider | None = None,
        target_entity: str = "businesses",
    ):
        self.model = model
        self.provider = provider
        self.target_entity = target_entity

    def generate_structured_nl(self, case: GenerationCase) -> str:
        if not case.structured_filters:
            return ""
        if self.model is None:
            return _deterministic_structured_nl(case.structured_filters, self.target_entity)
        payload = _ask_json(
            self.model,
            STRUCTURED_NL_PROMPT.format(
                schema_text=self._format_schema_for_prompt(),
                structured_constraints=self._format_structured_filters(case.structured_filters),
            ),
        )
        text = str(payload.get("structured_nl") or "").strip()
        return text or _deterministic_structured_nl(case.structured_filters, self.target_entity)

    def _format_schema_for_prompt(self) -> str:
        if self.provider is None:
            return "(schema unavailable)"
        if hasattr(self.provider, "get_schema_info"):
            schema = self.provider.get_schema_info()  # type: ignore[attr-defined]
            if schema:
                return self._format_schema_mapping(schema)
        if hasattr(self.provider, "get_tables") and hasattr(self.provider, "get_table_schema"):
            rows: list[str] = []
            for table in self.provider.get_tables():  # type: ignore[attr-defined]
                if table != "business":
                    continue
                columns = self.provider.get_table_schema(table)  # type: ignore[attr-defined]
                parts = []
                for column in columns:
                    name = column.get("Field") or column.get("name")
                    type_name = column.get("Type") or column.get("type")
                    if name:
                        parts.append(f"{name}({type_name})" if type_name else str(name))
                rows.append(f"- {table}: " + ", ".join(parts))
            if rows:
                return "\n".join(rows)
        fields: dict[str, str] = {}
        for record in self.provider.iter_businesses()[:50]:
            for key, value in record.fields.items():
                fields.setdefault(key, type(value).__name__)
        if not fields:
            return "(schema unavailable)"
        return "- business: " + ", ".join(f"{key}({value})" for key, value in sorted(fields.items()))

    @staticmethod
    def _format_schema_mapping(schema: Any) -> str:
        if not isinstance(schema, dict):
            return str(schema)
        rows: list[str] = []
        for table, desc in schema.items():
            if isinstance(desc, dict):
                fields = desc.get("fields") or []
                types = desc.get("types") or {}
                parts = [f"{field}({types[field]})" if field in types else str(field) for field in fields]
                rows.append(f"- {table}: " + ", ".join(parts))
            else:
                rows.append(f"- {table}: {desc}")
        return "\n".join(rows) if rows else "(schema unavailable)"

    @staticmethod
    def _format_structured_filters(filters: list[StructuredFilter]) -> str:
        rows: list[str] = []
        for predicate in filters:
            source = predicate.to_source_tuple()
            if isinstance(source, list) and len(source) >= 3:
                rows.append(f"- tag={source[2]}; snippet={source[0]}; value={source[1]}")
            else:
                rows.append(f"- field={predicate.field}; operator={predicate.operator}; value={predicate.value}")
        return "\n".join(rows)

    def compose(self, ctx: PipelineContext, case: GenerationCase) -> str:
        if case.final_query:
            ctx.final_query = case.final_query
            structured_nl = case.recall_query or _deterministic_structured_nl(case.structured_filters, self.target_entity)
            ctx.metadata["query_components"] = self._components(ctx, case, structured_nl=structured_nl)
            return ctx.final_query
        structured_nl = self.generate_structured_nl(case)
        components = self._components(ctx, case, structured_nl=structured_nl)
        ctx.metadata["query_components"] = components
        if self.model is None:
            ctx.final_query = _compose_without_model(components, self.target_entity)
            return ctx.final_query
        payload = _ask_json(
            self.model,
            COMPOSE_QUERY_PROMPT.format(
                structured_nl=components["structured"],
                geo_nl=components["geo"],
                text_nl=components["text"],
                image_nl=components["image"],
                kg_nl=components["kg"],
                target_entity=self.target_entity,
            ),
        )
        ctx.final_query = str(payload.get("nl_query") or payload.get("query") or payload.get("final_query") or "").strip()
        if not ctx.final_query:
            ctx.final_query = compose_final_query(case)
        return ctx.final_query

    @staticmethod
    def _components(ctx: PipelineContext, case: GenerationCase, *, structured_nl: str) -> dict[str, str]:
        geo = ctx.geo_query or (case.geo_constraint.nl_text if case.geo_constraint else "")
        return {
            "structured": structured_nl,
            "geo": geo,
            "text": ctx.text_query or case.text_query,
            "image": ctx.image_query or case.image_query,
            "kg": ctx.kg_query or case.kg_query,
        }
