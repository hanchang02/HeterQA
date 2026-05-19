"""Candidate expansion stage for answer-set-first construction."""

from __future__ import annotations

from typing import Any

from heterqa.construction.contracts import (
    BusinessRecord,
    CandidateState,
    ConstructionSettings,
    PipelineContext,
    StructuredFilter,
)
from heterqa.construction.providers import ConstructionDataProvider
from heterqa.construction.record_fields import _ask_json, verify_record_fields
from heterqa.providers.model_client import embed_visual_text_query


STRUCTURED_QUERY_PROMPT = (
    "You are a query composer. Convert the following structured constraints "
    "into a single, concise, natural-language user query that sounds like what a person would type when searching.\n\n"
    "The constraints come from a SQL-like system and each line is derived from a 3-tuple:\n"
    "- Lines with `type: filter` come from (query_sql, value, is_numeric_bool), e.g. "
    "  ('outdoor_seating = 1', 1, True).\n"
    "- Lines with `type: categories` come from (category_key, [selected_keywords], 'categories'), e.g. "
    "  ('Restaurants, Bars', ['Restaurants'], 'categories').\n\n"
    "Semantic rules:\n"
    "1) Be faithful — do NOT add, remove, or infer hard constraints that are not implied by the filters.\n"
    "2) The output MUST sound like a search query, starting with an action phrase such as "
    "'Find', 'Show me', or 'Look for'.\n"
    "3) Use plain natural language; never include SQL syntax, field names, or the word 'filter'.\n"
    "4) Keep it concise (one sentence, ideally under 20 words).\n"
    "5) Output format: JSON only.\n\n"
    "Special rules for categories:\n"
    "- When you see a line with \"type: categories\", only use the values inside 'selected_keywords'\n"
    "  to describe the business type.\n"
    "- Do NOT copy or paraphrase 'category_key'. It is provided only for context and must NOT appear in the final query.\n\n"
    "Style rules (important):\n"
    "- Make the query sound rich and user-intent-like, not minimal or robotic.\n"
    "- When you use category keywords, prefer phrases like '<keyword> service businesses', '<keyword> places', "
    "or '<keyword> venues' instead of just '<keyword> businesses'.\n"
    "- For group-related or convenience-related filters, you may use natural-language expansions that are "
    "consistent with the filters.\n\n"
    "Structured constraints:\n"
    "{constraints}\n\n"
    "Now return only JSON in the same format:\n"
    "{\"structured_nl\": \"<one concise user query>\"}"
)


def _format_filter(predicate: StructuredFilter) -> str:
    if predicate.operator == "semantic_category":
        category_key = ""
        if isinstance(predicate.raw, (list, tuple)) and predicate.raw:
            category_key = str(predicate.raw[0])
        return f"type: categories | selected_keywords: {predicate.value} | category_key: {category_key}"
    raw_sql = None
    if isinstance(predicate.raw, (list, tuple)) and predicate.raw:
        raw_sql = predicate.raw[0]
    sql = raw_sql or f"{predicate.field} {predicate.operator}"
    return f"type: filter | sql: {sql} | value: {predicate.value} | is_numeric: {bool(predicate.is_numeric)}"


def filters_to_nlp(filters: list[StructuredFilter], model: Any = None) -> str:
    """Generate the recall intent used by text/image/KG expansion."""

    constraints = "\n".join(f"- {_format_filter(item)}" for item in filters)
    if model is None:
        return "; ".join(_format_filter(item) for item in filters)
    prompt = STRUCTURED_QUERY_PROMPT.format(constraints=constraints)
    response = _ask_json(model, prompt)
    return str(response.get("structured_nl") or response.get("query") or "").strip()


class ExpansionTask:
    """Populate candidates from seeds, KG expansion, text recall, and image recall."""

    def __init__(
        self,
        ctx: PipelineContext,
        provider: ConstructionDataProvider,
        settings: ConstructionSettings,
        *,
        filters: list[StructuredFilter],
        model: Any = None,
    ):
        self.ctx = ctx
        self.provider = provider
        self.settings = settings
        self.filters = filters
        self.model = model

    def execute(self) -> None:
        self._add_initial_seeds()
        if not self.filters:
            raise ValueError("ExpansionTask requires structured filters; refusing to generate an unconstrained case.")
        self.ctx.recall_query = str(self.ctx.metadata.get("source_recall_query") or self.ctx.recall_query or "").strip()
        if not self.ctx.recall_query:
            self.ctx.recall_query = filters_to_nlp(self.filters, self.model)
        if self.settings.enabled_kg:
            self._add_new_candidates(self._get_kg_candidates(), "kg_expansion", "kg")
        if self.settings.enabled_text:
            self._add_new_candidates(self._get_text_vector_candidates(self.ctx.recall_query), "text_vector_recall", "text")
        if self.settings.enabled_image:
            self._add_new_candidates(self._get_image_vector_candidates(self.ctx.recall_query), "image_vector_recall", "image")
        self._hydrate_missing_metadata()
        checked, dropped = verify_record_fields(
            self.ctx,
            filters=self.filters,
            semantic_model=self.model,
            allow_missing=self.settings.allow_missing_structured_values,
        )
        self.ctx.stats["expansion"] = {
            "total_candidates": len(self.ctx.candidates),
            "record_field_checked": checked,
            "record_field_dropped": dropped,
            "recall_query": self.ctx.recall_query,
        }

    def _add_initial_seeds(self) -> None:
        seed_records = list(self.ctx.seed_records)
        if not seed_records:
            seed_ids = self.ctx.metadata.get("seed_business_ids", [])
            for business_id in seed_ids:
                record = self.provider.get_business(str(business_id))
                if record:
                    seed_records.append(record.fields)
        for row in seed_records:
            if "business_id" not in row:
                continue
            business_id = str(row["business_id"])
            if self.ctx.find_candidate(business_id):
                continue
            self.ctx.candidates.append(
                CandidateState(
                    business_id=business_id,
                    name=str(row.get("name", "")),
                    origin="initial_seed",
                    is_active=True,
                    metadata=dict(row),
                )
            )

    def _add_new_candidates(self, recalled: dict[str, float], origin: str, score_family: str) -> None:
        """Add IDs returned by an explicit expansion branch.

        The construction flow exposes three branches: KG feature expansion,
        review-vector expansion, and photo-vector expansion. This method only
        materializes their outputs; the recall logic itself is kept in the
        branch methods below rather than hidden inside provider.recall().
        """

        for business_id, score in recalled.items():
            business_id = str(business_id)
            candidate = self.ctx.find_candidate(business_id)
            if candidate:
                candidate.scores[f"{score_family}_recall"] = score
                continue
            self.ctx.candidates.append(
                CandidateState(
                    business_id=business_id,
                    origin=origin,
                    is_active=True,
                    scores={f"{score_family}_recall": score},
                )
            )

    def _get_kg_candidates(self) -> dict[str, float]:
        """KG expansion over selected category/field tokens.

        Production providers may expose feature-vector search and graph lookup
        methods.  File-backed public runs use KG evidence summaries through
        the same explicit graph-evidence branch, so the algorithm remains inspectable.
        """

        fields = self._kg_seed_fields()
        if not fields:
            return {}
        candidates: dict[str, float] = {}
        embedding_model = self.model.embedding if self.model is not None else None
        if embedding_model is not None:
            embeddings = _embed_texts(embedding_model, fields)
            for field_name, vector in zip(fields, embeddings, strict=False):
                feature_rows = self.provider.search_feature_embeddings(vector, limit=10)
                feature_docs = [{"text": str(row.get("feature_key") or row.get("text") or row)} for row in feature_rows]
                reranked = self._rerank(field_name, feature_docs, threshold=0.6)
                for row in reranked[:1]:
                    feature_name = str(row.get("text", ""))
                    if not feature_name:
                        continue
                    for business_id in self.provider.get_businesses_by_feature(feature_name, "pos"):
                        candidates[str(business_id)] = max(candidates.get(str(business_id), 0.0), float(row.get("rerank_score", 1.0) or 1.0))
            return candidates

        for record in self.provider.iter_businesses():
            evidence = self.provider.get_evidence("kg", record.business_id, self.ctx.recall_query, self.settings.evidence_limit)
            score = max((item.score or 0.0 for item in evidence), default=0.0)
            if score > 0:
                candidates[record.business_id] = score
        return dict(sorted(candidates.items(), key=lambda item: item[1], reverse=True)[: self.settings.top_k])

    def _get_text_vector_candidates(self, nlp_query: str) -> dict[str, float]:
        """Text vector expansion: embedding search followed by reranker filtering."""

        rows: list[dict[str, Any]]
        embedding_model = self.model.embedding if self.model is not None else None
        if embedding_model is not None:
            vector = _embed_texts(embedding_model, [nlp_query])[0]
            rows = list(self.provider.search_review_embeddings(vector, top_k=self.settings.top_k))
            for row in rows:
                row.setdefault("text", row.get("summary", ""))
        else:
            rows = self.provider.search_reviews(nlp_query, self._all_business_ids(), self.settings.top_k)

        docs = [
            {
                "business_id": str(row.get("business_id", "")),
                "text": str(row.get("text") or row.get("summary") or ""),
                "coarse_score": float(row.get("coarse_score", row.get("score", 0)) or 0),
            }
            for row in rows
            if row.get("business_id")
        ]
        reranked = self._rerank(nlp_query, docs, threshold=self.settings.text_rerank_thres)
        output: dict[str, float] = {}
        for row in reranked:
            business_id = str(row.get("business_id", ""))
            if business_id:
                output[business_id] = max(output.get(business_id, 0.0), float(row.get("rerank_score", row.get("coarse_score", 0)) or 0))
        return output

    def _get_image_vector_candidates(self, nlp_query: str) -> dict[str, float]:
        """Photo-vector expansion with coarse-score threshold gating."""

        rows: list[dict[str, Any]]
        vl_embedding_model = self.model.visual_embedding if self.model is not None else None
        if vl_embedding_model is not None:
            vector = _embed_visual_text(vl_embedding_model, nlp_query)
            rows = list(self.provider.search_photo_embeddings(vector, top_k=self.settings.top_k))
        else:
            rows = self.provider.search_photos(nlp_query, self._all_business_ids(), self.settings.top_k)

        output: dict[str, float] = {}
        for row in rows:
            business_id = str(row.get("business_id", ""))
            score = float(row.get("_score", row.get("score", 0)) or 0)
            if business_id and score >= self.settings.image_coarse_thres:
                output[business_id] = max(output.get(business_id, 0.0), score)
        return output

    def _kg_seed_fields(self) -> list[str]:
        fields: list[str] = []
        for predicate in self.filters:
            if predicate.field == "categories":
                values = predicate.value if isinstance(predicate.value, list) else [predicate.value]
                fields.extend(str(value) for value in values)
            else:
                fields.append(predicate.field)
        return [field for field in fields if field]

    def _all_business_ids(self) -> list[str]:
        return [record.business_id for record in self.provider.iter_businesses()]

    def _rerank(self, query: str, docs: list[dict[str, Any]], *, threshold: float) -> list[dict[str, Any]]:
        reranker = self.model.reranker if self.model is not None else None
        if reranker is not None:
            if not hasattr(reranker, "select_by_rerank_score"):
                raise ValueError("Configured reranker must expose select_by_rerank_score(...).")
            return list(reranker.select_by_rerank_score(query=query, documents_dict=docs, thres=threshold))
        return [doc for doc in docs if float(doc.get("coarse_score", doc.get("score", 0)) or 0) >= threshold or threshold <= 0]

    def _hydrate_missing_metadata(self) -> None:
        for candidate in self.ctx.candidates:
            if candidate.metadata:
                continue
            record: BusinessRecord | None = self.provider.get_business(candidate.business_id)
            if not record:
                candidate.drop("db_not_found")
                continue
            candidate.name = record.name
            candidate.metadata = dict(record.fields)

    def summary(self) -> dict[str, Any]:
        return {
            "candidate_count": len(self.ctx.candidates),
            "active_count": len(self.ctx.active_candidates()),
            "drop_reasons": self._drop_reasons(),
            "recall_query": self.ctx.recall_query,
        }

    def _drop_reasons(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for candidate in self.ctx.candidates:
            if not candidate.is_active:
                reason = candidate.drop_reason or "unknown"
                counts[reason] = counts.get(reason, 0) + 1
        return counts


def _embed_texts(embedding_model: Any, texts: list[str]) -> list[Any]:
    result = embedding_model.embed(texts)
    return list(result.get("embeddings", []))


def _embed_visual_text(vl_embedding_model: Any, text: str) -> Any:
    return embed_visual_text_query(vl_embedding_model, text)
