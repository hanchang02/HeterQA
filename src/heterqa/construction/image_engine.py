"""Image evidence stage with self/cross-verification union semantics."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any

from heterqa.construction.contracts import ConstructionSettings, LLMCallTrace, PipelineContext
from heterqa.construction.providers import ConstructionDataProvider
from heterqa.construction.text_engine import (
    TEXT_JUDGE_PROMPT,
    VLM_JUDGE_PROMPT,
    _call_json_model,
    _embed_visual_text,
    _keyword_payload,
    _support_result,
)


VLM_TEXT_PROMPT = """
Generate a single, high-quality, natural-language user query from a provided business image plus metadata.

You will receive:
1. A photo of a real-world business such as a store, restaurant, cafe, bakery, or service.
2. The business Name.
3. The business Category.
4. An optional short Caption.

Output format:
Return only standard JSON:
{{"query": "Your output query"}}

Hard constraints:
- Single aspect focus: the query must focus on exactly one prominent visible feature, such as one specific food item, seating style, lighting style, display setup, or service setup.
- Natural language: write like a real user would type in a search bar. Keep it concise.
- Strict anonymization: the query must not contain the provided Name, Category labels, direct caption quotes, addresses, brands, or chain names.
- Neutral reference: do not use "this business", "this restaurant", or similar phrases. Use general terms such as "places", "venues", or "spots".

Procedure:
1. Identify the most salient visible feature in the image.
2. Pick one perspective only; do not describe the whole scene.
3. Formulate a natural query as if a user saw the image and wanted to find a similar item, setup, or visual experience elsewhere.

Examples:
- Good: "Find places with outdoor wooden deck seating and string lights."
- Good: "Recommend spots that serve burgers on toasted brioche buns with thick-cut fries."
- Bad: "Find counter-service venues with retro wall signs, a marble-style counter, and open kitchens." This combines too many aspects.
- Bad: "Find Italian restaurants like Allegro Kitchen." This uses metadata.

Now analyze the image and output only the JSON object with a single field "query".

Name: {business_name}
Category: {business_category}
Caption: {photo_caption}
"""

VLM_AUDIT_PROMPT = """
Role:
You are a senior multimodal quality auditor. Evaluate the alignment between an original image and a generated search query.

Evaluation dimensions:

1. Visual faithfulness and focus
- Single aspect focus: the query must focus on exactly one prominent visible feature, such as a specific dish or seating style. Penalize if it combines multiple independent elements.
- Grounding: is the query content actually visible and salient in the photo?
- Utility filter: does it focus on a functional, searchable human need rather than incidental background details?

2. Search realism and anonymization
- Anonymization check: the query must not contain brand names, proper nouns, or specific labels provided in the metadata.
- Naturalness: does it sound like a real user search intent rather than a robotic description?

Generated query:
{query}

Output format:
Return strict JSON with exactly these fields:
{{
  "visual_faithfulness_score": 0,
  "intent_realism_score": 0,
  "is_single_aspect": true,
  "is_properly_anonymized": true,
  "audit_reasoning": "concise explanation in English"
}}
"""


class ImageSearchTask:
    def __init__(
        self,
        ctx: PipelineContext,
        provider: ConstructionDataProvider,
        settings: ConstructionSettings,
        *,
        model: Any = None,
        query: str = "",
    ):
        self.ctx = ctx
        self.provider = provider
        self.settings = settings
        self.model = model
        self.query = query

    def execute(self) -> None:
        if not self.settings.enabled_image or not self.ctx.candidates:
            return
        self.ctx.image_query = self.query or self.ctx.image_query or self._generate_query()
        if not self.ctx.image_query:
            self.ctx.stats["image_task_skipped"] = "no_image_query_generated"
            return
        active = self.ctx.active_candidates()
        active_ids = [candidate.business_id for candidate in active]
        photo_hits = self._photo_search(self.ctx.image_query, active_ids)
        if photo_hits:
            self._verify_photos(photo_hits)
        review_hits = self._cross_review_search(self.ctx.image_query, active_ids)
        if review_hits:
            self._verify_reviews(review_hits)
        survivors = 0
        dropped = 0
        for candidate in active:
            if candidate.image_verify is None and candidate.image_to_text_verify is None:
                candidate.drop("image_modal_all_evidence_failed")
                dropped += 1
            else:
                survivors += 1
        self.ctx.stats["image_task_survivors"] = survivors
        self.ctx.stats["image_task_dropped"] = dropped

    def _generate_query(self) -> str:
        seed_ids = [candidate.business_id for candidate in self.ctx.candidates if candidate.origin == "initial_seed" and candidate.is_active]
        for _ in range(self.settings.max_image_query_attempts):
            random.shuffle(seed_ids)
            for business_id in seed_ids:
                photos = self.provider.get_photos(business_id, limit=1)
                if not photos:
                    continue
                photo = photos[0]
                path = str(photo.get("path") or photo.get("source_locator") or "")
                caption = str(photo.get("caption") or photo.get("summary") or "")
                if self.model is None:
                    return caption[:240] if caption else ""
                prompt = VLM_TEXT_PROMPT.format(
                    business_name=str(photo.get("business_name") or photo.get("name") or "this place"),
                    business_category=str(photo.get("business_category") or photo.get("categories") or "this category"),
                    photo_caption=caption,
                )
                payload, trace = _call_json_model(self.model, prompt, image=path, stage="image_verification_query_generation")
                self.ctx.global_traces.append(trace)
                query = str(payload.get("query", "")).strip()
                if not query:
                    continue
                audit_payload, audit_trace = _call_json_model(
                    self.model,
                    VLM_AUDIT_PROMPT.format(query=query),
                    image=path,
                    stage="image_query_self_audit",
                )
                self.ctx.global_traces.append(audit_trace)
                if (
                    float(audit_payload.get("visual_faithfulness_score", 0) or 0) >= 8
                    and float(audit_payload.get("intent_realism_score", 0) or 0) >= 8
                    and audit_payload.get("is_single_aspect") is True
                    and audit_payload.get("is_properly_anonymized") is True
                ):
                    return query
        return ""

    def _photo_search(self, query: str, active_ids: list[str]) -> list[dict[str, Any]]:
        if self.model is not None and self.model.visual_embedding is not None:
            vector = _embed_visual_text(self.model.visual_embedding, query)
            rows = list(self.provider.search_photo_embeddings(vector, business_ids=active_ids, top_k=self.settings.top_k))
        else:
            rows = self.provider.search_photos(query, active_ids, self.settings.top_k)

        photos = []
        for row in rows:
            score = float(row.get("score", row.get("_score", 0)) or 0)
            if score <= self.settings.image_coarse_thres:
                continue
            photos.append(
                {
                    "business_id": str(row.get("business_id", "")),
                    "path": str(row.get("path") or row.get("source_locator") or row.get("photo_id", "")),
                    "caption": str(row.get("caption") or row.get("summary") or ""),
                    "score": score,
                    "source_locator_type": row.get("source_locator_type", "yelp_photo_id"),
                    "source_locator": row.get("source_locator", row.get("photo_id", row.get("path", ""))),
                }
            )
        return self._rerank_photos(query, photos)

    def _rerank_photos(self, query: str, photos: list[dict[str, Any]]) -> list[dict[str, Any]]:
        visual_reranker = self.model.visual_reranker if self.model is not None else None
        if visual_reranker is not None:
            if not hasattr(visual_reranker, "select_by_rerank_score"):
                raise ValueError("Configured visual_reranker must expose select_by_rerank_score(...).")
            docs = [
                {
                    "image": photo["path"],
                    "bid": photo["business_id"],
                    "vector_score": photo["score"],
                    **photo,
                }
                for photo in photos
            ]
            reranked = list(
                visual_reranker.select_by_rerank_score(
                    query=query,
                    documents_dict=docs,
                    thres=self.settings.image_reranker_thres,
                    top_n=None,
                )
            )
            output = []
            for row in reranked:
                item = dict(row)
                item["business_id"] = str(row.get("business_id") or row.get("bid", ""))
                item["path"] = str(row.get("path") or row.get("image", ""))
                item["score"] = float(row.get("vector_score", row.get("score", 0)) or 0)
                item["rerank_score"] = float(row.get("rerank_score", 0) or 0)
                output.append(item)
            return output
        output = []
        for photo in photos:
            score = float(photo.get("score", 0) or 0)
            if score >= self.settings.image_reranker_thres:
                item = dict(photo)
                item["rerank_score"] = score
                output.append(item)
        return output

    def _cross_review_search(self, query: str, active_ids: list[str]) -> list[dict[str, Any]]:
        embedding_model = self.model.embedding if self.model is not None else None
        if embedding_model is not None:
            vector = embedding_model.embed([query])["embeddings"][0] if embedding_model is not None else None
            rows = list(self.provider.hybrid_search_reviews(query, vector, active_ids, top_k=self.settings.top_k))
        else:
            rows = self.provider.search_reviews(query, active_ids, self.settings.top_k)
        docs = [
            {
                "business_id": str(row.get("business_id", "")),
                "text": str(row.get("text") or row.get("summary") or ""),
                "coarse_score": float(row.get("coarse_score", row.get("_score", row.get("score", 0))) or 0),
                "source_locator_type": row.get("source_locator_type", "yelp_review_id"),
                "source_locator": row.get("source_locator", row.get("review_id", "")),
            }
            for row in rows
            if row.get("business_id")
        ]
        return self._rerank_review_docs(query, docs)

    def _rerank_review_docs(self, query: str, docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        reranker = self.model.reranker if self.model is not None else None
        if reranker is not None:
            if not hasattr(reranker, "select_by_rerank_score"):
                raise ValueError("Configured reranker must expose select_by_rerank_score(...).")
            return list(
                reranker.select_by_rerank_score(
                    query=query,
                    documents_dict=docs,
                    thres=self.settings.text_rerank_thres,
                )
            )
        output = []
        for doc in docs:
            score = float(doc.get("coarse_score", doc.get("score", 0)) or 0)
            if score >= self.settings.text_rerank_thres:
                item = dict(doc)
                item["rerank_score"] = score
                output.append(item)
        return output

    def _verify_photos(self, photos: list[dict[str, Any]]) -> None:
        active = {candidate.business_id: candidate for candidate in self.ctx.active_candidates()}
        per_business: dict[str, int] = defaultdict(int)
        for photo in photos:
            business_id = str(photo.get("business_id", ""))
            candidate = active.get(business_id)
            if not candidate or per_business[business_id] >= 3:
                continue
            per_business[business_id] += 1
            path = str(photo.get("path") or photo.get("source_locator") or "")
            caption = str(photo.get("caption") or photo.get("summary") or "")
            if self.model is None:
                payload = _keyword_payload(self.ctx.image_query, caption)
                trace = LLMCallTrace(stage=f"image_self_verify_{business_id}", image_paths=[path])
            else:
                payload, trace = _call_json_model(
                    self.model,
                    VLM_JUDGE_PROMPT.format(query=self.ctx.image_query),
                    image=path,
                    stage=f"image_self_verify_{business_id}",
                )
            result = _support_result(
                payload,
                trace,
                locator_type=str(photo.get("source_locator_type", "yelp_photo_id")),
                locator=str(photo.get("source_locator", path)),
                summary=caption[:500] or "Photo evidence checked by visual verifier.",
                threshold=self.settings.llm_judge_threshold,
            )
            if result.is_passed(self.settings.llm_judge_threshold):
                current = candidate.image_verify
                if current is None or result.confidence > current.confidence:
                    candidate.set_verification("image_verify", result, "image")
                    candidate.scores["image_coarse"] = float(photo.get("score", 0) or 0)
                    candidate.scores["image_rerank"] = float(photo.get("rerank_score", 0) or 0)
                    candidate.scores["img_evidence_path"] = str(photo.get("source_locator", path))
            elif candidate.drop_reason is None:
                candidate.drop_reason = f"IMG_JUDGE_NO (Conf: {result.confidence:.2f})"
            self.ctx.global_traces.append(trace)

    def _verify_reviews(self, docs: list[dict[str, Any]]) -> None:
        active = {candidate.business_id: candidate for candidate in self.ctx.active_candidates()}
        per_business: dict[str, int] = defaultdict(int)
        for doc in docs:
            business_id = str(doc.get("business_id", ""))
            candidate = active.get(business_id)
            if not candidate or per_business[business_id] >= 2:
                continue
            per_business[business_id] += 1
            review_text = str(doc.get("text") or "")
            if self.model is None:
                payload = _keyword_payload(self.ctx.image_query, review_text)
                trace = LLMCallTrace(stage=f"image_to_text_cross_verify_{business_id}")
            else:
                payload, trace = _call_json_model(
                    self.model,
                    TEXT_JUDGE_PROMPT.format(input_review=review_text, input_query=self.ctx.image_query),
                    stage=f"image_to_text_cross_verify_{business_id}",
                )
            result = _support_result(
                payload,
                trace,
                locator_type=str(doc.get("source_locator_type", "yelp_review_id")),
                locator=str(doc.get("source_locator", "")),
                summary=review_text[:500],
                threshold=self.settings.llm_judge_threshold,
            )
            if result.is_passed(self.settings.llm_judge_threshold):
                current = candidate.image_to_text_verify
                if current is None or result.confidence > current.confidence:
                    candidate.set_verification("image_to_text_verify", result, "cross_modal")
                    candidate.scores["i2t_text_rerank"] = float(doc.get("rerank_score", doc.get("coarse_score", 0)) or 0)
                    candidate.scores["i2t_text_coarse"] = float(doc.get("coarse_score", 0) or 0)
            elif candidate.drop_reason is None:
                candidate.drop_reason = f"I2T_JUDGE_NO (Conf: {result.confidence:.2f})"
            self.ctx.global_traces.append(trace)
