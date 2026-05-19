"""Text evidence stage with answer-set-first union gating semantics."""

from __future__ import annotations

import json
import random
import re
import time
from collections import defaultdict
from typing import Any

from heterqa.construction.contracts import (
    ConstructionSettings,
    LLMCallTrace,
    PipelineContext,
    VerificationResult,
)
from heterqa.construction.providers import ConstructionDataProvider
from heterqa.construction.record_fields import _parse_json_response
from heterqa.providers.model_client import embed_visual_text_query


TEXT_QUERY_PROMPT = """
You are "Theme Analyst" and must output valid JSON only.

Your job:
- Read all user reviews.
- Identify the single most dominant, recurring theme.
- Produce a short, actionable request for recommendation based on that theme.

Hard rules:
- Output one JSON object only. No extra text, Markdown, headings, code fences, explanations, or summaries.
- The first character of your reply must be `{{` and the last character must be `}}`.
- Do not include fields other than the schema below.
- Forbidden phrases anywhere in output: "Summary", "Insights", "Sentiment", "Recommendation", "Based on", "I will", "I shall", "I can", "Here is".
- Avoid characters that could interfere with SQL or code execution, including semicolons, backslashes, and unnecessary quotation marks inside string values.
- If the reviews are insufficient to infer a dominant theme, return `{{"request": ""}}`.

Schema:
Return a JSON object with exactly one key:
- "request": string
  - Language: match the input reviews' language; if mixed, use English.
  - Content: short, clear, practical user request for recommendation that directly reflects the dominant theme.
  - No information not grounded in the reviews.
  - Keep it under 140 characters.

Examples:

Input:
[
  "The dry-aged ribeye was an absolute masterpiece, perfectly seared and bursting with flavor. Worth every penny. The wine pairing was spot on."
]
Output:
{{"request": "Recommend high-end steakhouses known for perfect dry-aged ribeye and expert wine pairings."}}

Input:
[
  "The blueberry tart is flaky, buttery, and heavenly. I dream about it. Best paired with a simple Americano."
]
Output:
{{"request": "Find places with great blueberry tarts and good Americano coffee."}}

Real Case:
Input:
{review_input}

Output:
"""

TEXT_JUDGE_PROMPT = """
You are an expert semantic relevance analyzer specialized in business reviews and user queries.
Your task is to determine whether a given user review is relevant to a specific query.
Relevance is defined as the review containing information that directly addresses the intent, context, or key entities such as products, services, or attributes mentioned in the query.

Guidelines:
1. Focus on semantic meaning rather than keyword matching.
2. Ignore irrelevant details in the review, including general descriptions unrelated to the query.
3. Consider the query's intent: recommendations, complaints, comparisons, or factual inquiries.
4. Output a JSON object containing:
   - "judgement": "Yes" or "No" indicating relevance.
   - "confidence": a decimal score between 0 and 1.
   - "reason": a concise explanation.
5. Confidence score should reflect:
   - 0.9-1.0: clear and direct relevance or irrelevance.
   - 0.7-0.89: strong evidence but some ambiguity.
   - 0.5-0.69: moderate relevance with significant ambiguity.
   - below 0.5: weak or unclear relevance.
6. Output only the JSON object with no additional explanations, headers, or text.

Example 1:
Input-Review: The hotel's pool was closed for maintenance, and the staff did not inform us at check-in. The room was clean but the disappointment ruined our stay.
Input-Query: Find hotels with well-maintained pools and responsive staff.
Output: {{"judgement": "No", "confidence": 0.95, "reason": "The review contradicts the pool and staff requirement."}}

Example 2:
Input-Review: The delivery was fast, and the packaging was eco-friendly. The product itself worked as described, but the instructions were unclear.
Input-Query: Recommend brands with sustainable packaging practices.
Output: {{"judgement": "Yes", "confidence": 0.85, "reason": "The review directly supports sustainable packaging."}}

Real Case:
Input-Review: {input_review}
Input-Query: {input_query}
Output:
"""

VLM_JUDGE_PROMPT = """
You are a helpful visual evaluator. Your task is to determine whether the information visible in the provided image can be used to answer the given text query.

Input:
- A text query from the user.
- An image.

Your job:
1. Carefully analyze the image and the query.
2. Decide if the visual content in the image is relevant and can potentially be used to answer the query.
3. Output a confidence score between 0 and 1:
   - 0 means the image is completely irrelevant and cannot help answer the query.
   - 1 means the image is highly relevant and likely contains information that can directly answer the query.
4. Output only JSON in this structure:
{{
  "can_answer": true,
  "confidence": 0.0,
  "reason": "concise explanation"
}}

Text query:
{query}
"""


def _call_json_model(model: Any, prompt: str, *, image: str | None = None, stage: str = "model_call") -> tuple[dict[str, Any], LLMCallTrace]:
    trace = LLMCallTrace(stage=stage, prompt=prompt, image_paths=[image] if image else [])
    start = time.time()
    target = getattr(model, "visual_judge", None) if image else getattr(model, "semantic_judge", None)
    if target is None:
        raise ValueError("A visual_judge is required for image calls." if image else "A semantic_judge is required for text calls.")
    if not hasattr(target, "ask_json"):
        raise ValueError("Configured model component must expose ask_json(prompt=..., image=..., temperature=..., max_tokens=...).")
    raw = target.ask_json(prompt=prompt, image=image, max_tokens=2000, temperature=0.7)
    if isinstance(raw, dict) and "json_text" in raw:
        trace.response = str(raw.get("json_text", ""))
        trace.token_usage = raw.get("metadata", {}).get("token_usage", {})
        payload = _parse_json_response(trace.response)
    else:
        trace.response = json.dumps(raw, ensure_ascii=False) if isinstance(raw, dict) else str(raw)
        payload = _parse_json_response(raw)
    trace.is_json_valid = bool(payload)
    trace.latency_ms = (time.time() - start) * 1000
    return payload, trace


def _support_result(
    payload: dict[str, Any],
    trace: LLMCallTrace,
    *,
    locator_type: str,
    locator: str,
    summary: str,
    threshold: float,
) -> VerificationResult:
    judgement = str(payload.get("judgement") or payload.get("can_answer") or payload.get("supports") or "").lower()
    try:
        confidence = float(payload.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    passed = judgement in {"yes", "true", "pass", "supports"} and confidence >= threshold
    return VerificationResult(
        judgement="yes" if passed else "no",
        confidence=confidence,
        reason=str(payload.get("reason", "")),
        trace=trace,
        evidence_locator_type=locator_type,
        evidence_locator=locator,
        evidence_summary=summary,
    )


def _keyword_payload(query: str, text: str) -> dict[str, Any]:
    query_tokens = {token for token in re.findall(r"[a-z0-9]+", query.lower()) if len(token) > 2}
    text_tokens = set(re.findall(r"[a-z0-9]+", text.lower()))
    overlap = len(query_tokens & text_tokens) / len(query_tokens) if query_tokens else 0.0
    return {
        "judgement": "yes" if overlap > 0 else "no",
        "confidence": overlap,
        "reason": "Lexical support score computed because no verifier model was configured.",
    }


class TextSearchTask:
    """Generate text query, verify reviews, verify images, then apply OR gating."""

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
        if not self.settings.enabled_text or not self.ctx.candidates:
            return
        self.ctx.text_query = self.query or self.ctx.text_query or self._generate_query()
        if not self.ctx.text_query:
            self.ctx.stats["text_task_skipped"] = "no_text_query_generated"
            return
        active = self.ctx.active_candidates()
        active_ids = [candidate.business_id for candidate in active]
        review_hits = self._dense_sparse_search(self.ctx.text_query, active_ids)
        if review_hits:
            self._verify_reviews(review_hits)
        photo_hits = self._cross_photo_search(self.ctx.text_query, active_ids)
        if photo_hits:
            self._verify_images(photo_hits)
        survivors = 0
        dropped = 0
        for candidate in active:
            if candidate.text_verify is None and candidate.text_to_image_verify is None:
                candidate.drop("text_modal_all_evidence_failed")
                dropped += 1
            else:
                survivors += 1
        self.ctx.stats["text_task_survivors"] = survivors
        self.ctx.stats["text_task_dropped"] = dropped

    def _generate_query(self) -> str:
        seed_ids = [candidate.business_id for candidate in self.ctx.candidates if candidate.origin == "initial_seed" and candidate.is_active]
        random.shuffle(seed_ids)
        for business_id in seed_ids:
            reviews = self.provider.get_reviews(business_id)
            if not reviews:
                continue
            review = random.choice(reviews)
            review_text = str(review.get("text") or review.get("summary") or "")
            if not review_text:
                continue
            if self.model is None:
                return review_text[:240]
            payload, trace = _call_json_model(
                self.model,
                TEXT_QUERY_PROMPT.format(review_input=review_text),
                stage="text_verification_query_generation",
            )
            self.ctx.global_traces.append(trace)
            request = str(payload.get("request", "")).strip()
            if request:
                return request
        return ""

    def _dense_sparse_search(self, query: str, active_ids: list[str]) -> list[dict[str, Any]]:
        """Hybrid text evidence retrieval over active candidates.

        Production providers can expose database-side hybrid RRF retrieval via
        hybrid_search_reviews(). File-backed providers use review search plus
        the same reranker gate.
        """

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
        return self._rerank_text_docs(query, docs)

    def _rerank_text_docs(self, query: str, docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        reranker = self.model.reranker if self.model is not None else None
        if reranker is not None:
            if not hasattr(reranker, "select_by_rerank_score"):
                raise ValueError("Configured reranker must expose select_by_rerank_score(...).")
            reranked = list(
                reranker.select_by_rerank_score(
                    query=query,
                    documents_dict=docs,
                    thres=self.settings.text_rerank_thres,
                )
            )
            return reranked
        output = []
        for doc in docs:
            score = float(doc.get("coarse_score", doc.get("score", 0)) or 0)
            if score >= self.settings.text_rerank_thres:
                item = dict(doc)
                item["rerank_score"] = score
                output.append(item)
        return output

    def _cross_photo_search(self, query: str, active_ids: list[str]) -> list[dict[str, Any]]:
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
        return self._rerank_photo_docs(query, photos)

    def _rerank_photo_docs(self, query: str, photos: list[dict[str, Any]]) -> list[dict[str, Any]]:
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

    def _verify_reviews(self, docs: list[dict[str, Any]]) -> None:
        active = {candidate.business_id: candidate for candidate in self.ctx.active_candidates()}
        quota_docs = []
        per_business: dict[str, int] = defaultdict(int)
        for doc in docs:
            business_id = str(doc.get("business_id", ""))
            if business_id not in active or per_business[business_id] >= 2:
                continue
            per_business[business_id] += 1
            quota_docs.append(doc)
        payloads = self._judge_reviews(quota_docs)
        for doc, payload, trace in payloads:
            business_id = str(doc.get("business_id", ""))
            candidate = active.get(business_id)
            if not candidate:
                continue
            review_text = str(doc.get("text") or "")
            result = _support_result(
                payload,
                trace,
                locator_type=str(doc.get("source_locator_type", "yelp_review_id")),
                locator=str(doc.get("source_locator", "")),
                summary=review_text[:500],
                threshold=self.settings.llm_judge_threshold,
            )
            if result.is_passed(self.settings.llm_judge_threshold):
                current = candidate.text_verify
                if current is None or result.confidence > current.confidence:
                    candidate.set_verification("text_verify", result, "text")
                    candidate.scores["text_rerank"] = float(doc.get("rerank_score", doc.get("score", doc.get("coarse_score", 0))) or 0)
                    candidate.scores["text_coarse"] = float(doc.get("coarse_score", 0) or 0)
            elif candidate.drop_reason is None:
                candidate.drop_reason = f"LLM_JUDGE_NO (Conf: {result.confidence:.2f})"
            self.ctx.global_traces.append(trace)

    def _judge_reviews(self, docs: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any], LLMCallTrace]]:
        if self.model is None:
            return [
                (
                    doc,
                    _keyword_payload(self.ctx.text_query, str(doc.get("text") or "")),
                    LLMCallTrace(stage=f"text_verify_{doc.get('business_id', '')}"),
                )
                for doc in docs
            ]
        semantic_judge = self.model.semantic_judge
        if hasattr(semantic_judge, "parallel_ask_json"):
            tasks = []
            prompts = []
            for doc in docs:
                prompt = TEXT_JUDGE_PROMPT.format(input_review=str(doc.get("text") or ""), input_query=self.ctx.text_query)
                prompts.append(prompt)
                tasks.append(
                    {
                        "bid": str(doc.get("business_id", "")),
                        "prompt": prompt,
                        "rerank_score": doc.get("rerank_score", 0),
                        "coarse_score": doc.get("coarse_score", 0),
                        "max_tokens": 2000,
                    }
                )
            start = time.time()
            outputs = semantic_judge.parallel_ask_json(tasks)
            latency = (time.time() - start) * 1000 / max(1, len(tasks))
            results = []
            for doc, prompt, output in zip(docs, prompts, outputs, strict=False):
                trace = LLMCallTrace(stage=f"text_verify_{doc.get('business_id', '')}", prompt=prompt, latency_ms=latency)
                metadata = output.get("metadata", {}) if isinstance(output, dict) else {}
                trace.response = str(output.get("json_text", output) if isinstance(output, dict) else output)
                trace.token_usage = metadata.get("token_usage", {})
                trace.error_msg = str(metadata.get("error")) if metadata.get("error") else None
                trace.is_json_valid = not bool(trace.error_msg)
                results.append((doc, _parse_json_response(trace.response), trace))
            return results
        results = []
        for doc in docs:
            payload, trace = _call_json_model(
                self.model,
                TEXT_JUDGE_PROMPT.format(input_review=str(doc.get("text") or ""), input_query=self.ctx.text_query),
                stage=f"text_verify_{doc.get('business_id', '')}",
            )
            results.append((doc, payload, trace))
        return results

    def _verify_images(self, photos: list[dict[str, Any]]) -> None:
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
                payload = _keyword_payload(self.ctx.text_query, caption)
                trace = LLMCallTrace(stage=f"text_to_image_cross_verify_{business_id}", image_paths=[path])
            else:
                payload, trace = _call_json_model(
                    self.model,
                    VLM_JUDGE_PROMPT.format(query=self.ctx.text_query),
                    image=path,
                    stage=f"text_to_image_cross_verify_{business_id}",
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
                current = candidate.text_to_image_verify
                if current is None or result.confidence > current.confidence:
                    candidate.set_verification("text_to_image_verify", result, "cross_modal")
                    candidate.scores["t2i_image_cos"] = float(photo.get("score", 0) or 0)
                    candidate.scores["t2i_image_rerank"] = float(photo.get("rerank_score", 0) or 0)
                    candidate.scores["t2i_evidence_path"] = str(photo.get("source_locator", path))
            elif candidate.drop_reason is None:
                candidate.drop_reason = f"T2I_JUDGE_NO (Conf: {result.confidence:.2f})"
            self.ctx.global_traces.append(trace)


def _embed_visual_text(vl_embedding_model: Any, text: str) -> Any:
    return embed_visual_text_query(vl_embedding_model, text)
