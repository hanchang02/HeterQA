"""Answer-set-first construction mainline for HeterQA."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from heterqa.construction.contracts import (
    ConstructionSettings,
    GenerationCase,
    GenerationResult,
    PipelineContext,
    StageSummary,
)
from heterqa.construction.expansion import ExpansionTask
from heterqa.construction.geo_engine import GeoSearchTask
from heterqa.construction.image_engine import ImageSearchTask
from heterqa.construction.kg_engine import KgSearchTask
from heterqa.construction.providers import ConstructionDataProvider, build_construction_provider
from heterqa.construction.query_generation import QueryComposer
from heterqa.construction.query_validation import validate_query_components
from heterqa.construction.structured_selection import (
    StructuredQuerySelector,
    StructuredSelectionSettings,
)
from heterqa.construction.text_engine import TextSearchTask
from heterqa.core.config import load_yaml_config
from heterqa.core.io import read_jsonl
from heterqa.providers.model_client import build_model_bundle


class HeterQAConstructionMainline:
    """Public data-generation mainline with waterfall candidate semantics."""

    def __init__(
        self,
        provider: ConstructionDataProvider,
        settings: ConstructionSettings,
        *,
        model: Any = None,
    ):
        self.provider = provider
        self.settings = settings
        self.model = model

    def run_case(self, case: GenerationCase) -> GenerationResult:
        ctx = self._new_context(case)
        summaries: list[StageSummary] = []

        expansion = ExpansionTask(
            ctx,
            self.provider,
            self.settings,
            filters=case.structured_filters,
            model=self.model,
        )
        before_expansion = len(ctx.candidates)
        expansion.execute()
        seed_count = len([candidate for candidate in ctx.candidates if candidate.origin == "initial_seed"])
        summaries.append(
            StageSummary(
                "record_field_initialization",
                input_candidates=before_expansion,
                output_candidates=seed_count,
            )
        )
        summaries.append(
            StageSummary(
                "heterogeneous_candidate_recall",
                input_candidates=seed_count,
                output_candidates=len(ctx.active_candidates()),
                dropped_candidates=len([candidate for candidate in ctx.candidates if not candidate.is_active]),
                metadata=expansion.summary(),
            )
        )
        self._run_stage(
            "geo_verification",
            ctx,
            summaries,
            GeoSearchTask(ctx, self.provider, self.settings, constraint=case.geo_constraint).execute,
        )
        self._run_stage(
            "text_verification",
            ctx,
            summaries,
            TextSearchTask(ctx, self.provider, self.settings, model=self.model, query=case.text_query).execute,
        )
        self._run_stage(
            "image_verification",
            ctx,
            summaries,
            ImageSearchTask(ctx, self.provider, self.settings, model=self.model, query=case.image_query).execute,
        )
        self._run_stage(
            "kg_verification",
            ctx,
            summaries,
            KgSearchTask(ctx, self.provider, self.settings, model=self.model, query=case.kg_query).execute,
        )

        composer = QueryComposer(self.model, provider=self.provider)
        final_query = composer.compose(ctx, case)
        query_validation = validate_query_components(
            final_query,
            dict(ctx.metadata.get("query_components") or {}),
            model=None if case.final_query else self.model,
        )
        ctx.metadata["query_validation"] = query_validation
        summaries.append(
            StageSummary(
                "final_query_generation",
                input_candidates=len(ctx.active_candidates()),
                output_candidates=len(ctx.active_candidates()),
                metadata={
                    "is_pass": query_validation["is_pass"],
                    "validation_mode": query_validation["mode"],
                    "missing_constraints": query_validation["missing_constraints"],
                },
            )
        )
        ctx.refined_results = [candidate for candidate in ctx.candidates if candidate.is_active]
        for candidate in ctx.refined_results:
            candidate.is_final_hit = True
            candidate.verdict = "yes"
        final_answer_ids = [candidate.business_id for candidate in ctx.refined_results]
        summaries.append(
            StageSummary(
                "retained_candidate_set",
                input_candidates=len(ctx.candidates),
                output_candidates=len(ctx.refined_results),
                dropped_candidates=len([candidate for candidate in ctx.candidates if not candidate.is_active]),
                metadata={
                    "target_min": case.target_answer_count_min,
                    "target_max": case.target_answer_count_max,
                    "within_target_range": case.target_answer_count_min
                    <= len(final_answer_ids)
                    <= case.target_answer_count_max,
                },
            )
        )
        return GenerationResult(
            case=case,
            candidates=list(ctx.candidates),
            final_answer_business_ids=final_answer_ids,
            stage_summaries=summaries,
            final_query=final_query,
            context=ctx,
        )

    def _new_context(self, case: GenerationCase) -> PipelineContext:
        seed_records = list(case.seed_records)
        if not seed_records:
            for business_id in case.seed_business_ids:
                record = self.provider.get_business(business_id)
                if record:
                    seed_records.append(record.fields)
        ctx = PipelineContext(
            task_id=f"{case.subset or 'case'}_{case.qid or uuid.uuid4().hex[:8]}",
            all_filters=[predicate.to_source_tuple() for predicate in case.structured_filters],
            seed_records=seed_records,
            geo_query=case.geo_constraint.nl_text if case.geo_constraint else "",
            text_query=case.text_query,
            image_query=case.image_query,
            kg_query=case.kg_query,
            final_query=case.final_query,
            metadata={"qid": case.qid, "subset": case.subset, "seed_business_ids": case.seed_business_ids},
        )
        if case.recall_query:
            ctx.metadata["source_recall_query"] = case.recall_query
            ctx.recall_query = case.recall_query
        return ctx

    @staticmethod
    def _run_stage(
        name: str,
        ctx: PipelineContext,
        summaries: list[StageSummary],
        fn: Any,
    ) -> None:
        before = len([candidate for candidate in ctx.candidates if candidate.is_active])
        dropped_before = len([candidate for candidate in ctx.candidates if not candidate.is_active])
        fn()
        after = len([candidate for candidate in ctx.candidates if candidate.is_active])
        dropped_after = len([candidate for candidate in ctx.candidates if not candidate.is_active])
        summaries.append(
            StageSummary(
                name=name,
                input_candidates=before,
                output_candidates=after,
                dropped_candidates=max(0, dropped_after - dropped_before),
            )
        )


def _settings_from_construction(config: dict[str, Any], settings: ConstructionSettings) -> StructuredSelectionSettings:
    raw = dict(config.get("structured_selection", {}))
    return StructuredSelectionSettings(
        compact=bool(raw.get("compact", settings.structured_selection_compact)),
        seeded=bool(raw.get("seeded", settings.structured_selection_seeded)),
        field_selection_mode=str(raw.get("field_selection_mode", settings.field_selection_mode)),
        max_retries=int(raw.get("max_retries", settings.structured_selection_max_retries)),
    )


def _subset_from_settings(settings: ConstructionSettings) -> str:
    enabled = []
    if settings.enabled_geo:
        enabled.append("Geo")
    if settings.enabled_text:
        enabled.append("Text")
    if settings.enabled_image:
        enabled.append("Image")
    if settings.enabled_kg:
        enabled.append("KG")
    return "_".join(enabled) if enabled else "Record_Field"


def _generate_cases(
    config: dict[str, Any],
    provider: ConstructionDataProvider,
    settings: ConstructionSettings,
) -> list[GenerationCase]:
    generation = dict(config.get("case_generation", {}))
    count = int(generation.get("count", 1))
    start_seed = generation.get("start_seed")
    subset = str(generation.get("subset") or _subset_from_settings(settings))
    qid_prefix = str(generation.get("qid_prefix", "generated"))
    selector = StructuredQuerySelector(provider, _settings_from_construction(config, settings))
    cases: list[GenerationCase] = []
    for index in range(count):
        seed = int(start_seed) + index if start_seed is not None else None
        selected = selector.select_fields(seed=seed)
        cases.append(
            GenerationCase(
                qid=f"{qid_prefix}-{index + 1}",
                subset=subset,
                structured_filters=selected.filters,
                seed_records=selected.seed_records,
                target_answer_count_min=settings.target_answer_count_min,
                target_answer_count_max=settings.target_answer_count_max,
                metadata={
                    "structured_selection": selected.metadata,
                    "source_structured_filters": selected.source_filters,
                    "selection_seed": seed,
                },
            )
        )
    return cases


def _load_cases(
    config: dict[str, Any],
    provider: ConstructionDataProvider,
    settings: ConstructionSettings,
) -> list[GenerationCase]:
    if config.get("cases_jsonl"):
        return [GenerationCase.from_raw(row) for row in read_jsonl(Path(config["cases_jsonl"]))]
    if config.get("case"):
        return [GenerationCase.from_raw(config["case"])]
    if config.get("case_generation") is not None:
        return _generate_cases(config, provider, settings)
    raise ValueError("Construction config must define case, cases_jsonl, or case_generation.")


def run_mainline_from_config(config_path: Path, output_dir: Path) -> Path:
    config = load_yaml_config(config_path)
    provider = build_construction_provider(config["data"])
    settings = ConstructionSettings.from_raw(config.get("settings"))
    model = build_model_bundle(config.get("model"))
    pipeline = HeterQAConstructionMainline(provider, settings, model=model)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = [pipeline.run_case(case) for case in _load_cases(config, provider, settings)]
    output_path = output_dir / "construction_cases.jsonl"
    with output_path.open("w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")

    summary_path = output_dir / "generation_summary.json"
    summary = {
        "case_count": len(results),
        "final_answer_pair_count": sum(len(result.final_answer_business_ids) for result in results),
        "cases": [
            {
                "qid": result.case.qid,
                "subset": result.case.subset,
                "final_answer_count": len(result.final_answer_business_ids),
                "stage_count": len(result.stage_summaries),
            }
            for result in results
        ],
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_path
