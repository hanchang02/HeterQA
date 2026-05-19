"""Knowledge-graph evidence stage with topology/text/image union semantics."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from heterqa.construction.contracts import ConstructionSettings, LLMCallTrace, PipelineContext, VerificationResult
from heterqa.construction.providers import ConstructionDataProvider
from heterqa.construction.text_engine import TextSearchTask


@dataclass
class FeatureConstraint:
    name: str
    sentiment: str
    logic: str
    source: str

    def display(self) -> str:
        prefix = "+" if self.logic == "require" else "avoid "
        return f"{prefix}{self.name}({self.sentiment})"

    def __str__(self) -> str:
        return self.display()


KG_QUERY_PROMPT = """
You are a Knowledge Graph intent generator.
We have identified a target group of venues based on specific feature constraints derived from graph analysis.

Context:
- Analysis path: {path_type} ({explanation})
- Target constraints:
{constraints}

Task:
Generate a specific, natural-language search query that a user would ask to find these venues.

Constraint semantics:
- Logic "require" with positive sentiment means the user explicitly wants this feature.
- Logic "avoid" with negative sentiment means the user explicitly wants to avoid this flaw, such as long waits or poor cleanliness.
- Collaborative constraints describe venues known for bridge features and popular with users who also like trait features.

Rules:
1) Preserve every graph-derived requirement unless it is redundant.
2) Use natural search language rather than graph terminology.
3) Do not mention internal path names, feature IDs, node IDs, graph mining, or topology.
4) Keep the query concise and actionable.
5) Output JSON only.

Output:
{{"query": "..."}}
"""


class KgSearchTask:
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
        self.graph = getattr(provider, "graph", None) or provider

    def execute(self) -> None:
        if not self.settings.enabled_kg or not self.ctx.candidates:
            return
        active = self.ctx.active_candidates()
        if not active:
            return
        state = self._mine_graph_state(active)
        if state:
            self._save_state(state)
            self._refine_by_topology(active, state)
        self.ctx.kg_query = self.query or self.ctx.kg_query or self._generate_query(state)
        if self.ctx.kg_query:
            text_branch = TextSearchTask(
                self.ctx,
                self.provider,
                self.settings,
                model=self.model,
                query=self.ctx.kg_query,
            )
            # Use TextSearchTask internals for KG multimodal evidence, but do not
            # apply text-stage gating here; KG has its own final union below.
            active_ids = [candidate.business_id for candidate in active]
            review_hits = text_branch._dense_sparse_search(self.ctx.kg_query, active_ids)
            if review_hits:
                text_branch._verify_reviews(review_hits)
            photo_hits = text_branch._cross_photo_search(self.ctx.kg_query, active_ids)
            if photo_hits:
                text_branch._verify_images(photo_hits)
        survivors = 0
        dropped = 0
        for candidate in active:
            has_topology = candidate.scores.get("kg_match", 0) == 1.0
            has_text = candidate.text_verify is not None
            has_image = candidate.text_to_image_verify is not None
            if has_topology or has_text or has_image:
                survivors += 1
                audit = candidate.metadata.setdefault("kg_audit", {})
                audit["survival_reason"] = []
                if has_topology:
                    audit["survival_reason"].append("topology")
                if has_text:
                    audit["survival_reason"].append("text_evidence")
                if has_image:
                    audit["survival_reason"].append("image_evidence")
            else:
                candidate.drop("KG_Union_Fail: No Topology/Text/Image Evidence")
                dropped += 1
        self.ctx.stats["kg_task_survivors"] = survivors
        self.ctx.stats["kg_task_dropped"] = dropped

    def _mine_graph_state(self, candidates: list[Any]) -> dict[str, Any] | None:
        for _ in range(self.settings.kg_max_retries):
            path_type = random.choice(["attribute", "collaborative"])
            constraints, meta, decision_log = self._mine_constraints(candidates, path_type)
            if not constraints:
                continue
            survivor_count = sum(1 for candidate in candidates if self._candidate_matches(candidate, constraints, path_type))
            if survivor_count >= self.settings.kg_min_survivors:
                return {
                    "path_type": path_type,
                    "constraints": constraints,
                    "meta_info": meta,
                    "decision_log": {**decision_log, "survivor_count": survivor_count},
                }
        return self._state_from_existing_kg_evidence(candidates)

    def _state_from_existing_kg_evidence(self, candidates: list[Any]) -> dict[str, Any] | None:
        constraints: list[FeatureConstraint] = []
        for candidate in candidates:
            rows = self.provider.get_evidence("kg", candidate.business_id, "", self.settings.evidence_limit)
            for row in rows:
                if row.supports is True:
                    constraints.append(FeatureConstraint(row.summary[:80], "pos", "require", "released_kg_evidence"))
                    break
        if not constraints:
            return None
        return {
            "path_type": "attribute",
            "constraints": constraints[:2],
            "meta_info": {"explanation": "Graph Logic: Existing KG evidence constraints"},
            "decision_log": {"path": "released_kg_evidence", "survivor_count": len(constraints)},
        }

    def _mine_constraints(
        self,
        candidates: list[Any],
        path_type: str,
    ) -> tuple[list[FeatureConstraint], dict[str, Any], dict[str, Any]]:
        if path_type == "attribute":
            return self._mine_attribute_constraints(candidates)
        return self._mine_collaborative_constraints(candidates)

    def _mine_attribute_constraints(
        self,
        candidates: list[Any],
    ) -> tuple[list[FeatureConstraint], dict[str, Any], dict[str, Any]]:
        constraints: list[FeatureConstraint] = []
        mode = random.choice(["intersection", "optimization"])
        decision_log: dict[str, Any] = {"path": "attribute", "strategy": mode, "counts": {}}
        anchor_pool: list[str] = []
        candidate_ids = {candidate.business_id for candidate in candidates}
        for candidate in candidates:
            anchor_pool.extend(list(self.graph.get_features_of_business(candidate.business_id, "pos")))
        anchors = self.graph.sample_features_by_distribution(anchor_pool, top_k=1)
        if not anchors:
            return [], {}, decision_log
        primary = str(anchors[0])
        constraints.append(FeatureConstraint(primary, "pos", "require", "attribute"))
        subset = set(self.graph.get_businesses_by_feature(primary, "pos")) or candidate_ids
        valid_ids = subset & candidate_ids
        explanation = f"Graph Logic: Positive Attribute ({primary})"

        if valid_ids and mode == "intersection":
            secondary_pool: list[str] = []
            for business_id in valid_ids:
                secondary_pool.extend(list(self.graph.get_features_of_business(business_id, "pos")))
            secondary_pool = [feature for feature in secondary_pool if feature != primary]
            secondary = self.graph.sample_features_by_distribution(secondary_pool, top_k=1)
            if secondary:
                feature = str(secondary[0])
                constraints.append(FeatureConstraint(feature, "pos", "require", "attribute"))
                explanation = f"Graph Logic: Intersection of Positive Attributes ({primary} & {feature})"
                decision_log["counts"] = {"pos_attr": 2, "neg_attr": 0}
        elif valid_ids and mode == "optimization":
            negative_pool: list[str] = []
            for business_id in valid_ids:
                negative_pool.extend(list(self.graph.get_features_of_business(business_id, "neg")))
            negative = self.graph.sample_features_by_distribution(negative_pool, top_k=1)
            if negative:
                feature = str(negative[0])
                constraints.append(FeatureConstraint(feature, "neg", "avoid", "attribute"))
                explanation = f"Graph Logic: Optimization (Require {primary}, Avoid {feature})"
                decision_log["counts"] = {"pos_attr": 1, "avoid_neg": 1}

        return constraints, {"explanation": explanation}, decision_log

    def _mine_collaborative_constraints(
        self,
        candidates: list[Any],
    ) -> tuple[list[FeatureConstraint], dict[str, Any], dict[str, Any]]:
        if random.random() < 0.5:
            if random.random() < 0.5:
                strategy, n_bridges, n_traits = "strict_double_bridge", 2, 1
            else:
                strategy, n_bridges, n_traits = "strict_double_trait", 1, 2
        else:
            strategy, n_bridges, n_traits = "base_homophily", 1, 1
        decision_log = {"path": "collaborative", "strategy": strategy, "counts": {"n_bridges": n_bridges, "n_traits": n_traits}}

        anchor_pool: list[str] = []
        for candidate in candidates:
            anchor_pool.extend(list(self.graph.get_features_of_business(candidate.business_id, "pos")))
        bridge_features = [str(item) for item in self.graph.sample_features_by_distribution(anchor_pool, top_k=n_bridges)]
        if len(bridge_features) < n_bridges:
            return [], {}, decision_log

        constraints = [FeatureConstraint(feature, "pos", "require", "shared_bridge") for feature in bridge_features]
        target_ids = {candidate.business_id for candidate in candidates}
        valid_owners = set(target_ids)
        for feature in bridge_features:
            owners = set(self.graph.get_businesses_by_feature(feature, "pos"))
            if owners:
                valid_owners &= owners
        if not valid_owners:
            return [], {}, decision_log

        cohort_users: set[str] = set()
        for business_id in valid_owners:
            user_sets = [
                set(self.graph.get_users_connected_via_feature(business_id, feature, "pos"))
                for feature in bridge_features
            ]
            if user_sets:
                cohort_users.update(set.intersection(*user_sets))
        if not cohort_users:
            return constraints, {
                "explanation": f"Graph Logic: Collaborative Filtering. Venues known for [{', '.join(bridge_features)}] with verified graph bridge features."
            }, decision_log

        if len(cohort_users) > 1000:
            cohort_users = set(random.sample(list(cohort_users), 1000))
        user_feature_pool: list[str] = []
        for user_id in cohort_users:
            user_feature_pool.extend(list(self.graph.get_features_of_user(user_id, sentiment="pos")))
        user_feature_pool = [feature for feature in user_feature_pool if feature not in bridge_features]
        trait_features = [str(item) for item in self.graph.sample_features_by_distribution(user_feature_pool, top_k=n_traits)]
        for feature in trait_features:
            constraints.append(FeatureConstraint(feature, "pos", "require", "user_trait"))
        if trait_features:
            explanation = (
                "Graph Logic: Collaborative Filtering. "
                f"Venues with [{', '.join(bridge_features)}] appealing to users who also love [{', '.join(trait_features)}]."
            )
        else:
            explanation = f"Graph Logic: Collaborative Filtering. Venues known for [{', '.join(bridge_features)}] with verified user clusters."
        return constraints, {"explanation": explanation}, decision_log

    def _candidate_matches(self, candidate: Any, constraints: list[FeatureConstraint], path_type: str) -> bool:
        positive = set(self.graph.get_features_of_business(candidate.business_id, "pos"))
        negative = set(self.graph.get_features_of_business(candidate.business_id, "neg"))
        if path_type == "collaborative":
            bridges = [constraint for constraint in constraints if constraint.source == "shared_bridge"]
            traits = [constraint for constraint in constraints if constraint.source == "user_trait"]
            if not bridges:
                return False
            for bridge in bridges:
                if bridge.name not in positive:
                    return False
            cohort_sets = [
                set(self.graph.get_users_connected_via_feature(candidate.business_id, bridge.name, "pos"))
                for bridge in bridges
            ]
            local_cohort = set.intersection(*cohort_sets) if cohort_sets else set()
            if not local_cohort:
                return False
            if not traits:
                return True
            trait_groups = [set(self.graph.get_users_by_feature(trait.name, trait.sentiment)) for trait in traits]
            trait_lovers = set.intersection(*trait_groups) if trait_groups else set()
            return bool(local_cohort & trait_lovers)
        for constraint in constraints:
            target = positive if constraint.sentiment == "pos" else negative
            if constraint.logic == "require" and constraint.name not in target:
                return False
            if constraint.logic == "avoid" and constraint.name in target:
                return False
        return True

    def _refine_by_topology(self, candidates: list[Any], state: dict[str, Any]) -> None:
        constraints: list[FeatureConstraint] = state["constraints"]
        explanation = state.get("meta_info", {}).get("explanation", "Graph topology analysis")
        passed = 0
        for candidate in candidates:
            if not candidate.is_active:
                continue
            if self._candidate_matches(candidate, constraints, state["path_type"]):
                candidate.scores["kg_match"] = 1.0
                reason = (
                    f"{explanation}\n"
                    f"Matched Constraints: [{', '.join(item.display() for item in constraints)}]"
                )
                candidate.set_verification(
                    "kg_verify",
                    VerificationResult(
                        judgement="yes",
                        confidence=1.0,
                        reason=reason,
                        trace=LLMCallTrace(stage="kg_topology_mining"),
                        evidence_locator_type="graph_feature_constraint",
                        evidence_locator=";".join(item.name for item in constraints),
                        evidence_summary=reason,
                    ),
                    "kg",
                )
                passed += 1
            else:
                candidate.metadata.setdefault("audit_fail_log", []).append("KG_Topo_Fail")
        self.ctx.stats["kg_topology_pass_count"] = passed

    def _generate_query(self, state: dict[str, Any] | None) -> str:
        if not state:
            return ""
        constraints = "\n".join(f"- {item.display()} | source={item.source}" for item in state["constraints"])
        prompt = KG_QUERY_PROMPT.format(
            path_type=state["path_type"],
            explanation=state.get("meta_info", {}).get("explanation", ""),
            constraints=constraints,
        )
        if self.model is None:
            return "Find places matching graph-derived features: " + ", ".join(item.name for item in state["constraints"])
        from heterqa.construction.text_engine import _call_json_model

        payload, trace = _call_json_model(self.model, prompt, stage="kg_query_gen")
        self.ctx.global_traces.append(trace)
        return str(payload.get("query", "")).strip()

    def _save_state(self, state: dict[str, Any]) -> None:
        self.ctx.metadata["kg_resolution_context"] = {
            "mining_path": {"selected_path": state["path_type"]},
            "reasoning": {
                "explanation": state.get("meta_info", {}).get("explanation", ""),
                "active_constraints": [
                    {
                        "name": item.name,
                        "sentiment": item.sentiment,
                        "logic": item.logic,
                        "source": item.source,
                        "display": item.display(),
                    }
                    for item in state["constraints"]
                ],
            },
            "generated_query": self.ctx.kg_query or None,
            "audit_status": "applied",
        }
