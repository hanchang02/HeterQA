"""Command line interface for the HeterQA data-generation pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from heterqa.audit.contradiction_detection import run_contradiction_detection
from heterqa.construction.pipeline import run_construction
from heterqa.data.yelp_oceanbase import add_data_cli, run_data_cli
from heterqa.finalize.certify_answer_set import certify_answer_set
from heterqa.graph.build import write_graph_feature_index
from heterqa.graph.canonicalization import canonicalize_feature_file
from heterqa.graph.feature_extraction import extract_feature_file
from heterqa.quality.metrics import human_rating_summary, query_metrics
from heterqa.release.export import export_hf_release
from heterqa.release.validate import validate_release, write_validation_report
from heterqa.review.manual_review import apply_manual_review, collect_manual_decisions, export_manual_review
from heterqa.core.config import load_yaml_config
from heterqa.providers.model_client import build_model_bundle


def _print_json(payload: object) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="heterqa")
    sub = parser.add_subparsers(dest="command", required=True)
    add_data_cli(sub)

    p_construct = sub.add_parser("construct")
    construct_sub = p_construct.add_subparsers(dest="construct_command", required=True)
    p_construct_run = construct_sub.add_parser("run")
    p_construct_run.add_argument("--config", required=True, type=Path)
    p_construct_run.add_argument("--output", required=True, type=Path)

    p_certify = sub.add_parser("certify")
    certify_sub = p_certify.add_subparsers(dest="certify_command", required=True)
    p_detect = certify_sub.add_parser("contradiction-detect")
    p_detect.add_argument("--config", required=False, type=Path)
    p_detect.add_argument("--input", required=True, type=Path)
    p_detect.add_argument("--output", required=True, type=Path)
    p_answer = certify_sub.add_parser("answer-set")
    p_answer.add_argument("--input", required=True, type=Path)
    p_answer.add_argument("--output", required=True, type=Path)
    p_answer.add_argument("--instance-index", required=False, type=Path)
    p_answer.add_argument("--expected-count", required=False, type=int)

    p_graph = sub.add_parser("graph")
    graph_sub = p_graph.add_subparsers(dest="graph_command", required=True)
    p_graph_extract = graph_sub.add_parser("extract-features")
    p_graph_extract.add_argument("--input", required=True, type=Path)
    p_graph_extract.add_argument("--output", required=True, type=Path)
    p_graph_extract.add_argument("--config", required=False, type=Path)
    p_graph_extract.add_argument("--text-field", default="text")
    p_graph_extract.add_argument("--source-locator-type", default="yelp_review_id")
    p_graph_extract.add_argument("--max-features-per-text", type=int, default=5)
    p_graph_canon = graph_sub.add_parser("canonicalize")
    p_graph_canon.add_argument("--input", required=True, type=Path)
    p_graph_canon.add_argument("--output", required=True, type=Path)
    p_graph_build = graph_sub.add_parser("build")
    p_graph_build.add_argument("--input", required=True, type=Path)
    p_graph_build.add_argument("--output-dir", required=True, type=Path)

    p_review = sub.add_parser("review")
    review_sub = p_review.add_subparsers(dest="review_command", required=True)
    p_review_export = review_sub.add_parser("export")
    p_review_export.add_argument("--input", required=True, type=Path)
    p_review_export.add_argument("--output", required=True, type=Path)
    p_collect_decisions = review_sub.add_parser("collect-decisions")
    p_collect_decisions.add_argument("--review-dir", required=True, type=Path)
    p_collect_decisions.add_argument("--output-csv", required=False, type=Path)
    p_apply = review_sub.add_parser("apply")
    p_apply.add_argument("--review-dir", required=True, type=Path)
    p_apply.add_argument("--input", required=True, type=Path)
    p_apply.add_argument("--output", required=True, type=Path)

    p_quality = sub.add_parser("quality")
    quality_sub = p_quality.add_subparsers(dest="quality_command", required=True)
    p_query_metrics = quality_sub.add_parser("query-metrics")
    p_query_metrics.add_argument("--input", required=True, type=Path)
    p_query_metrics.add_argument("--output", required=True, type=Path)
    p_human_summary = quality_sub.add_parser("human-summary")
    p_human_summary.add_argument("--ratings", required=True, type=Path)
    p_human_summary.add_argument("--queries", required=True, type=Path)
    p_human_summary.add_argument("--output", required=True, type=Path)

    p_release = sub.add_parser("release")
    release_sub = p_release.add_subparsers(dest="release_command", required=True)
    p_export = release_sub.add_parser("export-hf")
    p_export.add_argument("--input", required=True, type=Path)
    p_export.add_argument("--output", required=True, type=Path)
    p_export.add_argument("--skip-validation", action="store_true")
    p_validate = release_sub.add_parser("validate")
    p_validate.add_argument("--dataset-dir", required=True, type=Path)
    p_validate.add_argument("--report-output", required=False, type=Path)

    args = parser.parse_args(argv)

    if args.command == "data":
        _print_json(run_data_cli(args))
        return 0
    if args.command == "construct" and args.construct_command == "run":
        path = run_construction(args.config, args.output)
        _print_json({"output": str(path)})
        return 0
    if args.command == "certify" and args.certify_command == "contradiction-detect":
        _print_json({"output": str(run_contradiction_detection(args.input, args.output, args.config))})
        return 0
    if args.command == "certify" and args.certify_command == "answer-set":
        _print_json(
            {
                "output": str(
                    certify_answer_set(
                        args.input,
                        args.output,
                        instance_index=args.instance_index,
                        expected_count=args.expected_count,
                    )
                )
            }
        )
        return 0
    if args.command == "graph" and args.graph_command == "extract-features":
        config = load_yaml_config(args.config) if args.config else {}
        model = build_model_bundle(config.get("model"))
        _print_json(
            {
                "output": str(
                    extract_feature_file(
                        args.input,
                        args.output,
                        model=model,
                        text_field=args.text_field,
                        source_locator_type=args.source_locator_type,
                        max_features_per_text=args.max_features_per_text,
                    )
                )
            }
        )
        return 0
    if args.command == "graph" and args.graph_command == "canonicalize":
        _print_json({"output": str(canonicalize_feature_file(args.input, args.output))})
        return 0
    if args.command == "graph" and args.graph_command == "build":
        _print_json({"outputs": {key: str(value) for key, value in write_graph_feature_index(args.input, args.output_dir).items()}})
        return 0
    if args.command == "review" and args.review_command == "export":
        _print_json({"output": str(export_manual_review(args.input, args.output))})
        return 0
    if args.command == "review" and args.review_command == "collect-decisions":
        _print_json({"output": str(collect_manual_decisions(args.review_dir, args.output_csv))})
        return 0
    if args.command == "review" and args.review_command == "apply":
        _print_json({"output": str(apply_manual_review(args.review_dir, args.input, args.output))})
        return 0
    if args.command == "quality" and args.quality_command == "query-metrics":
        _print_json({"output": str(query_metrics(args.input, args.output))})
        return 0
    if args.command == "quality" and args.quality_command == "human-summary":
        _print_json({"output": str(human_rating_summary(args.ratings, args.queries, args.output))})
        return 0
    if args.command == "release" and args.release_command == "export-hf":
        hashes = export_hf_release(
            args.input,
            args.output,
            validate=not args.skip_validation,
        )
        _print_json({"files": hashes})
        return 0
    if args.command == "release" and args.release_command == "validate":
        report = validate_release(args.dataset_dir)
        if args.report_output:
            write_validation_report(args.report_output, report)
        print(report.to_markdown())
        return 0 if report.ok else 1
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
