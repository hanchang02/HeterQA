"""Public construction orchestrator for the HeterQA data-generation flow."""

from __future__ import annotations

from pathlib import Path

from heterqa.construction.mainline import HeterQAConstructionMainline, run_mainline_from_config

__all__ = ["HeterQAConstructionMainline", "run_mainline_from_config", "run_orchestrator"]


def run_orchestrator(config_path: Path, output_dir: Path) -> Path:
    return run_mainline_from_config(config_path, output_dir)

