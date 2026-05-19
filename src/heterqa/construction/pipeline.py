"""Answer-set-first construction entrypoint."""

from __future__ import annotations

from pathlib import Path

from heterqa.construction.mainline import run_mainline_from_config
from heterqa.core.config import load_yaml_config


def run_construction(config_path: Path, output_dir: Path) -> Path:
    config = load_yaml_config(config_path)
    mode = config.get("mode", "mainline")
    if mode != "mainline":
        raise ValueError("Public construction entrypoint supports mode=mainline.")
    return run_mainline_from_config(config_path, output_dir)
