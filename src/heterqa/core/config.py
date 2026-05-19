"""Configuration helpers for public HeterQA workflows."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml


def expand_env_values(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: expand_env_values(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_env_values(item) for item in value]
    if not isinstance(value, str):
        return value

    pattern = re.compile(r"\$\{([^}:]+)(?::-([^}]*))?\}")

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        default = match.group(2) or ""
        return os.environ.get(name, default)

    return pattern.sub(replace, value)


def load_yaml_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    return expand_env_values(data)


def require_keys(config: dict[str, Any], keys: list[str]) -> None:
    missing = [key for key in keys if key not in config]
    if missing:
        raise ValueError(f"Missing required configuration keys: {', '.join(missing)}")
