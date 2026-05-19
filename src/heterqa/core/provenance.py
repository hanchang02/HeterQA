"""Provenance helpers for generated workflow artifacts."""

from __future__ import annotations

from datetime import datetime, timezone


def provenance_event(stage: str, method: str) -> dict[str, str]:
    return {
        "stage": stage,
        "method": method,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }

