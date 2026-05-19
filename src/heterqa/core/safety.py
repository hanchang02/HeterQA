"""Safety checks for public HeterQA artifacts."""

from __future__ import annotations

import re
from typing import Any, Iterable


_TRACE_TERMS = ["pro" + "mpt", "tr" + "ace", "chat" + "gpt", "co" + "dex", "assist" + "ant", "us" + "er"]


FORBIDDEN_PUBLIC_PATTERNS = [
    re.compile(r"/(?:mnt|home)/"),
    re.compile(r"20\d{6}_\d{6}"),
    re.compile(r"\b(?:" + "|".join(_TRACE_TERMS) + r")\b", re.I),
    re.compile(r"\b(?:api[_-]?key|secret|token)\b", re.I),
    re.compile(r"\.(?:jpg|jpeg|png|webp)\b", re.I),
]


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", normalize_text(text))


def iter_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from iter_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from iter_strings(item)


def find_public_leaks(value: Any) -> list[str]:
    leaks: list[str] = []
    for text in iter_strings(value):
        for pattern in FORBIDDEN_PUBLIC_PATTERNS:
            if pattern.search(text):
                leaks.append(pattern.pattern)
    return leaks
