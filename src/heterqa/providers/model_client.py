"""Model-client provider boundary and configurable component loader."""

from __future__ import annotations

import importlib
import os
import re
import base64
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


class JsonJudgeClient(Protocol):
    def ask_json(self, *, prompt: str, temperature: float = 0.0, max_tokens: int = 2000, image: str | None = None) -> Any:
        ...


@dataclass
class ModelBundle:
    """Explicit model components used by the public construction flow."""

    semantic_judge: Any = None
    visual_judge: Any = None
    embedding: Any = None
    visual_embedding: Any = None
    reranker: Any = None
    visual_reranker: Any = None

    def has_any_component(self) -> bool:
        return any(
            component is not None
            for component in [
                self.semantic_judge,
                self.visual_judge,
                self.embedding,
                self.visual_embedding,
                self.reranker,
                self.visual_reranker,
            ]
        )

    def ask_json(self, *, prompt: str, temperature: float = 0.0, max_tokens: int = 2000, image: str | None = None) -> Any:
        client = self.visual_judge if image else self.semantic_judge
        if client is None:
            component = "visual_judge" if image else "semantic_judge"
            raise ValueError(f"ModelBundle has no configured {component}.")
        return client.ask_json(prompt=prompt, temperature=temperature, max_tokens=max_tokens, image=image)

    def parallel_ask_json(self, tasks: list[dict[str, Any]]) -> list[Any]:
        if self.semantic_judge is None:
            raise ValueError("ModelBundle has no configured semantic_judge for parallel JSON calls.")
        batcher = getattr(self.semantic_judge, "parallel_ask_json", None)
        if callable(batcher):
            return list(batcher(tasks))
        return [
            self.semantic_judge.ask_json(
                prompt=str(task["prompt"]),
                temperature=float(task.get("temperature", 0.0)),
                max_tokens=int(task.get("max_tokens", 2000)),
                image=task.get("image"),
            )
            for task in tasks
        ]


def _load_requests() -> Any:
    try:
        import requests  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Install heterqa[models] to use OpenAI-compatible model clients.") from exc
    return requests


def _api_key(explicit: str | None, env_name: str) -> str:
    key = explicit or os.environ.get(env_name, "")
    if not key:
        raise ValueError(f"Missing API key. Set {env_name} or pass api_key in the local config.")
    return key


def _image_url(image: str) -> str:
    if image.startswith(("http://", "https://", "data:")):
        return image
    path = Path(image)
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{payload}"


class OpenAICompatibleJsonJudge:
    """JSON judge for OpenAI-compatible chat-completions endpoints."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        api_key: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        endpoint: str = "/chat/completions",
        timeout: float = 120.0,
        response_format: bool = True,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = _api_key(api_key, api_key_env)
        self.endpoint = endpoint
        self.timeout = timeout
        self.response_format = response_format

    def ask_json(self, *, prompt: str, temperature: float = 0.0, max_tokens: int = 2000, image: str | None = None) -> Any:
        requests = _load_requests()
        if image:
            content: Any = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": _image_url(image)}},
            ]
        else:
            content = prompt
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if self.response_format:
            payload["response_format"] = {"type": "json_object"}
        response = requests.post(
            f"{self.base_url}{self.endpoint}",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        text = data["choices"][0]["message"]["content"]
        return {"json_text": text, "metadata": {"token_usage": data.get("usage", {})}}

    def parallel_ask_json(self, tasks: list[dict[str, Any]]) -> list[Any]:
        return [
            self.ask_json(
                prompt=str(task["prompt"]),
                temperature=float(task.get("temperature", 0.0)),
                max_tokens=int(task.get("max_tokens", 2000)),
                image=task.get("image"),
            )
            for task in tasks
        ]


class OpenAICompatibleEmbedding:
    """Text embedding client for OpenAI-compatible embedding endpoints."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        api_key: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        endpoint: str = "/embeddings",
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = _api_key(api_key, api_key_env)
        self.endpoint = endpoint
        self.timeout = timeout

    def embed(self, texts: list[str]) -> dict[str, Any]:
        requests = _load_requests()
        response = requests.post(
            f"{self.base_url}{self.endpoint}",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json={"model": self.model, "input": texts},
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        rows = sorted(data.get("data", []), key=lambda row: int(row.get("index", 0)))
        return {"embeddings": [row["embedding"] for row in rows], "metadata": {"token_usage": data.get("usage", {})}}

    def embed_text(self, texts: list[str]) -> dict[str, Any]:
        return self.embed(texts)


class ScoreReranker:
    """Deterministic score gate implementing the public reranker interface."""

    def select_by_rerank_score(
        self,
        *,
        query: str,
        documents_dict: list[dict[str, Any]],
        thres: float = 0.0,
        top_n: int | None = None,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for doc in documents_dict:
            item = dict(doc)
            score = float(
                item.get("rerank_score", item.get("coarse_score", item.get("vector_score", item.get("score", 0))))
                or 0
            )
            item["rerank_score"] = score
            if score >= thres:
                rows.append(item)
        rows.sort(key=lambda row: float(row.get("rerank_score") or 0), reverse=True)
        return rows if top_n is None else rows[:top_n]


def _first_embedding(result: Any) -> Any:
    if isinstance(result, dict):
        embeddings = result.get("embeddings")
        if isinstance(embeddings, list) and embeddings:
            return embeddings[0]
        if result.get("embedding") is not None:
            return result["embedding"]
    if isinstance(result, list) and result:
        first = result[0]
        if isinstance(first, dict) and first.get("embedding") is not None:
            return first["embedding"]
        return first
    raise ValueError("Embedding result did not contain an embedding vector.")


def embed_visual_text_query(vl_embedding_model: Any, text: str) -> Any:
    """Embed a text query with the public visual-text embedding interface."""

    if not hasattr(vl_embedding_model, "embed_text"):
        raise ValueError("Configured visual embedding model must expose embed_text([text]).")
    return _first_embedding(vl_embedding_model.embed_text([text]))


def _expand_env(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if not isinstance(value, str):
        return value

    pattern = re.compile(r"\$\{([^}:]+)(?::-([^}]*))?\}")

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        default = match.group(2) or ""
        return os.environ.get(name, default)

    return pattern.sub(replace, value)


def _import_object(path: str) -> Any:
    if ":" in path:
        module_name, attr_name = path.split(":", 1)
    else:
        module_name, attr_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    target: Any = module
    for part in attr_name.split("."):
        target = getattr(target, part)
    return target


def build_component(spec: Any) -> Any:
    """Instantiate one configured model/index component.

    Supported forms:
    - already-instantiated object
    - string import path
    - {"class": "pkg.ModClass", "kwargs": {...}}
    - {"factory": "pkg.module:create", "kwargs": {...}}
    """

    if spec is None or spec is False:
        return None
    if isinstance(spec, str):
        target = _import_object(spec)
        return target()
    if not isinstance(spec, dict):
        return spec

    spec = _expand_env(spec)
    if "factory" in spec:
        factory = _import_object(str(spec["factory"]))
        return factory(**dict(spec.get("kwargs") or {}))
    if "class" in spec:
        cls = _import_object(str(spec["class"]))
        return cls(**dict(spec.get("kwargs") or {}))
    if "object" in spec:
        return _import_object(str(spec["object"]))
    return None


def build_model_bundle(config: dict[str, Any] | None) -> Any:
    """Build the model object consumed by construction/audit stages.

    A top-level `factory` may return an object that implements the same public
    component attributes. Otherwise individual components are loaded into a
    ModelBundle.
    """

    if not config:
        return None
    if "factory" in config:
        return build_component({"factory": config["factory"], "kwargs": config.get("kwargs", {})})
    bundle = ModelBundle(
        semantic_judge=build_component(config.get("semantic_judge")),
        visual_judge=build_component(config.get("visual_judge")),
        embedding=build_component(config.get("embedding")),
        visual_embedding=build_component(config.get("visual_embedding")),
        reranker=build_component(config.get("reranker")),
        visual_reranker=build_component(config.get("visual_reranker")),
    )
    return bundle if bundle.has_any_component() else None
