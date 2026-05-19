from __future__ import annotations

import json
import os
from pathlib import Path

from heterqa.core.config import load_yaml_config
import heterqa.providers.model_client as model_client
from heterqa.providers.model_client import (
    OpenAICompatibleEmbedding,
    OpenAICompatibleJsonJudge,
    ScoreReranker,
    build_component,
    build_model_bundle,
    embed_visual_text_query,
)


class StaticJsonJudge:
    def __init__(self, answer: str = '{"ok": true}') -> None:
        self.answer = answer

    def ask_json(self, *, prompt: str, temperature: float = 0.0, max_tokens: int = 2000, image: str | None = None) -> str:
        assert prompt
        assert temperature == 0.0
        assert max_tokens == 2000
        assert image is None
        return self.answer


def make_static_json_judge(answer: str = '{"factory": true}') -> StaticJsonJudge:
    return StaticJsonJudge(answer)


def test_build_component_from_class_path() -> None:
    component = build_component(
        {
            "class": "tests.test_model_client.StaticJsonJudge",
            "kwargs": {"answer": '{"class": true}'},
        }
    )

    assert isinstance(component, StaticJsonJudge)
    assert json.loads(component.ask_json(prompt="prompt")) == {"class": True}


def test_build_component_from_factory_path() -> None:
    component = build_component(
        {
            "factory": "tests.test_model_client:make_static_json_judge",
            "kwargs": {"answer": '{"factory": true}'},
        }
    )

    assert isinstance(component, StaticJsonJudge)
    assert json.loads(component.ask_json(prompt="prompt")) == {"factory": True}


def test_build_model_bundle_expands_environment_values() -> None:
    os.environ["HETERQA_TEST_MODEL_ANSWER"] = '{"env": true}'
    bundle = build_model_bundle(
        {
            "semantic_judge": {
                "class": "tests.test_model_client.StaticJsonJudge",
                "kwargs": {"answer": "${HETERQA_TEST_MODEL_ANSWER}"},
            }
        }
    )

    assert isinstance(bundle.semantic_judge, StaticJsonJudge)
    assert json.loads(bundle.semantic_judge.ask_json(prompt="prompt")) == {"env": True}


def test_load_yaml_config_expands_environment_values(tmp_path: Path) -> None:
    os.environ["HETERQA_TEST_DB_HOST"] = "127.0.0.1"
    config_path = tmp_path / "config.yaml"
    config_path.write_text("data:\n  host: ${HETERQA_TEST_DB_HOST}\n  port: ${HETERQA_TEST_DB_PORT:-2881}\n", encoding="utf-8")

    config = load_yaml_config(config_path)

    assert config["data"]["host"] == "127.0.0.1"
    assert config["data"]["port"] == "2881"


def test_visual_embedding_helper_uses_public_embed_text_interface() -> None:
    class PublicVisualEmbedding:
        def embed_text(self, texts):
            assert texts == ["quiet patio"]
            return {"embeddings": [[0.1, 0.2, 0.3]]}

    assert embed_visual_text_query(PublicVisualEmbedding(), "quiet patio") == [0.1, 0.2, 0.3]


def test_openai_compatible_json_judge_uses_fixed_chat_payload(monkeypatch) -> None:
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": '{"ok": true}'}}], "usage": {"total_tokens": 7}}

    class FakeRequests:
        @staticmethod
        def post(url, headers, json, timeout):
            captured.update({"url": url, "headers": headers, "json": json, "timeout": timeout})
            return FakeResponse()

    monkeypatch.setattr(model_client, "_load_requests", lambda: FakeRequests)
    monkeypatch.setenv("HETERQA_UNIT_TEST_KEY", "unit-test-key")
    judge = OpenAICompatibleJsonJudge(model="json-model", api_key_env="HETERQA_UNIT_TEST_KEY", base_url="https://example.test/v1")

    out = judge.ask_json(prompt="Return JSON.", temperature=0.1, max_tokens=123)

    assert out["json_text"] == '{"ok": true}'
    assert captured["url"] == "https://example.test/v1/chat/completions"
    assert captured["json"]["messages"] == [{"role": "user", "content": "Return JSON."}]
    assert captured["json"]["response_format"] == {"type": "json_object"}
    assert captured["json"]["temperature"] == 0.1
    assert captured["json"]["max_tokens"] == 123


def test_openai_compatible_embedding_uses_fixed_embedding_payload(monkeypatch) -> None:
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"data": [{"index": 0, "embedding": [0.1, 0.2]}], "usage": {"total_tokens": 3}}

    class FakeRequests:
        @staticmethod
        def post(url, headers, json, timeout):
            captured.update({"url": url, "headers": headers, "json": json, "timeout": timeout})
            return FakeResponse()

    monkeypatch.setattr(model_client, "_load_requests", lambda: FakeRequests)
    monkeypatch.setenv("HETERQA_UNIT_TEST_KEY", "unit-test-key")
    embedding = OpenAICompatibleEmbedding(model="embedding-model", api_key_env="HETERQA_UNIT_TEST_KEY", base_url="https://example.test/v1")

    out = embedding.embed(["quiet patio"])

    assert out["embeddings"] == [[0.1, 0.2]]
    assert captured["url"] == "https://example.test/v1/embeddings"
    assert captured["json"] == {"model": "embedding-model", "input": ["quiet patio"]}


def test_score_reranker_implements_public_reranker_interface() -> None:
    rows = ScoreReranker().select_by_rerank_score(
        query="quiet patio",
        documents_dict=[{"business_id": "b1", "coarse_score": 0.8}, {"business_id": "b2", "coarse_score": 0.2}],
        thres=0.5,
    )

    assert rows == [{"business_id": "b1", "coarse_score": 0.8, "rerank_score": 0.8}]
