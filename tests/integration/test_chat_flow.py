from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings, load_settings
from app.main import app
from app.runtime import AppRuntime
from app.runtime import get_runtime
from app.schemas import ChatRequest


@pytest.fixture(autouse=True)
def isolated_app_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_CONFIG_PATH", "configs/app.yaml")
    monkeypatch.setenv("MODEL_PROVIDER", "mock")
    monkeypatch.setenv("RAG_LOCAL_STORE_PATH", str(tmp_path / "vector_store.json"))
    monkeypatch.setenv("RAG_BOOTSTRAP_ON_STARTUP", "true")
    get_settings.cache_clear()
    get_runtime.cache_clear()
    yield
    get_runtime.cache_clear()
    get_settings.cache_clear()


def test_healthz() -> None:
    client = TestClient(app)

    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_readyz() -> None:
    client = TestClient(app)

    response = client.get("/readyz")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "checks" in body
    assert body["checks"]["repository"]["backend"] == "sql"
    assert body["checks"]["planner"]["status"] == "ok"
    assert body["checks"]["agents"]["agent_count"] >= 1


def test_chat_qa_flow() -> None:
    client = TestClient(app)

    response = client.post(
        "/v1/chat",
        json={"query": "帮我总结知识库中关于部署流程的说明", "session_id": "itest-qa"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["error"] is None
    assert body["meta"]["request_id"]
    assert body["meta"]["session_id"] == "itest-qa"
    assert body["trace"]["request_id"] == body["meta"]["request_id"]
    assert body["trace"]["plan"]["intent"] == "qa"
    assert body["trace"]["planner_runs"]
    assert body["trace"]["planner_runs"][0]["source"] in {"model", "heuristic"}
    assert body["trace"]["agent_runs"]
    assert body["citations"]
    assert response.text.endswith("\n")


def test_chat_tool_flow() -> None:
    client = TestClient(app)

    response = client.post(
        "/v1/chat",
        json={"query": "请调用工具执行一次检查", "session_id": "itest-tool"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["trace"]["plan"]["intent"] == "tool"
    assert body["trace"]["tool_calls"]
    assert "技能工具收到请求" in body["answer"]


def test_chat_reflects_and_falls_back_when_tool_path_has_no_available_tool(tmp_path) -> None:
    config_path = tmp_path / "app.yaml"
    config_path.write_text(
        """
model:
  provider: mock
agents:
  planner:
    enabled: true
  qa_agent:
    enabled: true
  tool_agent:
    enabled: true
  fallback_agent:
    enabled: true
tools:
  timeout_seconds: 10
  max_calls: 3
  skill_tools_enabled: false
  mcp_tools_enabled: false
rag:
  enabled: false
  bootstrap_on_startup: false
database:
  enabled: false
app:
  max_reflections: 0
  request_timeout_seconds: 30
""".strip(),
        encoding="utf-8",
    )
    runtime = AppRuntime(load_settings(config_path))

    response = __import__("asyncio").run(
        runtime.handle_chat(
            ChatRequest(query="请调用工具执行一次检查", session_id="itest-fallback"),
            request_id="req-fallback-001",
        )
    )

    assert response.error is not None
    assert response.trace.plan is not None
    assert response.trace.plan.intent == "tool"
    assert response.trace.reflection_count == 0
    assert response.trace.reflections
    assert any(item.action == "fallback" for item in response.trace.reflections)
    assert response.trace.agents[-1] == "fallback_agent"
    assert "不能可靠完成" in response.answer
