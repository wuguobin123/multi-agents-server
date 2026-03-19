from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from app.agents.tool import ToolAgent
from app.config import load_settings
from app.errors import ErrorCode
from app.schemas import AgentExecutionContext
from app.tools import ToolRegistry


def _write_config(path: Path, body: str) -> Path:
    path.write_text(body.strip(), encoding="utf-8")
    return path


async def test_tool_registry_invokes_skill_tool() -> None:
    registry = ToolRegistry(load_settings("configs/app.yaml"))

    result, trace = await registry.invoke("local_echo", {"query": "hello", "intent": "tool"})

    assert result.success is True
    assert "hello" in result.output
    assert trace.name == "local_echo"
    assert trace.attempts == 1


async def test_tool_registry_validates_payload() -> None:
    registry = ToolRegistry(load_settings("configs/app.yaml"))

    result, trace = await registry.invoke("local_echo", {"session_id": "missing-query", "intent": "tool"})

    assert result.success is False
    assert result.error_code == ErrorCode.TOOL_VALIDATION_FAILED
    assert trace.validated is False


async def test_tool_registry_retries_retryable_failures(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path / "app.yaml",
        """
model:
  provider: mock
tools:
  retry_attempts: 2
  circuit_breaker_threshold: 3
  skill_tools_enabled: true
  mcp_tools_enabled: false
  skill_tools:
    - name: flaky_echo
      description: Fail once before succeeding.
      handler: flaky_echo
      enabled: true
      source: skill
      timeout_seconds: 1
      allowed_intents: ["tool"]
      input_schema:
        type: object
        properties:
          query:
            type: string
        required: ["query"]
database:
  enabled: false
rag:
  enabled: false
""",
    )
    registry = ToolRegistry(load_settings(config_path))

    result, trace = await registry.invoke(
        "flaky_echo",
        {
            "query": "retry me",
            "request_id": "retry-case-001",
            "intent": "tool",
            "succeed_on_attempt": 2,
        },
    )

    assert result.success is True
    assert trace.attempts == 2
    assert trace.metadata["retry_count"] == 1


async def test_tool_registry_opens_circuit_after_repeated_failures(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path / "app.yaml",
        """
model:
  provider: mock
tools:
  retry_attempts: 0
  circuit_breaker_threshold: 1
  circuit_breaker_cooldown_seconds: 60
  skill_tools_enabled: true
  mcp_tools_enabled: false
  skill_tools:
    - name: always_fail
      description: Always fail.
      handler: always_fail
      enabled: true
      source: skill
      timeout_seconds: 1
      allowed_intents: ["tool"]
      input_schema:
        type: object
        properties:
          query:
            type: string
        required: ["query"]
database:
  enabled: false
rag:
  enabled: false
""",
    )
    registry = ToolRegistry(load_settings(config_path))

    first_result, first_trace = await registry.invoke(
        "always_fail",
        {"query": "fail once", "request_id": "circuit-001", "intent": "tool"},
    )
    second_result, second_trace = await registry.invoke(
        "always_fail",
        {"query": "fail twice", "request_id": "circuit-002", "intent": "tool"},
    )

    assert first_result.success is False
    assert first_trace.metadata["circuit_state"] == "open"
    assert second_result.success is False
    assert second_result.error_code == ErrorCode.TOOL_CIRCUIT_OPEN
    assert second_trace.metadata["blocked_reason"] == "circuit_open"


async def test_tool_registry_enforces_intent_permission_and_timeout(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path / "app.yaml",
        """
model:
  provider: mock
tools:
  retry_attempts: 0
  circuit_breaker_threshold: 3
  skill_tools_enabled: true
  mcp_tools_enabled: false
  skill_tools:
    - name: slow_echo
      description: Slow tool for timeout checks.
      handler: slow_echo
      enabled: true
      source: skill
      timeout_seconds: 0
      allowed_intents: ["hybrid"]
      input_schema:
        type: object
        properties:
          query:
            type: string
        required: ["query"]
database:
  enabled: false
rag:
  enabled: false
""",
    )
    registry = ToolRegistry(load_settings(config_path))

    denied_result, denied_trace = await registry.invoke(
        "slow_echo",
        {"query": "deny me", "request_id": "perm-001", "intent": "tool"},
    )
    timeout_result, timeout_trace = await registry.invoke(
        "slow_echo",
        {"query": "timeout me", "request_id": "perm-002", "intent": "hybrid", "sleep_seconds": 0.05},
    )

    assert denied_result.success is False
    assert denied_result.error_code == ErrorCode.TOOL_PERMISSION_DENIED
    assert denied_trace.metadata["blocked_reason"] == "intent_not_allowed"
    assert timeout_result.success is False
    assert timeout_result.error_code == ErrorCode.REQUEST_TIMEOUT
    assert timeout_trace.attempts == 1


@pytest.fixture
def fake_browser_use(monkeypatch: pytest.MonkeyPatch) -> dict[str, type]:
    class FakeChatOpenAI:
        def __init__(self, *, model: str, api_key: str, base_url: str) -> None:
            self.model = model
            self.api_key = api_key
            self.base_url = base_url

    class FakeChatBrowserUse:
        def __init__(self, *, api_key: str, model: str) -> None:
            self.api_key = api_key
            self.model = model

    class FakeBrowser:
        last_instance: "FakeBrowser | None" = None

        def __init__(
            self,
            *,
            cdp_url: str | None = None,
            headers: dict[str, str] | None = None,
            allowed_domains: list[str] | None = None,
            use_cloud: bool = False,
            cloud_profile_id: str | None = None,
            cloud_proxy_country_code: str | None = None,
            cloud_timeout: int | None = None,
        ) -> None:
            self.cdp_url = cdp_url
            self.headers = headers
            self.allowed_domains = allowed_domains
            self.use_cloud = use_cloud
            self.cloud_profile_id = cloud_profile_id
            self.cloud_proxy_country_code = cloud_proxy_country_code
            self.cloud_timeout = cloud_timeout
            self.closed = False
            FakeBrowser.last_instance = self

        async def stop(self) -> None:
            self.closed = True

        async def close(self) -> None:
            self.closed = True

    class FakeHistory:
        def final_result(self) -> str:
            return "页面标题: Example Domain"

        def urls(self) -> list[str]:
            return ["https://example.com"]

        def model_actions(self) -> list[dict[str, str]]:
            return [{"action": "go_to_url", "url": "https://example.com"}]

        def errors(self) -> list[str]:
            return []

    class FakeAgent:
        last_instance: "FakeAgent | None" = None

        def __init__(self, *, task: str, llm: object, browser: object, use_vision: bool = True) -> None:
            self.task = task
            self.llm = llm
            self.browser = browser
            self.use_vision = use_vision
            self.max_steps: int | None = None
            FakeAgent.last_instance = self

        async def run(self, *, max_steps: int) -> FakeHistory:
            self.max_steps = max_steps
            return FakeHistory()

    monkeypatch.setitem(
        sys.modules,
        "browser_use",
        types.SimpleNamespace(
            Browser=FakeBrowser,
            Agent=FakeAgent,
            ChatOpenAI=FakeChatOpenAI,
            ChatBrowserUse=FakeChatBrowserUse,
        ),
    )
    return {
        "Browser": FakeBrowser,
        "Agent": FakeAgent,
        "ChatOpenAI": FakeChatOpenAI,
        "ChatBrowserUse": FakeChatBrowserUse,
    }


@pytest.fixture
def fake_browser_preflight(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    captured: dict[str, object] = {"requests": []}

    class FakeResponse:
        def __init__(self, *, status_code: int, text: str, json_body: dict[str, object] | None = None) -> None:
            self.status_code = status_code
            self.text = text
            self._json_body = json_body

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            if self._json_body is None:
                raise ValueError("no json")
            return self._json_body

    class FakeAsyncClient:
        def __init__(self, *, timeout: float, trust_env: bool) -> None:
            captured["timeout"] = timeout
            captured["trust_env"] = trust_env

        async def __aenter__(self) -> "FakeAsyncClient":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def get(self, url: str, *, headers: dict[str, str] | None = None) -> FakeResponse:
            captured["requests"].append({"url": url, "headers": headers or {}})
            if url.endswith("/health"):
                return FakeResponse(status_code=200, text="ok")
            return FakeResponse(
                status_code=200,
                text='{"browser":"chrome"}',
                json_body={
                    "browser": "chrome",
                    "cdp_http_url": "http://127.0.0.1:9222",
                    "cdp_ws_url": "ws://127.0.0.1:9222/devtools/browser/mock-browser",
                },
            )

    monkeypatch.setattr("app.tools.browser_use.httpx.AsyncClient", FakeAsyncClient)
    return captured


async def test_tool_registry_invokes_browser_use_skill_tool(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_browser_use: dict[str, type],
    fake_browser_preflight: dict[str, object],
) -> None:
    config_path = _write_config(
        tmp_path / "app.yaml",
        """
model:
  provider: minimax
  name: MiniMax-M2.5-highspeed
  api_key: test-minimax-key
  base_url: https://api.minimaxi.com/v1
tools:
  retry_attempts: 0
  circuit_breaker_threshold: 3
  skill_tools_enabled: true
  mcp_tools_enabled: false
  skill_tools:
    - name: browser_use
      description: 使用 browser-use 云端浏览器执行网页任务。
      handler: browser_use_task
      enabled: true
      source: skill
      timeout_seconds: 30
      allowed_intents: ["tool"]
      input_schema:
        type: object
        properties:
          query:
            type: string
          allowed_domains:
            type: array
        required: ["query"]
      metadata:
        session_timeout_minutes: 8
database:
  enabled: false
rag:
  enabled: false
""",
    )
    registry = ToolRegistry(load_settings(config_path))

    result, trace = await registry.invoke(
        "browser_use",
        {
            "query": "请用浏览器打开 example.com 看一下标题",
            "cdp_url": "http://127.0.0.1:9222",
            "allowed_domains": ["example.com"],
            "health_url": "http://127.0.0.1:8787/health",
            "info_url": "http://127.0.0.1:8787/v1/browser/info",
            "bearer_token": "remote-browser-token",
            "max_steps": 9,
            "intent": "tool",
        },
    )

    browser_instance = fake_browser_use["Browser"].last_instance
    agent_instance = fake_browser_use["Agent"].last_instance
    assert browser_instance is not None
    assert agent_instance is not None
    assert result.success is True
    assert "browser-use 远程浏览器任务执行完成" in result.output
    assert "example.com" in result.output
    assert trace.name == "browser_use"
    assert browser_instance.cdp_url == "http://127.0.0.1:9222"
    assert browser_instance.headers == {"Authorization": "Bearer remote-browser-token"}
    assert browser_instance.allowed_domains == ["example.com"]
    assert browser_instance.closed is True
    assert agent_instance.max_steps == 9
    assert "只能访问以下域名" in agent_instance.task
    assert fake_browser_preflight["requests"] == [
        {
            "url": "http://127.0.0.1:8787/health",
            "headers": {},
        },
        {
            "url": "http://127.0.0.1:8787/v1/browser/info",
            "headers": {"Authorization": "Bearer remote-browser-token"},
        },
    ]


async def test_browser_use_auto_increases_max_steps_for_complex_tasks(
    tmp_path: Path,
    fake_browser_use: dict[str, type],
    fake_browser_preflight: dict[str, object],
) -> None:
    config_path = _write_config(
        tmp_path / "app.yaml",
        """
model:
  provider: minimax
  name: MiniMax-M2.5-highspeed
  api_key: test-minimax-key
  base_url: https://api.minimaxi.com/v1
tools:
  retry_attempts: 0
  circuit_breaker_threshold: 3
  skill_tools_enabled: true
  mcp_tools_enabled: false
  skill_tools:
    - name: browser_use
      description: 使用 browser-use 云端浏览器执行网页任务。
      handler: browser_use_task
      enabled: true
      source: skill
      timeout_seconds: 30
      allowed_intents: ["tool"]
      input_schema:
        type: object
        properties:
          query:
            type: string
        required: ["query"]
      metadata:
        max_steps: 6
        use_vision: false
database:
  enabled: false
rag:
  enabled: false
""",
    )
    registry = ToolRegistry(load_settings(config_path))

    result, trace = await registry.invoke(
        "browser_use",
        {
            "query": "请使用浏览器打开小红书，登录后上传图片，并发布一条新帖子。",
            "cdp_url": "http://127.0.0.1:9222",
            "health_url": "http://127.0.0.1:8787/health",
            "info_url": "http://127.0.0.1:8787/v1/browser/info",
            "bearer_token": "remote-browser-token",
            "intent": "tool",
        },
    )

    agent_instance = fake_browser_use["Agent"].last_instance
    assert result.success is True
    assert trace.name == "browser_use"
    assert agent_instance is not None
    assert agent_instance.max_steps == 24
    assert agent_instance.use_vision is True


async def test_tool_registry_falls_back_to_info_cdp_url_when_configured_cdp_is_missing(
    tmp_path: Path,
    fake_browser_use: dict[str, type],
    fake_browser_preflight: dict[str, object],
) -> None:
    config_path = _write_config(
        tmp_path / "app.yaml",
        """
model:
  provider: minimax
  name: MiniMax-M2.5-highspeed
  api_key: test-minimax-key
  base_url: https://api.minimaxi.com/v1
tools:
  retry_attempts: 0
  circuit_breaker_threshold: 3
  skill_tools_enabled: true
  mcp_tools_enabled: false
  skill_tools:
    - name: browser_use
      description: 使用 browser-use 云端浏览器执行网页任务。
      handler: browser_use_task
      enabled: true
      source: skill
      timeout_seconds: 30
      allowed_intents: ["tool"]
      input_schema:
        type: object
        properties:
          query:
            type: string
        required: ["query"]
      metadata:
        health_url: http://127.0.0.1:8787/health
        info_url: http://127.0.0.1:8787/v1/browser/info
database:
  enabled: false
rag:
  enabled: false
""",
    )
    registry = ToolRegistry(load_settings(config_path))

    result, _trace = await registry.invoke(
        "browser_use",
        {
            "query": "请用浏览器打开 example.com 看一下标题",
            "bearer_token": "remote-browser-token",
            "intent": "tool",
        },
    )

    browser_instance = fake_browser_use["Browser"].last_instance
    assert browser_instance is not None
    assert result.success is True
    assert browser_instance.cdp_url == "ws://127.0.0.1:9222/devtools/browser/mock-browser"


async def test_tool_agent_prefers_browser_use_when_query_mentions_browser(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_browser_use: dict[str, type],
) -> None:
    config_path = _write_config(
        tmp_path / "app.yaml",
        """
model:
  provider: minimax
  name: MiniMax-M2.5-highspeed
  api_key: test-minimax-key
  base_url: https://api.minimaxi.com/v1
tools:
  retry_attempts: 0
  circuit_breaker_threshold: 3
  skill_tools_enabled: true
  mcp_tools_enabled: false
  skill_tools:
    - name: local_echo
      description: Echo the incoming query.
      handler: echo_query
      enabled: true
      source: skill
      timeout_seconds: 5
      allowed_intents: ["tool"]
      input_schema:
        type: object
        properties:
          query:
            type: string
        required: ["query"]
    - name: browser_use
      description: 使用 browser-use 云端浏览器执行网页任务。
      handler: browser_use_task
      enabled: true
      source: skill
      timeout_seconds: 30
      allowed_intents: ["tool"]
      input_schema:
        type: object
        properties:
          query:
            type: string
        required: ["query"]
      metadata:
        keywords: ["浏览器", "网页", "browser-use"]
database:
  enabled: false
rag:
  enabled: false
""",
    )
    registry = ToolRegistry(load_settings(config_path))
    agent = ToolAgent(registry)

    result = await agent.run(
        AgentExecutionContext(
            query="请调用浏览器访问 https://example.com 并告诉我页面标题",
            session_id="sess-browser-001",
            request_id="req-browser-001",
            plan_intent="tool",
        )
    )

    browser_instance = fake_browser_use["Browser"].last_instance
    assert browser_instance is not None
    assert result.success is True
    assert result.metadata["selected_tool"] == "browser_use"
    assert result.answer.startswith("browser-use 远程浏览器任务执行完成")
