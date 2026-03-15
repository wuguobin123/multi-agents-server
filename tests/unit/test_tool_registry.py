from __future__ import annotations

from pathlib import Path

from app.config import load_settings
from app.errors import ErrorCode
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
