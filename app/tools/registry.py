from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

from app.config import AppSettings
from app.errors import ErrorCode
from app.observability import get_logger
from app.schemas import IntentType, ToolCallTrace, ToolResult, ToolSpec
from app.tools.base import BaseToolAdapter
from app.tools.mcp_adapter import MCPToolAdapter
from app.tools.skill_adapter import SkillToolAdapter
from app.tools.validation import validate_payload


logger = get_logger(__name__)


@dataclass(slots=True)
class CircuitBreakerState:
    consecutive_failures: int = 0
    opened_at: float | None = None


class ToolRegistry:
    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings
        self._tools: dict[str, BaseToolAdapter] = {}
        self._circuit_breakers: dict[str, CircuitBreakerState] = {}
        self._build_registry()

    def _build_registry(self) -> None:
        if self._settings.tools.skill_tools_enabled:
            for config in self._settings.tools.skill_tools:
                if not config.enabled or not config.handler:
                    continue
                spec = ToolSpec(
                    name=config.name,
                    description=config.description,
                    source="skill",
                    enabled=config.enabled,
                    timeout_seconds=config.timeout_seconds,
                    input_schema=config.input_schema,
                    allowed_intents=config.allowed_intents or ["tool", "hybrid"],
                    metadata={**config.metadata, "handler": config.handler},
                )
                self._tools[spec.name] = SkillToolAdapter(spec, config.handler)
                self._circuit_breakers[spec.name] = CircuitBreakerState()
        if self._settings.tools.mcp_tools_enabled:
            for config in self._settings.tools.mcp_tools:
                if not config.enabled or not config.endpoint:
                    continue
                spec = ToolSpec(
                    name=config.name,
                    description=config.description,
                    source="mcp",
                    enabled=config.enabled,
                    timeout_seconds=config.timeout_seconds,
                    input_schema=config.input_schema,
                    allowed_intents=config.allowed_intents or ["tool", "hybrid"],
                    metadata={**config.metadata, "endpoint": config.endpoint, "method": config.method.upper()},
                )
                self._tools[spec.name] = MCPToolAdapter(spec, config.endpoint, config.method, config.static_payload)
                self._circuit_breakers[spec.name] = CircuitBreakerState()

    def list_specs(self, *, intent: IntentType | None = None) -> list[ToolSpec]:
        return [
            tool.spec
            for tool in self._tools.values()
            if self._is_enabled_by_policy(tool.spec, intent=intent)
        ]

    def get(self, name: str) -> BaseToolAdapter | None:
        return self._tools.get(name)

    def readiness(self) -> dict[str, Any]:
        open_circuits = [name for name, state in self._circuit_breakers.items() if state.opened_at is not None]
        return {
            "status": "ok",
            "tool_count": len(self._tools),
            "open_circuits": open_circuits,
        }

    async def invoke(self, name: str, payload: dict[str, Any]) -> tuple[ToolResult, ToolCallTrace]:
        tool = self.get(name)
        if tool is None:
            result = ToolResult(
                success=False,
                output="",
                error=f"Tool not found: {name}",
                error_code=ErrorCode.TOOL_NOT_FOUND,
                latency_ms=0,
            )
            return result, self._trace_for_failure(
                name=name,
                source="skill",
                payload=payload,
                error=result.error,
                error_code=result.error_code,
                validated=False,
                metadata={"blocked_reason": "tool_not_found"},
            )

        policy_error = self._check_policy(tool.spec, payload)
        if policy_error is not None:
            result = ToolResult(
                success=False,
                output="",
                error=policy_error["message"],
                error_code=policy_error["code"],
                latency_ms=0,
            )
            return result, self._trace_for_failure(
                name=tool.spec.name,
                source=tool.spec.source,
                payload=payload,
                error=result.error,
                error_code=result.error_code,
                validated=False,
                metadata=policy_error["metadata"],
            )

        validation_errors = validate_payload(payload, tool.spec.input_schema)
        if validation_errors:
            result = ToolResult(
                success=False,
                output="",
                error="; ".join(validation_errors),
                error_code=ErrorCode.TOOL_VALIDATION_FAILED,
                latency_ms=0,
            )
            return result, self._trace_for_failure(
                name=tool.spec.name,
                source=tool.spec.source,
                payload=payload,
                error=result.error,
                error_code=result.error_code,
                validated=False,
                metadata={"blocked_reason": "validation_failed"},
            )

        max_attempts = max(1, 1 + self._settings.tools.retry_attempts)
        attempts = 0
        last_result: ToolResult | None = None
        started_at = time.perf_counter()

        while attempts < max_attempts:
            attempts += 1
            result = await self._invoke_once(tool, payload)
            last_result = result
            if result.success:
                self._register_success(tool.spec.name)
                trace = ToolCallTrace(
                    name=tool.spec.name,
                    source=tool.spec.source,
                    input_summary=str(payload)[:200],
                    success=True,
                    latency_ms=int((time.perf_counter() - started_at) * 1000),
                    attempts=attempts,
                    validated=True,
                    metadata={
                        "retry_count": attempts - 1,
                        "circuit_state": self._circuit_state_name(tool.spec.name),
                        "tool_metadata": tool.spec.metadata,
                    },
                )
                logger.info(
                    "tool_invoked",
                    extra={
                        "tool_name": tool.spec.name,
                        "tool_source": tool.spec.source,
                        "success": True,
                        "latency_ms": trace.latency_ms,
                        "attempts": attempts,
                        "retry_count": attempts - 1,
                        "error_type": None,
                    },
                )
                return result, trace
            if not self._is_retryable(result.error_code) or attempts >= max_attempts:
                break

        assert last_result is not None  # guaranteed by max_attempts >= 1
        circuit_state = self._register_failure(tool.spec.name)
        trace = ToolCallTrace(
            name=tool.spec.name,
            source=tool.spec.source,
            input_summary=str(payload)[:200],
            success=False,
            latency_ms=int((time.perf_counter() - started_at) * 1000),
            attempts=attempts,
            validated=True,
            error=last_result.error,
            error_code=last_result.error_code,
            metadata={
                "retry_count": attempts - 1,
                "circuit_state": circuit_state,
                "tool_metadata": tool.spec.metadata,
            },
        )
        logger.info(
            "tool_invoked",
            extra={
                "tool_name": tool.spec.name,
                "tool_source": tool.spec.source,
                "success": False,
                "latency_ms": trace.latency_ms,
                "attempts": attempts,
                "retry_count": attempts - 1,
                "error_code": last_result.error_code,
                "error_type": last_result.error_code,
                "circuit_state": circuit_state,
            },
        )
        return last_result, trace

    async def _invoke_once(self, tool: BaseToolAdapter, payload: dict[str, Any]) -> ToolResult:
        started_at = time.perf_counter()
        try:
            async with asyncio.timeout(tool.spec.timeout_seconds):
                result = await tool.invoke(payload)
        except TimeoutError:
            result = ToolResult(
                success=False,
                output="",
                error=f"Tool timed out after {tool.spec.timeout_seconds}s",
                error_code=ErrorCode.REQUEST_TIMEOUT,
                latency_ms=int((time.perf_counter() - started_at) * 1000),
            )
        if not result.latency_ms:
            return result.model_copy(update={"latency_ms": int((time.perf_counter() - started_at) * 1000)})
        return result

    def _check_policy(self, spec: ToolSpec, payload: dict[str, Any]) -> dict[str, Any] | None:
        intent = payload.get("intent")
        if spec.source not in self._settings.tools.allowed_sources:
            return {
                "code": ErrorCode.TOOL_PERMISSION_DENIED,
                "message": f"Tool source is not allowed: {spec.source}",
                "metadata": {"blocked_reason": "source_blocked", "source": spec.source},
            }
        if self._settings.tools.allowed_tools and spec.name not in self._settings.tools.allowed_tools:
            return {
                "code": ErrorCode.TOOL_PERMISSION_DENIED,
                "message": f"Tool is not in the allowed list: {spec.name}",
                "metadata": {"blocked_reason": "tool_not_allowed"},
            }
        if spec.name in self._settings.tools.blocked_tools:
            return {
                "code": ErrorCode.TOOL_PERMISSION_DENIED,
                "message": f"Tool is blocked by configuration: {spec.name}",
                "metadata": {"blocked_reason": "tool_blocked"},
            }
        if spec.allowed_intents and intent not in spec.allowed_intents:
            return {
                "code": ErrorCode.TOOL_PERMISSION_DENIED,
                "message": f"Tool {spec.name} is not permitted for intent {intent}",
                "metadata": {"blocked_reason": "intent_not_allowed", "intent": intent},
            }
        circuit_state = self._circuit_breakers.setdefault(spec.name, CircuitBreakerState())
        if circuit_state.opened_at is not None:
            elapsed = time.monotonic() - circuit_state.opened_at
            if elapsed < self._settings.tools.circuit_breaker_cooldown_seconds:
                return {
                    "code": ErrorCode.TOOL_CIRCUIT_OPEN,
                    "message": f"Tool circuit is open for {spec.name}",
                    "metadata": {
                        "blocked_reason": "circuit_open",
                        "cooldown_seconds": self._settings.tools.circuit_breaker_cooldown_seconds,
                    },
                }
            circuit_state.consecutive_failures = 0
            circuit_state.opened_at = None
        return None

    def _is_enabled_by_policy(self, spec: ToolSpec, *, intent: IntentType | None) -> bool:
        if spec.source not in self._settings.tools.allowed_sources:
            return False
        if self._settings.tools.allowed_tools and spec.name not in self._settings.tools.allowed_tools:
            return False
        if spec.name in self._settings.tools.blocked_tools:
            return False
        if intent and spec.allowed_intents and intent not in spec.allowed_intents:
            return False
        return True

    def _is_retryable(self, error_code: str | None) -> bool:
        if error_code is None:
            return False
        return error_code in self._settings.tools.retryable_error_codes

    def _register_success(self, name: str) -> None:
        state = self._circuit_breakers.setdefault(name, CircuitBreakerState())
        state.consecutive_failures = 0
        state.opened_at = None

    def _register_failure(self, name: str) -> str:
        state = self._circuit_breakers.setdefault(name, CircuitBreakerState())
        state.consecutive_failures += 1
        if state.consecutive_failures >= self._settings.tools.circuit_breaker_threshold:
            state.opened_at = time.monotonic()
            return "open"
        return "closed"

    def _circuit_state_name(self, name: str) -> str:
        state = self._circuit_breakers.setdefault(name, CircuitBreakerState())
        return "open" if state.opened_at is not None else "closed"

    @staticmethod
    def _trace_for_failure(
        *,
        name: str,
        source: str,
        payload: dict[str, Any],
        error: str | None,
        error_code: str | None,
        validated: bool,
        metadata: dict[str, Any],
    ) -> ToolCallTrace:
        return ToolCallTrace(
            name=name,
            source=source,
            input_summary=str(payload)[:200],
            success=False,
            latency_ms=0,
            attempts=1,
            validated=validated,
            error=error,
            error_code=error_code,
            metadata=metadata,
        )
