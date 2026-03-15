from __future__ import annotations

import time
from typing import Any

import httpx

from app.errors import ErrorCode
from app.schemas import ToolResult, ToolSpec
from app.tools.base import BaseToolAdapter


class MCPToolAdapter(BaseToolAdapter):
    def __init__(self, spec: ToolSpec, endpoint: str, method: str = "POST", static_payload: dict[str, Any] | None = None) -> None:
        super().__init__(spec)
        self._endpoint = endpoint
        self._method = method.upper()
        self._static_payload = static_payload or {}

    async def invoke(self, payload: dict[str, Any]) -> ToolResult:
        started_at = time.perf_counter()
        request_payload = {**self._static_payload, **payload}
        try:
            async with httpx.AsyncClient(timeout=self.spec.timeout_seconds) as client:
                response = await client.request(self._method, self._endpoint, json=request_payload)
                response.raise_for_status()
            try:
                body = response.json()
            except ValueError:
                body = {"output": response.text}
            return ToolResult(
                success=True,
                output=body.get("output", str(body)),
                structured_data=body if isinstance(body, dict) else {"result": body},
                latency_ms=int((time.perf_counter() - started_at) * 1000),
            )
        except httpx.TimeoutException as exc:
            return ToolResult(
                success=False,
                output="",
                error=str(exc),
                error_code=ErrorCode.REQUEST_TIMEOUT,
                latency_ms=int((time.perf_counter() - started_at) * 1000),
            )
        except httpx.HTTPStatusError as exc:
            status_code = exc.response.status_code
            error_code = ErrorCode.DEPENDENCY_UNAVAILABLE if status_code >= 500 else ErrorCode.REQUEST_FAILED
            return ToolResult(
                success=False,
                output="",
                error=f"{status_code}: {exc.response.text[:200]}",
                error_code=error_code,
                latency_ms=int((time.perf_counter() - started_at) * 1000),
            )
        except Exception as exc:
            return ToolResult(
                success=False,
                output="",
                error=str(exc),
                error_code=ErrorCode.DEPENDENCY_UNAVAILABLE,
                latency_ms=int((time.perf_counter() - started_at) * 1000),
            )
