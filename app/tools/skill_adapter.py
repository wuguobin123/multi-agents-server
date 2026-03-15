from __future__ import annotations

import time
from typing import Any

from app.errors import ErrorCode
from app.schemas import ToolResult, ToolSpec
from app.tools.base import BaseToolAdapter
from app.tools.skill_handlers import SKILL_HANDLERS


class SkillToolAdapter(BaseToolAdapter):
    def __init__(self, spec: ToolSpec, handler_name: str) -> None:
        super().__init__(spec)
        if handler_name not in SKILL_HANDLERS:
            raise ValueError(f"Unknown skill handler: {handler_name}")
        self._handler = SKILL_HANDLERS[handler_name]

    async def invoke(self, payload: dict[str, Any]) -> ToolResult:
        started_at = time.perf_counter()
        try:
            result = await self._handler(payload)
            return ToolResult(
                success=True,
                output=result.get("output", ""),
                structured_data=result.get("structured_data"),
                latency_ms=int((time.perf_counter() - started_at) * 1000),
            )
        except Exception as exc:  # pragma: no cover - defensive guard
            return ToolResult(
                success=False,
                output="",
                error=str(exc),
                error_code=ErrorCode.REQUEST_FAILED,
                latency_ms=int((time.perf_counter() - started_at) * 1000),
            )
