from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from fastapi import Request

from app.errors import AppError, ErrorCode
from app.schemas import (
    BrowserTaskEvent,
    BrowserTaskRequest,
    BrowserTaskResult,
    BrowserTaskStatus,
    BrowserTaskSummary,
    ErrorDetail,
)
from app.tools import ToolRegistry


TERMINAL_BROWSER_TASK_STATUSES: set[BrowserTaskStatus] = {"succeeded", "failed"}


@dataclass(slots=True)
class BrowserTaskState:
    task_id: str
    request_id: str
    session_id: str
    query: str
    status: BrowserTaskStatus = "queued"
    result: BrowserTaskResult | None = None
    error: ErrorDetail | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None
    events: list[BrowserTaskEvent] = field(default_factory=list)
    event_seq: int = 0
    condition: asyncio.Condition = field(default_factory=asyncio.Condition)

    def to_summary(self) -> BrowserTaskSummary:
        return BrowserTaskSummary(
            task_id=self.task_id,
            status=self.status,
            session_id=self.session_id,
            query=self.query,
            result=self.result,
            error=self.error,
            created_at=self.created_at,
            updated_at=self.updated_at,
            completed_at=self.completed_at,
            status_url=f"/v1/browser/tasks/{self.task_id}",
            events_url=f"/v1/browser/tasks/{self.task_id}/events",
        )


class BrowserTaskManager:
    def __init__(self, tool_registry: ToolRegistry) -> None:
        self._tool_registry = tool_registry
        self._tasks: dict[str, BrowserTaskState] = {}
        self._lock = asyncio.Lock()

    async def create_task(self, request: BrowserTaskRequest, *, request_id: str | None = None) -> BrowserTaskSummary:
        task_id = uuid4().hex
        state = BrowserTaskState(
            task_id=task_id,
            request_id=request_id or uuid4().hex,
            session_id=request.session_id,
            query=request.query,
        )
        async with self._lock:
            self._tasks[task_id] = state
        await self._publish_event(state, "task_created", {"query": request.query})
        asyncio.create_task(self._run_task(state, request), name=f"browser-task-{task_id}")
        return state.to_summary()

    async def get_task(self, task_id: str) -> BrowserTaskSummary:
        return (await self._require_task(task_id)).to_summary()

    async def stream_events(self, task_id: str, request: Request):
        state = await self._require_task(task_id)
        index = 0

        while True:
            while index < len(state.events):
                event = state.events[index]
                index += 1
                yield self._format_sse(event)

            if state.status in TERMINAL_BROWSER_TASK_STATUSES:
                break
            if await request.is_disconnected():
                break

            async with state.condition:
                try:
                    await asyncio.wait_for(state.condition.wait(), timeout=10)
                except TimeoutError:
                    heartbeat = BrowserTaskEvent(seq=0, task_id=task_id, type="heartbeat")
                    yield self._format_sse(heartbeat)

    async def _run_task(self, state: BrowserTaskState, request: BrowserTaskRequest) -> None:
        await self._set_status(state, "running")
        await self._publish_event(state, "task_started", {})

        async def event_sink(event_type: str, data: dict[str, Any] | None = None) -> None:
            await self._publish_event(state, event_type, data or {})

        payload = request.model_dump(exclude_none=True)
        payload.update(
            {
                "intent": "tool",
                "request_id": state.request_id,
                "_event_sink": event_sink,
            }
        )

        result, trace = await self._tool_registry.invoke("browser_use", payload)
        if result.success:
            state.result = BrowserTaskResult(output=result.output, structured_data=result.structured_data)
            await self._set_status(state, "succeeded")
            await self._publish_event(
                state,
                "task_succeeded",
                {
                    "output": result.output,
                    "latency_ms": result.latency_ms,
                    "tool_trace": self._public_trace(trace),
                },
            )
            return

        state.error = ErrorDetail(
            code=result.error_code or ErrorCode.REQUEST_FAILED,
            message=result.error or "浏览器任务执行失败。",
            retryable=result.error_code == ErrorCode.REQUEST_TIMEOUT,
        )
        await self._set_status(state, "failed")
        await self._publish_event(
            state,
            "task_failed",
            {
                "error": state.error.model_dump(mode="json"),
                "latency_ms": result.latency_ms,
                "tool_trace": self._public_trace(trace),
            },
        )

    async def _require_task(self, task_id: str) -> BrowserTaskState:
        async with self._lock:
            state = self._tasks.get(task_id)
        if state is None:
            raise AppError(ErrorCode.INVALID_REQUEST, f"浏览器任务不存在: {task_id}", status_code=404)
        return state

    async def _set_status(self, state: BrowserTaskState, status: BrowserTaskStatus) -> None:
        state.status = status
        state.updated_at = datetime.now(UTC)
        if status in TERMINAL_BROWSER_TASK_STATUSES:
            state.completed_at = state.updated_at

    async def _publish_event(self, state: BrowserTaskState, event_type: str, data: dict[str, Any]) -> None:
        state.event_seq += 1
        state.updated_at = datetime.now(UTC)
        event = BrowserTaskEvent(
            seq=state.event_seq,
            task_id=state.task_id,
            type=event_type,  # type: ignore[arg-type]
            data=data,
        )
        async with state.condition:
            state.events.append(event)
            state.condition.notify_all()

    def _format_sse(self, event: BrowserTaskEvent) -> str:
        payload = event.model_dump(mode="json")
        lines = [f"event: {event.type}", f"id: {event.seq}", f"data: {json.dumps(payload, ensure_ascii=False)}", ""]
        return "\n".join(lines) + "\n"

    def _public_trace(self, trace: Any) -> dict[str, Any]:
        return {
            "call_id": getattr(trace, "call_id", None),
            "name": getattr(trace, "name", None),
            "source": getattr(trace, "source", None),
            "success": getattr(trace, "success", None),
            "latency_ms": getattr(trace, "latency_ms", None),
            "attempts": getattr(trace, "attempts", None),
            "error": getattr(trace, "error", None),
            "error_code": getattr(trace, "error_code", None),
        }
