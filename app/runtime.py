from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from functools import lru_cache
from uuid import uuid4

from app.agents import AgentRegistry, FallbackAgent, PlannerAgent, QAAgent, ToolAgent
from app.config import AppSettings, get_settings
from app.domain import AgentRunRecord, MessageRecord, SessionRecord, ToolCallRecord
from app.errors import ErrorCode
from app.graph import build_graph
from app.models import build_model_provider
from app.observability import get_logger, logging_context
from app.rag import RAGService
from app.repositories import ExecutionRepository, build_execution_repository
from app.schemas import AgentRunTrace, ChatRequest, ChatResponse, ChatTrace, Citation, ErrorDetail, ResponseMeta
from app.tools import ToolRegistry


logger = get_logger(__name__)


@dataclass(slots=True)
class RuntimeDependencies:
    settings: AppSettings
    rag_service: RAGService
    tool_registry: ToolRegistry
    planner: PlannerAgent
    agent_registry: AgentRegistry
    repository: ExecutionRepository
    graph: object


def build_runtime_dependencies(settings: AppSettings) -> RuntimeDependencies:
    provider = build_model_provider(settings)
    repository = build_execution_repository(settings)
    rag_service = RAGService(settings, repository)
    tool_registry = ToolRegistry(settings)
    planner = PlannerAgent(settings, provider)
    agent_registry = AgentRegistry(
        [
            QAAgent(rag_service, provider),
            ToolAgent(tool_registry),
            FallbackAgent(),
        ]
    )
    graph = build_graph(settings, planner, agent_registry)
    return RuntimeDependencies(
        settings=settings,
        rag_service=rag_service,
        tool_registry=tool_registry,
        planner=planner,
        agent_registry=agent_registry,
        repository=repository,
        graph=graph,
    )


class AppRuntime:
    def __init__(self, settings: AppSettings) -> None:
        self.dependencies = build_runtime_dependencies(settings)
        self.settings = self.dependencies.settings
        self.graph = self.dependencies.graph
        self.repository = self.dependencies.repository
        self.rag_service = self.dependencies.rag_service
        self.tool_registry = self.dependencies.tool_registry
        self.planner = self.dependencies.planner
        self.agent_registry = self.dependencies.agent_registry

    async def handle_chat(self, request: ChatRequest, *, request_id: str | None = None) -> ChatResponse:
        resolved_request_id = request_id or uuid4().hex
        started_at = time.perf_counter()
        initial_state = {
            "query": request.query,
            "request_id": resolved_request_id,
            "session_id": request.session_id,
            "chat_history": request.chat_history,
            "planner_runs": [],
            "selected_agents": [],
            "pending_agents": [],
            "executed_agents": [],
            "tool_calls": [],
            "agent_runs": [],
            "intermediate_results": [],
            "reflection_count": 0,
            "citations": [],
            "planner_hints": [],
            "reflections": [],
            "error": None,
            "final_answer": "",
        }
        self._persist_request_open(request, resolved_request_id)
        with logging_context(request_id=resolved_request_id, session_id=request.session_id):
            try:
                async with asyncio.timeout(self.settings.app.request_timeout_seconds):
                    result = await self.graph.ainvoke(initial_state)
            except TimeoutError:
                logger.exception("request_timeout")
                response = self._build_response(
                    request=request,
                    request_id=resolved_request_id,
                    result=initial_state,
                    answer="请求超时，系统未能在限定时间内完成处理。",
                    error=ErrorDetail(
                        code=ErrorCode.REQUEST_TIMEOUT,
                        message="请求超时，系统未能在限定时间内完成处理。",
                        retryable=True,
                    ),
                    duration_ms=int((time.perf_counter() - started_at) * 1000),
                )
                self._log_response_summary(response)
                self._persist_request_close(response)
                return response
            except Exception as exc:  # pragma: no cover - runtime guard
                logger.exception("request_failed")
                response = self._build_response(
                    request=request,
                    request_id=resolved_request_id,
                    result=initial_state,
                    answer="系统执行失败，请稍后重试。",
                    error=ErrorDetail(
                        code=ErrorCode.REQUEST_FAILED,
                        message=str(exc),
                        retryable=True,
                    ),
                    duration_ms=int((time.perf_counter() - started_at) * 1000),
                )
                self._log_response_summary(response)
                self._persist_request_close(response)
                return response

            response = self._build_response(
                request=request,
                request_id=resolved_request_id,
                result=result,
                answer=result.get("final_answer", ""),
                error=result.get("error"),
                duration_ms=int((time.perf_counter() - started_at) * 1000),
            )
            self._log_response_summary(response)
            self._persist_request_close(response)
            return response

    def readiness(self) -> dict[str, object]:
        tool_status = self.tool_registry.readiness()
        planner_status = {
            "status": "ok" if self.settings.agents.planner.enabled else "disabled",
            "enabled": self.settings.agents.planner.enabled,
            "provider": self.settings.model.provider,
        }
        agent_status = {
            "status": "ok",
            "agent_count": len(self.agent_registry.list_names()),
            "enabled_agents": self.agent_registry.list_names(),
        }
        overall_status = "ok"
        try:
            repository_status = self.repository.ping()
        except Exception as exc:  # pragma: no cover - readiness guard
            overall_status = "degraded"
            repository_status = {
                "backend": "sql",
                "status": "error",
                "error": str(exc),
            }
        try:
            rag_status = {
                "enabled": self.settings.rag.enabled,
                **self.rag_service.readiness(),
            }
            if rag_status.get("needs_rebuild"):
                overall_status = "degraded"
        except Exception as exc:  # pragma: no cover - readiness guard
            overall_status = "degraded"
            rag_status = {
                "enabled": self.settings.rag.enabled,
                "status": "error",
                "error": str(exc),
            }
        return {
            "status": overall_status,
            "checks": {
                "repository": repository_status,
                "planner": planner_status,
                "agents": agent_status,
                "tools": tool_status,
                "rag": rag_status,
            },
        }

    def _build_response(
        self,
        *,
        request: ChatRequest,
        request_id: str,
        result: dict[str, object],
        answer: str,
        error: ErrorDetail | None,
        duration_ms: int,
    ) -> ChatResponse:
        citations = [Citation.model_validate(item) for item in result.get("citations", [])]
        agent_runs = result.get("agent_runs", [])
        tool_calls = result.get("tool_calls", [])
        trace = ChatTrace(
            request_id=request_id,
            session_id=request.session_id,
            plan=result.get("plan"),
            planner_runs=result.get("planner_runs", []),
            agents=result.get("selected_agents", []),
            executed_agents=result.get("executed_agents", []),
            tool_calls=tool_calls,
            agent_runs=[
                AgentRunTrace(
                    agent_name=item.agent_name,
                    success=item.success,
                    answer_preview=item.answer[:120],
                    error_code=item.error_code,
                    metadata=item.metadata,
                )
                for item in agent_runs
            ],
            intermediate_results=result.get("intermediate_results", []),
            reflection_count=result.get("reflection_count", 0),
            reflections=result.get("reflections", []),
            error=error,
        )
        return ChatResponse(
            answer=answer,
            citations=citations,
            trace=trace,
            error=error,
            meta=ResponseMeta(
                request_id=request_id,
                session_id=request.session_id,
                duration_ms=duration_ms,
            ),
        )

    def _persist_request_open(self, request: ChatRequest, request_id: str) -> None:
        self.repository.upsert_session(
            SessionRecord(
                session_id=request.session_id,
                status="active",
                metadata={"last_query": request.query},
            )
        )
        self.repository.append_message(
            MessageRecord(
                session_id=request.session_id,
                request_id=request_id,
                role="user",
                content=request.query,
            )
        )

    def _persist_request_close(self, response: ChatResponse) -> None:
        status = "failed" if response.error else "completed"
        self.repository.upsert_session(
            SessionRecord(
                session_id=response.meta.session_id,
                status=status,
                metadata={"request_id": response.meta.request_id},
            )
        )
        self.repository.append_message(
            MessageRecord(
                session_id=response.meta.session_id,
                request_id=response.meta.request_id,
                role="assistant",
                content=response.answer,
                metadata={"error_code": response.error.code if response.error else None},
            )
        )
        self.repository.append_agent_runs(
            [
                AgentRunRecord(
                    session_id=response.meta.session_id,
                    request_id=response.meta.request_id,
                    agent_name=item.agent_name,
                    success=item.success,
                    answer=item.answer_preview,
                    error_code=item.error_code,
                    metadata=item.metadata,
                )
                for item in response.trace.agent_runs
            ]
        )
        self.repository.append_tool_calls(
            [
                ToolCallRecord(
                    session_id=response.meta.session_id,
                    request_id=response.meta.request_id,
                    call_id=item.call_id,
                    tool_name=item.name,
                    source=item.source,
                    success=item.success,
                    latency_ms=item.latency_ms,
                    input_summary=item.input_summary,
                    error=item.error,
                    error_code=item.error_code,
                    metadata={
                        **item.metadata,
                        "validated": item.validated,
                        "attempts": item.attempts,
                    },
                )
                for item in response.trace.tool_calls
            ]
        )

    @staticmethod
    def _log_response_summary(response: ChatResponse) -> None:
        logger.info(
            "request_completed",
            extra={
                "plan_intent": response.trace.plan.intent if response.trace.plan else None,
                "selected_agents": response.trace.agents,
                "latency_ms": response.meta.duration_ms,
                "error_type": response.error.code if response.error else None,
            },
        )


@lru_cache(maxsize=1)
def get_runtime() -> AppRuntime:
    return AppRuntime(get_settings())
