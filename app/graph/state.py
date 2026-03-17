from __future__ import annotations

from typing_extensions import TypedDict

from app.schemas import (
    AgentExecutionResult,
    ChatHistoryMessage,
    Citation,
    ErrorDetail,
    Plan,
    PlannerRunTrace,
    ReflectionTrace,
    ToolCallTrace,
)


class GraphState(TypedDict, total=False):
    query: str
    request_id: str
    session_id: str
    knowledge_base_id: str | None
    chat_history: list[ChatHistoryMessage]
    plan: Plan
    planner_runs: list[PlannerRunTrace]
    selected_agents: list[str]
    pending_agents: list[str]
    executed_agents: list[str]
    tool_calls: list[ToolCallTrace]
    agent_runs: list[AgentExecutionResult]
    intermediate_results: list[dict[str, object]]
    reflection_count: int
    citations: list[Citation]
    current_agent: str
    planner_hints: list[str]
    reflections: list[ReflectionTrace]
    final_answer: str
    error: ErrorDetail | None
    next_step: str
