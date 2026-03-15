from __future__ import annotations

from app.tools import ToolRegistry
from app.schemas import AgentDescriptor, AgentExecutionContext, AgentExecutionResult, ToolSpec


class ToolAgent:
    name = "tool_agent"
    description = "Select and execute the most relevant tool for operational requests."
    capabilities = ["tool_invocation", "skill_execution", "mcp_execution"]

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    async def run(self, context: AgentExecutionContext) -> AgentExecutionResult:
        tool = self._select_tool(context.query, context.planner_hints, context.plan_intent)
        if tool is None:
            return AgentExecutionResult(
                agent_name=self.name,
                success=False,
                answer="当前没有可用工具可以处理这个请求。",
                error_code="tool_missing",
                metadata={"reason": "tool_missing"},
            )
        payload = {
            "query": context.query,
            "session_id": context.session_id,
            "request_id": context.request_id,
            "intent": context.plan_intent,
            "hints": context.planner_hints,
            "chat_history": [item.model_dump() for item in context.chat_history],
        }
        tool_result, trace = await self._registry.invoke(tool.name, payload)
        answer = tool_result.output if tool_result.output else "工具调用失败，未得到可用结果。"
        return AgentExecutionResult(
            agent_name=self.name,
            success=tool_result.success,
            answer=answer,
            tool_result=tool_result,
            tool_calls=[trace],
            error_code=tool_result.error_code,
            metadata={"selected_tool": tool.name},
        )

    def describe(self) -> AgentDescriptor:
        return AgentDescriptor(
            name=self.name,
            description=self.description,
            capabilities=self.capabilities,
        )

    def _select_tool(self, query: str, tool_hints: list[str] | None, intent: str | None) -> ToolSpec | None:
        specs = self._registry.list_specs(intent=intent)
        if not specs:
            return None
        lowered = query.lower()
        hint_text = " ".join(tool_hints or []).lower()
        ranked: list[tuple[int, ToolSpec]] = []
        for spec in specs:
            score = 0
            if spec.name.lower() in lowered or spec.name.lower() in hint_text:
                score += 3
            if any(token in lowered for token in spec.description.lower().split()):
                score += 1
            if "query" in spec.input_schema.get("required", []):
                score += 1
            ranked.append((score, spec))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return ranked[0][1]
