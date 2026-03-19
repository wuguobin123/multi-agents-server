from __future__ import annotations

import re

from app.tools import ToolRegistry
from app.schemas import AgentDescriptor, AgentExecutionContext, AgentExecutionResult, ToolSpec


class ToolAgent:
    name = "tool_agent"
    description = "Select and execute tools for operational requests, including browser automation, web navigation, and external integrations."
    capabilities = ["tool_invocation", "skill_execution", "mcp_execution", "browser_automation", "web_navigation"]

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
        answer = tool_result.output if tool_result.output else (tool_result.error or "工具调用失败，未得到可用结果。")
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
            canonical_names = {
                spec.name.lower(),
                spec.name.lower().replace("_", "-"),
                spec.name.lower().replace("_", " "),
            }
            if any(name in lowered or name in hint_text for name in canonical_names):
                score += 3

            keywords = [
                str(item).strip().lower()
                for item in spec.metadata.get("keywords", [])
                if str(item).strip()
            ]
            if any(keyword in lowered or keyword in hint_text for keyword in keywords):
                score += 4

            description_tokens = [
                token
                for token in re.split(r"[\s,，。；;、/()]+", spec.description.lower())
                if len(token) >= 2
            ]
            if any(token in lowered for token in description_tokens):
                score += 1
            if "query" in spec.input_schema.get("required", []):
                score += 1
            ranked.append((score, spec))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return ranked[0][1]
