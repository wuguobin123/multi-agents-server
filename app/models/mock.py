from __future__ import annotations

import json
from collections.abc import AsyncIterator

from app.models.base import ChatMessage, ModelProvider, ModelResponse


class MockProvider(ModelProvider):
    async def chat(self, messages: list[ChatMessage], *, config: dict | None = None) -> ModelResponse:
        user_message = next((message.content for message in reversed(messages) if message.role == "user"), "")
        system_message = next((message.content for message in messages if message.role == "system"), "")
        if "return json only" in system_message.lower() and "multi-agent system" in system_message.lower():
            return ModelResponse(content=self._planner_response(user_message), raw={"provider": "mock"})
        if "citation" in system_message.lower():
            content = f"基于当前检索结果，问题可以概括为：{user_message[:160]}"
        else:
            content = f"MockProvider 响应：{user_message[:200]}"
        return ModelResponse(content=content, raw={"provider": "mock"})

    async def stream(self, messages: list[ChatMessage], *, config: dict | None = None) -> AsyncIterator[str]:
        response = await self.chat(messages, config=config)
        yield response.content

    def supports_tool_calling(self) -> bool:
        return False

    def supports_structured_output(self) -> bool:
        return True

    @staticmethod
    def _planner_response(user_message: str) -> str:
        try:
            payload = json.loads(user_message)
        except json.JSONDecodeError:
            payload = {"query": user_message, "planner_hints": [], "available_agents": []}
        query = str(payload.get("query", ""))
        hints = [str(item) for item in payload.get("planner_hints", [])]
        lowered = query.lower()
        hint_text = " ".join(hints).lower()
        available_agents = {item.get("name", "") for item in payload.get("available_agents", []) if item.get("name")}

        def resolve(candidates: list[str]) -> list[str]:
            resolved = [candidate for candidate in candidates if candidate in available_agents]
            if resolved:
                return resolved
            if "fallback_agent" in available_agents:
                return ["fallback_agent"]
            return []

        tool_terms = {"tool", "工具", "skill", "mcp", "执行", "调用", "run", "list", "search"}
        rag_terms = {"知识库", "文档", "部署", "总结", "docs", "kb", "manual", "guide"}
        has_tool_signal = any(term in lowered for term in tool_terms)
        has_rag_signal = any(term in lowered for term in rag_terms)

        if "tool_agent_failed" in hint_text and not has_rag_signal:
            plan = {
                "intent": "fallback",
                "requires_rag": False,
                "requires_tools": False,
                "agents": resolve(["fallback_agent"]),
                "success_criteria": "清楚说明当前无法完成的原因并给出下一步建议",
                "notes": "此前工具调用失败，且没有知识库兜底路径。",
            }
        elif has_tool_signal and has_rag_signal:
            plan = {
                "intent": "hybrid",
                "requires_rag": True,
                "requires_tools": True,
                "agents": resolve(["tool_agent", "qa_agent"]),
                "success_criteria": "先执行工具，再结合知识库给出最终回答",
                "notes": "混合任务，需要串联工具结果和知识库内容。",
            }
        elif has_tool_signal:
            plan = {
                "intent": "tool",
                "requires_rag": False,
                "requires_tools": True,
                "agents": resolve(["tool_agent"]),
                "success_criteria": "完成工具调用并返回结构化结果",
                "notes": "优先走工具链路。",
            }
        else:
            plan = {
                "intent": "qa",
                "requires_rag": True,
                "requires_tools": False,
                "agents": resolve(["qa_agent"]),
                "success_criteria": "基于知识库返回答案并附带引用",
                "notes": "优先走 RAG 链路。",
            }
        return json.dumps(plan, ensure_ascii=False)
