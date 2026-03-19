from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from app.config import AppSettings
from app.models import ChatMessage, ModelProvider, build_model_provider
from app.observability import get_logger
from app.schemas import AgentDescriptor, Plan, PlannerRunTrace


logger = get_logger(__name__)


class PlannerAgent:
    def __init__(self, settings: AppSettings, provider: ModelProvider | None = None) -> None:
        self._settings = settings
        self._provider = provider or build_model_provider(settings)

    async def plan(self, query: str, *, hints: list[str] | None = None, available_agents: list[str] | None = None) -> Plan:
        plan, _ = await self.plan_with_trace(query, hints=hints, available_agents=available_agents)
        return plan

    async def plan_with_trace(
        self,
        query: str,
        *,
        hints: list[str] | None = None,
        available_agents: list[str] | list[AgentDescriptor] | None = None,
        attempt: int = 1,
    ) -> tuple[Plan, PlannerRunTrace]:
        descriptors = self._normalize_agents(available_agents)
        enabled_agents = {descriptor.name for descriptor in descriptors}
        model_error: str | None = None
        raw_output: str | None = None

        if self._settings.agents.planner.enabled:
            try:
                response = await self._provider.chat(
                    self._build_messages(query, hints or [], descriptors),
                    config={"temperature": 0},
                )
                raw_output = response.content.strip()
                plan = self._normalize_plan(self._parse_model_plan(raw_output), enabled_agents)
                return plan, PlannerRunTrace(
                    attempt=attempt,
                    source="model",
                    success=True,
                    available_agents=sorted(enabled_agents),
                    hints=list(hints or []),
                    raw_output=raw_output[:1000] if raw_output else None,
                )
            except Exception as exc:  # pragma: no cover - planner guard
                model_error = str(exc)
                logger.warning("planner_model_fallback", extra={"error": model_error})

        plan = self._heuristic_plan(query, hints or [], enabled_agents)
        return plan, PlannerRunTrace(
            attempt=attempt,
            source="heuristic",
            success=model_error is None,
            available_agents=sorted(enabled_agents),
            hints=list(hints or []),
            raw_output=raw_output[:1000] if raw_output else None,
            error=model_error,
        )

    def _build_messages(
        self,
        query: str,
        hints: list[str],
        descriptors: list[AgentDescriptor],
    ) -> list[ChatMessage]:
        agent_catalog = [
            {
                "name": descriptor.name,
                "description": descriptor.description,
                "capabilities": descriptor.capabilities,
            }
            for descriptor in descriptors
        ]
        prompt = {
            "query": query,
            "planner_hints": hints,
            "available_agents": agent_catalog,
            "response_schema": {
                "intent": "qa|tool|hybrid|fallback",
                "requires_rag": "boolean",
                "requires_tools": "boolean",
                "agents": ["agent_name"],
                "success_criteria": "string",
                "notes": "string|null",
            },
        }
        return [
            ChatMessage(
                role="system",
                content=(
                    "You are the planner of a multi-agent system. "
                    "Return JSON only. Choose agents only from the provided catalog. "
                    "Use fallback_agent when no reliable execution path exists. "
                    "If the query asks to browse a website, open a webpage, operate a browser, or perform web automation, "
                    "prefer tool_agent when it is available."
                ),
            ),
            ChatMessage(role="user", content=json.dumps(prompt, ensure_ascii=False)),
        ]

    @staticmethod
    def _parse_model_plan(raw_output: str) -> Plan:
        payload = raw_output.strip()
        if payload.startswith("```"):
            lines = [line for line in payload.splitlines() if not line.strip().startswith("```")]
            payload = "\n".join(lines).strip()
        start = payload.find("{")
        end = payload.rfind("}")
        if start == -1 or end == -1 or end < start:
            raise ValueError("Planner model response does not contain a JSON object.")
        try:
            return Plan.model_validate(json.loads(payload[start : end + 1]))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise ValueError(f"Planner model response is invalid: {exc}") from exc

    def _normalize_plan(self, plan: Plan, enabled_agents: set[str]) -> Plan:
        resolved_agents = self._resolve_agents(plan.agents, enabled_agents)
        return plan.model_copy(update={"agents": resolved_agents})

    def _heuristic_plan(self, query: str, hints: list[str], enabled_agents: set[str]) -> Plan:
        lowered = query.lower()
        tool_terms = {
            "tool",
            "工具",
            "skill",
            "mcp",
            "执行",
            "调用",
            "run",
            "list",
            "search",
            "browser",
            "browser-use",
            "浏览器",
            "网页",
            "网站",
            "页面",
            "打开网页",
            "访问网站",
        }
        rag_terms = {"知识库", "文档", "部署", "总结", "docs", "kb", "manual", "guide"}
        has_tool_signal = any(term in lowered for term in tool_terms)
        has_rag_signal = any(term in lowered for term in rag_terms)
        hint_text = " ".join(hints).lower()

        if "tool_agent_failed" in hint_text and not has_rag_signal:
            return Plan(
                intent="fallback",
                requires_rag=False,
                requires_tools=False,
                agents=self._resolve_agents(["fallback_agent"], enabled_agents),
                success_criteria="清楚说明当前无法完成的原因并给出下一步建议",
                notes="此前工具调用失败，且没有知识库兜底路径。",
            )
        if has_tool_signal and has_rag_signal:
            return Plan(
                intent="hybrid",
                requires_rag=True,
                requires_tools=True,
                agents=self._resolve_agents(["tool_agent", "qa_agent"], enabled_agents),
                success_criteria="先执行工具，再结合知识库给出最终回答",
                notes="混合任务，需要串联工具结果和知识库内容。",
            )
        if has_tool_signal:
            return Plan(
                intent="tool",
                requires_rag=False,
                requires_tools=True,
                agents=self._resolve_agents(["tool_agent"], enabled_agents),
                success_criteria="完成工具调用并返回结构化结果",
                notes="优先走工具链路。",
            )
        return Plan(
            intent="qa",
            requires_rag=True,
            requires_tools=False,
            agents=self._resolve_agents(["qa_agent"], enabled_agents),
            success_criteria="基于知识库返回答案并附带引用",
            notes="优先走 RAG 链路。",
        )

    @staticmethod
    def _normalize_agents(available_agents: list[str] | list[AgentDescriptor] | None) -> list[AgentDescriptor]:
        default_agents = [
            AgentDescriptor(name="qa_agent", description="Knowledge-grounded QA", capabilities=["rag"]),
            AgentDescriptor(name="tool_agent", description="Tool execution", capabilities=["tool_invocation"]),
            AgentDescriptor(name="fallback_agent", description="Safe fallback", capabilities=["fallback_response"]),
        ]
        if not available_agents:
            return default_agents
        first = available_agents[0]
        if isinstance(first, str):
            return [
                AgentDescriptor(name=name, description=f"{name} agent", capabilities=[])
                for name in available_agents
            ]
        return [AgentDescriptor.model_validate(item) for item in available_agents]

    @staticmethod
    def _resolve_agents(candidates: list[str], enabled_agents: set[str]) -> list[str]:
        resolved = [candidate for candidate in candidates if candidate in enabled_agents]
        if resolved:
            return resolved
        if "fallback_agent" in enabled_agents:
            return ["fallback_agent"]
        return []
