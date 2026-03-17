from __future__ import annotations

import json

from app.agents import PlannerAgent
from app.config import load_settings
from app.models.base import ChatMessage, ModelProvider, ModelResponse
from app.models.mock import MockProvider


class InvalidPlannerProvider(ModelProvider):
    async def chat(self, messages: list[ChatMessage], *, config: dict | None = None) -> ModelResponse:
        return ModelResponse(content="not-json", raw={"provider": "invalid"})

    async def stream(self, messages: list[ChatMessage], *, config: dict | None = None):
        yield "not-json"

    def supports_tool_calling(self) -> bool:
        return False

    def supports_structured_output(self) -> bool:
        return False


class HybridPlannerProvider(ModelProvider):
    async def chat(self, messages: list[ChatMessage], *, config: dict | None = None) -> ModelResponse:
        return ModelResponse(
            content=json.dumps(
                {
                    "intent": "hybrid",
                    "requires_rag": True,
                    "requires_tools": True,
                    "agents": ["tool_agent", "qa_agent"],
                    "success_criteria": "先执行工具，再结合知识库给出最终回答",
                    "notes": "模型生成的结构化规划。",
                },
                ensure_ascii=False,
            ),
            raw={"provider": "stub"},
        )

    async def stream(self, messages: list[ChatMessage], *, config: dict | None = None):
        yield ""

    def supports_tool_calling(self) -> bool:
        return False

    def supports_structured_output(self) -> bool:
        return True


def test_planner_routes_to_qa_for_knowledge_query() -> None:
    planner = PlannerAgent(load_settings("configs/app.yaml"), provider=MockProvider())

    plan, trace = __import__("asyncio").run(planner.plan_with_trace("帮我总结知识库中的部署流程"))

    assert plan.intent == "qa"
    assert plan.requires_rag is True
    assert plan.agents == ["qa_agent"]
    assert trace.source == "model"


def test_planner_routes_to_tool_for_tool_query() -> None:
    planner = PlannerAgent(load_settings("configs/app.yaml"), provider=MockProvider())

    plan = __import__("asyncio").run(planner.plan("请调用工具执行一次本地检查"))

    assert plan.intent == "tool"
    assert plan.requires_tools is True
    assert plan.agents == ["tool_agent"]


def test_planner_falls_back_when_target_agent_is_unavailable() -> None:
    planner = PlannerAgent(load_settings("configs/app.yaml"), provider=MockProvider())

    plan = __import__("asyncio").run(
        planner.plan("请调用工具执行一次本地检查", available_agents=["qa_agent", "fallback_agent"])
    )

    assert plan.agents == ["fallback_agent"]


def test_planner_uses_model_output_when_schema_is_valid() -> None:
    planner = PlannerAgent(load_settings("configs/app.yaml"), provider=HybridPlannerProvider())

    plan, trace = __import__("asyncio").run(
        planner.plan_with_trace(
            "先调用工具再结合知识库总结部署方式",
            available_agents=["qa_agent", "tool_agent", "fallback_agent"],
        )
    )

    assert plan.intent == "hybrid"
    assert plan.agents == ["tool_agent", "qa_agent"]
    assert trace.source == "model"
    assert trace.error is None


def test_planner_falls_back_to_heuristic_when_model_output_is_invalid() -> None:
    planner = PlannerAgent(load_settings("configs/app.yaml"), provider=InvalidPlannerProvider())

    plan, trace = __import__("asyncio").run(
        planner.plan_with_trace(
            "请调用工具执行一次本地检查",
            available_agents=["qa_agent", "tool_agent", "fallback_agent"],
        )
    )

    assert plan.intent == "tool"
    assert plan.agents == ["tool_agent"]
    assert trace.source == "heuristic"
    assert trace.error
