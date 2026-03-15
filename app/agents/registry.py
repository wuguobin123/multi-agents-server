from __future__ import annotations

from app.agents.base import ExecutionAgent
from app.schemas import AgentDescriptor, AgentExecutionContext, AgentExecutionResult


class AgentRegistry:
    def __init__(self, agents: list[ExecutionAgent]) -> None:
        self._agents = {agent.name: agent for agent in agents}

    def list_names(self) -> list[str]:
        return list(self._agents)

    def list_descriptors(self) -> list[AgentDescriptor]:
        return [agent.describe() for agent in self._agents.values()]

    def get(self, name: str) -> ExecutionAgent | None:
        return self._agents.get(name)

    async def run(self, name: str, context: AgentExecutionContext) -> AgentExecutionResult:
        agent = self.get(name)
        if agent is None:
            return AgentExecutionResult(
                agent_name=name,
                success=False,
                answer="指定的 Agent 不存在或未启用。",
                error_code="agent_not_found",
                metadata={"reason": "agent_not_found"},
            )
        return await agent.run(context)
