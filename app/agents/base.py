from __future__ import annotations

from typing import Protocol

from app.schemas import AgentDescriptor, AgentExecutionContext, AgentExecutionResult


class ExecutionAgent(Protocol):
    name: str
    description: str
    capabilities: list[str]

    async def run(self, context: AgentExecutionContext) -> AgentExecutionResult: ...

    def describe(self) -> AgentDescriptor: ...
