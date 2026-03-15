from __future__ import annotations

from app.schemas import AgentDescriptor, AgentExecutionContext, AgentExecutionResult


class FallbackAgent:
    name = "fallback_agent"
    description = "Provide a safe fallback answer when primary execution paths fail."
    capabilities = ["fallback_response", "failure_explanation"]

    async def run(self, context: AgentExecutionContext) -> AgentExecutionResult:
        reason_text = "；".join(reason for reason in context.planner_hints if reason) or "当前规划或执行链路无法稳定完成请求。"
        answer = f"我暂时不能可靠完成这个请求。原因：{reason_text}。建议补充更明确的目标、可用工具或知识库内容后重试。"
        return AgentExecutionResult(
            agent_name=self.name,
            success=True,
            answer=answer,
            metadata={"reasons": context.planner_hints},
        )

    def describe(self) -> AgentDescriptor:
        return AgentDescriptor(
            name=self.name,
            description=self.description,
            capabilities=self.capabilities,
        )
