from __future__ import annotations

from app.models import ChatMessage, ModelProvider
from app.rag import RAGService
from app.schemas import AgentDescriptor, AgentExecutionContext, AgentExecutionResult, Citation


class QAAgent:
    name = "qa_agent"
    description = "Answer knowledge-grounded questions using retrieved citations."
    capabilities = ["rag", "citation_synthesis", "knowledge_qa"]

    def __init__(self, rag_service: RAGService, provider: ModelProvider) -> None:
        self._rag = rag_service
        self._provider = provider

    async def run(self, context: AgentExecutionContext) -> AgentExecutionResult:
        citations = await self._rag.search(context.query, knowledge_base_id=context.knowledge_base_id)
        if not citations:
            return AgentExecutionResult(
                agent_name=self.name,
                success=False,
                answer="知识库没有检索到足够内容，无法生成可靠回答。",
                citations=[],
                error_code="rag_empty",
                metadata={"reason": "rag_empty", "knowledge_base_id": context.knowledge_base_id},
            )
        answer = await self._synthesize_answer(context.query, citations)
        return AgentExecutionResult(
            agent_name=self.name,
            success=True,
            answer=answer,
            citations=citations,
            metadata={"citation_count": len(citations), "knowledge_base_id": context.knowledge_base_id},
        )

    def describe(self) -> AgentDescriptor:
        return AgentDescriptor(
            name=self.name,
            description=self.description,
            capabilities=self.capabilities,
        )

    async def _synthesize_answer(self, query: str, citations: list[Citation]) -> str:
        context = "\n\n".join(f"[{item.source}] {item.snippet}" for item in citations)
        messages = [
            ChatMessage(role="system", content="Answer using the provided citations only. If evidence is weak, say so explicitly."),
            ChatMessage(role="user", content=f"问题：{query}\n\n可用引用：\n{context}"),
        ]
        response = await self._provider.chat(messages)
        if response.raw.get("provider") == "mock":
            return self._mock_summary(citations)
        if response.content.strip():
            return response.content.strip()
        sources = "；".join(sorted({citation.source for citation in citations}))
        return f"根据知识库内容，相关结论主要来自：{sources}。"

    @staticmethod
    def _mock_summary(citations: list[Citation]) -> str:
        snippets = []
        for citation in citations[:3]:
            cleaned = " ".join(citation.snippet.split())
            snippets.append(cleaned[:120])
        merged = "；".join(snippets)
        sources = "；".join(sorted({citation.source for citation in citations}))
        return f"根据知识库检索结果，关键信息包括：{merged}。引用来源：{sources}。"
