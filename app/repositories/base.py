from __future__ import annotations

from typing import Protocol, runtime_checkable

from app.domain import (
    AgentRunRecord,
    IngestionJobRecord,
    KnowledgeChunkRecord,
    KnowledgeDocumentRecord,
    MessageRecord,
    SessionRecord,
    ToolCallRecord,
)


@runtime_checkable
class ExecutionRepository(Protocol):
    def upsert_session(self, session: SessionRecord) -> None: ...

    def append_message(self, message: MessageRecord) -> None: ...

    def append_agent_runs(self, records: list[AgentRunRecord]) -> None: ...

    def append_tool_calls(self, records: list[ToolCallRecord]) -> None: ...

    def replace_knowledge_base(
        self,
        documents: list[KnowledgeDocumentRecord],
        chunks: list[KnowledgeChunkRecord],
    ) -> None: ...

    def create_ingestion_job(self, job: IngestionJobRecord) -> int: ...

    def update_ingestion_job(self, job_id: int, *, status: str, message: str | None = None) -> None: ...

    def ping(self) -> dict[str, int | str]: ...
