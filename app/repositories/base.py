from __future__ import annotations

from typing import Protocol, runtime_checkable

from app.domain import (
    AgentRunRecord,
    IngestionJobRecord,
    KnowledgeBaseRecord,
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

    def upsert_knowledge_base(self, knowledge_base: KnowledgeBaseRecord) -> None: ...

    def get_knowledge_base(self, knowledge_base_id: str) -> KnowledgeBaseRecord | None: ...

    def get_knowledge_base_by_code(self, code: str) -> KnowledgeBaseRecord | None: ...

    def list_knowledge_bases(self) -> list[KnowledgeBaseRecord]: ...

    def upsert_knowledge_document(self, document: KnowledgeDocumentRecord) -> None: ...

    def get_knowledge_document(self, document_id: str) -> KnowledgeDocumentRecord | None: ...

    def list_knowledge_documents(self, knowledge_base_id: str) -> list[KnowledgeDocumentRecord]: ...

    def replace_document_chunks(self, document: KnowledgeDocumentRecord, chunks: list[KnowledgeChunkRecord]) -> None: ...

    def list_document_chunks(self, document_id: str) -> list[KnowledgeChunkRecord]: ...

    def delete_knowledge_document(self, document_id: str) -> KnowledgeDocumentRecord | None: ...

    def create_ingestion_job(self, job: IngestionJobRecord) -> int: ...

    def update_ingestion_job(
        self,
        job_id: int,
        *,
        status: str,
        stage: str | None = None,
        message: str | None = None,
        error_message: str | None = None,
    ) -> None: ...

    def get_ingestion_job(self, job_id: int) -> IngestionJobRecord | None: ...

    def ping(self) -> dict[str, int | str]: ...
