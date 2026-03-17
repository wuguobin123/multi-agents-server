from __future__ import annotations

from datetime import UTC, datetime

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


class InMemoryExecutionRepository:
    def __init__(self) -> None:
        self._sessions: dict[str, SessionRecord] = {}
        self._messages: list[MessageRecord] = []
        self._agent_runs: list[AgentRunRecord] = []
        self._tool_calls: list[ToolCallRecord] = []
        self._knowledge_bases: dict[str, KnowledgeBaseRecord] = {}
        self._documents: dict[str, KnowledgeDocumentRecord] = {}
        self._chunks: dict[str, KnowledgeChunkRecord] = {}
        self._ingestion_jobs: dict[int, IngestionJobRecord] = {}
        self._next_job_id = 1

    def upsert_session(self, session: SessionRecord) -> None:
        existing = self._sessions.get(session.session_id)
        if existing is not None:
            session.created_at = existing.created_at
        self._sessions[session.session_id] = session

    def append_message(self, message: MessageRecord) -> None:
        self._messages.append(message)

    def append_agent_runs(self, records: list[AgentRunRecord]) -> None:
        self._agent_runs.extend(records)

    def append_tool_calls(self, records: list[ToolCallRecord]) -> None:
        self._tool_calls.extend(records)

    def upsert_knowledge_base(self, knowledge_base: KnowledgeBaseRecord) -> None:
        existing = self._knowledge_bases.get(knowledge_base.knowledge_base_id)
        if existing is not None:
            knowledge_base.created_at = existing.created_at
        self._knowledge_bases[knowledge_base.knowledge_base_id] = knowledge_base

    def get_knowledge_base(self, knowledge_base_id: str) -> KnowledgeBaseRecord | None:
        return self._knowledge_bases.get(knowledge_base_id)

    def get_knowledge_base_by_code(self, code: str) -> KnowledgeBaseRecord | None:
        for knowledge_base in self._knowledge_bases.values():
            if knowledge_base.code == code:
                return knowledge_base
        return None

    def list_knowledge_bases(self) -> list[KnowledgeBaseRecord]:
        return sorted(self._knowledge_bases.values(), key=lambda item: item.created_at)

    def upsert_knowledge_document(self, document: KnowledgeDocumentRecord) -> None:
        existing = self._documents.get(document.document_id)
        if existing is not None:
            document.created_at = existing.created_at
        for record in list(self._documents.values()):
            if (
                record.knowledge_base_id == document.knowledge_base_id
                and record.original_filename == document.original_filename
                and record.document_id != document.document_id
                and document.is_latest
            ):
                self._documents[record.document_id] = record.model_copy(update={"is_latest": False, "updated_at": document.updated_at})
        self._documents[document.document_id] = document

    def get_knowledge_document(self, document_id: str) -> KnowledgeDocumentRecord | None:
        return self._documents.get(document_id)

    def list_knowledge_documents(self, knowledge_base_id: str) -> list[KnowledgeDocumentRecord]:
        return sorted(
            [item for item in self._documents.values() if item.knowledge_base_id == knowledge_base_id],
            key=lambda item: (item.original_filename, item.version, item.created_at),
        )

    def replace_document_chunks(self, document: KnowledgeDocumentRecord, chunks: list[KnowledgeChunkRecord]) -> None:
        self.upsert_knowledge_document(document)
        stale_ids = [chunk_id for chunk_id, chunk in self._chunks.items() if chunk.document_id == document.document_id]
        for chunk_id in stale_ids:
            self._chunks.pop(chunk_id, None)
        for chunk in chunks:
            self._chunks[chunk.chunk_id] = chunk

    def list_document_chunks(self, document_id: str) -> list[KnowledgeChunkRecord]:
        return sorted(
            [item for item in self._chunks.values() if item.document_id == document_id],
            key=lambda item: item.chunk_index,
        )

    def delete_knowledge_document(self, document_id: str) -> KnowledgeDocumentRecord | None:
        document = self._documents.pop(document_id, None)
        if document is None:
            return None
        stale_ids = [chunk_id for chunk_id, chunk in self._chunks.items() if chunk.document_id == document_id]
        for chunk_id in stale_ids:
            self._chunks.pop(chunk_id, None)
        return document

    def create_ingestion_job(self, job: IngestionJobRecord) -> int:
        job_id = self._next_job_id
        self._next_job_id += 1
        self._ingestion_jobs[job_id] = job
        return job_id

    def update_ingestion_job(
        self,
        job_id: int,
        *,
        status: str,
        stage: str | None = None,
        message: str | None = None,
        error_message: str | None = None,
    ) -> None:
        job = self._ingestion_jobs.get(job_id)
        if job is None:
            return
        now = datetime.now(UTC)
        updated = {
            "status": status,
            "stage": stage or job.stage,
            "message": message,
            "error_message": error_message,
            "updated_at": now,
        }
        if status == "running" and job.started_at is None:
            updated["started_at"] = now
        if status in {"completed", "failed"}:
            updated["finished_at"] = now
        self._ingestion_jobs[job_id] = job.model_copy(update=updated)

    def get_ingestion_job(self, job_id: int) -> IngestionJobRecord | None:
        return self._ingestion_jobs.get(job_id)

    def ping(self) -> dict[str, int | str]:
        return {
            "backend": "memory",
            "sessions": len(self._sessions),
            "messages": len(self._messages),
            "agent_runs": len(self._agent_runs),
            "tool_calls": len(self._tool_calls),
            "knowledge_bases": len(self._knowledge_bases),
            "knowledge_documents": len(self._documents),
            "knowledge_chunks": len(self._chunks),
            "ingestion_jobs": len(self._ingestion_jobs),
        }
