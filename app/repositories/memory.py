from __future__ import annotations

from app.domain import (
    AgentRunRecord,
    IngestionJobRecord,
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

    def replace_knowledge_base(
        self,
        documents: list[KnowledgeDocumentRecord],
        chunks: list[KnowledgeChunkRecord],
    ) -> None:
        self._documents = {item.document_id: item for item in documents}
        self._chunks = {item.chunk_id: item for item in chunks}

    def create_ingestion_job(self, job: IngestionJobRecord) -> int:
        job_id = self._next_job_id
        self._next_job_id += 1
        self._ingestion_jobs[job_id] = job
        return job_id

    def update_ingestion_job(self, job_id: int, *, status: str, message: str | None = None) -> None:
        if job_id not in self._ingestion_jobs:
            return
        job = self._ingestion_jobs[job_id].model_copy(update={"status": status, "message": message})
        self._ingestion_jobs[job_id] = job

    def ping(self) -> dict[str, int | str]:
        return {
            "backend": "memory",
            "sessions": len(self._sessions),
            "messages": len(self._messages),
            "agent_runs": len(self._agent_runs),
            "tool_calls": len(self._tool_calls),
            "knowledge_documents": len(self._documents),
            "knowledge_chunks": len(self._chunks),
        }
