from __future__ import annotations

import json
from typing import Any

from sqlalchemy import create_engine, delete, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from app.config import DatabaseSettings
from app.domain import (
    AgentRunRecord,
    IngestionJobRecord,
    KnowledgeChunkRecord,
    KnowledgeDocumentRecord,
    MessageRecord,
    SessionRecord,
    ToolCallRecord,
)
from app.repositories.models import (
    AgentRunModel,
    ChatMessageModel,
    ChatSessionModel,
    IngestionJobModel,
    KnowledgeChunkModel,
    KnowledgeDocumentModel,
    ToolCallModel,
)
from app.repositories.schema import ensure_schema


def _json_dump(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def build_engine(settings: DatabaseSettings) -> Engine:
    connect_args: dict[str, Any] = {}
    engine_kwargs: dict[str, Any] = {
        "echo": settings.echo,
        "pool_pre_ping": True,
    }
    if settings.url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
    else:
        engine_kwargs["pool_recycle"] = settings.pool_recycle_seconds
        engine_kwargs["pool_size"] = settings.pool_size
        engine_kwargs["max_overflow"] = settings.max_overflow
    if connect_args:
        engine_kwargs["connect_args"] = connect_args
    return create_engine(settings.url, **engine_kwargs)


class SQLExecutionRepository:
    def __init__(self, settings: DatabaseSettings) -> None:
        self._settings = settings
        self._engine = build_engine(settings)
        self._session_factory = sessionmaker(bind=self._engine, autoflush=False, expire_on_commit=False)
        if settings.auto_init:
            ensure_schema(self._engine)

    @property
    def engine(self) -> Engine:
        return self._engine

    def upsert_session(self, session: SessionRecord) -> None:
        with self._session_factory.begin() as db:
            model = db.get(ChatSessionModel, session.session_id)
            if model is None:
                db.add(
                    ChatSessionModel(
                        session_id=session.session_id,
                        status=session.status,
                        metadata_json=_json_dump(session.metadata),
                        created_at=session.created_at,
                        updated_at=session.updated_at,
                    )
                )
                return
            model.status = session.status
            model.metadata_json = _json_dump(session.metadata)
            model.updated_at = session.updated_at

    def append_message(self, message: MessageRecord) -> None:
        with self._session_factory.begin() as db:
            db.add(
                ChatMessageModel(
                    session_id=message.session_id,
                    request_id=message.request_id,
                    role=message.role,
                    content=message.content,
                    metadata_json=_json_dump(message.metadata),
                    created_at=message.created_at,
                )
            )

    def append_agent_runs(self, records: list[AgentRunRecord]) -> None:
        if not records:
            return
        with self._session_factory.begin() as db:
            db.add_all(
                [
                    AgentRunModel(
                        session_id=record.session_id,
                        request_id=record.request_id,
                        agent_name=record.agent_name,
                        success=record.success,
                        answer=record.answer,
                        error_code=record.error_code,
                        metadata_json=_json_dump(record.metadata),
                        created_at=record.created_at,
                    )
                    for record in records
                ]
            )

    def append_tool_calls(self, records: list[ToolCallRecord]) -> None:
        if not records:
            return
        with self._session_factory.begin() as db:
            db.add_all(
                [
                    ToolCallModel(
                        call_id=record.call_id,
                        session_id=record.session_id,
                        request_id=record.request_id,
                        tool_name=record.tool_name,
                        source=record.source,
                        success=record.success,
                        latency_ms=record.latency_ms,
                        input_summary=record.input_summary,
                        error=record.error,
                        error_code=record.error_code,
                        metadata_json=_json_dump(record.metadata),
                        created_at=record.created_at,
                    )
                    for record in records
                ]
            )

    def replace_knowledge_base(
        self,
        documents: list[KnowledgeDocumentRecord],
        chunks: list[KnowledgeChunkRecord],
    ) -> None:
        with self._session_factory.begin() as db:
            db.execute(delete(KnowledgeChunkModel))
            db.execute(delete(KnowledgeDocumentModel))
            db.add_all(
                [
                    KnowledgeDocumentModel(
                        document_id=record.document_id,
                        source=record.source,
                        metadata_json=_json_dump(record.metadata),
                        created_at=record.created_at,
                    )
                    for record in documents
                ]
            )
            db.add_all(
                [
                    KnowledgeChunkModel(
                        chunk_id=record.chunk_id,
                        document_id=record.document_id,
                        source=record.source,
                        text=record.text,
                        chunk_index=record.chunk_index,
                        metadata_json=_json_dump(record.metadata),
                        created_at=record.created_at,
                    )
                    for record in chunks
                ]
            )

    def create_ingestion_job(self, job: IngestionJobRecord) -> int:
        with self._session_factory.begin() as db:
            model = IngestionJobModel(
                source=job.source,
                status=job.status,
                message=job.message,
                created_at=job.created_at,
                updated_at=job.updated_at,
            )
            db.add(model)
            db.flush()
            return int(model.id)

    def update_ingestion_job(self, job_id: int, *, status: str, message: str | None = None) -> None:
        with self._session_factory.begin() as db:
            model = db.get(IngestionJobModel, job_id)
            if model is None:
                return
            model.status = status
            model.message = message

    def ping(self) -> dict[str, int | str]:
        with self._engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {
            "backend": "sql",
            "dialect": self._engine.dialect.name,
            "status": "ok",
        }
