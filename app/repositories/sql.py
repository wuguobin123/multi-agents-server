from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import create_engine, delete, select, text, update
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from app.config import DatabaseSettings
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
from app.repositories.models import (
    AgentRunModel,
    ChatMessageModel,
    ChatSessionModel,
    IngestionJobModel,
    KnowledgeBaseModel,
    KnowledgeChunkModel,
    KnowledgeDocumentModel,
    ToolCallModel,
)
from app.repositories.schema import ensure_schema


def _json_dump(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _json_load(payload: str | None) -> dict[str, Any]:
    if not payload:
        return {}
    return json.loads(payload)


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


def _to_knowledge_base_record(model: KnowledgeBaseModel) -> KnowledgeBaseRecord:
    return KnowledgeBaseRecord(
        knowledge_base_id=model.knowledge_base_id,
        code=model.code,
        name=model.name,
        description=model.description,
        status=model.status,
        embedding_provider=model.embedding_provider,
        embedding_model=model.embedding_model,
        embedding_dimension=model.embedding_dimension,
        vector_backend=model.vector_backend,
        vector_collection=model.vector_collection,
        storage_backend=model.storage_backend,
        metadata=_json_load(model.metadata_json),
        created_at=model.created_at,
        updated_at=model.updated_at,
    )


def _to_knowledge_document_record(model: KnowledgeDocumentModel) -> KnowledgeDocumentRecord:
    updated_at = model.updated_at or model.created_at or datetime.now(UTC)
    return KnowledgeDocumentRecord(
        document_id=model.document_id,
        knowledge_base_id=model.knowledge_base_id,
        source=model.source,
        original_filename=model.original_filename,
        storage_path=model.storage_path,
        file_url=model.file_url,
        file_hash=model.file_hash,
        file_size=model.file_size,
        mime_type=model.mime_type,
        parser_type=model.parser_type,
        chunking_strategy=model.chunking_strategy,
        chunking_config=_json_load(model.chunking_config_json),
        embedding_provider=model.embedding_provider,
        embedding_model=model.embedding_model,
        embedding_dimension=model.embedding_dimension,
        status=model.status,
        version=model.version,
        is_latest=model.is_latest,
        metadata=_json_load(model.metadata_json),
        created_at=model.created_at,
        updated_at=updated_at,
    )


def _to_knowledge_chunk_record(model: KnowledgeChunkModel) -> KnowledgeChunkRecord:
    return KnowledgeChunkRecord(
        chunk_id=model.chunk_id,
        knowledge_base_id=model.knowledge_base_id,
        document_id=model.document_id,
        source=model.source,
        text=model.text,
        chunk_index=model.chunk_index,
        vector_point_id=model.vector_point_id,
        token_count=model.token_count,
        metadata=_json_load(model.metadata_json),
        created_at=model.created_at,
    )


def _to_ingestion_job_record(model: IngestionJobModel) -> IngestionJobRecord:
    return IngestionJobRecord(
        knowledge_base_id=model.knowledge_base_id,
        document_id=model.document_id,
        source=model.source,
        status=model.status,
        stage=model.stage,
        message=model.message,
        error_message=model.error_message,
        created_at=model.created_at,
        started_at=model.started_at,
        finished_at=model.finished_at,
        updated_at=model.updated_at,
    )


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

    def upsert_knowledge_base(self, knowledge_base: KnowledgeBaseRecord) -> None:
        with self._session_factory.begin() as db:
            model = db.get(KnowledgeBaseModel, knowledge_base.knowledge_base_id)
            if model is None:
                db.add(
                    KnowledgeBaseModel(
                        knowledge_base_id=knowledge_base.knowledge_base_id,
                        code=knowledge_base.code,
                        name=knowledge_base.name,
                        description=knowledge_base.description,
                        status=knowledge_base.status,
                        embedding_provider=knowledge_base.embedding_provider,
                        embedding_model=knowledge_base.embedding_model,
                        embedding_dimension=knowledge_base.embedding_dimension,
                        vector_backend=knowledge_base.vector_backend,
                        vector_collection=knowledge_base.vector_collection,
                        storage_backend=knowledge_base.storage_backend,
                        metadata_json=_json_dump(knowledge_base.metadata),
                        created_at=knowledge_base.created_at,
                        updated_at=knowledge_base.updated_at,
                    )
                )
                return
            model.code = knowledge_base.code
            model.name = knowledge_base.name
            model.description = knowledge_base.description
            model.status = knowledge_base.status
            model.embedding_provider = knowledge_base.embedding_provider
            model.embedding_model = knowledge_base.embedding_model
            model.embedding_dimension = knowledge_base.embedding_dimension
            model.vector_backend = knowledge_base.vector_backend
            model.vector_collection = knowledge_base.vector_collection
            model.storage_backend = knowledge_base.storage_backend
            model.metadata_json = _json_dump(knowledge_base.metadata)
            model.updated_at = knowledge_base.updated_at

    def get_knowledge_base(self, knowledge_base_id: str) -> KnowledgeBaseRecord | None:
        with self._session_factory() as db:
            model = db.get(KnowledgeBaseModel, knowledge_base_id)
            return _to_knowledge_base_record(model) if model is not None else None

    def get_knowledge_base_by_code(self, code: str) -> KnowledgeBaseRecord | None:
        with self._session_factory() as db:
            model = db.scalar(select(KnowledgeBaseModel).where(KnowledgeBaseModel.code == code))
            return _to_knowledge_base_record(model) if model is not None else None

    def list_knowledge_bases(self) -> list[KnowledgeBaseRecord]:
        with self._session_factory() as db:
            models = db.scalars(select(KnowledgeBaseModel).order_by(KnowledgeBaseModel.created_at.asc())).all()
            return [_to_knowledge_base_record(model) for model in models]

    def upsert_knowledge_document(self, document: KnowledgeDocumentRecord) -> None:
        with self._session_factory.begin() as db:
            if document.is_latest:
                db.execute(
                    update(KnowledgeDocumentModel)
                    .where(
                        KnowledgeDocumentModel.knowledge_base_id == document.knowledge_base_id,
                        KnowledgeDocumentModel.original_filename == document.original_filename,
                        KnowledgeDocumentModel.document_id != document.document_id,
                    )
                    .values(is_latest=False, updated_at=document.updated_at)
                )
            model = db.get(KnowledgeDocumentModel, document.document_id)
            if model is None:
                db.add(
                    KnowledgeDocumentModel(
                        document_id=document.document_id,
                        knowledge_base_id=document.knowledge_base_id,
                        source=document.source,
                        original_filename=document.original_filename,
                        storage_path=document.storage_path,
                        file_url=document.file_url,
                        file_hash=document.file_hash,
                        file_size=document.file_size,
                        mime_type=document.mime_type,
                        parser_type=document.parser_type,
                        chunking_strategy=document.chunking_strategy,
                        chunking_config_json=_json_dump(document.chunking_config),
                        embedding_provider=document.embedding_provider,
                        embedding_model=document.embedding_model,
                        embedding_dimension=document.embedding_dimension,
                        status=document.status,
                        version=document.version,
                        is_latest=document.is_latest,
                        metadata_json=_json_dump(document.metadata),
                        created_at=document.created_at,
                        updated_at=document.updated_at,
                    )
                )
                return
            model.knowledge_base_id = document.knowledge_base_id
            model.source = document.source
            model.original_filename = document.original_filename
            model.storage_path = document.storage_path
            model.file_url = document.file_url
            model.file_hash = document.file_hash
            model.file_size = document.file_size
            model.mime_type = document.mime_type
            model.parser_type = document.parser_type
            model.chunking_strategy = document.chunking_strategy
            model.chunking_config_json = _json_dump(document.chunking_config)
            model.embedding_provider = document.embedding_provider
            model.embedding_model = document.embedding_model
            model.embedding_dimension = document.embedding_dimension
            model.status = document.status
            model.version = document.version
            model.is_latest = document.is_latest
            model.metadata_json = _json_dump(document.metadata)
            model.updated_at = document.updated_at

    def get_knowledge_document(self, document_id: str) -> KnowledgeDocumentRecord | None:
        with self._session_factory() as db:
            model = db.get(KnowledgeDocumentModel, document_id)
            return _to_knowledge_document_record(model) if model is not None else None

    def list_knowledge_documents(self, knowledge_base_id: str) -> list[KnowledgeDocumentRecord]:
        with self._session_factory() as db:
            models = db.scalars(
                select(KnowledgeDocumentModel)
                .where(KnowledgeDocumentModel.knowledge_base_id == knowledge_base_id)
                .order_by(KnowledgeDocumentModel.original_filename.asc(), KnowledgeDocumentModel.version.asc())
            ).all()
            return [_to_knowledge_document_record(model) for model in models]

    def replace_document_chunks(self, document: KnowledgeDocumentRecord, chunks: list[KnowledgeChunkRecord]) -> None:
        with self._session_factory.begin() as db:
            if document.is_latest:
                db.execute(
                    update(KnowledgeDocumentModel)
                    .where(
                        KnowledgeDocumentModel.knowledge_base_id == document.knowledge_base_id,
                        KnowledgeDocumentModel.original_filename == document.original_filename,
                        KnowledgeDocumentModel.document_id != document.document_id,
                    )
                    .values(is_latest=False, updated_at=document.updated_at)
                )
            model = db.get(KnowledgeDocumentModel, document.document_id)
            if model is None:
                db.add(
                    KnowledgeDocumentModel(
                        document_id=document.document_id,
                        knowledge_base_id=document.knowledge_base_id,
                        source=document.source,
                        original_filename=document.original_filename,
                        storage_path=document.storage_path,
                        file_url=document.file_url,
                        file_hash=document.file_hash,
                        file_size=document.file_size,
                        mime_type=document.mime_type,
                        parser_type=document.parser_type,
                        chunking_strategy=document.chunking_strategy,
                        chunking_config_json=_json_dump(document.chunking_config),
                        embedding_provider=document.embedding_provider,
                        embedding_model=document.embedding_model,
                        embedding_dimension=document.embedding_dimension,
                        status=document.status,
                        version=document.version,
                        is_latest=document.is_latest,
                        metadata_json=_json_dump(document.metadata),
                        created_at=document.created_at,
                        updated_at=document.updated_at,
                    )
                )
            else:
                model.knowledge_base_id = document.knowledge_base_id
                model.source = document.source
                model.original_filename = document.original_filename
                model.storage_path = document.storage_path
                model.file_url = document.file_url
                model.file_hash = document.file_hash
                model.file_size = document.file_size
                model.mime_type = document.mime_type
                model.parser_type = document.parser_type
                model.chunking_strategy = document.chunking_strategy
                model.chunking_config_json = _json_dump(document.chunking_config)
                model.embedding_provider = document.embedding_provider
                model.embedding_model = document.embedding_model
                model.embedding_dimension = document.embedding_dimension
                model.status = document.status
                model.version = document.version
                model.is_latest = document.is_latest
                model.metadata_json = _json_dump(document.metadata)
                model.updated_at = document.updated_at

            db.execute(delete(KnowledgeChunkModel).where(KnowledgeChunkModel.document_id == document.document_id))
            db.add_all(
                [
                    KnowledgeChunkModel(
                        chunk_id=record.chunk_id,
                        knowledge_base_id=record.knowledge_base_id,
                        document_id=record.document_id,
                        source=record.source,
                        text=record.text,
                        chunk_index=record.chunk_index,
                        vector_point_id=record.vector_point_id,
                        token_count=record.token_count,
                        metadata_json=_json_dump(record.metadata),
                        created_at=record.created_at,
                    )
                    for record in chunks
                ]
            )

    def list_document_chunks(self, document_id: str) -> list[KnowledgeChunkRecord]:
        with self._session_factory() as db:
            models = db.scalars(
                select(KnowledgeChunkModel)
                .where(KnowledgeChunkModel.document_id == document_id)
                .order_by(KnowledgeChunkModel.chunk_index.asc())
            ).all()
            return [_to_knowledge_chunk_record(model) for model in models]

    def delete_knowledge_document(self, document_id: str) -> KnowledgeDocumentRecord | None:
        with self._session_factory.begin() as db:
            model = db.get(KnowledgeDocumentModel, document_id)
            if model is None:
                return None
            record = _to_knowledge_document_record(model)
            db.execute(delete(KnowledgeChunkModel).where(KnowledgeChunkModel.document_id == document_id))
            db.delete(model)
            return record

    def create_ingestion_job(self, job: IngestionJobRecord) -> int:
        with self._session_factory.begin() as db:
            model = IngestionJobModel(
                knowledge_base_id=job.knowledge_base_id,
                document_id=job.document_id,
                source=job.source,
                status=job.status,
                stage=job.stage,
                message=job.message,
                error_message=job.error_message,
                created_at=job.created_at,
                started_at=job.started_at,
                finished_at=job.finished_at,
                updated_at=job.updated_at,
            )
            db.add(model)
            db.flush()
            return int(model.id)

    def update_ingestion_job(
        self,
        job_id: int,
        *,
        status: str,
        stage: str | None = None,
        message: str | None = None,
        error_message: str | None = None,
    ) -> None:
        with self._session_factory.begin() as db:
            model = db.get(IngestionJobModel, job_id)
            if model is None:
                return
            now = datetime.now(UTC)
            model.status = status
            if stage is not None:
                model.stage = stage
            model.message = message
            model.error_message = error_message
            if status == "running" and model.started_at is None:
                model.started_at = now
            if status in {"completed", "failed"}:
                model.finished_at = now
            model.updated_at = now

    def get_ingestion_job(self, job_id: int) -> IngestionJobRecord | None:
        with self._session_factory() as db:
            model = db.get(IngestionJobModel, job_id)
            return _to_ingestion_job_record(model) if model is not None else None

    def ping(self) -> dict[str, int | str]:
        with self._engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {
            "backend": "sql",
            "dialect": self._engine.dialect.name,
            "status": "ok",
        }
