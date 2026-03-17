from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import inspect, select, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from app.repositories.models import Base, KnowledgeBaseModel, SchemaMigrationModel


INITIAL_SCHEMA_VERSION = 1
INITIAL_SCHEMA_NAME = "0001_initial_schema"
KNOWLEDGE_BASE_SCHEMA_VERSION = 2
KNOWLEDGE_BASE_SCHEMA_NAME = "0002_knowledge_base_schema"


def ensure_schema(engine: Engine) -> None:
    Base.metadata.create_all(engine)
    inspector = inspect(engine)
    if "schema_migration" not in inspector.get_table_names():
        Base.metadata.tables["schema_migration"].create(engine)

    with Session(engine) as session:
        existing_versions = set(session.scalars(select(SchemaMigrationModel.version)).all())

        if INITIAL_SCHEMA_VERSION not in existing_versions:
            session.add(
                SchemaMigrationModel(
                    version=INITIAL_SCHEMA_VERSION,
                    name=INITIAL_SCHEMA_NAME,
                    applied_at=datetime.now(UTC),
                )
            )
            session.commit()
            existing_versions.add(INITIAL_SCHEMA_VERSION)

        if KNOWLEDGE_BASE_SCHEMA_VERSION not in existing_versions:
            _apply_knowledge_base_schema(engine)
            session.add(
                SchemaMigrationModel(
                    version=KNOWLEDGE_BASE_SCHEMA_VERSION,
                    name=KNOWLEDGE_BASE_SCHEMA_NAME,
                    applied_at=datetime.now(UTC),
                )
            )
            session.commit()


def _apply_knowledge_base_schema(engine: Engine) -> None:
    KnowledgeBaseModel.__table__.create(engine, checkfirst=True)
    _ensure_columns(
        engine,
        "knowledge_document",
        {
            "knowledge_base_id": "knowledge_base_id VARCHAR(128) DEFAULT 'kb-default'",
            "original_filename": "original_filename VARCHAR(255) DEFAULT ''",
            "storage_path": "storage_path VARCHAR(255) DEFAULT ''",
            "file_url": "file_url VARCHAR(512)",
            "file_hash": "file_hash VARCHAR(64)",
            "file_size": "file_size INTEGER DEFAULT 0",
            "mime_type": "mime_type VARCHAR(128)",
            "parser_type": "parser_type VARCHAR(64) DEFAULT 'text'",
            "chunking_strategy": "chunking_strategy VARCHAR(64)",
            "chunking_config_json": "chunking_config_json TEXT",
            "embedding_provider": "embedding_provider VARCHAR(64)",
            "embedding_model": "embedding_model VARCHAR(128)",
            "embedding_dimension": "embedding_dimension INTEGER",
            "status": "status VARCHAR(32) DEFAULT 'pending'",
            "version": "version INTEGER DEFAULT 1",
            "is_latest": "is_latest BOOLEAN DEFAULT 1",
            "updated_at": "updated_at DATETIME",
        },
    )
    _backfill_null_text_column(engine, "knowledge_document", "chunking_config_json", "{}")
    _ensure_columns(
        engine,
        "knowledge_chunk",
        {
            "knowledge_base_id": "knowledge_base_id VARCHAR(128) DEFAULT 'kb-default'",
            "vector_point_id": "vector_point_id VARCHAR(128)",
            "token_count": "token_count INTEGER DEFAULT 0",
        },
    )
    _ensure_columns(
        engine,
        "ingestion_job",
        {
            "knowledge_base_id": "knowledge_base_id VARCHAR(128) DEFAULT 'kb-default'",
            "document_id": "document_id VARCHAR(128)",
            "stage": "stage VARCHAR(64) DEFAULT 'uploaded'",
            "error_message": "error_message TEXT",
            "started_at": "started_at DATETIME",
            "finished_at": "finished_at DATETIME",
        },
    )


def _ensure_columns(engine: Engine, table_name: str, columns: dict[str, str]) -> None:
    inspector = inspect(engine)
    if table_name not in inspector.get_table_names():
        return
    existing = {column["name"] for column in inspector.get_columns(table_name)}
    statements = [
        f"ALTER TABLE {table_name} ADD COLUMN {definition}"
        for column_name, definition in columns.items()
        if column_name not in existing
    ]
    if not statements:
        return
    with engine.begin() as conn:
        for statement in statements:
            conn.execute(text(statement))


def _backfill_null_text_column(engine: Engine, table_name: str, column_name: str, value: str) -> None:
    inspector = inspect(engine)
    if table_name not in inspector.get_table_names():
        return
    existing = {column["name"] for column in inspector.get_columns(table_name)}
    if column_name not in existing:
        return
    with engine.begin() as conn:
        conn.execute(
            text(f"UPDATE {table_name} SET {column_name} = :value WHERE {column_name} IS NULL"),
            {"value": value},
        )
