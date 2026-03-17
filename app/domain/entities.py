from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.schemas import ToolSource


def utc_now() -> datetime:
    return datetime.now(UTC)


class SessionRecord(BaseModel):
    session_id: str
    status: Literal["active", "completed", "failed"] = "active"
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class MessageRecord(BaseModel):
    session_id: str
    request_id: str
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class AgentRunRecord(BaseModel):
    session_id: str
    request_id: str
    agent_name: str
    success: bool
    answer: str
    error_code: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class ToolCallRecord(BaseModel):
    session_id: str
    request_id: str
    call_id: str
    tool_name: str
    source: ToolSource
    success: bool
    latency_ms: int
    input_summary: str
    error: str | None = None
    error_code: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class KnowledgeBaseRecord(BaseModel):
    knowledge_base_id: str
    code: str
    name: str
    description: str | None = None
    status: Literal["active", "disabled"] = "active"
    embedding_provider: str = "mock"
    embedding_model: str = "mock-embedding"
    embedding_dimension: int = 256
    vector_backend: str = "local"
    vector_collection: str
    storage_backend: str = "local"
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class KnowledgeDocumentRecord(BaseModel):
    document_id: str
    knowledge_base_id: str
    source: str
    original_filename: str
    storage_path: str
    file_url: str | None = None
    file_hash: str | None = None
    file_size: int = 0
    mime_type: str | None = None
    parser_type: str = "text"
    chunking_strategy: str | None = None
    chunking_config: dict[str, Any] = Field(default_factory=dict)
    embedding_provider: str | None = None
    embedding_model: str | None = None
    embedding_dimension: int | None = None
    status: Literal["pending", "processing", "ready", "failed", "duplicate", "deleted"] = "pending"
    version: int = 1
    is_latest: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class KnowledgeChunkRecord(BaseModel):
    chunk_id: str
    knowledge_base_id: str
    document_id: str
    source: str
    text: str
    chunk_index: int
    vector_point_id: str | None = None
    token_count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class IngestionJobRecord(BaseModel):
    knowledge_base_id: str
    document_id: str | None = None
    source: str
    status: Literal["queued", "running", "completed", "failed"]
    stage: str = "uploaded"
    message: str | None = None
    error_message: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    updated_at: datetime = Field(default_factory=utc_now)
