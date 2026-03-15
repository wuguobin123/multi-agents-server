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


class KnowledgeDocumentRecord(BaseModel):
    document_id: str
    source: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class KnowledgeChunkRecord(BaseModel):
    chunk_id: str
    document_id: str
    source: str
    text: str
    chunk_index: int
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class IngestionJobRecord(BaseModel):
    source: str
    status: Literal["running", "completed", "failed"]
    message: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
