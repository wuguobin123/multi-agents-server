from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class RAGDocument(BaseModel):
    document_id: str
    source: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChunkRecord(BaseModel):
    chunk_id: str
    document_id: str
    source: str
    text: str
    chunk_index: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class VectorPoint(BaseModel):
    point_id: str
    vector: list[float]
    payload: dict[str, Any]


class SearchResult(BaseModel):
    chunk_id: str
    source: str
    text: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class IndexMetadata(BaseModel):
    embedding_provider: str
    embedding_model: str
    embedding_dimension: int
    embedding_fingerprint: str
    indexed_at: datetime
    chunk_count: int
    vector_store_backend: str
    collection_name: str | None = None
