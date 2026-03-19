from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from app.config.settings import ChunkingSettings


IntentType = Literal["qa", "tool", "hybrid", "fallback"]
ToolSource = Literal["skill", "mcp"]
ChatRole = Literal["system", "user", "assistant", "tool"]
BrowserTaskStatus = Literal["queued", "running", "succeeded", "failed"]
BrowserTaskEventType = Literal[
    "task_created",
    "task_started",
    "preflight_started",
    "preflight_completed",
    "agent_started",
    "browser_step",
    "browser_done",
    "task_succeeded",
    "task_failed",
    "heartbeat",
]


class Citation(BaseModel):
    source: str
    snippet: str
    score: float = 0.0
    chunk_id: str | None = None
    document_id: str | None = None
    knowledge_base_id: str | None = None


class Plan(BaseModel):
    intent: IntentType
    requires_rag: bool = False
    requires_tools: bool = False
    agents: list[str] = Field(default_factory=list)
    success_criteria: str
    notes: str | None = None


class AgentDescriptor(BaseModel):
    name: str
    description: str
    capabilities: list[str] = Field(default_factory=list)


class PlannerRunTrace(BaseModel):
    attempt: int
    source: Literal["model", "heuristic"]
    success: bool
    available_agents: list[str] = Field(default_factory=list)
    hints: list[str] = Field(default_factory=list)
    raw_output: str | None = None
    error: str | None = None


class ReflectionTrace(BaseModel):
    attempt: int
    failed_agent: str
    reason: str
    action: Literal["route", "replan", "fallback", "finish"]
    added_hint: str | None = None


class ToolSpec(BaseModel):
    name: str
    description: str
    source: ToolSource
    enabled: bool = True
    timeout_seconds: int = 10
    input_schema: dict[str, Any] = Field(default_factory=dict)
    allowed_intents: list[IntentType] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolCallTrace(BaseModel):
    call_id: str = Field(default_factory=lambda: uuid4().hex)
    name: str
    source: ToolSource
    input_summary: str
    success: bool
    latency_ms: int
    attempts: int = 1
    validated: bool = True
    error: str | None = None
    error_code: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    success: bool
    output: str
    structured_data: dict[str, Any] | None = None
    error: str | None = None
    error_code: str | None = None
    latency_ms: int


class ChatHistoryMessage(BaseModel):
    role: ChatRole
    content: str = Field(min_length=1)


class ErrorDetail(BaseModel):
    code: str
    message: str
    retryable: bool = False


class AgentExecutionContext(BaseModel):
    query: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    request_id: str = Field(min_length=1)
    knowledge_base_id: str | None = None
    plan_intent: IntentType | None = None
    chat_history: list[ChatHistoryMessage] = Field(default_factory=list)
    planner_hints: list[str] = Field(default_factory=list)


class AgentExecutionResult(BaseModel):
    agent_name: str
    success: bool
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    tool_result: ToolResult | None = None
    tool_calls: list[ToolCallTrace] = Field(default_factory=list)
    error_code: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatRequest(BaseModel):
    query: str = Field(min_length=1)
    session_id: str
    knowledge_base_id: str | None = None
    chat_history: list[ChatHistoryMessage] = Field(default_factory=list)


class BrowserTaskRequest(BaseModel):
    query: str = Field(min_length=1)
    session_id: str
    allowed_domains: list[str] = Field(default_factory=list)
    cdp_url: str | None = None
    health_url: str | None = None
    info_url: str | None = None
    bearer_token: str | None = None
    max_steps: int | None = None
    max_failures: int | None = None
    retry_delay: int | None = None
    max_actions_per_step: int | None = None
    use_vision: bool | None = None
    disable_env_proxy: bool | None = None


class BrowserTaskResult(BaseModel):
    output: str
    structured_data: dict[str, Any] | None = None


class BrowserTaskEvent(BaseModel):
    seq: int
    task_id: str
    type: BrowserTaskEventType
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    data: dict[str, Any] = Field(default_factory=dict)


class BrowserTaskSummary(BaseModel):
    task_id: str
    status: BrowserTaskStatus
    session_id: str
    query: str
    result: BrowserTaskResult | None = None
    error: ErrorDetail | None = None
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None = None
    status_url: str
    events_url: str


class AgentRunTrace(BaseModel):
    agent_name: str
    success: bool
    answer_preview: str
    error_code: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatTrace(BaseModel):
    request_id: str
    session_id: str
    plan: Plan | None = None
    planner_runs: list[PlannerRunTrace] = Field(default_factory=list)
    agents: list[str] = Field(default_factory=list)
    executed_agents: list[str] = Field(default_factory=list)
    tool_calls: list[ToolCallTrace] = Field(default_factory=list)
    agent_runs: list[AgentRunTrace] = Field(default_factory=list)
    intermediate_results: list[dict[str, Any]] = Field(default_factory=list)
    reflection_count: int = 0
    reflections: list[ReflectionTrace] = Field(default_factory=list)
    error: ErrorDetail | None = None


class ResponseMeta(BaseModel):
    request_id: str
    session_id: str
    duration_ms: int
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    trace: ChatTrace
    error: ErrorDetail | None = None
    meta: ResponseMeta


class RAGRebuildRequest(BaseModel):
    chunking: ChunkingSettings | None = None


class RAGConfigResponse(BaseModel):
    enabled: bool
    docs_path: str
    top_k: int
    vector_store_backend: str
    collection_name: str
    chunking: dict[str, Any]
    available_chunking_strategies: list[dict[str, Any]] = Field(default_factory=list)
    index_status: dict[str, Any] = Field(default_factory=dict)


class RAGRebuildResponse(BaseModel):
    chunk_count: int
    chunking: dict[str, Any]
    index_status: dict[str, Any] = Field(default_factory=dict)


class KnowledgeBaseCreateRequest(BaseModel):
    code: str = Field(min_length=2, max_length=64)
    name: str = Field(min_length=1, max_length=128)
    description: str | None = None
    embedding_provider: str | None = None
    embedding_model: str | None = None
    embedding_dimension: int | None = None
    vector_backend: str | None = None


class KnowledgeBaseSummary(BaseModel):
    knowledge_base_id: str
    code: str
    name: str
    description: str | None = None
    status: str
    embedding_provider: str
    embedding_model: str
    embedding_dimension: int
    vector_backend: str
    vector_collection: str
    storage_backend: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime


class KnowledgeBaseListResponse(BaseModel):
    items: list[KnowledgeBaseSummary] = Field(default_factory=list)


class KnowledgeDocumentSummary(BaseModel):
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
    status: str
    version: int = 1
    is_latest: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime


class KnowledgeDocumentListResponse(BaseModel):
    items: list[KnowledgeDocumentSummary] = Field(default_factory=list)


class IngestionJobResponse(BaseModel):
    job_id: int
    knowledge_base_id: str
    document_id: str | None = None
    source: str
    status: str
    stage: str
    message: str | None = None
    error_message: str | None = None
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    updated_at: datetime


class DocumentUploadResponse(BaseModel):
    knowledge_base_id: str
    document: KnowledgeDocumentSummary
    job: IngestionJobResponse


class DocumentDeleteResponse(BaseModel):
    knowledge_base_id: str
    document_id: str
    deleted: bool
