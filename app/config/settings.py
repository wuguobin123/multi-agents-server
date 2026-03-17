from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT_DIR / "configs" / "app.yaml"


class ModelSettings(BaseModel):
    provider: str = "mock"
    name: str = "mock-chat"
    temperature: float = 0.3
    max_tokens: int = 4096
    stream: bool = False
    timeout_seconds: int = 20
    api_key: str | None = None
    base_url: str | None = None


class AgentToggle(BaseModel):
    enabled: bool = True


class AgentsSettings(BaseModel):
    planner: AgentToggle = Field(default_factory=AgentToggle)
    qa_agent: AgentToggle = Field(default_factory=AgentToggle)
    tool_agent: AgentToggle = Field(default_factory=AgentToggle)
    fallback_agent: AgentToggle = Field(default_factory=AgentToggle)


class ToolConfig(BaseModel):
    name: str
    description: str
    enabled: bool = True
    source: str
    timeout_seconds: int = 10
    input_schema: dict[str, Any] = Field(default_factory=dict)
    allowed_intents: list[str] = Field(default_factory=list)
    handler: str | None = None
    endpoint: str | None = None
    method: str = "POST"
    static_payload: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolsSettings(BaseModel):
    timeout_seconds: int = 10
    max_calls: int = 3
    retry_attempts: int = 1
    retryable_error_codes: list[str] = Field(
        default_factory=lambda: ["dependency_unavailable", "request_timeout", "request_failed"]
    )
    circuit_breaker_threshold: int = 3
    circuit_breaker_cooldown_seconds: int = 30
    allowed_sources: list[str] = Field(default_factory=lambda: ["skill", "mcp"])
    allowed_tools: list[str] = Field(default_factory=list)
    blocked_tools: list[str] = Field(default_factory=list)
    skill_tools_enabled: bool = True
    mcp_tools_enabled: bool = True
    skill_tools: list[ToolConfig] = Field(default_factory=list)
    mcp_tools: list[ToolConfig] = Field(default_factory=list)


ChunkingStrategy = Literal["recursive_character", "qa_pair"]


class ChunkingSettings(BaseModel):
    strategy: ChunkingStrategy = "recursive_character"
    chunk_size: int = 700
    chunk_overlap: int = 80
    qa_question_prefixes: list[str] = Field(
        default_factory=lambda: ["Q:", "Q：", "问:", "问：", "Question:"]
    )
    qa_answer_prefixes: list[str] = Field(
        default_factory=lambda: ["A:", "A：", "答:", "答：", "Answer:"]
    )
    qa_fallback_to_recursive: bool = True

    def profile(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class RAGSettings(BaseModel):
    enabled: bool = True
    docs_path: str = "data/knowledge_base"
    uploads_path: str = "data/uploads"
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    top_k: int = 4
    vector_store_backend: str = "local"
    local_store_path: str = "data/vector_store.json"
    collection_name: str = "knowledge_chunks"
    storage_backend: str = "local"
    storage_public_base_url: str | None = None
    embedding_provider: str = "mock"
    embedding_model: str = "mock-embedding"
    embedding_dimension: int = 256
    embedding_api_key: str | None = None
    embedding_base_url: str | None = None
    embedding_batch_size: int = 32
    embedding_timeout_seconds: int = 20
    embedding_max_retries: int = 2
    bootstrap_on_startup: bool = True
    request_timeout_seconds: int = 10
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _normalize_chunking_settings(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        normalized = dict(data)
        raw_chunking = normalized.get("chunking")
        chunking = dict(raw_chunking) if isinstance(raw_chunking, dict) else {}
        legacy_mapping = {
            "splitter_strategy": "strategy",
            "chunk_size": "chunk_size",
            "chunk_overlap": "chunk_overlap",
            "qa_question_prefixes": "qa_question_prefixes",
            "qa_answer_prefixes": "qa_answer_prefixes",
            "qa_fallback_to_recursive": "qa_fallback_to_recursive",
        }
        for legacy_key, chunking_key in legacy_mapping.items():
            if legacy_key in normalized and chunking_key not in chunking:
                chunking[chunking_key] = normalized[legacy_key]
        if chunking:
            normalized["chunking"] = chunking
        return normalized


class ServerSettings(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8080
    log_level: str = "INFO"


class ObservabilitySettings(BaseModel):
    json_logs: bool = True


class AppRuntimeSettings(BaseModel):
    max_reflections: int = 2
    request_timeout_seconds: int = 30


class DatabaseSettings(BaseModel):
    enabled: bool = True
    url: str = "sqlite:///data/runtime.db"
    auto_init: bool = True
    echo: bool = False
    pool_size: int = 5
    max_overflow: int = 10
    pool_recycle_seconds: int = 1800


class AppSettings(BaseModel):
    model: ModelSettings = Field(default_factory=ModelSettings)
    agents: AgentsSettings = Field(default_factory=AgentsSettings)
    tools: ToolsSettings = Field(default_factory=ToolsSettings)
    rag: RAGSettings = Field(default_factory=RAGSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    server: ServerSettings = Field(default_factory=ServerSettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)
    app: AppRuntimeSettings = Field(default_factory=AppRuntimeSettings)


def _read_yaml_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file_obj:
        data = yaml.safe_load(file_obj) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")
    return data


def _apply_env_overrides(config: dict[str, Any]) -> dict[str, Any]:
    def first_env(*names: str) -> str | None:
        for name in names:
            value = os.getenv(name)
            if value not in (None, ""):
                return value
        return None

    result = dict(config)
    result.setdefault("model", {})
    result.setdefault("database", {})
    result.setdefault("rag", {})
    result["rag"].setdefault("chunking", {})
    result.setdefault("server", {})
    result.setdefault("app", {})

    env_pairs = {
        ("model", "provider"): os.getenv("MODEL_PROVIDER"),
        ("model", "name"): os.getenv("MODEL_NAME"),
        ("model", "timeout_seconds"): os.getenv("MODEL_TIMEOUT_SECONDS"),
        ("model", "api_key"): os.getenv("MINIMAX_API_KEY"),
        ("model", "base_url"): os.getenv("MINIMAX_BASE_URL"),
        ("database", "enabled"): os.getenv("DATABASE_ENABLED"),
        ("database", "url"): os.getenv("DATABASE_URL"),
        ("database", "auto_init"): os.getenv("DATABASE_AUTO_INIT"),
        ("rag", "vector_store_backend"): os.getenv("RAG_VECTOR_STORE_BACKEND"),
        ("rag", "local_store_path"): os.getenv("RAG_LOCAL_STORE_PATH"),
        ("rag", "collection_name"): os.getenv("RAG_COLLECTION_NAME"),
        ("rag", "uploads_path"): os.getenv("RAG_UPLOADS_PATH"),
        ("rag", "storage_backend"): os.getenv("RAG_STORAGE_BACKEND"),
        ("rag", "storage_public_base_url"): os.getenv("RAG_STORAGE_PUBLIC_BASE_URL"),
        ("rag", "embedding_provider"): first_env("EMBEDDING_PROVIDER", "RAG_EMBEDDING_PROVIDER"),
        ("rag", "embedding_model"): first_env("EMBEDDING_MODEL", "RAG_EMBEDDING_MODEL"),
        ("rag", "embedding_dimension"): first_env("EMBEDDING_DIMENSION", "RAG_EMBEDDING_DIMENSION"),
        ("rag", "embedding_api_key"): first_env("EMBEDDING_API_KEY", "RAG_EMBEDDING_API_KEY"),
        ("rag", "embedding_base_url"): first_env("EMBEDDING_BASE_URL", "RAG_EMBEDDING_BASE_URL"),
        ("rag", "embedding_batch_size"): first_env("EMBEDDING_BATCH_SIZE", "RAG_EMBEDDING_BATCH_SIZE"),
        ("rag", "embedding_timeout_seconds"): first_env(
            "EMBEDDING_TIMEOUT_SECONDS",
            "RAG_EMBEDDING_TIMEOUT_SECONDS",
        ),
        ("rag", "embedding_max_retries"): first_env("EMBEDDING_MAX_RETRIES", "RAG_EMBEDDING_MAX_RETRIES"),
        ("rag", "bootstrap_on_startup"): os.getenv("RAG_BOOTSTRAP_ON_STARTUP"),
        ("rag", "qdrant_url"): os.getenv("QDRANT_URL"),
        ("rag", "qdrant_api_key"): os.getenv("QDRANT_API_KEY"),
        ("server", "host"): os.getenv("SERVER_HOST"),
        ("server", "port"): os.getenv("SERVER_PORT"),
        ("server", "log_level"): os.getenv("LOG_LEVEL"),
        ("app", "request_timeout_seconds"): os.getenv("APP_REQUEST_TIMEOUT_SECONDS"),
    }
    for (section, key), value in env_pairs.items():
        if value in (None, ""):
            continue
        result[section][key] = value

    chunking_env_pairs = {
        "strategy": os.getenv("RAG_CHUNKING_STRATEGY"),
        "chunk_size": os.getenv("RAG_CHUNK_SIZE"),
        "chunk_overlap": os.getenv("RAG_CHUNK_OVERLAP"),
        "qa_question_prefixes": os.getenv("RAG_QA_QUESTION_PREFIXES"),
        "qa_answer_prefixes": os.getenv("RAG_QA_ANSWER_PREFIXES"),
        "qa_fallback_to_recursive": os.getenv("RAG_QA_FALLBACK_TO_RECURSIVE"),
    }
    for key, value in chunking_env_pairs.items():
        if value in (None, ""):
            continue
        if key in {"qa_question_prefixes", "qa_answer_prefixes"}:
            result["rag"]["chunking"][key] = [item.strip() for item in value.split(",") if item.strip()]
            continue
        result["rag"]["chunking"][key] = value
    return result


def load_settings(config_path: str | Path | None = None) -> AppSettings:
    path = Path(config_path or os.getenv("APP_CONFIG_PATH", DEFAULT_CONFIG_PATH))
    if not path.is_absolute():
        path = ROOT_DIR / path
    config = _apply_env_overrides(_read_yaml_config(path))
    return AppSettings.model_validate(config)


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    return load_settings()
