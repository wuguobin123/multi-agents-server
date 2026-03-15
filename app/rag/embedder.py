from __future__ import annotations

import math
import time
from typing import Protocol

import httpx

from app.config.settings import AppSettings, RAGSettings
from app.observability import get_logger


logger = get_logger(__name__)


class EmbeddingError(RuntimeError):
    pass


class Embedder(Protocol):
    @property
    def dimension(self) -> int: ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...

    def fingerprint(self) -> str: ...


class MockEmbedder:
    def __init__(self, dimension: int, model: str = "mock-embedding") -> None:
        self._dimension = dimension
        self._model = model

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def fingerprint(self) -> str:
        return f"mock:{self._model}:{self._dimension}"

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self._dimension
        for token in self._tokenize(text):
            index = hash(token) % self._dimension
            vector[index] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [round(value / norm, 8) for value in vector]

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        ascii_buffer: list[str] = []
        tokens: set[str] = set()
        cjk_chars: list[str] = []

        def flush_ascii() -> None:
            if not ascii_buffer:
                return
            token = "".join(ascii_buffer).lower()
            if len(token) > 1:
                tokens.add(token)
            ascii_buffer.clear()

        for char in text:
            if "\u4e00" <= char <= "\u9fff":
                flush_ascii()
                cjk_chars.append(char)
            elif char.isalnum():
                ascii_buffer.append(char)
            else:
                flush_ascii()
        flush_ascii()

        for index, char in enumerate(cjk_chars):
            tokens.add(char)
            if index + 1 < len(cjk_chars):
                tokens.add(char + cjk_chars[index + 1])
            if index + 2 < len(cjk_chars):
                tokens.add(char + cjk_chars[index + 1] + cjk_chars[index + 2])
        return tokens


class OpenAICompatibleEmbedder:
    def __init__(
        self,
        *,
        provider: str,
        model: str,
        dimension: int,
        api_key: str,
        base_url: str,
        timeout_seconds: int,
        max_retries: int,
    ) -> None:
        self._provider = provider
        self._model = model
        self._dimension = dimension
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._max_retries = max_retries

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._request_embeddings(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._request_embeddings([text])[0]

    def fingerprint(self) -> str:
        return f"{self._provider}:{self._model}:{self._dimension}"

    def _request_embeddings(self, texts: list[str]) -> list[list[float]]:
        payload = {
            "model": self._model,
            "input": texts,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            started_at = time.perf_counter()
            try:
                response = httpx.post(
                    f"{self._base_url}/embeddings",
                    json=payload,
                    headers=headers,
                    timeout=self._timeout_seconds,
                )
                response.raise_for_status()
                data = response.json().get("data", [])
                vectors = [item["embedding"] for item in data]
                if len(vectors) != len(texts):
                    raise EmbeddingError("Embedding result count does not match request batch size.")
                for vector in vectors:
                    if len(vector) != self._dimension:
                        raise EmbeddingError(
                            f"Embedding dimension mismatch: expected {self._dimension}, got {len(vector)}."
                        )
                logger.info(
                    "embedding_request_succeeded",
                    extra={
                        "provider": self._provider,
                        "model": self._model,
                        "batch_size": len(texts),
                        "latency_ms": int((time.perf_counter() - started_at) * 1000),
                    },
                )
                return vectors
            except (httpx.HTTPError, KeyError, ValueError, EmbeddingError) as exc:
                last_error = exc
                logger.warning(
                    "embedding_request_failed",
                    extra={
                        "provider": self._provider,
                        "model": self._model,
                        "batch_size": len(texts),
                        "attempt": attempt + 1,
                        "max_attempts": self._max_retries + 1,
                        "error": str(exc),
                    },
                )
                if attempt >= self._max_retries:
                    break
        raise EmbeddingError(f"Embedding request failed for provider {self._provider}: {last_error}") from last_error


def build_embedder(settings: AppSettings | RAGSettings) -> Embedder:
    rag_settings = settings.rag if isinstance(settings, AppSettings) else settings
    provider = rag_settings.embedding_provider.lower()
    if provider == "mock":
        return MockEmbedder(rag_settings.embedding_dimension, rag_settings.embedding_model)
    if provider in {"dashscope_openai_compatible", "openai_compatible"}:
        if not rag_settings.embedding_api_key or not rag_settings.embedding_base_url:
            raise ValueError(
                f"Embedding provider '{provider}' requires embedding_api_key and embedding_base_url."
            )
        return OpenAICompatibleEmbedder(
            provider=provider,
            model=rag_settings.embedding_model,
            dimension=rag_settings.embedding_dimension,
            api_key=rag_settings.embedding_api_key,
            base_url=rag_settings.embedding_base_url,
            timeout_seconds=rag_settings.embedding_timeout_seconds,
            max_retries=rag_settings.embedding_max_retries,
        )
    raise ValueError(f"Unsupported embedding provider: {rag_settings.embedding_provider}")
