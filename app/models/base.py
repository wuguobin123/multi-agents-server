from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str
    content: str


class ModelResponse(BaseModel):
    content: str
    finish_reason: str = "stop"
    raw: dict[str, Any] = Field(default_factory=dict)


class ModelProvider(ABC):
    @abstractmethod
    async def chat(self, messages: list[ChatMessage], *, config: dict[str, Any] | None = None) -> ModelResponse:
        raise NotImplementedError

    @abstractmethod
    async def stream(
        self,
        messages: list[ChatMessage],
        *,
        config: dict[str, Any] | None = None,
    ) -> AsyncIterator[str]:
        raise NotImplementedError

    @abstractmethod
    def supports_tool_calling(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def supports_structured_output(self) -> bool:
        raise NotImplementedError
