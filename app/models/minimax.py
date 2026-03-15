from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx

from app.config import AppSettings
from app.models.base import ChatMessage, ModelProvider, ModelResponse


class MiniMaxProvider(ModelProvider):
    """
    Uses MiniMax native text chat completions endpoint.
    """

    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings.model

    async def chat(self, messages: list[ChatMessage], *, config: dict[str, Any] | None = None) -> ModelResponse:
        request_config = config or {}
        payload = {
            "model": request_config.get("model", self._settings.name),
            "messages": [message.model_dump() for message in messages],
            "temperature": request_config.get("temperature", self._settings.temperature),
            "top_p": request_config.get("top_p", 0.95),
            "max_completion_tokens": request_config.get(
                "max_completion_tokens",
                request_config.get("max_tokens", self._settings.max_tokens),
            ),
            "stream": False,
        }
        headers = {
            "Authorization": f"Bearer {self._settings.api_key}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=self._settings.timeout_seconds) as client:
            response = await client.post(
                f"{self._settings.base_url.rstrip('/')}/text/chatcompletion_v2",
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
        body = response.json()
        base_resp = body.get("base_resp") or {}
        choices = body.get("choices")
        if not choices:
            status_code = base_resp.get("status_code")
            status_msg = base_resp.get("status_msg") or "MiniMax response missing choices."
            if status_code not in (None, 0):
                raise RuntimeError(f"MiniMax API error {status_code}: {status_msg}")
            raise RuntimeError(status_msg)
        choice = choices[0]
        message = choice.get("message", {})
        return ModelResponse(content=message.get("content", ""), finish_reason=choice.get("finish_reason", "stop"), raw=body)

    async def stream(self, messages: list[ChatMessage], *, config: dict[str, Any] | None = None) -> AsyncIterator[str]:
        response = await self.chat(messages, config=config)
        yield response.content

    def supports_tool_calling(self) -> bool:
        return False

    def supports_structured_output(self) -> bool:
        return False
