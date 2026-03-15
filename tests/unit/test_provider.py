import pytest

from app.config import load_settings
from app.models import ChatMessage, build_model_provider
from app.models.minimax import MiniMaxProvider


async def test_mock_provider_returns_content() -> None:
    settings = load_settings()
    settings.model.provider = "mock"
    settings.model.name = "mock-chat"
    provider = build_model_provider(settings)

    response = await provider.chat([ChatMessage(role="user", content="hello world")])

    assert "hello world" in response.content


def test_minimax_provider_requires_credentials() -> None:
    settings = load_settings()
    settings.model.provider = "minimax"
    settings.model.api_key = None
    settings.model.base_url = "https://api.minimaxi.com/v1"

    with pytest.raises(ValueError, match="MiniMax config incomplete"):
        build_model_provider(settings)


async def test_minimax_provider_uses_native_chatcompletion_v2(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": "native minimax response",
                        },
                    }
                ]
            }

    class FakeAsyncClient:
        def __init__(self, *, timeout: int) -> None:
            captured["timeout"] = timeout

        async def __aenter__(self) -> "FakeAsyncClient":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def post(self, url: str, *, json: dict[str, object], headers: dict[str, str]) -> FakeResponse:
            captured["url"] = url
            captured["json"] = json
            captured["headers"] = headers
            return FakeResponse()

    monkeypatch.setattr("app.models.minimax.httpx.AsyncClient", FakeAsyncClient)

    settings = load_settings()
    settings.model.provider = "minimax"
    settings.model.name = "MiniMax-M2.5-highspeed"
    settings.model.api_key = "test-key"
    settings.model.base_url = "https://api.minimaxi.com/v1"
    provider = MiniMaxProvider(settings)

    response = await provider.chat([ChatMessage(role="user", content="hello minimax")])

    assert response.content == "native minimax response"
    assert captured["url"] == "https://api.minimaxi.com/v1/text/chatcompletion_v2"
    assert captured["headers"] == {
        "Authorization": "Bearer test-key",
        "Content-Type": "application/json",
    }
    assert captured["json"] == {
        "model": "MiniMax-M2.5-highspeed",
        "messages": [{"role": "user", "content": "hello minimax"}],
        "temperature": settings.model.temperature,
        "top_p": 0.95,
        "max_completion_tokens": settings.model.max_tokens,
        "stream": False,
    }


async def test_minimax_provider_raises_clear_error_when_model_is_not_supported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {
                "choices": None,
                "base_resp": {
                    "status_code": 2061,
                    "status_msg": "your current code plan not support model, MiniMax-M2.5-highspeed",
                },
            }

    class FakeAsyncClient:
        def __init__(self, *, timeout: int) -> None:
            self.timeout = timeout

        async def __aenter__(self) -> "FakeAsyncClient":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def post(self, url: str, *, json: dict[str, object], headers: dict[str, str]) -> FakeResponse:
            return FakeResponse()

    monkeypatch.setattr("app.models.minimax.httpx.AsyncClient", FakeAsyncClient)

    settings = load_settings()
    settings.model.provider = "minimax"
    settings.model.name = "MiniMax-M2.5-highspeed"
    settings.model.api_key = "test-key"
    settings.model.base_url = "https://api.minimaxi.com/v1"
    provider = MiniMaxProvider(settings)

    with pytest.raises(RuntimeError, match="MiniMax API error 2061"):
        await provider.chat([ChatMessage(role="user", content="hello minimax")])
