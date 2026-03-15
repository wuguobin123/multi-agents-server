from __future__ import annotations

from app.config import AppSettings
from app.models.base import ModelProvider
from app.models.minimax import MiniMaxProvider
from app.models.mock import MockProvider
from app.observability import get_logger


logger = get_logger(__name__)


def build_model_provider(settings: AppSettings) -> ModelProvider:
    provider = settings.model.provider.lower()
    if provider == "mock":
        return MockProvider()
    if provider == "minimax":
        if not settings.model.api_key or not settings.model.base_url:
            raise ValueError("MiniMax config incomplete: MINIMAX_API_KEY and MINIMAX_BASE_URL are required.")
        return MiniMaxProvider(settings)
    raise ValueError(f"Unsupported model provider: {settings.model.provider}")
