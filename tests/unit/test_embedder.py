from pathlib import Path

import pytest

from app.config import load_settings
from app.rag.embedder import MockEmbedder, build_embedder


def test_build_embedder_returns_mock_by_default(tmp_path: Path) -> None:
    config_path = tmp_path / "app.yaml"
    config_path.write_text(
        """
rag:
  embedding_provider: mock
  embedding_dimension: 128
database:
  enabled: false
""".strip(),
        encoding="utf-8",
    )

    settings = load_settings(config_path)
    embedder = build_embedder(settings)

    assert isinstance(embedder, MockEmbedder)
    assert embedder.dimension == 128
    assert embedder.fingerprint() == "mock:mock-embedding:128"


def test_build_embedder_requires_credentials_for_openai_compatible_provider(tmp_path: Path) -> None:
    config_path = tmp_path / "app.yaml"
    config_path.write_text(
        """
rag:
  embedding_provider: dashscope_openai_compatible
  embedding_model: text-embedding-v4
  embedding_dimension: 1024
database:
  enabled: false
""".strip(),
        encoding="utf-8",
    )

    settings = load_settings(config_path)

    with pytest.raises(ValueError, match="embedding_api_key"):
        build_embedder(settings)
