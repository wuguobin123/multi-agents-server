from __future__ import annotations

import io
from contextlib import redirect_stdout
from pathlib import Path

from app.config import get_settings
from app.config import load_settings
from app.rag import RAGService
from scripts import bootstrap_runtime


def test_bootstrap_runtime_initializes_local_knowledge_base(tmp_path: Path, monkeypatch) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "deploy.md").write_text("部署流程包括构建镜像、启动容器、执行健康检查。", encoding="utf-8")
    vector_path = tmp_path / "vectors.json"
    config_path = tmp_path / "app.yaml"
    config_path.write_text(
        f"""
model:
  provider: mock
rag:
  enabled: true
  docs_path: {docs_dir}
  vector_store_backend: local
  local_store_path: {vector_path}
  bootstrap_on_startup: false
database:
  enabled: false
""".strip(),
        encoding="utf-8",
    )

    monkeypatch.setenv("APP_CONFIG_PATH", str(config_path))
    monkeypatch.setenv("BOOTSTRAP_KB_ON_STARTUP", "true")
    get_settings.cache_clear()

    stdout = io.StringIO()
    with redirect_stdout(stdout):
        bootstrap_runtime.main()

    assert vector_path.exists()
    assert "knowledge base initialized" in stdout.getvalue()

    get_settings.cache_clear()


def test_bootstrap_runtime_skips_rebuild_when_vectors_exist(tmp_path: Path, monkeypatch) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "deploy.md").write_text("部署流程包括构建镜像。", encoding="utf-8")
    vector_path = tmp_path / "vectors.json"
    config_path = tmp_path / "app.yaml"
    config_path.write_text(
        f"""
model:
  provider: mock
rag:
  enabled: true
  docs_path: {docs_dir}
  vector_store_backend: local
  local_store_path: {vector_path}
  bootstrap_on_startup: false
database:
  enabled: false
""".strip(),
        encoding="utf-8",
    )
    rag = RAGService(load_settings(config_path))
    rag.rebuild()

    monkeypatch.setenv("APP_CONFIG_PATH", str(config_path))
    monkeypatch.setenv("BOOTSTRAP_KB_ON_STARTUP", "true")
    get_settings.cache_clear()

    stdout = io.StringIO()
    with redirect_stdout(stdout):
        bootstrap_runtime.main()

    assert "knowledge base already initialized" in stdout.getvalue()

    get_settings.cache_clear()


def test_bootstrap_runtime_rebuilds_when_embedding_fingerprint_changes(tmp_path: Path, monkeypatch) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "deploy.md").write_text("部署流程包括构建镜像。", encoding="utf-8")
    vector_path = tmp_path / "vectors.json"
    initial_config_path = tmp_path / "initial-app.yaml"
    initial_config_path.write_text(
        f"""
model:
  provider: mock
rag:
  enabled: true
  docs_path: {docs_dir}
  vector_store_backend: local
  local_store_path: {vector_path}
  embedding_dimension: 128
  bootstrap_on_startup: false
database:
  enabled: false
""".strip(),
        encoding="utf-8",
    )
    rag = RAGService(load_settings(initial_config_path))
    rag.rebuild()

    switched_config_path = tmp_path / "switched-app.yaml"
    switched_config_path.write_text(
        f"""
model:
  provider: mock
rag:
  enabled: true
  docs_path: {docs_dir}
  vector_store_backend: local
  local_store_path: {vector_path}
  embedding_dimension: 64
  bootstrap_on_startup: false
database:
  enabled: false
""".strip(),
        encoding="utf-8",
    )

    monkeypatch.setenv("APP_CONFIG_PATH", str(switched_config_path))
    monkeypatch.setenv("BOOTSTRAP_KB_ON_STARTUP", "true")
    get_settings.cache_clear()

    stdout = io.StringIO()
    with redirect_stdout(stdout):
        bootstrap_runtime.main()

    output = stdout.getvalue()
    assert "knowledge base initialized" in output
    assert "embedding_fingerprint_changed" in output

    get_settings.cache_clear()
