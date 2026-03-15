from pathlib import Path

import pytest

from app.config import load_settings
from app.rag import RAGService


async def test_rag_returns_empty_when_manifest_has_no_chunks(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    config_path = tmp_path / "app.yaml"
    config_path.write_text(
        f"""
model:
  provider: mock
rag:
  enabled: true
  docs_path: {docs_dir}
  vector_store_backend: local
  local_store_path: {tmp_path / "vectors.json"}
database:
  enabled: false
""".strip(),
        encoding="utf-8",
    )

    settings = load_settings(config_path)
    rag = RAGService(settings)

    citations = await rag.search("anything")

    assert citations == []


async def test_rag_rebuilds_and_returns_citations_from_local_vector_store(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "deploy.md").write_text("部署流程包括构建镜像、启动容器、执行健康检查。", encoding="utf-8")
    config_path = tmp_path / "app.yaml"
    config_path.write_text(
        f"""
model:
  provider: mock
rag:
  enabled: true
  docs_path: {docs_dir}
  vector_store_backend: local
  local_store_path: {tmp_path / "vectors.json"}
  bootstrap_on_startup: false
database:
  enabled: false
""".strip(),
        encoding="utf-8",
    )

    settings = load_settings(config_path)
    rag = RAGService(settings)
    chunks = rag.rebuild()
    citations = await rag.search("总结部署流程")

    assert chunks
    assert citations
    assert citations[0].source.endswith("deploy.md")
    readiness = rag.readiness()
    assert readiness["needs_rebuild"] is False
    assert readiness["index_metadata"] is not None


async def test_rag_blocks_search_when_embedding_fingerprint_changes(tmp_path: Path) -> None:
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
  embedding_dimension: 128
  bootstrap_on_startup: false
database:
  enabled: false
""".strip(),
        encoding="utf-8",
    )

    settings = load_settings(config_path)
    rag = RAGService(settings)
    rag.rebuild()

    switched_config_path = tmp_path / "app-switched.yaml"
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

    switched_settings = load_settings(switched_config_path)
    switched_rag = RAGService(switched_settings)

    with pytest.raises(RuntimeError, match="rebuild required"):
        await switched_rag.search("总结部署流程")
