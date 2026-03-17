from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings, load_settings
from app.main import app
from app.runtime import AppRuntime
from app.runtime import get_runtime
from app.schemas import ChatRequest


@pytest.fixture(autouse=True)
def isolated_app_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_CONFIG_PATH", "configs/app.yaml")
    monkeypatch.setenv("MODEL_PROVIDER", "mock")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'runtime.db'}")
    monkeypatch.setenv("RAG_LOCAL_STORE_PATH", str(tmp_path / "vector_store.json"))
    monkeypatch.setenv("RAG_BOOTSTRAP_ON_STARTUP", "true")
    get_settings.cache_clear()
    get_runtime.cache_clear()
    yield
    get_runtime.cache_clear()
    get_settings.cache_clear()


def test_healthz() -> None:
    client = TestClient(app)

    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_readyz() -> None:
    client = TestClient(app)

    response = client.get("/readyz")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "checks" in body
    assert body["checks"]["repository"]["backend"] == "sql"
    assert body["checks"]["planner"]["status"] == "ok"
    assert body["checks"]["agents"]["agent_count"] >= 1


def test_chat_qa_flow() -> None:
    client = TestClient(app)

    response = client.post(
        "/v1/chat",
        json={"query": "帮我总结知识库中关于部署流程的说明", "session_id": "itest-qa"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["error"] is None
    assert body["meta"]["request_id"]
    assert body["meta"]["session_id"] == "itest-qa"
    assert body["trace"]["request_id"] == body["meta"]["request_id"]
    assert body["trace"]["plan"]["intent"] == "qa"
    assert body["trace"]["planner_runs"]
    assert body["trace"]["planner_runs"][0]["source"] in {"model", "heuristic"}
    assert body["trace"]["agent_runs"]
    assert body["citations"]
    assert response.text.endswith("\n")


def test_chat_tool_flow() -> None:
    client = TestClient(app)

    response = client.post(
        "/v1/chat",
        json={"query": "请调用工具执行一次检查", "session_id": "itest-tool"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["trace"]["plan"]["intent"] == "tool"
    assert body["trace"]["tool_calls"]
    assert "技能工具收到请求" in body["answer"]


def test_rag_config_endpoint_reports_chunking_options() -> None:
    client = TestClient(app)

    response = client.get("/v1/rag/config")

    assert response.status_code == 200
    body = response.json()
    assert body["enabled"] is True
    assert body["chunking"]["strategy"] == "recursive_character"
    assert any(item["name"] == "qa_pair" for item in body["available_chunking_strategies"])


def test_rag_rebuild_endpoint_accepts_chunking_override() -> None:
    client = TestClient(app)

    response = client.post(
        "/v1/rag/rebuild",
        json={
            "chunking": {
                "strategy": "qa_pair",
                "chunk_size": 700,
                "chunk_overlap": 80,
                "qa_question_prefixes": ["Q:", "问："],
                "qa_answer_prefixes": ["A:", "答："],
                "qa_fallback_to_recursive": True,
            }
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["chunk_count"] >= 1
    assert body["chunking"]["strategy"] == "qa_pair"
    assert body["index_status"]["needs_rebuild"] is False


def test_chat_reflects_and_falls_back_when_tool_path_has_no_available_tool(tmp_path) -> None:
    config_path = tmp_path / "app.yaml"
    config_path.write_text(
        """
model:
  provider: mock
agents:
  planner:
    enabled: true
  qa_agent:
    enabled: true
  tool_agent:
    enabled: true
  fallback_agent:
    enabled: true
tools:
  timeout_seconds: 10
  max_calls: 3
  skill_tools_enabled: false
  mcp_tools_enabled: false
rag:
  enabled: false
  bootstrap_on_startup: false
database:
  enabled: false
app:
  max_reflections: 0
  request_timeout_seconds: 30
""".strip(),
        encoding="utf-8",
    )
    runtime = AppRuntime(load_settings(config_path))

    response = __import__("asyncio").run(
        runtime.handle_chat(
            ChatRequest(query="请调用工具执行一次检查", session_id="itest-fallback"),
            request_id="req-fallback-001",
        )
    )

    assert response.error is not None
    assert response.trace.plan is not None
    assert response.trace.plan.intent == "tool"
    assert response.trace.reflection_count == 0
    assert response.trace.reflections
    assert any(item.action == "fallback" for item in response.trace.reflections)
    assert response.trace.agents[-1] == "fallback_agent"
    assert "不能可靠完成" in response.answer


def test_knowledge_base_upload_and_chat_flow() -> None:
    client = TestClient(app)

    create_response = client.post(
        "/v1/knowledge-bases",
        json={"code": "ops", "name": "Ops KB", "embedding_provider": "mock", "embedding_model": "mock-embedding"},
    )
    assert create_response.status_code == 200
    knowledge_base_id = create_response.json()["knowledge_base_id"]

    upload_response = client.post(
        f"/v1/knowledge-bases/{knowledge_base_id}/documents",
        data={"chunking_strategy": "recursive_character", "chunk_size": "300", "chunk_overlap": "20"},
        files={"file": ("ops.md", "发布流程先构建镜像，再灰度发布，并校验健康检查。", "text/markdown")},
    )
    assert upload_response.status_code == 200
    document_id = upload_response.json()["document"]["document_id"]
    job_id = upload_response.json()["job"]["job_id"]

    job_response = client.get(f"/v1/ingestion-jobs/{job_id}")
    assert job_response.status_code == 200
    assert job_response.json()["status"] == "completed"
    assert job_response.json()["stage"] == "indexed"

    list_response = client.get(f"/v1/knowledge-bases/{knowledge_base_id}/documents")
    assert list_response.status_code == 200
    items = list_response.json()["items"]
    assert len(items) == 1
    assert items[0]["document_id"] == document_id
    assert items[0]["status"] == "ready"
    assert items[0]["version"] == 1

    chat_response = client.post(
        "/v1/chat",
        json={
            "query": "总结发布流程",
            "session_id": "itest-kb-chat",
            "knowledge_base_id": knowledge_base_id,
        },
    )
    assert chat_response.status_code == 200
    body = chat_response.json()
    assert body["error"] is None
    assert body["citations"]
    assert all(item["knowledge_base_id"] == knowledge_base_id for item in body["citations"])


def test_duplicate_upload_marks_new_version_as_duplicate() -> None:
    client = TestClient(app)

    kb_response = client.post("/v1/knowledge-bases", json={"code": "faq", "name": "FAQ KB"})
    knowledge_base_id = kb_response.json()["knowledge_base_id"]

    first_upload = client.post(
        f"/v1/knowledge-bases/{knowledge_base_id}/documents",
        files={"file": ("faq.md", "Q: 口令是什么？\nA: 蓝莓管道-20260315", "text/markdown")},
        data={"chunking_strategy": "qa_pair"},
    )
    assert first_upload.status_code == 200

    second_upload = client.post(
        f"/v1/knowledge-bases/{knowledge_base_id}/documents",
        files={"file": ("faq.md", "Q: 口令是什么？\nA: 蓝莓管道-20260315", "text/markdown")},
        data={"chunking_strategy": "qa_pair"},
    )
    assert second_upload.status_code == 200
    second_job = second_upload.json()["job"]
    assert second_job["status"] == "completed"
    assert second_job["stage"] == "deduplicated"

    list_response = client.get(f"/v1/knowledge-bases/{knowledge_base_id}/documents")
    items = list_response.json()["items"]
    assert len(items) == 2
    assert items[0]["version"] == 1
    assert items[1]["version"] == 2
    assert items[1]["status"] == "duplicate"


def test_reindex_and_delete_document_endpoints() -> None:
    client = TestClient(app)

    kb_response = client.post("/v1/knowledge-bases", json={"code": "docs", "name": "Docs KB"})
    knowledge_base_id = kb_response.json()["knowledge_base_id"]

    upload_response = client.post(
        f"/v1/knowledge-bases/{knowledge_base_id}/documents",
        files={"file": ("deploy.md", "部署步骤：先构建镜像，再启动服务。", "text/markdown")},
    )
    document_id = upload_response.json()["document"]["document_id"]

    reindex_response = client.post(f"/v1/knowledge-bases/{knowledge_base_id}/documents/{document_id}/reindex")
    assert reindex_response.status_code == 200
    assert reindex_response.json()["status"] in {"queued", "completed"}

    delete_response = client.delete(f"/v1/knowledge-bases/{knowledge_base_id}/documents/{document_id}")
    assert delete_response.status_code == 200
    assert delete_response.json()["deleted"] is True

    list_response = client.get(f"/v1/knowledge-bases/{knowledge_base_id}/documents")
    assert list_response.status_code == 200
    assert list_response.json()["items"] == []
