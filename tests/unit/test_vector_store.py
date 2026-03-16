import httpx

from app.rag.vector_store import QdrantHttpVectorStore


def test_qdrant_vector_store_disables_env_proxy(monkeypatch) -> None:
    captured: list[dict[str, object]] = []

    def fake_request(method: str, url: str, **kwargs):
        captured.append(
            {
                "method": method,
                "url": url,
                "trust_env": kwargs.get("trust_env"),
            }
        )
        return httpx.Response(
            200,
            request=httpx.Request(method, url),
            json={"result": True, "status": "ok", "time": 0.0},
        )

    monkeypatch.setattr(httpx, "request", fake_request)

    QdrantHttpVectorStore(
        base_url="http://127.0.0.1:6333",
        collection_name="knowledge_chunks",
        vector_size=256,
    )

    assert captured
    assert captured[0]["method"] == "PUT"
    assert captured[0]["trust_env"] is False
