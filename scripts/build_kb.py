from __future__ import annotations

from app.config import get_settings
from app.rag import RAGService
from app.repositories import build_execution_repository


def main() -> None:
    settings = get_settings()
    repository = build_execution_repository(settings)
    rag_settings = settings.model_copy(deep=True)
    rag_settings.rag.bootstrap_on_startup = False
    rag_service = RAGService(rag_settings, repository)
    profile = rag_service.embedding_profile()
    print(
        "knowledge base rebuild starting",
        {
            "provider": profile["provider"],
            "model": profile["model"],
            "dimension": profile["dimension"],
            "batch_size": profile["batch_size"],
            "fingerprint": profile["fingerprint"],
        },
    )
    chunks = rag_service.rebuild()
    status = rag_service.index_status()
    print(
        "knowledge base rebuild completed",
        {
            "chunk_count": len(chunks),
            "provider": profile["provider"],
            "model": profile["model"],
            "dimension": profile["dimension"],
            "fingerprint": profile["fingerprint"],
            "backend": settings.rag.vector_store_backend,
            "manifest_path": status["manifest_path"],
        },
    )


if __name__ == "__main__":
    main()
