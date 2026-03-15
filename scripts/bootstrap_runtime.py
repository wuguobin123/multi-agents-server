from __future__ import annotations

import os

from app.config import get_settings
from app.rag import RAGService
from app.repositories import build_execution_repository


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def main() -> None:
    settings = get_settings()
    repository = build_execution_repository(settings)

    if settings.database.enabled:
        status = repository.ping()
        print(
            "database initialized",
            {
                "backend": status["backend"],
                "dialect": status.get("dialect"),
            },
        )

    if not settings.rag.enabled or not _env_flag("BOOTSTRAP_KB_ON_STARTUP", True):
        return

    rag_settings = settings.model_copy(deep=True)
    rag_settings.rag.bootstrap_on_startup = False
    rag_service = RAGService(rag_settings, repository)
    force_rebuild = _env_flag("FORCE_KB_REBUILD", False)
    index_status = rag_service.index_status(force_rebuild=force_rebuild)
    if index_status["needs_rebuild"]:
        chunks = rag_service.rebuild()
        print(
            "knowledge base initialized",
            {
                "chunk_count": len(chunks),
                "backend": settings.rag.vector_store_backend,
                "reasons": index_status["reasons"],
                "fingerprint": rag_service.embedding_profile()["fingerprint"],
            },
        )
    else:
        print(
            "knowledge base already initialized",
            {
                "point_count": index_status["point_count"],
                "backend": settings.rag.vector_store_backend,
                "fingerprint": rag_service.embedding_profile()["fingerprint"],
            },
        )


if __name__ == "__main__":
    main()
