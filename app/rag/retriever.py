from __future__ import annotations

from app.rag.embedder import Embedder
from app.rag.vector_store import VectorStore
from app.schemas import Citation


class Retriever:
    def __init__(self, embedder: Embedder, vector_store: VectorStore, *, default_top_k: int) -> None:
        self._embedder = embedder
        self._vector_store = vector_store
        self._default_top_k = default_top_k

    def search(self, query: str, *, top_k: int | None = None) -> list[Citation]:
        query_vector = self._embedder.embed_query(query)
        results = self._vector_store.query(query_vector, limit=top_k or self._default_top_k)
        return [
            Citation(
                source=item.source,
                snippet=item.text[:240],
                score=item.score,
            )
            for item in results
        ]
