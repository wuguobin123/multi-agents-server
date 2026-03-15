from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from app.config import AppSettings
from app.domain import IngestionJobRecord, KnowledgeChunkRecord, KnowledgeDocumentRecord
from app.observability import get_logger
from app.rag.embedder import build_embedder
from app.rag.index_manifest import IndexManifestStore
from app.rag.loader import FileSystemDocumentLoader
from app.rag.retriever import Retriever
from app.rag.splitter import DocumentSplitter
from app.rag.types import ChunkRecord, IndexMetadata, VectorPoint
from app.rag.vector_store import LocalVectorStore, QdrantHttpVectorStore, VectorStore
from app.repositories.base import ExecutionRepository
from app.schemas import Citation


logger = get_logger(__name__)


class RAGService:
    def __init__(self, settings: AppSettings, repository: ExecutionRepository | None = None) -> None:
        self._settings = settings.rag
        self._root_dir = Path(__file__).resolve().parents[2]
        self._repository = repository
        self._loader = FileSystemDocumentLoader(self._root_dir, self._settings.docs_path)
        self._splitter = DocumentSplitter(
            chunk_size=self._settings.chunk_size,
            chunk_overlap=self._settings.chunk_overlap,
        )
        self._embedder = build_embedder(settings)
        self._vector_store = self._build_vector_store()
        self._manifest = IndexManifestStore(self._build_manifest_path())
        self._retriever = Retriever(self._embedder, self._vector_store, default_top_k=self._settings.top_k)
        if self._settings.bootstrap_on_startup and self.index_status()["needs_rebuild"]:
            self.rebuild()

    def _build_vector_store(self) -> VectorStore:
        if self._settings.vector_store_backend == "qdrant":
            if not self._settings.qdrant_url:
                raise ValueError("Qdrant backend requires rag.qdrant_url")
            return QdrantHttpVectorStore(
                base_url=self._settings.qdrant_url,
                collection_name=self._settings.collection_name,
                vector_size=self._settings.embedding_dimension,
                api_key=self._settings.qdrant_api_key,
                timeout_seconds=self._settings.request_timeout_seconds,
            )
        return LocalVectorStore(self._root_dir / self._settings.local_store_path)

    def _build_manifest_path(self) -> Path:
        if self._settings.vector_store_backend == "local":
            store_path = self._root_dir / self._settings.local_store_path
            return store_path.parent / f"{store_path.stem}.manifest.json"
        return self._root_dir / "data" / f"{self._settings.collection_name}.manifest.json"

    @property
    def docs_path(self) -> Path:
        return self._loader.docs_path

    def embedding_profile(self) -> dict[str, object]:
        return {
            "provider": self._settings.embedding_provider,
            "model": self._settings.embedding_model,
            "dimension": self._embedder.dimension,
            "fingerprint": self._embedder.fingerprint(),
            "batch_size": self._settings.embedding_batch_size,
        }

    def index_metadata(self) -> IndexMetadata | None:
        return self._manifest.read()

    def index_status(self, *, force_rebuild: bool = False) -> dict[str, object]:
        manifest = self.index_metadata()
        point_count = self._vector_store.count()
        vector_size = self._vector_store.vector_size()
        reasons: list[str] = []

        if force_rebuild:
            reasons.append("force_rebuild")
        if point_count == 0:
            reasons.append("vector_store_empty")
        if manifest is None:
            reasons.append("index_manifest_missing")
        else:
            if manifest.embedding_fingerprint != self._embedder.fingerprint():
                reasons.append("embedding_fingerprint_changed")
            if manifest.embedding_dimension != self._embedder.dimension:
                reasons.append("manifest_dimension_mismatch")
            if manifest.chunk_count != point_count:
                reasons.append("chunk_count_mismatch")
            if manifest.vector_store_backend != self._settings.vector_store_backend:
                reasons.append("vector_store_backend_changed")
            if manifest.collection_name != self._settings.collection_name:
                reasons.append("collection_name_changed")
        if vector_size is not None and vector_size != self._embedder.dimension:
            reasons.append("vector_dimension_mismatch")

        deduped_reasons = list(dict.fromkeys(reasons))
        return {
            "needs_rebuild": bool(deduped_reasons),
            "reasons": deduped_reasons,
            "manifest_path": str(self._manifest.path),
            "point_count": point_count,
            "vector_size": vector_size,
            "current_embedding": self.embedding_profile(),
            "index_metadata": manifest.model_dump(mode="json") if manifest is not None else None,
        }

    def rebuild(self) -> list[ChunkRecord]:
        if not self._settings.enabled:
            return []
        job_id = None
        if self._repository is not None:
            job_id = self._repository.create_ingestion_job(
                IngestionJobRecord(
                    source=str(self._loader.docs_path),
                    status="running",
                    message="knowledge_base_rebuild_started",
                )
            )
        try:
            documents = self._loader.load()
            chunks = self._splitter.split(documents)
            metadata = IndexMetadata(
                embedding_provider=self._settings.embedding_provider,
                embedding_model=self._settings.embedding_model,
                embedding_dimension=self._embedder.dimension,
                embedding_fingerprint=self._embedder.fingerprint(),
                indexed_at=datetime.now(timezone.utc),
                chunk_count=len(chunks),
                vector_store_backend=self._settings.vector_store_backend,
                collection_name=self._settings.collection_name,
            )
            self._manifest.clear()
            self._vector_store.reset()
            batch_size = max(1, self._settings.embedding_batch_size)
            for batch_start in range(0, len(chunks), batch_size):
                batch = chunks[batch_start : batch_start + batch_size]
                try:
                    vectors = self._embedder.embed_documents([chunk.text for chunk in batch])
                except Exception:
                    logger.exception(
                        "knowledge_base_batch_embedding_failed",
                        extra={
                            "batch_start": batch_start,
                            "batch_size": len(batch),
                            "embedding_provider": self._settings.embedding_provider,
                            "embedding_model": self._settings.embedding_model,
                        },
                    )
                    raise
                points = [
                    VectorPoint(
                        point_id=str(uuid5(NAMESPACE_URL, chunk.chunk_id)),
                        vector=vector,
                        payload={
                            "chunk_id": chunk.chunk_id,
                            "document_id": chunk.document_id,
                            "source": chunk.source,
                            "text": chunk.text,
                            "chunk_index": chunk.chunk_index,
                            "metadata": chunk.metadata,
                        },
                    )
                    for chunk, vector in zip(batch, vectors, strict=False)
                ]
                self._vector_store.upsert(points)
            self._manifest.write(metadata)
            if self._repository is not None:
                self._repository.replace_knowledge_base(
                    [
                        KnowledgeDocumentRecord(
                            document_id=document.document_id,
                            source=document.source,
                            metadata=document.metadata,
                        )
                        for document in documents
                    ],
                    [
                        KnowledgeChunkRecord(
                            chunk_id=chunk.chunk_id,
                            document_id=chunk.document_id,
                            source=chunk.source,
                            text=chunk.text,
                            chunk_index=chunk.chunk_index,
                            metadata=chunk.metadata,
                        )
                        for chunk in chunks
                    ],
                )
                if job_id is not None:
                    self._repository.update_ingestion_job(
                        job_id,
                        status="completed",
                        message=f"ingested {len(chunks)} chunks with {metadata.embedding_fingerprint}",
                    )
            logger.info(
                "rag_rebuilt",
                extra={
                    "document_count": len(documents),
                    "chunk_count": len(chunks),
                    "vector_backend": self._settings.vector_store_backend,
                    "embedding_provider": self._settings.embedding_provider,
                    "embedding_model": self._settings.embedding_model,
                    "embedding_dimension": self._embedder.dimension,
                    "embedding_fingerprint": self._embedder.fingerprint(),
                },
            )
            return chunks
        except Exception as exc:
            if self._repository is not None and job_id is not None:
                self._repository.update_ingestion_job(job_id, status="failed", message=str(exc))
            raise

    async def search(self, query: str, *, top_k: int | None = None) -> list[Citation]:
        if not self._settings.enabled:
            return []
        status = self.index_status()
        if int(status["point_count"]) == 0:
            return []
        if status["needs_rebuild"]:
            raise RuntimeError(
                f"RAG index is incompatible with the configured embedder; rebuild required: {status['reasons']}"
            )
        return self._retriever.search(query, top_k=top_k)

    def readiness(self) -> dict[str, object]:
        status = self._vector_store.readiness()
        index_status = self.index_status()
        status["docs_path"] = str(self._loader.docs_path)
        status["point_count"] = index_status["point_count"]
        status["manifest_path"] = index_status["manifest_path"]
        status["needs_rebuild"] = index_status["needs_rebuild"]
        status["rebuild_reasons"] = index_status["reasons"]
        status["current_embedding"] = index_status["current_embedding"]
        status["index_metadata"] = index_status["index_metadata"]
        return status
