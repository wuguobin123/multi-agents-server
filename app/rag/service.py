from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import NAMESPACE_URL, uuid4, uuid5

from app.config import AppSettings, RAGSettings
from app.config.settings import ChunkingSettings
from app.domain import IngestionJobRecord, KnowledgeBaseRecord, KnowledgeChunkRecord, KnowledgeDocumentRecord
from app.observability import get_logger
from app.rag.embedder import Embedder, build_embedder
from app.rag.index_manifest import IndexManifestStore
from app.rag.loader import FileSystemDocumentLoader, build_document_record
from app.rag.retriever import Retriever
from app.rag.splitter import DocumentSplitter, available_chunking_strategies
from app.rag.types import ChunkRecord, IndexMetadata, VectorPoint
from app.rag.vector_store import LocalVectorStore, QdrantHttpVectorStore, VectorStore
from app.repositories.base import ExecutionRepository
from app.schemas import Citation
from app.storage import LocalFileStorage


logger = get_logger(__name__)


class RAGService:
    default_knowledge_base_id = "kb-default"
    default_knowledge_base_code = "default"
    default_knowledge_base_name = "Default Knowledge Base"

    def __init__(self, settings: AppSettings, repository: ExecutionRepository | None = None) -> None:
        self._app_settings = settings
        self._settings = settings.rag
        self._root_dir = Path(__file__).resolve().parents[2]
        self._repository = repository
        self._loader = FileSystemDocumentLoader(self._root_dir, self._settings.docs_path)
        self._set_chunking_settings(self._settings.chunking)
        self._storage = LocalFileStorage(
            self._root_dir,
            self._settings.uploads_path,
            public_base_url=self._settings.storage_public_base_url,
        )
        self._default_embedder = build_embedder(settings)
        self._ensure_default_knowledge_base()
        if self._settings.bootstrap_on_startup and self.index_status()["needs_rebuild"]:
            self.rebuild()

    @property
    def docs_path(self) -> Path:
        return self._loader.docs_path

    def _repository_required(self) -> ExecutionRepository:
        if self._repository is None:
            raise RuntimeError("Repository is required for knowledge-base management operations.")
        return self._repository

    def _set_chunking_settings(self, chunking: ChunkingSettings) -> None:
        self._settings.chunking = chunking.model_copy(deep=True)
        self._splitter = DocumentSplitter(self._settings.chunking)

    def _default_knowledge_base(self) -> KnowledgeBaseRecord:
        return KnowledgeBaseRecord(
            knowledge_base_id=self.default_knowledge_base_id,
            code=self.default_knowledge_base_code,
            name=self.default_knowledge_base_name,
            description="System bootstrap knowledge base.",
            embedding_provider=self._settings.embedding_provider,
            embedding_model=self._settings.embedding_model,
            embedding_dimension=self._settings.embedding_dimension,
            vector_backend=self._settings.vector_store_backend,
            vector_collection=self._settings.collection_name,
            storage_backend=self._settings.storage_backend,
        )

    def _ensure_default_knowledge_base(self) -> None:
        if self._repository is None:
            return
        record = self._default_knowledge_base()
        existing = self._repository.get_knowledge_base(record.knowledge_base_id)
        if existing is None:
            self._repository.upsert_knowledge_base(record)
            return
        merged = existing.model_copy(
            update={
                "code": record.code,
                "name": record.name,
                "description": record.description,
                "embedding_provider": record.embedding_provider,
                "embedding_model": record.embedding_model,
                "embedding_dimension": record.embedding_dimension,
                "vector_backend": record.vector_backend,
                "vector_collection": record.vector_collection,
                "storage_backend": record.storage_backend,
                "updated_at": datetime.now(UTC),
            }
        )
        self._repository.upsert_knowledge_base(merged)

    def _knowledge_base_or_default(self, knowledge_base_id: str | None = None) -> KnowledgeBaseRecord:
        if knowledge_base_id in (None, self.default_knowledge_base_id):
            if self._repository is not None:
                existing = self._repository.get_knowledge_base(self.default_knowledge_base_id)
                if existing is not None:
                    return existing
            return self._default_knowledge_base()
        repository = self._repository_required()
        knowledge_base = repository.get_knowledge_base(knowledge_base_id)
        if knowledge_base is None:
            raise ValueError(f"Knowledge base not found: {knowledge_base_id}")
        return knowledge_base

    def _rag_settings_for_knowledge_base(self, knowledge_base: KnowledgeBaseRecord) -> RAGSettings:
        rag_settings = self._settings.model_copy(deep=True)
        rag_settings.vector_store_backend = knowledge_base.vector_backend
        rag_settings.collection_name = knowledge_base.vector_collection
        rag_settings.embedding_provider = knowledge_base.embedding_provider
        rag_settings.embedding_model = knowledge_base.embedding_model
        rag_settings.embedding_dimension = knowledge_base.embedding_dimension
        return rag_settings

    def _build_embedder_for_knowledge_base(self, knowledge_base: KnowledgeBaseRecord) -> Embedder:
        return build_embedder(self._rag_settings_for_knowledge_base(knowledge_base))

    def _local_store_path_for_collection(self, collection_name: str) -> Path:
        base_store = self._root_dir / self._settings.local_store_path
        if collection_name == self._settings.collection_name:
            return base_store
        return base_store.parent / f"{collection_name}.json"

    def _build_vector_store(self, knowledge_base: KnowledgeBaseRecord, *, vector_size: int) -> VectorStore:
        if knowledge_base.vector_backend == "qdrant":
            if not self._settings.qdrant_url:
                raise ValueError("Qdrant backend requires rag.qdrant_url")
            return QdrantHttpVectorStore(
                base_url=self._settings.qdrant_url,
                collection_name=knowledge_base.vector_collection,
                vector_size=vector_size,
                api_key=self._settings.qdrant_api_key,
                timeout_seconds=self._settings.request_timeout_seconds,
            )
        return LocalVectorStore(self._local_store_path_for_collection(knowledge_base.vector_collection))

    def _build_manifest_path(self, collection_name: str) -> Path:
        if self._settings.vector_store_backend == "local":
            store_path = self._local_store_path_for_collection(collection_name)
            return store_path.parent / f"{store_path.stem}.manifest.json"
        return self._root_dir / "data" / f"{collection_name}.manifest.json"

    def _manifest_store(self, collection_name: str) -> IndexManifestStore:
        return IndexManifestStore(self._build_manifest_path(collection_name))

    def chunking_profile(self) -> dict[str, object]:
        return {
            "active": self._splitter.profile(),
            "available_strategies": available_chunking_strategies(),
        }

    def embedding_profile(self) -> dict[str, object]:
        return {
            "provider": self._settings.embedding_provider,
            "model": self._settings.embedding_model,
            "dimension": self._default_embedder.dimension,
            "fingerprint": self._default_embedder.fingerprint(),
            "batch_size": self._settings.embedding_batch_size,
        }

    def _collection_index_status(
        self,
        *,
        knowledge_base: KnowledgeBaseRecord,
        embedder: Embedder,
        compare_chunking: bool,
        chunking: ChunkingSettings | None = None,
        force_rebuild: bool = False,
    ) -> dict[str, object]:
        manifest = self._manifest_store(knowledge_base.vector_collection).read()
        vector_store = self._build_vector_store(knowledge_base, vector_size=embedder.dimension)
        point_count = vector_store.count()
        vector_size = vector_store.vector_size()
        reasons: list[str] = []
        if force_rebuild:
            reasons.append("force_rebuild")
        if point_count == 0:
            reasons.append("vector_store_empty")
        if manifest is None:
            reasons.append("index_manifest_missing")
        else:
            if manifest.embedding_fingerprint != embedder.fingerprint():
                reasons.append("embedding_fingerprint_changed")
            if manifest.embedding_dimension != embedder.dimension:
                reasons.append("manifest_dimension_mismatch")
            if manifest.chunk_count != point_count:
                reasons.append("chunk_count_mismatch")
            if manifest.vector_store_backend != knowledge_base.vector_backend:
                reasons.append("vector_store_backend_changed")
            if manifest.collection_name != knowledge_base.vector_collection:
                reasons.append("collection_name_changed")
            if compare_chunking:
                active_chunking = chunking or self._settings.chunking
                if manifest.chunking_strategy != active_chunking.strategy:
                    reasons.append("chunking_strategy_changed")
                if manifest.chunking_config != active_chunking.profile():
                    reasons.append("chunking_config_changed")
        if vector_size is not None and vector_size != embedder.dimension:
            reasons.append("vector_dimension_mismatch")
        return {
            "needs_rebuild": bool(reasons),
            "reasons": list(dict.fromkeys(reasons)),
            "manifest_path": str(self._build_manifest_path(knowledge_base.vector_collection)),
            "point_count": point_count,
            "vector_size": vector_size,
            "current_embedding": {
                "provider": knowledge_base.embedding_provider,
                "model": knowledge_base.embedding_model,
                "dimension": embedder.dimension,
                "fingerprint": embedder.fingerprint(),
                "batch_size": self._settings.embedding_batch_size,
            },
            "current_chunking": self.chunking_profile(),
            "index_metadata": manifest.model_dump(mode="json") if manifest is not None else None,
        }

    def index_status(self, *, force_rebuild: bool = False) -> dict[str, object]:
        knowledge_base = self._knowledge_base_or_default()
        return self._collection_index_status(
            knowledge_base=knowledge_base,
            embedder=self._default_embedder,
            compare_chunking=True,
            chunking=self._settings.chunking,
            force_rebuild=force_rebuild,
        )

    def list_knowledge_bases(self) -> list[KnowledgeBaseRecord]:
        repository = self._repository_required()
        self._ensure_default_knowledge_base()
        return repository.list_knowledge_bases()

    def create_knowledge_base(
        self,
        *,
        code: str,
        name: str,
        description: str | None = None,
        embedding_provider: str | None = None,
        embedding_model: str | None = None,
        embedding_dimension: int | None = None,
        vector_backend: str | None = None,
    ) -> KnowledgeBaseRecord:
        repository = self._repository_required()
        existing = repository.get_knowledge_base_by_code(code)
        if existing is not None:
            raise ValueError(f"Knowledge base code already exists: {code}")
        now = datetime.now(UTC)
        record = KnowledgeBaseRecord(
            knowledge_base_id=f"kb-{uuid4().hex[:12]}",
            code=code,
            name=name,
            description=description,
            embedding_provider=embedding_provider or self._settings.embedding_provider,
            embedding_model=embedding_model or self._settings.embedding_model,
            embedding_dimension=embedding_dimension or self._settings.embedding_dimension,
            vector_backend=vector_backend or self._settings.vector_store_backend,
            vector_collection=f"knowledge_{code}",
            storage_backend=self._settings.storage_backend,
            created_at=now,
            updated_at=now,
        )
        repository.upsert_knowledge_base(record)
        return record

    def list_knowledge_documents(self, knowledge_base_id: str) -> list[KnowledgeDocumentRecord]:
        repository = self._repository_required()
        return repository.list_knowledge_documents(knowledge_base_id)

    def get_ingestion_job(self, job_id: int) -> IngestionJobRecord | None:
        repository = self._repository_required()
        return repository.get_ingestion_job(job_id)

    def register_document_upload(
        self,
        *,
        knowledge_base_id: str,
        filename: str,
        content: bytes,
        mime_type: str | None,
        chunking: ChunkingSettings | None = None,
        embedding_provider: str | None = None,
        embedding_model: str | None = None,
        embedding_dimension: int | None = None,
    ) -> tuple[KnowledgeDocumentRecord, int, bool]:
        repository = self._repository_required()
        knowledge_base = self._knowledge_base_or_default(knowledge_base_id)
        latest_documents = [doc for doc in repository.list_knowledge_documents(knowledge_base_id) if doc.is_latest]

        requested_embedding = {
            "embedding_provider": embedding_provider or knowledge_base.embedding_provider,
            "embedding_model": embedding_model or knowledge_base.embedding_model,
            "embedding_dimension": embedding_dimension or knowledge_base.embedding_dimension,
        }
        if latest_documents and any(
            getattr(knowledge_base, field_name) != field_value for field_name, field_value in requested_embedding.items()
        ):
            raise ValueError("Embedding profile cannot be changed after documents have been ingested into a knowledge base.")
        if not latest_documents and any(
            getattr(knowledge_base, field_name) != field_value for field_name, field_value in requested_embedding.items()
        ):
            knowledge_base = knowledge_base.model_copy(update={**requested_embedding, "updated_at": datetime.now(UTC)})
            repository.upsert_knowledge_base(knowledge_base)

        document_id = f"doc-{uuid4().hex}"
        stored_file = self._storage.save_bytes(
            knowledge_base_id=knowledge_base_id,
            document_id=document_id,
            filename=filename,
            content=content,
            mime_type=mime_type,
        )
        same_name_versions = [doc.version for doc in latest_documents if doc.original_filename == filename]
        version = (max(same_name_versions) + 1) if same_name_versions else 1
        duplicate = next(
            (
                doc
                for doc in latest_documents
                if doc.file_hash == stored_file.file_hash and doc.status in {"ready", "duplicate"}
            ),
            None,
        )
        active_chunking = chunking or self._settings.chunking
        now = datetime.now(UTC)
        document = KnowledgeDocumentRecord(
            document_id=document_id,
            knowledge_base_id=knowledge_base_id,
            source=stored_file.source,
            original_filename=filename,
            storage_path=stored_file.storage_path,
            file_url=stored_file.file_url,
            file_hash=stored_file.file_hash,
            file_size=stored_file.file_size,
            mime_type=stored_file.mime_type,
            chunking_strategy=active_chunking.strategy,
            chunking_config=active_chunking.profile(),
            embedding_provider=knowledge_base.embedding_provider,
            embedding_model=knowledge_base.embedding_model,
            embedding_dimension=knowledge_base.embedding_dimension,
            status="duplicate" if duplicate is not None else "pending",
            version=version,
            is_latest=True,
            metadata={"duplicate_of": duplicate.document_id} if duplicate is not None else {},
            created_at=now,
            updated_at=now,
        )
        repository.upsert_knowledge_document(document)
        job = IngestionJobRecord(
            knowledge_base_id=knowledge_base_id,
            document_id=document_id,
            source=document.source,
            status="completed" if duplicate is not None else "queued",
            stage="deduplicated" if duplicate is not None else "uploaded",
            message="duplicate_document_skipped" if duplicate is not None else "document_uploaded",
            created_at=now,
            updated_at=now,
            finished_at=now if duplicate is not None else None,
        )
        job_id = repository.create_ingestion_job(job)
        return document, job_id, duplicate is None

    def reindex_document(self, document_id: str) -> int:
        repository = self._repository_required()
        document = repository.get_knowledge_document(document_id)
        if document is None:
            raise ValueError(f"Document not found: {document_id}")
        now = datetime.now(UTC)
        repository.upsert_knowledge_document(document.model_copy(update={"status": "pending", "updated_at": now}))
        return repository.create_ingestion_job(
            IngestionJobRecord(
                knowledge_base_id=document.knowledge_base_id,
                document_id=document_id,
                source=document.source,
                status="queued",
                stage="uploaded",
                message="document_reindex_requested",
                created_at=now,
                updated_at=now,
            )
        )

    def _chunking_from_document(self, document: KnowledgeDocumentRecord) -> ChunkingSettings:
        if not document.chunking_config:
            return self._settings.chunking.model_copy(deep=True)
        return ChunkingSettings.model_validate(document.chunking_config)

    def _to_chunk_records(
        self,
        *,
        knowledge_base_id: str,
        chunks: list[ChunkRecord],
    ) -> tuple[list[KnowledgeChunkRecord], list[VectorPoint]]:
        chunk_records: list[KnowledgeChunkRecord] = []
        points: list[VectorPoint] = []
        for chunk in chunks:
            vector_point_id = str(uuid5(NAMESPACE_URL, f"{knowledge_base_id}:{chunk.chunk_id}"))
            chunk_records.append(
                KnowledgeChunkRecord(
                    chunk_id=chunk.chunk_id,
                    knowledge_base_id=knowledge_base_id,
                    document_id=chunk.document_id,
                    source=chunk.source,
                    text=chunk.text,
                    chunk_index=chunk.chunk_index,
                    vector_point_id=vector_point_id,
                    token_count=max(1, len(chunk.text.split())),
                    metadata=chunk.metadata,
                )
            )
            points.append(
                VectorPoint(
                    point_id=vector_point_id,
                    vector=[],
                    payload={
                        "chunk_id": chunk.chunk_id,
                        "knowledge_base_id": knowledge_base_id,
                        "document_id": chunk.document_id,
                        "source": chunk.source,
                        "text": chunk.text,
                        "chunk_index": chunk.chunk_index,
                        "metadata": chunk.metadata,
                    },
                )
            )
        return chunk_records, points

    def _write_manifest(
        self,
        *,
        knowledge_base: KnowledgeBaseRecord,
        embedder: Embedder,
        chunking: ChunkingSettings,
        chunk_count: int,
    ) -> None:
        metadata = IndexMetadata(
            embedding_provider=knowledge_base.embedding_provider,
            embedding_model=knowledge_base.embedding_model,
            embedding_dimension=embedder.dimension,
            embedding_fingerprint=embedder.fingerprint(),
            indexed_at=datetime.now(UTC),
            chunk_count=chunk_count,
            vector_store_backend=knowledge_base.vector_backend,
            collection_name=knowledge_base.vector_collection,
            chunking_strategy=chunking.strategy,
            chunking_config=chunking.profile(),
        )
        self._manifest_store(knowledge_base.vector_collection).write(metadata)

    def run_ingestion_job(self, job_id: int) -> dict[str, object]:
        repository = self._repository_required()
        job = repository.get_ingestion_job(job_id)
        if job is None:
            raise ValueError(f"Ingestion job not found: {job_id}")
        if job.document_id is None:
            raise ValueError(f"Ingestion job has no document: {job_id}")
        document = repository.get_knowledge_document(job.document_id)
        if document is None:
            raise ValueError(f"Document not found: {job.document_id}")
        if document.status == "duplicate":
            return {"job_id": job_id, "status": "completed", "stage": "deduplicated", "chunk_count": 0}

        knowledge_base = self._knowledge_base_or_default(document.knowledge_base_id)
        chunking = self._chunking_from_document(document)
        embedder = self._build_embedder_for_knowledge_base(knowledge_base)
        vector_store = self._build_vector_store(knowledge_base, vector_size=embedder.dimension)

        repository.update_ingestion_job(job_id, status="running", stage="parsing", message="document_parsing_started")
        repository.upsert_knowledge_document(document.model_copy(update={"status": "processing", "updated_at": datetime.now(UTC)}))

        try:
            rag_document = build_document_record(
                path=Path(document.storage_path),
                document_id=document.document_id,
                source=document.source,
                metadata={
                    "source_path": document.source,
                    "original_filename": document.original_filename,
                    "knowledge_base_id": document.knowledge_base_id,
                },
            )
            repository.update_ingestion_job(
                job_id,
                status="running",
                stage="chunking",
                message="document_chunking_started",
            )
            splitter = DocumentSplitter(chunking)
            chunks = splitter.split([rag_document])

            repository.update_ingestion_job(
                job_id,
                status="running",
                stage="embedding",
                message=f"embedding_{len(chunks)}_chunks",
            )
            chunk_records, points = self._to_chunk_records(knowledge_base_id=document.knowledge_base_id, chunks=chunks)
            vectors = embedder.embed_documents([chunk.text for chunk in chunks])
            hydrated_points = [
                point.model_copy(update={"vector": vector})
                for point, vector in zip(points, vectors, strict=False)
            ]

            stale_chunks = repository.list_document_chunks(document.document_id)
            stale_point_ids = [chunk.vector_point_id for chunk in stale_chunks if chunk.vector_point_id]
            if stale_point_ids:
                vector_store.delete(stale_point_ids)
            vector_store.upsert(hydrated_points)

            parsed_document = document.model_copy(
                update={
                    "parser_type": rag_document.metadata.get("parser_type", document.parser_type),
                    "status": "ready",
                    "updated_at": datetime.now(UTC),
                }
            )
            repository.replace_document_chunks(parsed_document, chunk_records)
            self._write_manifest(
                knowledge_base=knowledge_base,
                embedder=embedder,
                chunking=chunking,
                chunk_count=vector_store.count(),
            )
            repository.update_ingestion_job(
                job_id,
                status="completed",
                stage="indexed",
                message=f"indexed_{len(chunks)}_chunks",
            )
            logger.info(
                "document_ingested",
                extra={
                    "knowledge_base_id": document.knowledge_base_id,
                    "document_id": document.document_id,
                    "chunk_count": len(chunks),
                    "collection": knowledge_base.vector_collection,
                },
            )
            return {"job_id": job_id, "status": "completed", "stage": "indexed", "chunk_count": len(chunks)}
        except Exception as exc:
            repository.upsert_knowledge_document(
                document.model_copy(update={"status": "failed", "updated_at": datetime.now(UTC), "metadata": {**document.metadata, "error": str(exc)}})
            )
            repository.update_ingestion_job(
                job_id,
                status="failed",
                stage="failed",
                message=str(exc),
                error_message=str(exc),
            )
            raise

    def rebuild(self, chunking: ChunkingSettings | None = None) -> list[ChunkRecord]:
        if not self._settings.enabled:
            return []
        if chunking is not None:
            self._set_chunking_settings(chunking)
        knowledge_base = self._knowledge_base_or_default()
        embedder = self._default_embedder
        vector_store = self._build_vector_store(knowledge_base, vector_size=embedder.dimension)
        active_chunking = self._settings.chunking.model_copy(deep=True)
        documents = self._loader.load()
        splitter = DocumentSplitter(active_chunking)
        chunks = splitter.split(documents)

        self._manifest_store(knowledge_base.vector_collection).clear()
        vector_store.reset()
        batch_size = max(1, self._settings.embedding_batch_size)
        points: list[VectorPoint] = []
        for batch_start in range(0, len(chunks), batch_size):
            batch = chunks[batch_start : batch_start + batch_size]
            vectors = embedder.embed_documents([chunk.text for chunk in batch])
            for chunk, vector in zip(batch, vectors, strict=False):
                point_id = str(uuid5(NAMESPACE_URL, f"{knowledge_base.knowledge_base_id}:{chunk.chunk_id}"))
                points.append(
                    VectorPoint(
                        point_id=point_id,
                        vector=vector,
                        payload={
                            "chunk_id": chunk.chunk_id,
                            "knowledge_base_id": knowledge_base.knowledge_base_id,
                            "document_id": chunk.document_id,
                            "source": chunk.source,
                            "text": chunk.text,
                            "chunk_index": chunk.chunk_index,
                            "metadata": chunk.metadata,
                        },
                    )
                )
        vector_store.upsert(points)
        self._write_manifest(
            knowledge_base=knowledge_base,
            embedder=embedder,
            chunking=active_chunking,
            chunk_count=vector_store.count(),
        )

        if self._repository is not None:
            repository = self._repository
            repository.upsert_knowledge_base(knowledge_base.model_copy(update={"updated_at": datetime.now(UTC)}))
            for document in repository.list_knowledge_documents(knowledge_base.knowledge_base_id):
                repository.delete_knowledge_document(document.document_id)
            document_records: dict[str, KnowledgeDocumentRecord] = {}
            for document in documents:
                source_path = self._root_dir / document.source if Path(document.source).is_relative_to(Path(".")) else Path(document.source)
                document_records[document.document_id] = KnowledgeDocumentRecord(
                    document_id=document.document_id,
                    knowledge_base_id=knowledge_base.knowledge_base_id,
                    source=document.source,
                    original_filename=Path(document.source).name,
                    storage_path=str(source_path),
                    file_url=document.source,
                    file_hash=None,
                    file_size=len(document.content.encode("utf-8")),
                    mime_type="text/markdown" if document.source.endswith(".md") else "text/plain",
                    parser_type=str(document.metadata.get("parser_type", "text")),
                    chunking_strategy=active_chunking.strategy,
                    chunking_config=active_chunking.profile(),
                    embedding_provider=knowledge_base.embedding_provider,
                    embedding_model=knowledge_base.embedding_model,
                    embedding_dimension=knowledge_base.embedding_dimension,
                    status="ready",
                    version=1,
                    is_latest=True,
                    metadata=document.metadata,
                    created_at=datetime.now(UTC),
                    updated_at=datetime.now(UTC),
                )
            chunk_groups: dict[str, list[KnowledgeChunkRecord]] = {document_id: [] for document_id in document_records}
            for chunk in chunks:
                vector_point_id = str(uuid5(NAMESPACE_URL, f"{knowledge_base.knowledge_base_id}:{chunk.chunk_id}"))
                chunk_groups.setdefault(chunk.document_id, []).append(
                    KnowledgeChunkRecord(
                        chunk_id=chunk.chunk_id,
                        knowledge_base_id=knowledge_base.knowledge_base_id,
                        document_id=chunk.document_id,
                        source=chunk.source,
                        text=chunk.text,
                        chunk_index=chunk.chunk_index,
                        vector_point_id=vector_point_id,
                        token_count=max(1, len(chunk.text.split())),
                        metadata=chunk.metadata,
                    )
                )
            for document_id, record in document_records.items():
                repository.replace_document_chunks(record, chunk_groups.get(document_id, []))
        logger.info(
            "rag_rebuilt",
            extra={
                "knowledge_base_id": knowledge_base.knowledge_base_id,
                "document_count": len(documents),
                "chunk_count": len(chunks),
                "vector_backend": knowledge_base.vector_backend,
                "embedding_provider": knowledge_base.embedding_provider,
                "embedding_model": knowledge_base.embedding_model,
            },
        )
        return chunks

    async def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        knowledge_base_id: str | None = None,
    ) -> list[Citation]:
        if not self._settings.enabled:
            return []
        knowledge_base = self._knowledge_base_or_default(knowledge_base_id)
        embedder = self._default_embedder if knowledge_base.knowledge_base_id == self.default_knowledge_base_id else self._build_embedder_for_knowledge_base(knowledge_base)
        compare_chunking = knowledge_base.knowledge_base_id == self.default_knowledge_base_id
        status = self._collection_index_status(
            knowledge_base=knowledge_base,
            embedder=embedder,
            compare_chunking=compare_chunking,
            chunking=self._settings.chunking,
        )
        if int(status["point_count"]) == 0:
            return []
        if compare_chunking and status["needs_rebuild"]:
            raise RuntimeError(
                f"RAG index is incompatible with the current configuration; rebuild required: {status['reasons']}"
            )
        if not compare_chunking and status["vector_size"] not in {None, embedder.dimension}:
            raise RuntimeError("RAG index is incompatible with the current embedding profile.")
        vector_store = self._build_vector_store(knowledge_base, vector_size=embedder.dimension)
        retriever = Retriever(embedder, vector_store, default_top_k=self._settings.top_k)
        return retriever.search(query, top_k=top_k)

    def delete_document(self, document_id: str) -> KnowledgeDocumentRecord | None:
        repository = self._repository_required()
        document = repository.get_knowledge_document(document_id)
        if document is None:
            return None
        knowledge_base = self._knowledge_base_or_default(document.knowledge_base_id)
        embedder = self._build_embedder_for_knowledge_base(knowledge_base)
        vector_store = self._build_vector_store(knowledge_base, vector_size=embedder.dimension)
        chunks = repository.list_document_chunks(document_id)
        point_ids = [chunk.vector_point_id for chunk in chunks if chunk.vector_point_id]
        if point_ids:
            vector_store.delete(point_ids)
            self._write_manifest(
                knowledge_base=knowledge_base,
                embedder=embedder,
                chunking=self._settings.chunking,
                chunk_count=vector_store.count(),
            )
        repository.delete_knowledge_document(document_id)
        return document

    def readiness(self) -> dict[str, object]:
        knowledge_base = self._knowledge_base_or_default()
        status = self._build_vector_store(knowledge_base, vector_size=self._default_embedder.dimension).readiness()
        index_status = self.index_status()
        status["docs_path"] = str(self._loader.docs_path)
        status["point_count"] = index_status["point_count"]
        status["manifest_path"] = index_status["manifest_path"]
        status["needs_rebuild"] = index_status["needs_rebuild"]
        status["rebuild_reasons"] = index_status["reasons"]
        status["index_metadata"] = index_status["index_metadata"]
        status["current_embedding"] = index_status["current_embedding"]
        status["current_chunking"] = index_status["current_chunking"]
        if self._repository is not None:
            status["knowledge_base_count"] = len(self._repository.list_knowledge_bases())
        return status
