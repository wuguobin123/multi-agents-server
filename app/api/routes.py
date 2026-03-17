from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, Request, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.api.responses import JSONLineResponse
from app.runtime import AppRuntime, get_runtime
from app.schemas import (
    ChatRequest,
    ChatResponse,
    DocumentDeleteResponse,
    DocumentUploadResponse,
    IngestionJobResponse,
    KnowledgeBaseCreateRequest,
    KnowledgeBaseListResponse,
    KnowledgeBaseSummary,
    KnowledgeDocumentListResponse,
    RAGConfigResponse,
    RAGRebuildRequest,
    RAGRebuildResponse,
)
from app.config.settings import ChunkingSettings


router = APIRouter()


def _chunking_from_form(
    *,
    strategy: str | None,
    chunk_size: int | None,
    chunk_overlap: int | None,
    qa_question_prefixes: str | None,
    qa_answer_prefixes: str | None,
    qa_fallback_to_recursive: bool | None,
) -> ChunkingSettings | None:
    payload: dict[str, object] = {}
    if strategy:
        payload["strategy"] = strategy
    if chunk_size is not None:
        payload["chunk_size"] = chunk_size
    if chunk_overlap is not None:
        payload["chunk_overlap"] = chunk_overlap
    if qa_question_prefixes:
        payload["qa_question_prefixes"] = [item.strip() for item in qa_question_prefixes.split(",") if item.strip()]
    if qa_answer_prefixes:
        payload["qa_answer_prefixes"] = [item.strip() for item in qa_answer_prefixes.split(",") if item.strip()]
    if qa_fallback_to_recursive is not None:
        payload["qa_fallback_to_recursive"] = qa_fallback_to_recursive
    if not payload:
        return None
    return ChunkingSettings.model_validate(payload)


@router.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/readyz")
async def readyz(runtime: AppRuntime = Depends(get_runtime)) -> JSONLineResponse:
    payload = runtime.readiness()
    status_code = 200 if payload.get("status") == "ok" else 503
    return JSONLineResponse(status_code=status_code, content=payload)


@router.post("/v1/chat", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    request: Request,
    runtime: AppRuntime = Depends(get_runtime),
) -> ChatResponse:
    request_id = getattr(request.state, "request_id", None)
    return await runtime.handle_chat(payload, request_id=request_id)


@router.get("/v1/rag/config", response_model=RAGConfigResponse)
async def rag_config(runtime: AppRuntime = Depends(get_runtime)) -> RAGConfigResponse:
    return RAGConfigResponse.model_validate(runtime.rag_configuration())


@router.post("/v1/rag/rebuild", response_model=RAGRebuildResponse)
async def rebuild_rag(
    payload: RAGRebuildRequest,
    runtime: AppRuntime = Depends(get_runtime),
) -> RAGRebuildResponse:
    result = await run_in_threadpool(runtime.rebuild_knowledge_base, payload.chunking)
    return RAGRebuildResponse.model_validate(result)


@router.get("/v1/knowledge-bases", response_model=KnowledgeBaseListResponse)
async def list_knowledge_bases(runtime: AppRuntime = Depends(get_runtime)) -> KnowledgeBaseListResponse:
    return await run_in_threadpool(runtime.list_knowledge_bases)


@router.post("/v1/knowledge-bases", response_model=KnowledgeBaseSummary)
async def create_knowledge_base(
    payload: KnowledgeBaseCreateRequest,
    runtime: AppRuntime = Depends(get_runtime),
) -> KnowledgeBaseSummary:
    return await run_in_threadpool(runtime.create_knowledge_base, payload)


@router.get("/v1/knowledge-bases/{knowledge_base_id}/documents", response_model=KnowledgeDocumentListResponse)
async def list_documents(
    knowledge_base_id: str,
    runtime: AppRuntime = Depends(get_runtime),
) -> KnowledgeDocumentListResponse:
    return await run_in_threadpool(runtime.list_documents, knowledge_base_id)


@router.post("/v1/knowledge-bases/{knowledge_base_id}/documents", response_model=DocumentUploadResponse)
async def upload_document(
    knowledge_base_id: str,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    chunking_strategy: str | None = Form(default=None),
    chunk_size: int | None = Form(default=None),
    chunk_overlap: int | None = Form(default=None),
    qa_question_prefixes: str | None = Form(default=None),
    qa_answer_prefixes: str | None = Form(default=None),
    qa_fallback_to_recursive: bool | None = Form(default=None),
    embedding_provider: str | None = Form(default=None),
    embedding_model: str | None = Form(default=None),
    embedding_dimension: int | None = Form(default=None),
    runtime: AppRuntime = Depends(get_runtime),
) -> DocumentUploadResponse:
    chunking = _chunking_from_form(
        strategy=chunking_strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        qa_question_prefixes=qa_question_prefixes,
        qa_answer_prefixes=qa_answer_prefixes,
        qa_fallback_to_recursive=qa_fallback_to_recursive,
    )
    content = await file.read()
    response, should_ingest = await run_in_threadpool(
        runtime.upload_document,
        knowledge_base_id=knowledge_base_id,
        filename=file.filename or "document",
        content=content,
        mime_type=file.content_type,
        chunking=chunking,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        embedding_dimension=embedding_dimension,
    )
    if should_ingest:
        background_tasks.add_task(runtime.process_ingestion_job, response.job.job_id)
    return response


@router.get("/v1/ingestion-jobs/{job_id}", response_model=IngestionJobResponse)
async def get_ingestion_job(job_id: int, runtime: AppRuntime = Depends(get_runtime)) -> IngestionJobResponse:
    response = await run_in_threadpool(runtime.get_ingestion_job, job_id)
    if response is None:
        return IngestionJobResponse(
            job_id=job_id,
            knowledge_base_id="",
            source="",
            status="failed",
            stage="missing",
            message="job_not_found",
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
    return response


@router.post("/v1/knowledge-bases/{knowledge_base_id}/documents/{document_id}/reindex", response_model=IngestionJobResponse)
async def reindex_document(
    knowledge_base_id: str,
    document_id: str,
    background_tasks: BackgroundTasks,
    runtime: AppRuntime = Depends(get_runtime),
) -> IngestionJobResponse:
    job = await run_in_threadpool(runtime.reindex_document, document_id)
    if job.knowledge_base_id == knowledge_base_id:
        background_tasks.add_task(runtime.process_ingestion_job, job.job_id)
    return job


@router.delete("/v1/knowledge-bases/{knowledge_base_id}/documents/{document_id}", response_model=DocumentDeleteResponse)
async def delete_document(
    knowledge_base_id: str,
    document_id: str,
    runtime: AppRuntime = Depends(get_runtime),
) -> DocumentDeleteResponse:
    response = await run_in_threadpool(runtime.delete_document, document_id)
    if response.knowledge_base_id:
        return response
    return DocumentDeleteResponse(knowledge_base_id=knowledge_base_id, document_id=document_id, deleted=False)
