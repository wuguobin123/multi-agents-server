from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from app.api.responses import JSONLineResponse
from app.schemas import ChatRequest, ChatResponse
from app.runtime import AppRuntime, get_runtime


router = APIRouter()


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
