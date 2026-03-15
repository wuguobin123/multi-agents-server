from __future__ import annotations

import time
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError

from app.api.responses import JSONLineResponse
from app.api.routes import router
from app.config import get_settings
from app.errors import AppError, ErrorCode
from app.observability import configure_logging, get_logger, logging_context


settings = get_settings()
configure_logging(settings.server.log_level, settings.observability.json_logs)
logger = get_logger(__name__)

app = FastAPI(title="Multi Agents Server", version="0.1.0", default_response_class=JSONLineResponse)
app.include_router(router)


@app.middleware("http")
async def attach_request_context(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or uuid4().hex
    request.state.request_id = request_id
    started_at = time.perf_counter()
    with logging_context(request_id=request_id):
        response = await call_next(request)
    logger.info(
        "http_request_completed",
        extra={
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "latency_ms": int((time.perf_counter() - started_at) * 1000),
        },
    )
    response.headers["x-request-id"] = request_id
    return response


@app.exception_handler(AppError)
async def handle_app_error(request: Request, exc: AppError) -> JSONLineResponse:
    request_id = getattr(request.state, "request_id", None)
    return JSONLineResponse(
        status_code=exc.status_code,
        content={
            "error": exc.to_detail().model_dump(),
            "request_id": request_id,
        },
    )


@app.exception_handler(RequestValidationError)
async def handle_validation_error(request: Request, exc: RequestValidationError) -> JSONLineResponse:
    request_id = getattr(request.state, "request_id", None)
    return JSONLineResponse(
        status_code=422,
        content={
            "error": {
                "code": ErrorCode.INVALID_REQUEST,
                "message": "请求参数校验失败。",
                "retryable": False,
                "details": exc.errors(),
            },
            "request_id": request_id,
        },
    )
