from __future__ import annotations

from enum import StrEnum

from app.schemas import ErrorDetail


class ErrorCode(StrEnum):
    INVALID_REQUEST = "invalid_request"
    REQUEST_TIMEOUT = "request_timeout"
    REQUEST_FAILED = "request_failed"
    TOOL_NOT_FOUND = "tool_not_found"
    TOOL_VALIDATION_FAILED = "tool_validation_failed"
    TOOL_PERMISSION_DENIED = "tool_permission_denied"
    TOOL_CIRCUIT_OPEN = "tool_circuit_open"
    DEPENDENCY_UNAVAILABLE = "dependency_unavailable"
    EMPTY_RESULT = "empty_result"
    MAX_REFLECTIONS_REACHED = "max_reflections_reached"


class AppError(Exception):
    def __init__(self, code: ErrorCode, message: str, *, retryable: bool = False, status_code: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable
        self.status_code = status_code

    def to_detail(self) -> ErrorDetail:
        return ErrorDetail(code=self.code, message=self.message, retryable=self.retryable)
