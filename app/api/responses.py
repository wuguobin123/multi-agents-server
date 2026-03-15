from __future__ import annotations

from typing import Any

from fastapi.responses import JSONResponse


class JSONLineResponse(JSONResponse):
    def render(self, content: Any) -> bytes:
        body = super().render(content)
        if body.endswith(b"\n"):
            return body
        return body + b"\n"
