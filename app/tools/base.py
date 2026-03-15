from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from app.schemas import ToolResult, ToolSpec


class BaseToolAdapter(ABC):
    def __init__(self, spec: ToolSpec) -> None:
        self.spec = spec

    @abstractmethod
    async def invoke(self, payload: dict[str, Any]) -> ToolResult:
        raise NotImplementedError
