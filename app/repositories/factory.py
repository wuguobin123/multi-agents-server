from __future__ import annotations

from app.config import AppSettings
from app.repositories.base import ExecutionRepository
from app.repositories.memory import InMemoryExecutionRepository
from app.repositories.sql import SQLExecutionRepository


def build_execution_repository(settings: AppSettings) -> ExecutionRepository:
    if not settings.database.enabled:
        return InMemoryExecutionRepository()
    return SQLExecutionRepository(settings.database)
