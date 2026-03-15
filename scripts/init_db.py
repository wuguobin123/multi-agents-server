from __future__ import annotations

from app.config import get_settings
from app.repositories.sql import SQLExecutionRepository


def main() -> None:
    settings = get_settings()
    repository = SQLExecutionRepository(settings.database)
    status = repository.ping()
    print(
        "database initialized",
        {
            "backend": status["backend"],
            "dialect": status.get("dialect"),
        },
    )


if __name__ == "__main__":
    main()
