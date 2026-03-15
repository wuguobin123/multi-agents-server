from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import inspect, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from app.repositories.models import Base, SchemaMigrationModel


INITIAL_SCHEMA_VERSION = 1
INITIAL_SCHEMA_NAME = "0001_initial_schema"


def ensure_schema(engine: Engine) -> None:
    Base.metadata.create_all(engine)
    inspector = inspect(engine)
    if "schema_migration" not in inspector.get_table_names():
        Base.metadata.tables["schema_migration"].create(engine)

    with Session(engine) as session:
        exists = session.scalar(
            select(SchemaMigrationModel.version).where(SchemaMigrationModel.version == INITIAL_SCHEMA_VERSION)
        )
        if exists is None:
            session.add(
                SchemaMigrationModel(
                    version=INITIAL_SCHEMA_VERSION,
                    name=INITIAL_SCHEMA_NAME,
                    applied_at=datetime.now(UTC),
                )
            )
            session.commit()
