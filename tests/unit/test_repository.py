from pathlib import Path

from sqlalchemy import text

from app.config import load_settings
from app.domain import AgentRunRecord, MessageRecord, SessionRecord, ToolCallRecord
from app.repositories import SQLExecutionRepository


def test_sql_repository_initializes_schema_and_persists_records(tmp_path: Path) -> None:
    db_path = tmp_path / "runtime.db"
    config_path = tmp_path / "app.yaml"
    config_path.write_text(
        f"""
model:
  provider: mock
database:
  enabled: true
  url: sqlite:///{db_path}
  auto_init: true
""".strip(),
        encoding="utf-8",
    )

    repository = SQLExecutionRepository(load_settings(config_path).database)

    repository.upsert_session(SessionRecord(session_id="s1", status="active", metadata={"source": "test"}))
    repository.append_message(MessageRecord(session_id="s1", request_id="r1", role="user", content="hello"))
    repository.append_agent_runs(
        [
            AgentRunRecord(
                session_id="s1",
                request_id="r1",
                agent_name="qa_agent",
                success=True,
                answer="answer",
            )
        ]
    )
    repository.append_tool_calls(
        [
            ToolCallRecord(
                session_id="s1",
                request_id="r1",
                call_id="call-1",
                tool_name="local_echo",
                source="skill",
                success=True,
                latency_ms=1,
                input_summary="query=hello",
            )
        ]
    )

    with repository.engine.connect() as conn:
        assert conn.execute(text("SELECT COUNT(*) FROM schema_migration")).scalar_one() == 2
        assert conn.execute(text("SELECT COUNT(*) FROM chat_session")).scalar_one() == 1
        assert conn.execute(text("SELECT COUNT(*) FROM chat_message")).scalar_one() == 1
        assert conn.execute(text("SELECT COUNT(*) FROM agent_run")).scalar_one() == 1
        assert conn.execute(text("SELECT COUNT(*) FROM tool_call")).scalar_one() == 1


def test_sql_repository_schema_initialization_is_idempotent(tmp_path: Path) -> None:
    db_path = tmp_path / "runtime.db"
    config_path = tmp_path / "app.yaml"
    config_path.write_text(
        f"""
model:
  provider: mock
database:
  enabled: true
  url: sqlite:///{db_path}
  auto_init: true
""".strip(),
        encoding="utf-8",
    )

    settings = load_settings(config_path).database
    first = SQLExecutionRepository(settings)
    second = SQLExecutionRepository(settings)

    with first.engine.connect() as conn:
        assert conn.execute(text("SELECT COUNT(*) FROM schema_migration")).scalar_one() == 2
    with second.engine.connect() as conn:
        assert conn.execute(text("SELECT COUNT(*) FROM schema_migration")).scalar_one() == 2
