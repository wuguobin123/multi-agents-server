from __future__ import annotations

import logging

from app.observability.logging import JsonFormatter, RequestContextFilter, logging_context


def test_json_formatter_includes_structured_runtime_fields() -> None:
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="test.logger",
        level=logging.INFO,
        pathname=__file__,
        lineno=10,
        msg="request_completed",
        args=(),
        exc_info=None,
    )
    record.request_id = "req-001"
    record.session_id = "sess-001"
    record.plan_intent = "tool"
    record.selected_agents = ["tool_agent"]
    record.tool_name = "local_echo"
    record.latency_ms = 12
    record.error_type = None

    rendered = formatter.format(record)

    assert '"request_id": "req-001"' in rendered
    assert '"session_id": "sess-001"' in rendered
    assert '"plan_intent": "tool"' in rendered
    assert '"selected_agents": ["tool_agent"]' in rendered
    assert '"tool_name": "local_echo"' in rendered
    assert '"latency_ms": 12' in rendered


def test_logging_context_sets_request_and_session() -> None:
    with logging_context(request_id="req-ctx", session_id="sess-ctx"):
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname=__file__,
            lineno=32,
            msg="context_check",
            args=(),
            exc_info=None,
        )
        RequestContextFilter().filter(record)

        rendered = formatter.format(record)

    assert '"request_id": "req-ctx"' in rendered
    assert '"session_id": "sess-ctx"' in rendered
