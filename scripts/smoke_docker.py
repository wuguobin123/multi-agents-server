from __future__ import annotations

import json
import sys
from urllib import error, request


def fetch_json(url: str, *, method: str = "GET", payload: dict | None = None) -> tuple[int, dict]:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = request.Request(url, data=data, headers=headers, method=method)
    try:
        with request.urlopen(req, timeout=10) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{url} returned {exc.code}: {body}") from exc


def assert_condition(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    base_url = "http://127.0.0.1:8080"

    health_status, health_body = fetch_json(f"{base_url}/healthz")
    assert_condition(health_status == 200, "healthz did not return 200")
    assert_condition(health_body.get("status") == "ok", "healthz status is not ok")

    ready_status, ready_body = fetch_json(f"{base_url}/readyz")
    assert_condition(ready_status == 200, "readyz did not return 200")
    assert_condition(ready_body.get("status") == "ok", "readyz status is not ok")
    assert_condition(ready_body["checks"]["repository"]["status"] == "ok", "repository is not ready")
    assert_condition(ready_body["checks"]["planner"]["status"] in {"ok", "disabled"}, "planner is not ready")
    assert_condition(int(ready_body["checks"]["agents"].get("agent_count", 0)) > 0, "agents are not ready")
    assert_condition(ready_body["checks"]["tools"]["status"] == "ok", "tools are not ready")
    if ready_body["checks"]["rag"]["enabled"]:
        assert_condition(ready_body["checks"]["rag"]["status"] == "ok", "rag is not ready")
        assert_condition(int(ready_body["checks"]["rag"].get("point_count", 0)) > 0, "rag has no indexed points")

    qa_status, qa_body = fetch_json(
        f"{base_url}/v1/chat",
        method="POST",
        payload={"query": "帮我总结知识库中关于部署流程的说明", "session_id": "smoke-qa-001"},
    )
    assert_condition(qa_status == 200, "qa chat did not return 200")
    assert_condition(qa_body.get("error") is None, "qa chat returned an error")
    assert_condition(qa_body["trace"]["plan"]["intent"] == "qa", "qa chat plan intent mismatch")
    assert_condition(bool(qa_body.get("citations")), "qa chat did not return citations")

    tool_status, tool_body = fetch_json(
        f"{base_url}/v1/chat",
        method="POST",
        payload={"query": "请调用工具执行一次检查", "session_id": "smoke-tool-001"},
    )
    assert_condition(tool_status == 200, "tool chat did not return 200")
    assert_condition(tool_body.get("error") is None, "tool chat returned an error")
    assert_condition(tool_body["trace"]["plan"]["intent"] == "tool", "tool chat plan intent mismatch")
    assert_condition(bool(tool_body["trace"].get("tool_calls")), "tool chat did not invoke a tool")

    print(
        json.dumps(
            {
                "healthz": health_body,
                "readyz": ready_body,
                "qa_trace": {
                    "intent": qa_body["trace"]["plan"]["intent"],
                    "planner_source": qa_body["trace"]["planner_runs"][0]["source"],
                    "citation_count": len(qa_body.get("citations", [])),
                },
                "tool_trace": {
                    "intent": tool_body["trace"]["plan"]["intent"],
                    "planner_source": tool_body["trace"]["planner_runs"][0]["source"],
                    "tool_attempts": tool_body["trace"]["tool_calls"][0]["attempts"],
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - script entrypoint
        print(f"smoke failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
