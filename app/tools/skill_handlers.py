from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable


SkillHandler = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


async def echo_query(payload: dict[str, Any]) -> dict[str, Any]:
    query = payload.get("query", "")
    return {
        "output": f"技能工具收到请求：{query}",
        "structured_data": {"echo": query},
    }


_FLAKY_ATTEMPTS: dict[str, int] = {}


async def flaky_echo(payload: dict[str, Any]) -> dict[str, Any]:
    request_id = str(payload.get("request_id", "default"))
    succeed_on_attempt = int(payload.get("succeed_on_attempt", 2))
    current_attempt = _FLAKY_ATTEMPTS.get(request_id, 0) + 1
    _FLAKY_ATTEMPTS[request_id] = current_attempt
    if current_attempt < succeed_on_attempt:
        raise RuntimeError(f"transient failure on attempt {current_attempt}")
    return {
        "output": f"flaky tool succeeded on attempt {current_attempt}",
        "structured_data": {"attempt": current_attempt},
    }


async def always_fail(payload: dict[str, Any]) -> dict[str, Any]:
    message = str(payload.get("error", "forced tool failure"))
    raise RuntimeError(message)


async def slow_echo(payload: dict[str, Any]) -> dict[str, Any]:
    sleep_seconds = float(payload.get("sleep_seconds", 0.2))
    await asyncio.sleep(sleep_seconds)
    return {
        "output": f"slow tool finished after {sleep_seconds} seconds",
        "structured_data": {"sleep_seconds": sleep_seconds},
    }


SKILL_HANDLERS: dict[str, SkillHandler] = {
    "echo_query": echo_query,
    "flaky_echo": flaky_echo,
    "always_fail": always_fail,
    "slow_echo": slow_echo,
}
