from __future__ import annotations

import inspect
import json
import os
from collections.abc import Iterable
from contextlib import contextmanager
from typing import Any
from urllib.parse import urlparse

import httpx
from openai import APIConnectionError, APIStatusError, RateLimitError
from openai.types.chat import ChatCompletionContentPartTextParam
from openai.types.shared_params.response_format_json_schema import JSONSchema, ResponseFormatJSONSchema
from pydantic import BaseModel


def _coerce_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _coerce_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_string_list(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _resolve_setting(
    payload: dict[str, Any],
    metadata: dict[str, Any],
    payload_key: str,
    *,
    env_name: str | None = None,
    metadata_key: str | None = None,
    default: Any = None,
) -> Any:
    if payload_key in payload and payload[payload_key] not in (None, ""):
        return payload[payload_key]
    resolved_metadata_key = metadata_key or payload_key
    if resolved_metadata_key in metadata and metadata[resolved_metadata_key] not in (None, ""):
        return metadata[resolved_metadata_key]
    if env_name:
        env_value = os.getenv(env_name)
        if env_value not in (None, ""):
            return env_value
    return default


class MiniMaxBrowserUseChatOpenAI:
    """browser-use ChatOpenAI wrapper that always requests separated reasoning for MiniMax."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    async def _repair_structured_output(
        self,
        *,
        raw_content: str,
        response_format: JSONSchema,
        model_params: dict[str, Any],
    ) -> str:
        repair_messages = [
            {
                "role": "system",
                "content": (
                    "Return valid JSON only. Do not include markdown, XML tags, or any explanatory text. "
                    "The output must exactly match the provided JSON schema."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Convert the following browser agent response into valid JSON.\n\n"
                    f"Schema:\n{json.dumps(response_format, ensure_ascii=False)}\n\n"
                    f"Response:\n{raw_content}"
                ),
            },
        ]
        response = await self._inner.get_client().chat.completions.create(
            model=self._inner.model,
            messages=repair_messages,
            response_format=ResponseFormatJSONSchema(json_schema=response_format, type="json_schema"),
            extra_body={"reasoning_split": True},
            **model_params,
        )
        choice = response.choices[0] if response.choices else None
        if choice is None or choice.message.content is None:
            raise ValueError("MiniMax structured output repair returned an empty response")
        return choice.message.content

    async def ainvoke(self, messages: list[Any], output_format: type[BaseModel] | None = None, **kwargs: Any) -> Any:
        from browser_use.llm.exceptions import ModelProviderError, ModelRateLimitError
        from browser_use.llm.messages import BaseMessage
        from browser_use.llm.openai.serializer import OpenAIMessageSerializer
        from browser_use.llm.schema import SchemaOptimizer
        from browser_use.llm.views import ChatInvokeCompletion

        # Preserve the same serialization and error semantics as browser-use ChatOpenAI,
        # but force MiniMax to return reasoning in `reasoning_details` instead of inline <think>.
        typed_messages = [message for message in messages if isinstance(message, BaseMessage)]
        openai_messages = OpenAIMessageSerializer.serialize_messages(typed_messages)

        try:
            model_params: dict[str, Any] = {}

            if self._inner.temperature is not None:
                model_params["temperature"] = self._inner.temperature

            if self._inner.frequency_penalty is not None:
                model_params["frequency_penalty"] = self._inner.frequency_penalty

            if self._inner.max_completion_tokens is not None:
                model_params["max_completion_tokens"] = self._inner.max_completion_tokens

            if self._inner.top_p is not None:
                model_params["top_p"] = self._inner.top_p

            if self._inner.seed is not None:
                model_params["seed"] = self._inner.seed

            if self._inner.service_tier is not None:
                model_params["service_tier"] = self._inner.service_tier

            if self._inner.reasoning_models and any(
                str(model_name).lower() in str(self._inner.model).lower()
                for model_name in self._inner.reasoning_models
            ):
                model_params["reasoning_effort"] = self._inner.reasoning_effort
                model_params.pop("temperature", None)
                model_params.pop("frequency_penalty", None)

            extra_body = {"reasoning_split": True}

            if output_format is None:
                response = await self._inner.get_client().chat.completions.create(
                    model=self._inner.model,
                    messages=openai_messages,
                    extra_body=extra_body,
                    **model_params,
                )
                choice = response.choices[0] if response.choices else None
                if choice is None:
                    raise ModelProviderError(
                        message="Invalid MiniMax chat completion response: missing or empty `choices`.",
                        status_code=502,
                        model=self._inner.name,
                    )
                usage = self._inner._get_usage(response)
                return ChatInvokeCompletion(
                    completion=choice.message.content or "",
                    usage=usage,
                    stop_reason=choice.finish_reason,
                )

            response_format: JSONSchema = {
                "name": "agent_output",
                "strict": True,
                "schema": SchemaOptimizer.create_optimized_json_schema(
                    output_format,
                    remove_min_items=self._inner.remove_min_items_from_schema,
                    remove_defaults=self._inner.remove_defaults_from_schema,
                ),
            }

            if self._inner.add_schema_to_system_prompt and openai_messages and openai_messages[0]["role"] == "system":
                schema_text = f"\n<json_schema>\n{response_format}\n</json_schema>"
                if isinstance(openai_messages[0]["content"], str):
                    openai_messages[0]["content"] += schema_text
                elif isinstance(openai_messages[0]["content"], Iterable):
                    openai_messages[0]["content"] = list(openai_messages[0]["content"]) + [
                        ChatCompletionContentPartTextParam(text=schema_text, type="text")
                    ]

            if self._inner.dont_force_structured_output:
                response = await self._inner.get_client().chat.completions.create(
                    model=self._inner.model,
                    messages=openai_messages,
                    extra_body=extra_body,
                    **model_params,
                )
            else:
                response = await self._inner.get_client().chat.completions.create(
                    model=self._inner.model,
                    messages=openai_messages,
                    response_format=ResponseFormatJSONSchema(json_schema=response_format, type="json_schema"),
                    extra_body=extra_body,
                    **model_params,
                )

            choice = response.choices[0] if response.choices else None
            if choice is None or choice.message.content is None:
                raise ModelProviderError(
                    message="Failed to parse structured output from MiniMax response",
                    status_code=500,
                    model=self._inner.name,
                )

            usage = self._inner._get_usage(response)
            raw_content = choice.message.content
            try:
                parsed = output_format.model_validate_json(raw_content)
            except Exception:
                repaired_content = await self._repair_structured_output(
                    raw_content=raw_content,
                    response_format=response_format,
                    model_params=model_params,
                )
                parsed = output_format.model_validate_json(repaired_content)
            return ChatInvokeCompletion(
                completion=parsed,
                usage=usage,
                stop_reason=choice.finish_reason,
            )
        except ModelProviderError:
            raise
        except RateLimitError as exc:
            raise ModelRateLimitError(message=exc.message, model=self._inner.name) from exc
        except APIConnectionError as exc:
            raise ModelProviderError(message=str(exc), model=self._inner.name) from exc
        except APIStatusError as exc:
            raise ModelProviderError(message=exc.message, status_code=exc.status_code, model=self._inner.name) from exc
        except Exception as exc:
            raise ModelProviderError(message=str(exc), model=self._inner.name) from exc


def _call_with_supported_kwargs(callable_obj: Any, **kwargs: Any) -> Any:
    signature = inspect.signature(callable_obj)
    filtered_kwargs = {key: value for key, value in kwargs.items() if key in signature.parameters and value is not None}
    return callable_obj(**filtered_kwargs)


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


async def _call_history_accessor(history: Any, accessor: str) -> Any:
    target = getattr(history, accessor, None)
    if target is None:
        return None
    if callable(target):
        return await _maybe_await(target())
    return await _maybe_await(target)


async def _emit_browser_use_event(payload: dict[str, Any], event_type: str, data: dict[str, Any] | None = None) -> None:
    sink = payload.get("_event_sink")
    if callable(sink):
        await _maybe_await(sink(event_type, data or {}))


def _browser_use_step_callback(payload: dict[str, Any]):
    async def _callback(browser_state: Any, agent_output: Any, step_number: int) -> None:
        await _emit_browser_use_event(
            payload,
            "browser_step",
            {
                "step": step_number,
                "url": getattr(browser_state, "url", None),
                "title": getattr(browser_state, "title", None),
                "actions": str(agent_output)[:1000] if agent_output is not None else None,
            },
        )

    return _callback


def _browser_use_done_callback(payload: dict[str, Any]):
    async def _callback(history: Any) -> None:
        final_result = await _call_history_accessor(history, "final_result")
        rendered_result, _ = _serialize_browser_use_value(final_result)
        await _emit_browser_use_event(payload, "browser_done", {"result": rendered_result})

    return _callback


def _serialize_browser_use_value(value: Any) -> tuple[str, Any]:
    if value is None:
        return "", None
    if isinstance(value, str):
        return value, value
    if hasattr(value, "model_dump"):
        dumped = value.model_dump(mode="json")
        return json.dumps(dumped, ensure_ascii=False, indent=2), dumped
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, indent=2), value
    return str(value), str(value)


@contextmanager
def _temporary_env(overrides: dict[str, str | None]):
    previous = {key: os.environ.get(key) for key in overrides}
    try:
        for key, value in overrides.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


async def _preflight_remote_browser(
    *,
    health_url: str | None,
    info_url: str | None,
    bearer_token: str | None,
) -> dict[str, Any]:
    if not health_url and not info_url:
        return {}

    summary: dict[str, Any] = {}
    headers = {"Authorization": f"Bearer {bearer_token}"} if bearer_token else {}
    async with httpx.AsyncClient(timeout=10.0, trust_env=False) as client:
        if health_url:
            try:
                response = await client.get(health_url)
                response.raise_for_status()
                summary["health"] = {"ok": True, "status_code": response.status_code, "body": response.text[:500]}
            except Exception as exc:  # pragma: no cover - defensive
                summary["health"] = {"ok": False, "error": str(exc)}
        if info_url:
            try:
                response = await client.get(info_url, headers=headers)
                response.raise_for_status()
                try:
                    summary["info"] = response.json()
                except ValueError:
                    summary["info"] = {"raw": response.text[:1000]}
            except Exception as exc:  # pragma: no cover - defensive
                summary["info"] = {"ok": False, "error": str(exc)}
    return summary


def _resolve_runtime_model(payload: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    runtime_model = metadata.get("runtime_model", {})
    if not isinstance(runtime_model, dict):
        runtime_model = {}

    provider = _resolve_setting(
        payload,
        runtime_model,
        "llm_provider",
        env_name="MODEL_PROVIDER",
        metadata_key="provider",
        default="minimax",
    )
    model_name = _resolve_setting(
        payload,
        runtime_model,
        "llm_model",
        env_name="MODEL_NAME",
        metadata_key="name",
    )
    api_key = _resolve_setting(
        payload,
        runtime_model,
        "llm_api_key",
        env_name="MINIMAX_API_KEY" if str(provider).lower() == "minimax" else "BROWSER_USE_LLM_API_KEY",
        metadata_key="api_key",
    )
    base_url = _resolve_setting(
        payload,
        runtime_model,
        "llm_base_url",
        env_name="MINIMAX_BASE_URL" if str(provider).lower() == "minimax" else "BROWSER_USE_LLM_BASE_URL",
        metadata_key="base_url",
    )
    return {
        "provider": str(provider or "").lower(),
        "model": model_name,
        "api_key": api_key,
        "base_url": base_url,
    }


def _build_browser_use_llm(payload: dict[str, Any], metadata: dict[str, Any]) -> Any:
    try:
        from browser_use import ChatBrowserUse, ChatOpenAI
    except ImportError as exc:  # pragma: no cover - exercised in runtime env
        raise RuntimeError("browser-use 未安装，请先执行 uv sync。") from exc

    model_settings = _resolve_runtime_model(payload, metadata)
    provider = model_settings["provider"]

    if provider in {"minimax", "openai", "openai_compatible", "custom_openai"}:
        if not model_settings["model"] or not model_settings["api_key"] or not model_settings["base_url"]:
            raise RuntimeError("browser-use 缺少 LLM 配置，当前至少需要 model/api_key/base_url。")
        llm = _call_with_supported_kwargs(
            ChatOpenAI,
            model=model_settings["model"],
            api_key=model_settings["api_key"],
            base_url=model_settings["base_url"],
        )
        if provider == "minimax":
            return MiniMaxBrowserUseChatOpenAI(llm)
        return llm

    browser_use_api_key = _resolve_setting(
        payload,
        metadata,
        "browser_use_api_key",
        env_name="BROWSER_USE_API_KEY",
        metadata_key="browser_use_api_key",
    )
    if provider in {"browser_use", "browser-use", "browseruse"}:
        if not browser_use_api_key:
            raise RuntimeError("provider=browser_use 时必须提供 BROWSER_USE_API_KEY。")
        return _call_with_supported_kwargs(
            ChatBrowserUse,
            api_key=browser_use_api_key,
            model=model_settings["model"] or "gpt-4.1-mini",
        )

    raise RuntimeError(f"browser-use 暂不支持当前 LLM provider: {provider or 'unknown'}")


def _build_browser(
    payload: dict[str, Any],
    metadata: dict[str, Any],
    *,
    resolved_cdp_url: str | None = None,
    allowed_domains: list[str] | None = None,
    bearer_token: str | None = None,
) -> tuple[Any, dict[str, Any]]:
    try:
        from browser_use import Browser
    except ImportError as exc:  # pragma: no cover - exercised in runtime env
        raise RuntimeError("browser-use 未安装，请先执行 uv sync。") from exc

    cdp_url = resolved_cdp_url or _resolve_setting(
        payload,
        metadata,
        "cdp_url",
        env_name="BROWSER_USE_CDP_URL",
        metadata_key="cdp_url",
    )
    profile_id = _resolve_setting(
        payload,
        metadata,
        "profile_id",
        env_name="BROWSER_USE_PROFILE_ID",
        metadata_key="profile_id",
    )
    proxy_country_code = _resolve_setting(
        payload,
        metadata,
        "proxy_country_code",
        env_name="BROWSER_USE_PROXY_COUNTRY_CODE",
        metadata_key="proxy_country_code",
    )
    timeout_minutes = _coerce_int(
        _resolve_setting(
            payload,
            metadata,
            "session_timeout_minutes",
            env_name="BROWSER_USE_SESSION_TIMEOUT_MINUTES",
            metadata_key="session_timeout_minutes",
        )
    )
    headers = {"Authorization": f"Bearer {bearer_token}"} if bearer_token and cdp_url else None

    browser = _call_with_supported_kwargs(
        Browser,
        cdp_url=cdp_url,
        headers=headers,
        allowed_domains=allowed_domains,
        use_cloud=bool(not cdp_url and (profile_id or proxy_country_code or timeout_minutes)),
        cloud_profile_id=profile_id,
        cloud_proxy_country_code=proxy_country_code,
        cloud_timeout=timeout_minutes,
    )
    return browser, {
        "cdp_url": cdp_url,
        "profile_id": profile_id,
        "proxy_country_code": proxy_country_code,
        "session_timeout_minutes": timeout_minutes,
        "has_headers": bool(headers),
    }


def _resolve_cdp_url(payload: dict[str, Any], metadata: dict[str, Any]) -> str | None:
    return _resolve_setting(
        payload,
        metadata,
        "cdp_url",
        env_name="BROWSER_USE_CDP_URL",
        metadata_key="cdp_url",
    )


def _adjust_browser_task_limits(
    *,
    task: str,
    max_steps: int,
    use_vision: bool,
    max_failures: int,
    retry_delay: int,
    max_actions_per_step: int,
    explicit_max_steps: bool,
    explicit_use_vision: bool,
) -> tuple[int, bool, int, int, int]:
    task_lower = task.lower()
    simple_page_query = any(keyword in task_lower for keyword in ("标题", "title", "页面标题"))
    if simple_page_query and "总结" not in task and "summary" not in task_lower:
        if not explicit_use_vision:
            use_vision = False
        if not explicit_max_steps:
            max_steps = min(max_steps, 4)
        max_actions_per_step = min(max_actions_per_step, 3)
        max_failures = min(max_failures, 1)
        retry_delay = min(retry_delay, 1)
        return max_steps, use_vision, max_failures, retry_delay, max_actions_per_step

    complex_action_keywords = (
        "发布",
        "上传",
        "登录",
        "注册",
        "填写",
        "表单",
        "提交",
        "创建",
        "发送",
        "发帖",
        "发一条",
        "私信",
        "评论",
        "点赞",
        "关注",
        "下单",
        "预约",
        "支付",
        "purchase",
        "checkout",
        "publish",
        "upload",
        "login",
        "sign in",
        "sign up",
        "fill",
        "submit",
        "create",
        "send",
        "comment",
        "like",
        "follow",
        "book",
        "pay",
        "post a",
    )
    connector_keywords = ("并", "然后", "接着", "之后", "最后", "再", "同时", " and ", " then ", " after ")

    complexity_score = sum(1 for keyword in complex_action_keywords if keyword in task_lower)
    complexity_score += sum(1 for keyword in connector_keywords if keyword in task_lower)

    if complexity_score >= 2:
        if not explicit_max_steps:
            max_steps = max(max_steps, 18 if complexity_score < 4 else 24)
        if not explicit_use_vision:
            use_vision = True
        max_actions_per_step = max(max_actions_per_step, 5 if complexity_score < 4 else 6)
        max_failures = max(max_failures, 2)
        retry_delay = max(retry_delay, 2)

    return max_steps, use_vision, max_failures, retry_delay, max_actions_per_step


async def run_browser_use_task(payload: dict[str, Any]) -> dict[str, Any]:
    metadata = payload.get("_tool_metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}

    task = str(payload.get("task") or payload.get("query") or "").strip()
    if not task:
        raise RuntimeError("browser-use 工具缺少任务描述。")

    await _emit_browser_use_event(payload, "preflight_started", {"task": task})

    allowed_domains = _coerce_string_list(
        _resolve_setting(
            payload,
            metadata,
            "allowed_domains",
            env_name="BROWSER_USE_ALLOWED_DOMAINS",
            metadata_key="allowed_domains",
            default=[],
        )
    )
    explicit_max_steps = payload.get("max_steps") not in (None, "")
    explicit_use_vision = payload.get("use_vision") is not None
    max_steps = _coerce_int(
        _resolve_setting(
            payload,
            metadata,
            "max_steps",
            env_name="BROWSER_USE_MAX_STEPS",
            metadata_key="max_steps",
            default=6,
        )
    ) or 6
    use_vision = _coerce_bool(
        _resolve_setting(
            payload,
            metadata,
            "use_vision",
            env_name="BROWSER_USE_USE_VISION",
            metadata_key="use_vision",
            default=False,
        ),
        default=False,
    )
    max_failures = _coerce_int(
        _resolve_setting(
            payload,
            metadata,
            "max_failures",
            env_name="BROWSER_USE_MAX_FAILURES",
            metadata_key="max_failures",
            default=1,
        )
    ) or 1
    retry_delay = _coerce_int(
        _resolve_setting(
            payload,
            metadata,
            "retry_delay",
            env_name="BROWSER_USE_RETRY_DELAY",
            metadata_key="retry_delay",
            default=2,
        )
    ) or 2
    max_actions_per_step = _coerce_int(
        _resolve_setting(
            payload,
            metadata,
            "max_actions_per_step",
            env_name="BROWSER_USE_MAX_ACTIONS_PER_STEP",
            metadata_key="max_actions_per_step",
            default=4,
        )
    ) or 4
    max_steps, use_vision, max_failures, retry_delay, max_actions_per_step = _adjust_browser_task_limits(
        task=task,
        max_steps=max_steps,
        use_vision=use_vision,
        max_failures=max_failures,
        retry_delay=retry_delay,
        max_actions_per_step=max_actions_per_step,
        explicit_max_steps=explicit_max_steps,
        explicit_use_vision=explicit_use_vision,
    )

    health_url = _resolve_setting(
        payload,
        metadata,
        "health_url",
        env_name="BROWSER_USE_REMOTE_HEALTH_URL",
        metadata_key="health_url",
    )
    info_url = _resolve_setting(
        payload,
        metadata,
        "info_url",
        env_name="BROWSER_USE_REMOTE_INFO_URL",
        metadata_key="info_url",
    )
    bearer_token = _resolve_setting(
        payload,
        metadata,
        "bearer_token",
        env_name="BROWSER_USE_REMOTE_BEARER_TOKEN",
        metadata_key="bearer_token",
    )

    preflight = await _preflight_remote_browser(
        health_url=health_url,
        info_url=info_url,
        bearer_token=bearer_token,
    )
    await _emit_browser_use_event(payload, "preflight_completed", preflight)
    info_payload = preflight.get("info", {})
    discovered_cdp_url = None
    if isinstance(info_payload, dict):
        discovered_cdp_url = (
            info_payload.get("cdp_ws_url")
            or info_payload.get("cdp_json_version_url")
            or info_payload.get("cdp_http_url")
        )
    configured_cdp_url = _resolve_cdp_url(payload, metadata)
    effective_cdp_url = configured_cdp_url or discovered_cdp_url

    llm = _build_browser_use_llm(payload, metadata)
    browser, browser_settings = _build_browser(
        payload,
        metadata,
        resolved_cdp_url=effective_cdp_url,
        allowed_domains=allowed_domains,
        bearer_token=bearer_token,
    )
    disable_env_proxy = _coerce_bool(
        _resolve_setting(
            payload,
            metadata,
            "disable_env_proxy",
            env_name="BROWSER_USE_DISABLE_ENV_PROXY",
            metadata_key="disable_env_proxy",
            default=bool(browser_settings.get("cdp_url")),
        ),
        default=bool(browser_settings.get("cdp_url")),
    )

    effective_task = task
    if allowed_domains:
        effective_task = (
            f"{task}\n\n"
            f"约束: 只能访问以下域名或其子域名: {', '.join(allowed_domains)}。"
        )

    try:
        from browser_use import Agent
    except ImportError as exc:  # pragma: no cover - exercised in runtime env
        raise RuntimeError("browser-use 未安装，请先执行 uv sync。") from exc

    agent = _call_with_supported_kwargs(
        Agent,
        task=effective_task,
        llm=llm,
        browser=browser,
        use_vision=use_vision,
        max_failures=max_failures,
        retry_delay=retry_delay,
        max_actions_per_step=max_actions_per_step,
        register_new_step_callback=_browser_use_step_callback(payload),
        register_done_callback=_browser_use_done_callback(payload),
    )
    await _emit_browser_use_event(
        payload,
        "agent_started",
        {
            "task": task,
            "effective_task": effective_task,
            "max_steps": max_steps,
            "use_vision": use_vision,
        },
    )

    env_overrides: dict[str, str | None] = {}
    if disable_env_proxy and browser_settings.get("cdp_url"):
        hostname = urlparse(str(browser_settings["cdp_url"])).hostname or ""
        for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
            env_overrides[key] = None
        if hostname:
            env_overrides["NO_PROXY"] = hostname
            env_overrides["no_proxy"] = hostname

    history = None
    try:
        with _temporary_env(env_overrides):
            history = await _maybe_await(_call_with_supported_kwargs(agent.run, max_steps=max_steps))
    finally:
        browser_stop = getattr(browser, "stop", None)
        if callable(browser_stop):
            await _maybe_await(browser_stop())
        browser_close = getattr(browser, "close", None)
        if callable(browser_close):
            await _maybe_await(browser_close())

    final_result = await _call_history_accessor(history, "final_result")
    visited_urls = await _call_history_accessor(history, "urls")
    actions = await _call_history_accessor(history, "model_actions")
    errors = await _call_history_accessor(history, "errors")

    rendered_result, structured_result = _serialize_browser_use_value(final_result)
    await _emit_browser_use_event(
        payload,
        "browser_done",
        {
            "result": rendered_result,
            "visited_urls": visited_urls,
        },
    )

    summary_lines = [
        "browser-use 远程浏览器任务执行完成。",
        f"任务: {task}",
    ]
    if browser_settings.get("cdp_url"):
        summary_lines.append(f"CDP: {browser_settings['cdp_url']}")
    if allowed_domains:
        summary_lines.append(f"域名限制: {', '.join(allowed_domains)}")
    if rendered_result:
        summary_lines.extend(["结果:", rendered_result])
    else:
        summary_lines.append("结果: browser-use 未返回最终文本结果。")

    return {
        "output": "\n".join(summary_lines),
        "structured_data": {
            "provider": "browser-use",
            "mode": "remote-cdp" if browser_settings.get("cdp_url") else "cloud",
            "task": task,
            "effective_task": effective_task,
            "allowed_domains": allowed_domains,
            "max_steps": max_steps,
            "use_vision": use_vision,
            "disable_env_proxy": disable_env_proxy,
            "max_failures": max_failures,
            "retry_delay": retry_delay,
            "max_actions_per_step": max_actions_per_step,
            "browser": browser_settings,
            "configured_cdp_url": configured_cdp_url,
            "discovered_cdp_url": discovered_cdp_url,
            "preflight": preflight,
            "result": structured_result,
            "visited_urls": visited_urls,
            "actions": actions,
            "errors": errors,
        },
    }
