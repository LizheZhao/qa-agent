"""LangChain chat model for the enterprise generation gateway."""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

import httpx
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import ConfigDict, Field, PrivateAttr, model_validator

from enterprise_llm.errors import (
    GatewayConfigurationError,
    GatewayEnvelopeError,
    GatewayHTTPError,
    GatewayResponseError,
    GatewayTimeoutError,
    GatewayTransportError,
)

Vendor = Literal["claude", "openai"]
JsonObject = dict[str, Any]


def _content_text(content: str | list[str | dict[str, Any]]) -> str:
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, Mapping):
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "\n".join(part for part in parts if part)


def normalize_tools(
    tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
) -> list[JsonObject]:
    """Use LangChain's maintained converter, then normalize to Anthropic definitions."""

    normalized: list[JsonObject] = []
    for tool in tools:
        converted = convert_to_openai_tool(tool)
        function = converted.get("function", converted)
        normalized.append(
            {
                "name": function["name"],
                "description": function.get("description", ""),
                "input_schema": function.get("parameters", {"type": "object", "properties": {}}),
            }
        )
    return normalized


def _forced_tool_name(tool_choice: Any, tools: Sequence[JsonObject]) -> str | None:
    if isinstance(tool_choice, str):
        if tool_choice not in {"auto", "none", "required", "any"}:
            return tool_choice
        if tool_choice in {"required", "any"} and len(tools) == 1:
            return str(tools[0]["name"])
    if isinstance(tool_choice, Mapping):
        name = tool_choice.get("name")
        if isinstance(name, str):
            return name
        function = tool_choice.get("function")
        if isinstance(function, Mapping) and isinstance(function.get("name"), str):
            return str(function["name"])
    return None


def _serialize_tool_decision(message: AIMessage) -> str:
    calls = [
        {"name": call["name"], "arguments": call.get("args", {})} for call in message.tool_calls
    ]
    return json.dumps({"tool_calls": calls}, separators=(",", ":"))


def _message_parts(messages: Sequence[BaseMessage]) -> list[tuple[str, str]]:
    parts: list[tuple[str, str]] = []
    for message in messages:
        text = _content_text(message.content)
        if isinstance(message, SystemMessage):
            role = "system"
        elif isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, AIMessage):
            role = "assistant"
            if message.tool_calls:
                decision = _serialize_tool_decision(message)
                text = f"{text}\n{decision}".strip()
        elif isinstance(message, ToolMessage):
            role = "user"
            text = f"Tool result for call {message.tool_call_id}: {text}"
        else:
            role = "user"
        parts.append((role, text))
    return parts


def _merge_roles(parts: Sequence[tuple[str, str]]) -> list[tuple[str, str]]:
    merged: list[tuple[str, str]] = []
    for role, text in parts:
        if merged and merged[-1][0] == role:
            old_role, old_text = merged[-1]
            merged[-1] = (old_role, f"{old_text}\n{text}".strip())
        else:
            merged.append((role, text))
    return merged


def _claude_contents(messages: Sequence[BaseMessage]) -> list[JsonObject]:
    parts = _message_parts(messages)
    system = "\n".join(text for role, text in parts if role == "system" and text)
    conversational = [(role, text) for role, text in parts if role != "system"]
    if system:
        if conversational and conversational[0][0] == "user":
            role, text = conversational[0]
            conversational[0] = (role, f"{system}\n\n{text}".strip())
        else:
            conversational.insert(0, ("user", system))
    if not conversational:
        conversational = [("user", system or "Continue.")]
    merged = _merge_roles(conversational)
    if merged[-1][0] == "assistant":
        merged.append(("user", "Continue."))
    return [{"role": role, "content": [{"type": "text", "text": text}]} for role, text in merged]


def _openai_contents(messages: Sequence[BaseMessage]) -> list[JsonObject]:
    contents: list[JsonObject] = []
    for role, text in _merge_roles(_message_parts(messages)):
        content_type = "output_text" if role == "assistant" else "input_text"
        contents.append(
            {"role": role, "type": "message", "content": [{"text": text, "type": content_type}]}
        )
    return contents


def _arguments(value: Any) -> JsonObject:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise GatewayResponseError("Tool arguments are not valid JSON") from exc
        if isinstance(parsed, dict):
            return parsed
    raise GatewayResponseError("Tool arguments must be a JSON object")


def _normalize_native_call(raw: Mapping[str, Any]) -> JsonObject:
    function = raw.get("function")
    source = function if isinstance(function, Mapping) else raw
    name = source.get("name")
    arguments = source.get("arguments", source.get("input", {}))
    if not isinstance(name, str):
        raise GatewayResponseError("Native tool call is missing a name")
    call_id = raw.get("id") or raw.get("call_id") or f"call_{uuid.uuid4().hex}"
    return {"name": name, "args": _arguments(arguments), "id": str(call_id), "type": "tool_call"}


def _native_tool_calls(data: Mapping[str, Any], response: Any) -> list[JsonObject]:
    """Read the first populated native-call representation in gateway precedence order."""

    direct = data.get("tool_calls")
    candidates = _mapping_items(direct)
    if candidates:
        return _normalize_distinct_native_calls(candidates)

    function_call = data.get("function_call")
    if isinstance(function_call, Mapping):
        return _normalize_distinct_native_calls([function_call])

    candidates = _function_call_items(data.get("output"))
    if candidates:
        return _normalize_distinct_native_calls(candidates)

    if isinstance(response, Mapping):
        candidates = _function_call_items(response.get("output"))
        if candidates:
            return _normalize_distinct_native_calls(candidates)

        content = _mapping_items(response.get("content"))
        candidates = [item for item in content if item.get("type") == "tool_use"]
        if candidates:
            return _normalize_distinct_native_calls(candidates)

    return []


def _mapping_items(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _function_call_items(value: Any) -> list[Mapping[str, Any]]:
    return [item for item in _mapping_items(value) if item.get("type") == "function_call"]


def _normalize_distinct_native_calls(candidates: Sequence[Mapping[str, Any]]) -> list[JsonObject]:
    calls: list[JsonObject] = []
    seen_provider_ids: set[str] = set()
    for candidate in candidates:
        provider_id = candidate.get("id") or candidate.get("call_id")
        if provider_id is not None:
            normalized_id = str(provider_id)
            if normalized_id in seen_provider_ids:
                continue
            seen_provider_ids.add(normalized_id)
        calls.append(_normalize_native_call(candidate))
    return calls


def _response_text(response: Any) -> str:
    if isinstance(response, str):
        return response
    if not isinstance(response, Mapping):
        raise GatewayResponseError("data.response must be a string or provider response object")
    pieces: list[str] = []
    content = response.get("content")
    if isinstance(content, list):
        pieces.extend(
            str(block["text"])
            for block in content
            if isinstance(block, Mapping) and block.get("type") == "text" and "text" in block
        )
    output = response.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, Mapping):
                continue
            blocks = item.get("content")
            if isinstance(blocks, list):
                pieces.extend(
                    str(block["text"])
                    for block in blocks
                    if isinstance(block, Mapping) and isinstance(block.get("text"), str)
                )
    return "\n".join(pieces)


def _usage_and_model(data: Mapping[str, Any], response: Any) -> tuple[JsonObject, str | None]:
    response_map = response if isinstance(response, Mapping) else {}
    usage_value = response_map.get("usage", data.get("usage", {}))
    usage = usage_value if isinstance(usage_value, Mapping) else {}
    cache = int(usage.get("cache_creation_input_tokens", 0) or 0) + int(
        usage.get("cache_read_input_tokens", 0) or 0
    )
    input_tokens = int(usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0) + cache
    output_tokens = int(usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0)
    total = int(usage.get("total_tokens", input_tokens + output_tokens) or 0)
    if cache and "total_tokens" in usage:
        total += cache
    normalized = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total,
    }
    model = response_map.get("model", data.get("model"))
    return normalized, str(model) if model is not None else None


class ChatEnterpriseGateway(BaseChatModel):
    """Explicitly configured LangChain model backed by an enterprise HTTP gateway."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    base_url: str
    vendor: Vendor
    model_name: str
    client_code: str
    token: str = Field(repr=False)
    user: str = Field(repr=False)
    timeout_seconds: float = Field(default=180.0, gt=0)
    max_tokens: int = 4096

    _client: httpx.Client | None = PrivateAttr(default=None)
    _async_client: httpx.AsyncClient | None = PrivateAttr(default=None)

    def __init__(self, **data: Any) -> None:
        client = data.pop("http_client", None)
        async_client = data.pop("async_http_client", None)
        super().__init__(**data)
        self._client = client
        self._async_client = async_client

    @model_validator(mode="after")
    def validate_configuration(self) -> ChatEnterpriseGateway:
        missing = [
            name
            for name in ("base_url", "model_name", "client_code", "token", "user")
            if not getattr(self, name).strip()
        ]
        if missing:
            raise GatewayConfigurationError(
                "Missing required enterprise gateway values: " + ", ".join(missing)
            )
        return self

    @property
    def _llm_type(self) -> str:
        return "enterprise-gateway"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {
            "base_url": self.base_url.rstrip("/"),
            "vendor": self.vendor,
            "model_name": self.model_name,
            "client_code": self.client_code,
        }

    @property
    def endpoint(self) -> str:
        return f"{self.base_url.rstrip('/')}/client/{self.client_code}/{self.vendor}/generate"

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "X-HTTP-AUTH-TOKEN": self.token,
            "X-HTTP-AUTH-USER": self.user,
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: Any = None,
        **kwargs: Any,
    ) -> Runnable[Any, BaseMessage]:
        return self.bind(tools=list(tools), tool_choice=tool_choice, **kwargs)

    def _build_payload(
        self,
        messages: Sequence[BaseMessage],
        tools: Sequence[JsonObject],
        tool_choice: Any,
    ) -> JsonObject:
        effective_messages = list(messages)
        if self.vendor == "claude":
            payload: JsonObject = {
                "contents": _claude_contents(effective_messages),
                "max_tokens": self.max_tokens,
                "verbose": True,
                "model": self.model_name,
            }
            if tools:
                payload["tools"] = tools
                forced = _forced_tool_name(tool_choice, tools)
                if forced:
                    payload["tool_choice"] = {"type": "tool", "name": forced}
                elif tool_choice in {"required", "any"}:
                    payload["tool_choice"] = {"type": "any"}
            return payload

        payload = {"contents": _openai_contents(effective_messages), "model": self.model_name}
        if tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["input_schema"],
                }
                for tool in tools
            ]
            forced = _forced_tool_name(tool_choice, tools)
            if forced:
                payload["tool_choice"] = {"type": "function", "name": forced}
            elif tool_choice == "none":
                payload["tool_choice"] = "none"
            elif tool_choice in {"required", "any"}:
                payload["tool_choice"] = "required"
        return payload

    def _decode_response(self, body: Any) -> AIMessage:
        if not isinstance(body, Mapping):
            raise GatewayResponseError("Gateway body must be a JSON object")
        if body.get("status_code") != 200:
            raise GatewayEnvelopeError(_envelope_message(body))
        result = body.get("result")
        if not isinstance(result, Mapping):
            raise GatewayResponseError("Gateway body is missing result object")
        if result.get("error") is True:
            message = str(result.get("errorMessage") or "Gateway provider reported an error")
            raise GatewayEnvelopeError(message)
        data = body.get("data")
        if not isinstance(data, Mapping):
            # Temporary compatibility for callers that adopted the bootstrap's original nesting.
            data = result.get("data")
        if not isinstance(data, Mapping):
            raise GatewayResponseError("Gateway body is missing top-level data object")
        response = data.get("response")
        text = _response_text(response)
        calls = _native_tool_calls(data, response)
        usage, provider_model = _usage_and_model(data, response)
        return AIMessage(
            content=text,
            tool_calls=calls,
            response_metadata={"model_name": provider_model or self.model_name, "usage": usage},
            usage_metadata=usage,
        )

    def _sync_request(self, payload: JsonObject) -> Any:
        client = self._client or httpx.Client(timeout=self.timeout_seconds)
        close = self._client is None
        try:
            response = client.post(self.endpoint, headers=self._headers, json=payload)
            if response.status_code >= 400:
                message = _http_message(response)
                raise GatewayHTTPError(
                    f"Gateway HTTP {response.status_code}: {message}",
                    status_code=response.status_code,
                )
            try:
                return response.json()
            except ValueError as exc:
                raise GatewayResponseError("Gateway response was not valid JSON") from exc
        except httpx.TimeoutException as exc:
            raise GatewayTimeoutError("Enterprise gateway request timed out") from exc
        except httpx.HTTPError as exc:
            raise GatewayTransportError(f"Enterprise gateway transport failure: {exc}") from exc
        finally:
            if close:
                client.close()

    async def _async_request(self, payload: JsonObject) -> Any:
        client = self._async_client or httpx.AsyncClient(timeout=self.timeout_seconds)
        close = self._async_client is None
        try:
            response = await client.post(self.endpoint, headers=self._headers, json=payload)
            if response.status_code >= 400:
                message = _http_message(response)
                raise GatewayHTTPError(
                    f"Gateway HTTP {response.status_code}: {message}",
                    status_code=response.status_code,
                )
            try:
                return response.json()
            except ValueError as exc:
                raise GatewayResponseError("Gateway response was not valid JSON") from exc
        except httpx.TimeoutException as exc:
            raise GatewayTimeoutError("Enterprise gateway request timed out") from exc
        except httpx.HTTPError as exc:
            raise GatewayTransportError(f"Enterprise gateway transport failure: {exc}") from exc
        finally:
            if close:
                await client.aclose()

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        del stop, run_manager
        tools = normalize_tools(kwargs.pop("tools", []))
        tool_choice = kwargs.pop("tool_choice", None)
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise GatewayConfigurationError(f"Unsupported invocation options: {unexpected}")
        body = self._sync_request(self._build_payload(messages, tools, tool_choice))
        message = self._decode_response(body)
        return ChatResult(generations=[ChatGeneration(message=message)])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        del stop, run_manager
        tools = normalize_tools(kwargs.pop("tools", []))
        tool_choice = kwargs.pop("tool_choice", None)
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise GatewayConfigurationError(f"Unsupported invocation options: {unexpected}")
        body = await self._async_request(self._build_payload(messages, tools, tool_choice))
        message = self._decode_response(body)
        return ChatResult(generations=[ChatGeneration(message=message)])


def _envelope_message(body: Mapping[str, Any]) -> str:
    result = body.get("result")
    if isinstance(result, Mapping) and result.get("errorMessage"):
        return str(result["errorMessage"])
    return f"Gateway envelope status_code was {body.get('status_code')!r}, expected 200"


def _http_message(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return response.reason_phrase or "request failed"
    if isinstance(body, Mapping):
        result = body.get("result")
        if isinstance(result, Mapping) and result.get("errorMessage"):
            return str(result["errorMessage"])
        for key in ("errorMessage", "message", "detail"):
            if body.get(key):
                return str(body[key])
    return response.reason_phrase or "request failed"
