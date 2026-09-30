from __future__ import annotations

from typing import Any

import httpx
import pytest
from enterprise_llm import (
    ChatEnterpriseGateway,
    GatewayEnvelopeError,
    GatewayHTTPError,
    GatewayResponseError,
    GatewayTimeoutError,
    GatewayTransportError,
)
from langchain_core.messages import HumanMessage


def tool(value: int) -> int:
    """A test tool."""

    return value


def envelope(response: Any, **data_values: Any) -> dict[str, Any]:
    return {
        "status_code": 200,
        "result": {"error": False},
        "data": {"response": response, **data_values},
    }


def model_with_handler(handler: Any, **overrides: Any) -> ChatEnterpriseGateway:
    values: dict[str, Any] = {
        "base_url": "https://gateway.test",
        "vendor": "claude",
        "model_name": "requested",
        "client_code": "client",
        "token": "secret-token",
        "user": "secret-user",
        "http_client": httpx.Client(transport=httpx.MockTransport(handler)),
        "async_http_client": httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    }
    values.update(overrides)
    return ChatEnterpriseGateway(**values)


@pytest.mark.unit
def test_anthropic_tool_use_text_usage_and_model_extraction() -> None:
    gateway = model_with_handler(lambda request: httpx.Response(200, json=envelope("unused")))
    message = gateway._decode_response(
        envelope(
            {
                "model": "claude-provider",
                "content": [
                    {"type": "text", "text": "thinking"},
                    {"type": "tool_use", "id": "call-a", "name": "tool", "input": {"value": 3}},
                ],
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 4,
                    "cache_creation_input_tokens": 2,
                    "cache_read_input_tokens": 1,
                },
            }
        ),
    )
    assert message.content == "thinking"
    assert message.tool_calls[0] == {
        "name": "tool",
        "args": {"value": 3},
        "id": "call-a",
        "type": "tool_call",
    }
    assert message.usage_metadata == {"input_tokens": 13, "output_tokens": 4, "total_tokens": 17}
    assert message.response_metadata["model_name"] == "claude-provider"


@pytest.mark.unit
def test_openai_function_call_shapes_and_response_text() -> None:
    gateway = model_with_handler(lambda request: httpx.Response(200, json=envelope("unused")))
    body = envelope(
        {
            "model": "gpt-provider",
            "output": [
                {"type": "function_call", "id": "c1", "name": "tool", "arguments": '{"value":5}'},
                {"type": "message", "content": [{"type": "output_text", "text": "text"}]},
            ],
            "usage": {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9},
        },
        function_call={"id": "c2", "name": "tool", "arguments": {"value": 6}},
    )
    message = gateway._decode_response(body)
    assert message.content == "text"
    assert [call["args"] for call in message.tool_calls] == [{"value": 6}]
    assert message.usage_metadata == {"input_tokens": 7, "output_tokens": 2, "total_tokens": 9}


@pytest.mark.unit
def test_top_level_data_is_authoritative_over_legacy_nested_data() -> None:
    gateway = model_with_handler(lambda request: httpx.Response(200, json=envelope("unused")))
    body = envelope("top-level")
    body["result"]["data"] = {"response": "legacy-nested"}
    message = gateway._decode_response(body)
    assert message.content == "top-level"


@pytest.mark.unit
def test_legacy_nested_data_remains_a_temporary_fallback() -> None:
    gateway = model_with_handler(lambda request: httpx.Response(200, json=envelope("unused")))
    body = {
        "status_code": 200,
        "result": {"error": False, "data": {"response": "legacy-nested"}},
    }
    message = gateway._decode_response(body)
    assert message.content == "legacy-nested"


@pytest.mark.unit
def test_native_tool_call_precedence_prevents_duplicate_execution() -> None:
    gateway = model_with_handler(lambda request: httpx.Response(200, json=envelope("unused")))
    duplicate = {"type": "tool_use", "id": "same-call", "name": "tool", "input": {"value": 9}}
    body = envelope(
        {"content": [duplicate]},
        tool_calls=[
            {"id": "same-call", "function": {"name": "tool", "arguments": '{"value":9}'}},
            {"id": "same-call", "function": {"name": "tool", "arguments": '{"value":9}'}},
        ],
    )
    message = gateway._decode_response(body)
    assert message.tool_calls == [
        {"name": "tool", "args": {"value": 9}, "id": "same-call", "type": "tool_call"}
    ]


@pytest.mark.unit
def test_plain_json_text_is_not_parsed_as_a_tool_call() -> None:
    gateway = model_with_handler(lambda request: httpx.Response(200, json=envelope("unused")))
    text = '{"tool_calls":[{"name":"tool","arguments":{}}]}'
    message = gateway._decode_response(envelope(text))
    assert message.content == text
    assert message.tool_calls == []


@pytest.mark.unit
def test_native_capability_error_is_not_retried() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(400, json={"message": "tool use unsupported"})

    gateway = model_with_handler(handler)
    with pytest.raises(GatewayHTTPError):
        gateway.bind_tools([tool]).invoke([HumanMessage("use it")])
    assert len(requests) == 1
    assert "tools" in requests[0].read().decode()


@pytest.mark.unit
@pytest.mark.parametrize("failure", ["timeout", "http", "transport"])
def test_native_request_is_not_retried_for_failures(failure: str) -> None:
    count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal count
        count += 1
        if failure == "timeout":
            raise httpx.ReadTimeout("timed out", request=request)
        if failure == "transport":
            raise httpx.ConnectError("connection failed", request=request)
        return httpx.Response(500, json={"message": "ordinary failure"})

    gateway = model_with_handler(handler)
    expected = {
        "timeout": GatewayTimeoutError,
        "http": GatewayHTTPError,
        "transport": GatewayTransportError,
    }[failure]
    with pytest.raises(expected):
        gateway.bind_tools([tool]).invoke("go")
    assert count == 1


@pytest.mark.unit
@pytest.mark.parametrize(
    ("body", "error"),
    [
        ([], GatewayResponseError),
        (
            {"status_code": 503, "result": {"errorMessage": "provider unavailable"}},
            GatewayEnvelopeError,
        ),
        (
            {"status_code": 200, "result": {"error": True, "errorMessage": "bad provider"}},
            GatewayEnvelopeError,
        ),
        ({"status_code": 200, "result": {"error": False}}, GatewayResponseError),
    ],
)
def test_malformed_envelopes_and_provider_errors(body: Any, error: type[Exception]) -> None:
    gateway = model_with_handler(lambda request: httpx.Response(200, json=envelope("unused")))
    with pytest.raises(error):
        gateway._decode_response(body)


@pytest.mark.unit
async def test_sync_and_async_langchain_calls() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=envelope("ok"))

    gateway = model_with_handler(handler)
    assert gateway.invoke("sync").content == "ok"
    assert (await gateway.ainvoke("async")).content == "ok"


@pytest.mark.unit
def test_non_json_http_response_is_normalized() -> None:
    gateway = model_with_handler(lambda request: httpx.Response(200, text="not json"))
    with pytest.raises(GatewayResponseError, match="not valid JSON"):
        gateway.invoke("hello")
