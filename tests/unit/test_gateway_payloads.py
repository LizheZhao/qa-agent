from __future__ import annotations

from typing import Any

import httpx
import pytest
from enterprise_llm.adapter import ChatEnterpriseGateway, normalize_tools
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, Field


class LookupInput(BaseModel):
    query: str = Field(description="Lookup query")


def lookup(query: str) -> str:
    """Look up harmless test data."""

    return query


def make_model(vendor: str = "claude", **overrides: Any) -> ChatEnterpriseGateway:
    values: dict[str, Any] = {
        "base_url": "https://gateway.test/root/",
        "vendor": vendor,
        "model_name": "test-model",
        "client_code": "client-123",
        "token": "never-print-token",
        "user": "user-123",
    }
    values.update(overrides)
    return ChatEnterpriseGateway(**values)


@pytest.mark.unit
def test_timeout_defaults_to_enterprise_parity_and_is_configurable() -> None:
    assert make_model().timeout_seconds == 180.0
    assert make_model(timeout_seconds=45).timeout_seconds == 45.0


@pytest.mark.unit
def test_tool_normalization_uses_langchain_schema_conversion() -> None:
    tool = normalize_tools([lookup])[0]
    assert tool["name"] == "lookup"
    assert tool["input_schema"]["properties"]["query"]["type"] == "string"


@pytest.mark.unit
def test_claude_message_history_and_native_shape() -> None:
    model = make_model()
    tools = normalize_tools([lookup])
    messages = [
        SystemMessage("system rules"),
        HumanMessage([{"type": "text", "text": "first"}]),
        HumanMessage("second"),
        AIMessage(
            "",
            tool_calls=[{"name": "lookup", "args": {"query": "x"}, "id": "c1"}],
        ),
        ToolMessage("found", tool_call_id="c1"),
        AIMessage("answer draft"),
    ]
    payload = model._build_payload(messages, tools, "lookup")
    assert payload["model"] == "test-model"
    assert payload["max_tokens"] == 4096
    assert payload["tools"] == tools
    assert payload["tool_choice"] == {"type": "tool", "name": "lookup"}
    contents = payload["contents"]
    assert contents[0]["role"] == "user"
    assert "system rules\n\nfirst\nsecond" in contents[0]["content"][0]["text"]
    assert '"tool_calls"' in contents[1]["content"][0]["text"]
    assert "Tool result for call c1" in contents[2]["content"][0]["text"]
    assert contents[-1] == {
        "role": "user",
        "content": [{"type": "text", "text": "Continue."}],
    }


@pytest.mark.unit
def test_openai_responses_shape_and_native_tools() -> None:
    model = make_model("openai")
    tools = normalize_tools([lookup])
    payload = model._build_payload(
        [SystemMessage("rules"), HumanMessage("hello"), AIMessage("hi")],
        tools,
        "required",
    )
    assert payload["contents"][0] == {
        "role": "system",
        "type": "message",
        "content": [{"text": "rules", "type": "input_text"}],
    }
    assert payload["contents"][2]["content"][0]["type"] == "output_text"
    assert payload["tools"][0]["type"] == "function"
    assert payload["tools"][0]["parameters"] == tools[0]["input_schema"]
    assert payload["tool_choice"] == {"type": "function", "name": "lookup"}


@pytest.mark.unit
@pytest.mark.parametrize(
    ("vendor", "choice", "expected"),
    [
        ("claude", "required", {"type": "tool", "name": "lookup"}),
        ("claude", "any", {"type": "tool", "name": "lookup"}),
        ("claude", "auto", None),
        ("claude", "none", None),
        ("openai", "required", {"type": "function", "name": "lookup"}),
        ("openai", "any", {"type": "function", "name": "lookup"}),
        ("openai", "auto", None),
        ("openai", "none", "none"),
        ("openai", {"type": "function", "name": "lookup"}, {"type": "function", "name": "lookup"}),
    ],
)
def test_meaningful_tool_choice_mappings(vendor: str, choice: Any, expected: Any) -> None:
    model = make_model(vendor)
    payload = model._build_payload([HumanMessage("go")], normalize_tools([lookup]), choice)
    assert payload.get("tool_choice") == expected


@pytest.mark.unit
def test_required_with_multiple_claude_tools_maps_to_any() -> None:
    def other(value: int) -> int:
        """Other test operation."""

        return value

    payload = make_model()._build_payload(
        [HumanMessage("go")], normalize_tools([lookup, other]), "required"
    )
    assert payload["tool_choice"] == {"type": "any"}


@pytest.mark.unit
def test_headers_are_sent_but_not_in_identifying_params() -> None:
    captured: httpx.Request | None = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured
        captured = request
        return httpx.Response(
            200,
            json={"status_code": 200, "result": {"error": False, "data": {"response": "ok"}}},
        )

    model = make_model(http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert model.invoke("hello").content == "ok"
    assert captured is not None
    assert captured.headers["X-HTTP-AUTH-TOKEN"] == "never-print-token"
    assert captured.url.path == "/root/client/client-123/claude/generate"
    assert "token" not in str(model._identifying_params)
