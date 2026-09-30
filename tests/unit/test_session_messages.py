import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from agentic_orchestration.sessions.messages import decode_message, encode_message


@pytest.mark.unit
def test_message_codec_round_trips_supported_message_fields() -> None:
    messages = [
        SystemMessage(
            content="rules",
            id="system-1",
            name="policy",
            additional_kwargs={"scope": "session"},
        ),
        HumanMessage(
            content=[{"type": "text", "text": "hello"}],
            id="human-1",
            response_metadata={"source": "api"},
        ),
        AIMessage(
            content="",
            id="ai-1",
            tool_calls=[{"name": "lookup", "args": {"query": "x"}, "id": "call-1"}],
            usage_metadata={"input_tokens": 4, "output_tokens": 2, "total_tokens": 6},
        ),
        ToolMessage(
            content="found",
            id="tool-1",
            tool_call_id="call-1",
            artifact={"rows": 1},
            status="success",
        ),
    ]

    restored = [decode_message(encode_message(message)) for message in messages]

    assert [message.model_dump(mode="json") for message in restored] == [
        message.model_dump(mode="json") for message in messages
    ]


@pytest.mark.unit
def test_message_codec_requires_persistent_message_id() -> None:
    with pytest.raises(ValueError, match="must have an ID"):
        encode_message(HumanMessage(content="missing"))
