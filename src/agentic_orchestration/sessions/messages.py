"""Lossless conversion between LangChain messages and controlled stored documents."""

from typing import Any, cast

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from agentic_orchestration.sessions.schema import StoredMessage


def encode_message(message: BaseMessage) -> StoredMessage:
    if message.id is None:
        raise ValueError("Messages must have an ID before they are persisted")
    if not isinstance(message, (SystemMessage, HumanMessage, AIMessage, ToolMessage)):
        raise ValueError(f"Unsupported message type: {message.type}")

    # JSON mode rejects opaque Python objects and produces values that are safe to
    # pass to BSON. Large artifacts will receive a separate storage contract later.
    data = message.model_dump(mode="json")
    stored = StoredMessage(
        id=message.id,
        type=cast(Any, message.type),
        content=data["content"],
        name=data.get("name"),
        example=data.get("example"),
        additional_kwargs=data.get("additional_kwargs", {}),
        response_metadata=data.get("response_metadata", {}),
    )
    if isinstance(message, AIMessage):
        return stored.model_copy(
            update={
                "tool_calls": data.get("tool_calls", []),
                "invalid_tool_calls": data.get("invalid_tool_calls", []),
                "usage_metadata": data.get("usage_metadata"),
            }
        )
    if isinstance(message, ToolMessage):
        return stored.model_copy(
            update={
                "tool_call_id": message.tool_call_id,
                "artifact": data.get("artifact"),
                "tool_status": data.get("status"),
            }
        )
    return stored


def decode_message(stored: StoredMessage) -> BaseMessage:
    common: dict[str, Any] = {
        "content": stored.content,
        "additional_kwargs": stored.additional_kwargs,
        "response_metadata": stored.response_metadata,
        "name": stored.name,
        "id": stored.id,
    }
    if stored.type == "system":
        return SystemMessage(**common)
    if stored.type == "human":
        return HumanMessage(**common, example=bool(stored.example))
    if stored.type == "ai":
        return AIMessage(
            **common,
            example=bool(stored.example),
            tool_calls=stored.tool_calls,
            invalid_tool_calls=stored.invalid_tool_calls,
            usage_metadata=stored.usage_metadata,
        )
    if stored.tool_call_id is None:
        raise ValueError("Stored tool messages must include tool_call_id")
    return ToolMessage(
        **common,
        tool_call_id=stored.tool_call_id,
        artifact=stored.artifact,
        status=cast(Any, stored.tool_status),
    )
