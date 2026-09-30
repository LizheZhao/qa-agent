"""Bounded, secret-aware conversion of runtime values into trace content."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from hashlib import sha256
from typing import Any

from langchain_core.messages import BaseMessage
from orchestration_core import CapturedContent, ContentCaptureMode
from pydantic import BaseModel

DEFAULT_MAX_INLINE_BYTES = 64 * 1024
MAX_CAPTURE_DEPTH = 12
# An oversized value is retried as a preview: long strings keep their head, long lists their first
# items, so the start of a readout or a row sample stays readable in the trace.
PREVIEW_STRING_CHARS = 1_000
PREVIEW_LIST_ITEMS = 5
REDACTED = "[REDACTED]"
_SECRET_KEYS = {
    "api_key",
    "authorization",
    "cookie",
    "credentials",
    "gateway_token",
    "password",
    "secret",
    "set_cookie",
    "token",
}


def capture_content(
    value: Any,
    *,
    content_type: str = "application/json",
    max_inline_bytes: int = DEFAULT_MAX_INLINE_BYTES,
) -> CapturedContent:
    """Capture a BSON/JSON-safe projection or an explicit omission marker."""

    try:
        normalized = _normalize(value, depth=0)
        encoded = _encode(normalized)
    except (TypeError, ValueError, RecursionError):
        return CapturedContent(
            mode=ContentCaptureMode.OMITTED,
            content_type=content_type,
            reason="unsupported_value",
        )
    digest = sha256(encoded).hexdigest()
    if len(encoded) > max_inline_bytes:
        preview = _preview(normalized)
        encoded_preview = _encode(preview)
        if len(encoded_preview) > max_inline_bytes:
            return CapturedContent(
                mode=ContentCaptureMode.OMITTED,
                content_type=content_type,
                byte_size=len(encoded),
                sha256=digest,
                reason="inline_size_limit_exceeded",
            )
        # byte_size and sha256 describe the stored preview; metadata names the full value.
        return CapturedContent(
            mode=ContentCaptureMode.INLINE,
            content_type=content_type,
            value=preview,
            byte_size=len(encoded_preview),
            sha256=sha256(encoded_preview).hexdigest(),
            metadata={
                "preview": True,
                "original_byte_size": len(encoded),
                "original_sha256": digest,
            },
        )
    return CapturedContent(
        mode=ContentCaptureMode.INLINE,
        content_type=content_type,
        value=normalized,
        byte_size=len(encoded),
        sha256=digest,
    )


def _encode(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )


def _preview(value: Any) -> Any:
    """Shorten an already normalized value, marking every cut so it is never mistaken for whole."""

    if isinstance(value, str) and len(value) > PREVIEW_STRING_CHARS:
        cut = len(value) - PREVIEW_STRING_CHARS
        return f"{value[:PREVIEW_STRING_CHARS]}... [truncated {cut} chars]"
    if isinstance(value, dict):
        return {key: _preview(item) for key, item in value.items()}
    if isinstance(value, list):
        head = [_preview(item) for item in value[:PREVIEW_LIST_ITEMS]]
        if len(value) > PREVIEW_LIST_ITEMS:
            head.append(f"[truncated {len(value) - PREVIEW_LIST_ITEMS} more items]")
        return head
    return value


def _normalize(value: Any, *, depth: int) -> Any:
    if depth > MAX_CAPTURE_DEPTH:
        raise ValueError("capture nesting is too deep")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, BaseMessage):
        return _message(value, depth=depth)
    if isinstance(value, BaseModel):
        return _normalize(value.model_dump(mode="python"), depth=depth + 1)
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for raw_key, item in value.items():
            key = str(raw_key)
            normalized[key] = REDACTED if _is_secret_key(key) else _normalize(item, depth=depth + 1)
        return normalized
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_normalize(item, depth=depth + 1) for item in value]
    raise TypeError(f"unsupported capture value: {type(value).__name__}")


def _is_secret_key(key: str) -> bool:
    normalized = key.lower().replace("-", "_")
    return normalized in _SECRET_KEYS or normalized.endswith(
        ("_api_key", "_authorization", "_cookie", "_password", "_secret", "_token")
    )


def _message(message: BaseMessage, *, depth: int) -> dict[str, Any]:
    projection: dict[str, Any] = {
        "id": message.id,
        "role": message.type,
        "content": _normalize(message.content, depth=depth + 1),
    }
    if message.name is not None:
        projection["name"] = message.name
    if message.additional_kwargs:
        projection["additional_kwargs"] = _normalize(message.additional_kwargs, depth=depth + 1)
    if message.response_metadata:
        projection["response_metadata"] = _normalize(message.response_metadata, depth=depth + 1)
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        projection["tool_calls"] = _normalize(tool_calls, depth=depth + 1)
    usage_metadata = getattr(message, "usage_metadata", None)
    if usage_metadata:
        projection["usage_metadata"] = _normalize(usage_metadata, depth=depth + 1)
    tool_call_id = getattr(message, "tool_call_id", None)
    if tool_call_id is not None:
        projection["tool_call_id"] = tool_call_id
    return projection
