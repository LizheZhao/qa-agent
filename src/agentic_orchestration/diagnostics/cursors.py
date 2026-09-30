"""Versioned opaque cursor encoding for deterministic diagnostic pages."""

from __future__ import annotations

import base64
import json
from datetime import datetime
from typing import Literal, cast


class InvalidCursorError(ValueError):
    pass


def encode_cursor(kind: str, values: list[str | int]) -> str:
    payload = json.dumps(
        {"v": 1, "kind": kind, "values": values},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return base64.urlsafe_b64encode(payload).decode().rstrip("=")


def decode_cursor(value: str | None, *, kind: str, size: int) -> list[str | int] | None:
    if value is None:
        return None
    try:
        padded = value + "=" * (-len(value) % 4)
        payload = json.loads(base64.b64decode(padded, altchars=b"-_", validate=True))
    except ValueError as exc:
        raise InvalidCursorError("Cursor is not valid base64url JSON") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("v") != 1
        or payload.get("kind") != kind
        or not isinstance(payload.get("values"), list)
        or len(payload["values"]) != size
        or any(not isinstance(item, (str, int)) for item in payload["values"])
    ):
        raise InvalidCursorError("Cursor is incompatible with this resource")
    return cast(list[str | int], payload["values"])


def cursor_datetime(value: str | int) -> datetime:
    if not isinstance(value, str):
        raise InvalidCursorError("Cursor timestamp is invalid")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise InvalidCursorError("Cursor timestamp is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise InvalidCursorError("Cursor timestamp must include a timezone")
    return parsed


def cursor_string(value: str | int) -> str:
    if not isinstance(value, str) or not value:
        raise InvalidCursorError("Cursor ID is invalid")
    return value


def cursor_integer(value: str | int) -> int:
    if not isinstance(value, int) or value < 1:
        raise InvalidCursorError("Cursor number is invalid")
    return value


def encode_attempt_cursor(
    kind: Literal["session_attempts", "recent_attempts"],
    *,
    attempted_turn_number: int,
    started_at: datetime,
    trace_id: str,
) -> str:
    values: list[str | int] = [started_at.isoformat(), trace_id]
    if kind == "session_attempts":
        values.insert(0, attempted_turn_number)
    return encode_cursor(kind, values)
