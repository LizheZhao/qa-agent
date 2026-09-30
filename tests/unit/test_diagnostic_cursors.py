from datetime import UTC, datetime

import pytest

from agentic_orchestration.diagnostics.cursors import (
    InvalidCursorError,
    cursor_datetime,
    cursor_integer,
    cursor_string,
    decode_cursor,
    encode_attempt_cursor,
    encode_cursor,
)


@pytest.mark.unit
def test_cursor_round_trip_preserves_kind_and_values() -> None:
    encoded = encode_cursor("sessions", ["2026-08-18T12:00:00+00:00", "session-b"])

    assert decode_cursor(encoded, kind="sessions", size=2) == [
        "2026-08-18T12:00:00+00:00",
        "session-b",
    ]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("value", "kind", "size"),
    [
        ("not-base64", "sessions", 2),
        (encode_cursor("turns", [1]), "sessions", 1),
        (encode_cursor("sessions", ["timestamp"]), "sessions", 2),
    ],
)
def test_cursor_rejects_invalid_encoding_kind_or_size(value: str, kind: str, size: int) -> None:
    with pytest.raises(InvalidCursorError):
        decode_cursor(value, kind=kind, size=size)


@pytest.mark.unit
def test_cursor_rejects_unknown_version() -> None:
    version_two = "eyJraW5kIjoic2Vzc2lvbnMiLCJ2IjoyLCJ2YWx1ZXMiOlsiYSIsImIiXX0"

    with pytest.raises(InvalidCursorError, match="incompatible"):
        decode_cursor(version_two, kind="sessions", size=2)


@pytest.mark.unit
def test_cursor_timestamp_requires_timezone() -> None:
    with pytest.raises(InvalidCursorError, match="timezone"):
        cursor_datetime("2026-08-18T12:00:00")

    assert cursor_datetime("2026-08-18T12:00:00+00:00") == datetime(2026, 8, 18, 12, tzinfo=UTC)


@pytest.mark.unit
def test_cursor_fields_require_expected_types() -> None:
    assert cursor_string("session-a") == "session-a"
    assert cursor_integer(3) == 3

    with pytest.raises(InvalidCursorError):
        cursor_string(3)
    with pytest.raises(InvalidCursorError):
        cursor_integer("3")


@pytest.mark.unit
def test_attempt_cursor_uses_the_resource_sort_key() -> None:
    started_at = datetime(2026, 8, 18, 12, tzinfo=UTC)
    session_cursor = encode_attempt_cursor(
        "session_attempts",
        attempted_turn_number=3,
        started_at=started_at,
        trace_id="trace-a",
    )
    recent_cursor = encode_attempt_cursor(
        "recent_attempts",
        attempted_turn_number=3,
        started_at=started_at,
        trace_id="trace-a",
    )

    assert decode_cursor(session_cursor, kind="session_attempts", size=3) == [
        3,
        started_at.isoformat(),
        "trace-a",
    ]
    assert decode_cursor(recent_cursor, kind="recent_attempts", size=2) == [
        started_at.isoformat(),
        "trace-a",
    ]
