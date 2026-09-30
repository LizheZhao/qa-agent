"""Non-durable lifecycle event delivery seam for the future diagnostic UI."""

from collections.abc import Sequence
from typing import Protocol

from orchestration_core import ExecutionEvent


class ExecutionEventPublisher(Protocol):
    """Accept a live event without placing subscriber I/O on the graph path."""

    def publish(self, event: ExecutionEvent) -> None: ...


class NullExecutionEventPublisher:
    """Discard events when no live diagnostic transport is configured."""

    def publish(self, event: ExecutionEvent) -> None:
        del event


class MemoryExecutionEventPublisher:
    """Concurrency-safe-enough event collector for single-loop tests."""

    def __init__(self) -> None:
        self._events: dict[str, list[ExecutionEvent]] = {}

    def publish(self, event: ExecutionEvent) -> None:
        events = self._events.setdefault(event.trace_id, [])
        expected = len(events) + 1
        if event.sequence != expected:
            raise ValueError(
                f"event sequence for trace {event.trace_id!r} must be {expected}, "
                f"received {event.sequence}"
            )
        events.append(event)

    def events(self, trace_id: str) -> Sequence[ExecutionEvent]:
        return tuple(self._events.get(trace_id, ()))
