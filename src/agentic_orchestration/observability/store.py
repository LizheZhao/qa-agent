"""Snapshot storage seam and deterministic process-memory adapter."""

import asyncio
from collections.abc import Sequence
from typing import Protocol

from orchestration_core import ExecutionSpan, ExecutionTrace, GraphDefinition

type ExecutionSnapshot = ExecutionTrace | ExecutionSpan


class ExecutionTraceStore(Protocol):
    async def save_snapshots(self, snapshots: Sequence[ExecutionSnapshot]) -> None: ...

    async def save_graph_definition(self, definition: GraphDefinition) -> None: ...

    async def get_trace(self, trace_id: str) -> ExecutionTrace | None: ...

    async def spans(self, trace_id: str) -> Sequence[ExecutionSpan]: ...

    async def span(self, trace_id: str, span_id: str) -> ExecutionSpan | None: ...

    async def graph_definition(self, definition_id: str) -> GraphDefinition | None: ...


class MemoryExecutionTraceStore:
    """Concurrency-safe adapter used until the Mongo implementation is introduced."""

    def __init__(self) -> None:
        self._traces: dict[str, ExecutionTrace] = {}
        self._spans: dict[tuple[str, str], ExecutionSpan] = {}
        self._definitions: dict[str, GraphDefinition] = {}
        self._lock = asyncio.Lock()

    async def save_snapshots(self, snapshots: Sequence[ExecutionSnapshot]) -> None:
        async with self._lock:
            for snapshot in snapshots:
                if isinstance(snapshot, ExecutionTrace):
                    current_trace = self._traces.get(snapshot.trace_id)
                    if (
                        current_trace is None
                        or current_trace.snapshot_sequence <= snapshot.snapshot_sequence
                    ):
                        self._traces[snapshot.trace_id] = snapshot
                else:
                    key = (snapshot.trace_id, snapshot.span_id)
                    current_span = self._spans.get(key)
                    if (
                        current_span is None
                        or current_span.snapshot_sequence <= snapshot.snapshot_sequence
                    ):
                        self._spans[key] = snapshot

    async def save_graph_definition(self, definition: GraphDefinition) -> None:
        async with self._lock:
            self._definitions[definition.definition_id] = definition

    async def get_trace(self, trace_id: str) -> ExecutionTrace | None:
        async with self._lock:
            return self._traces.get(trace_id)

    async def traces(self) -> Sequence[ExecutionTrace]:
        async with self._lock:
            return tuple(self._traces.values())

    async def spans(self, trace_id: str) -> Sequence[ExecutionSpan]:
        async with self._lock:
            return tuple(
                sorted(
                    (span for (key, _), span in self._spans.items() if key == trace_id),
                    key=lambda span: span.sequence_started,
                )
            )

    async def span(self, trace_id: str, span_id: str) -> ExecutionSpan | None:
        async with self._lock:
            return self._spans.get((trace_id, span_id))

    async def graph_definition(self, definition_id: str) -> GraphDefinition | None:
        async with self._lock:
            return self._definitions.get(definition_id)
