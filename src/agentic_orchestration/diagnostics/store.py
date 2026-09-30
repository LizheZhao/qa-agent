"""Read-only diagnostic query seam."""

from typing import Protocol

from orchestration_core import GraphDefinition

from agentic_orchestration.diagnostics.contracts import (
    AttemptSummary,
    Page,
    SessionSummary,
    SpanDetail,
    SpanSummary,
    TraceDetail,
    TurnSummary,
)


class DiagnosticReadError(RuntimeError):
    pass


class DiagnosticReadStore(Protocol):
    async def sessions(self, *, limit: int, cursor: str | None) -> Page[SessionSummary]: ...

    async def session(self, session_id: str) -> SessionSummary | None: ...

    async def turns(
        self, session_id: str, *, limit: int, cursor: str | None
    ) -> Page[TurnSummary]: ...

    async def session_attempts(
        self, session_id: str, *, limit: int, cursor: str | None
    ) -> Page[AttemptSummary]: ...

    async def recent_attempts(self, *, limit: int, cursor: str | None) -> Page[AttemptSummary]: ...

    async def trace(self, trace_id: str) -> TraceDetail | None: ...

    async def spans(self, trace_id: str) -> list[SpanSummary]: ...

    async def span(self, trace_id: str, span_id: str) -> SpanDetail | None: ...

    async def graph_definition(self, definition_id: str) -> GraphDefinition | None: ...
