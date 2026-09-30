"""Deterministic process-memory diagnostic adapter."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from orchestration_core import ExecutionTrace, GraphDefinition

from agentic_orchestration.diagnostics.contracts import (
    AttemptSummary,
    ConversationMessage,
    Page,
    SessionSummary,
    SpanDetail,
    SpanSummary,
    TraceDetail,
    TurnSummary,
    conversation_title,
    text_content,
)
from agentic_orchestration.diagnostics.cursors import (
    cursor_datetime,
    cursor_integer,
    cursor_string,
    decode_cursor,
    encode_attempt_cursor,
    encode_cursor,
)
from agentic_orchestration.diagnostics.projections import (
    attempt_summary,
    span_detail,
    span_summary,
    trace_detail,
)
from agentic_orchestration.observability.store import MemoryExecutionTraceStore
from agentic_orchestration.sessions.contracts import SessionSnapshot
from agentic_orchestration.sessions.memory import MemorySessionStore


class MemoryDiagnosticReadStore:
    def __init__(
        self,
        session_store: MemorySessionStore,
        execution_store: MemoryExecutionTraceStore,
    ) -> None:
        self._session_store = session_store
        self._execution_store = execution_store

    async def sessions(self, *, limit: int, cursor: str | None) -> Page[SessionSummary]:
        boundary = decode_cursor(cursor, kind="sessions", size=2)
        snapshots = sorted(
            await self._session_store.snapshots(),
            key=lambda item: (item.updated_at or datetime.min.replace(tzinfo=UTC), item.session_id),
            reverse=True,
        )
        if boundary is not None:
            updated_at = cursor_datetime(boundary[0])
            session_id = cursor_string(boundary[1])
            snapshots = [
                item
                for item in snapshots
                if (item.updated_at or datetime.min.replace(tzinfo=UTC), item.session_id)
                < (updated_at, session_id)
            ]
        page = snapshots[: limit + 1]
        items = [_session_summary(item) for item in page[:limit]]
        return Page(
            items=items,
            next_cursor=(
                encode_cursor("sessions", [items[-1].updated_at.isoformat(), items[-1].session_id])
                if len(page) > limit and items
                else None
            ),
        )

    async def session(self, session_id: str) -> SessionSummary | None:
        snapshot = await self._session_store.load(session_id)
        if snapshot is None:
            return None
        return _session_summary(snapshot)

    async def turns(self, session_id: str, *, limit: int, cursor: str | None) -> Page[TurnSummary]:
        boundary = decode_cursor(cursor, kind="turns", size=1)
        snapshot = await self._session_store.load(session_id)
        turns = (
            sorted(snapshot.turns, key=lambda item: item.turn_number, reverse=True)
            if snapshot
            else []
        )
        if boundary is not None:
            turn_number = cursor_integer(boundary[0])
            turns = [turn for turn in turns if turn.turn_number < turn_number]
        page = turns[: limit + 1]
        items = [
            TurnSummary(
                turn_number=turn.turn_number,
                committed_at=turn.committed_at or datetime.min.replace(tzinfo=UTC),
                input=ConversationMessage(
                    role="user", content=text_content(turn.input_message.content)
                ),
                response=ConversationMessage(
                    role="assistant",
                    content=text_content(
                        next(
                            message
                            for message in turn.generated_messages
                            if message.id == turn.final_message_id
                        ).content
                    ),
                ),
                routing_outcome=turn.routing_outcome,
                selected_agent_id=turn.selected_agent_id,
                trace_id=turn.trace_id,
            )
            for turn in page[:limit]
        ]
        return Page(
            items=items,
            next_cursor=(
                encode_cursor("turns", [items[-1].turn_number])
                if len(page) > limit and items
                else None
            ),
        )

    async def session_attempts(
        self, session_id: str, *, limit: int, cursor: str | None
    ) -> Page[AttemptSummary]:
        boundary = decode_cursor(cursor, kind="session_attempts", size=3)
        traces = [
            trace
            for trace in await self._execution_store.traces()
            if trace.session_id == session_id
        ]
        traces.sort(
            key=lambda item: (item.attempted_turn_number, item.started_at, item.trace_id),
            reverse=True,
        )
        if boundary is not None:
            key = (
                cursor_integer(boundary[0]),
                cursor_datetime(boundary[1]),
                cursor_string(boundary[2]),
            )
            traces = [
                trace
                for trace in traces
                if (trace.attempted_turn_number, trace.started_at, trace.trace_id) < key
            ]
        return _attempt_page(
            traces,
            limit=limit,
            kind="session_attempts",
            input_previews=await self._attempt_input_previews(traces),
        )

    async def recent_attempts(self, *, limit: int, cursor: str | None) -> Page[AttemptSummary]:
        boundary = decode_cursor(cursor, kind="recent_attempts", size=2)
        traces = sorted(
            await self._execution_store.traces(),
            key=lambda item: (item.started_at, item.trace_id),
            reverse=True,
        )
        if boundary is not None:
            key = (cursor_datetime(boundary[0]), cursor_string(boundary[1]))
            traces = [trace for trace in traces if (trace.started_at, trace.trace_id) < key]
        return _attempt_page(
            traces,
            limit=limit,
            kind="recent_attempts",
            input_previews=await self._attempt_input_previews(traces),
        )

    async def trace(self, trace_id: str) -> TraceDetail | None:
        trace = await self._execution_store.get_trace(trace_id)
        return trace_detail(trace) if trace is not None else None

    async def spans(self, trace_id: str) -> list[SpanSummary]:
        return [span_summary(span) for span in await self._execution_store.spans(trace_id)]

    async def span(self, trace_id: str, span_id: str) -> SpanDetail | None:
        span = await self._execution_store.span(trace_id, span_id)
        return span_detail(span) if span is not None else None

    async def graph_definition(self, definition_id: str) -> GraphDefinition | None:
        return await self._execution_store.graph_definition(definition_id)

    async def _attempt_input_previews(self, traces: list[ExecutionTrace]) -> dict[str, str]:
        snapshots = {
            snapshot.session_id: snapshot for snapshot in await self._session_store.snapshots()
        }
        previews: dict[str, str] = {}
        for trace in traces:
            snapshot = snapshots.get(trace.session_id)
            if snapshot is None:
                continue
            turn = next(
                (item for item in snapshot.turns if item.trace_id == trace.trace_id),
                None,
            )
            if turn is not None:
                previews[trace.trace_id] = conversation_title(
                    text_content(turn.input_message.content)
                )
        return previews


def _attempt_page(
    traces: list[ExecutionTrace],
    *,
    limit: int,
    kind: Literal["session_attempts", "recent_attempts"],
    input_previews: dict[str, str],
) -> Page[AttemptSummary]:
    typed = traces[: limit + 1]
    items = [
        attempt_summary(trace).model_copy(
            update={"input_preview": input_previews.get(trace.trace_id)}
        )
        for trace in typed[:limit]
    ]
    if len(typed) <= limit or not items:
        next_cursor = None
    else:
        last = items[-1]
        next_cursor = encode_attempt_cursor(
            kind,
            attempted_turn_number=last.attempted_turn_number,
            started_at=last.started_at,
            trace_id=last.trace_id,
        )
    return Page(items=items, next_cursor=next_cursor)


def _session_summary(snapshot: SessionSnapshot) -> SessionSummary:
    return SessionSummary(
        session_id=snapshot.session_id,
        agent_id=snapshot.agent_id,
        display_title=(
            conversation_title(text_content(snapshot.turns[0].input_message.content))
            if snapshot.turns
            else "Untitled conversation"
        ),
        revision=snapshot.revision,
        status=snapshot.status,
        created_at=snapshot.created_at or datetime.min.replace(tzinfo=UTC),
        updated_at=snapshot.updated_at or datetime.min.replace(tzinfo=UTC),
    )
