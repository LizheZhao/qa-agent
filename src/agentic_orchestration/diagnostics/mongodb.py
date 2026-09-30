"""MongoDB diagnostic projections and cursor-paged reads."""

from __future__ import annotations

from typing import Any, Literal

from orchestration_core import ExecutionTrace, GraphDefinition
from pydantic import ValidationError
from pymongo import DESCENDING
from pymongo.asynchronous.database import AsyncDatabase
from pymongo.errors import PyMongoError

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
from agentic_orchestration.diagnostics.store import DiagnosticReadError
from agentic_orchestration.observability.mongodb import (
    EXECUTION_TRACES_COLLECTION,
    ExecutionTraceStorageError,
    promote_observability_document,
)
from agentic_orchestration.observability.store import ExecutionTraceStore
from agentic_orchestration.sessions.schema import (
    SESSION_TURNS_COLLECTION,
    SESSIONS_COLLECTION,
    SessionDocument,
    TurnDocument,
    validate_turn_document,
)


class MongoDiagnosticReadStore:
    def __init__(
        self, database: AsyncDatabase[dict[str, Any]], execution_store: ExecutionTraceStore
    ) -> None:
        self._sessions = database[SESSIONS_COLLECTION]
        self._turns = database[SESSION_TURNS_COLLECTION]
        self._traces = database[EXECUTION_TRACES_COLLECTION]
        self._execution_store = execution_store

    async def sessions(self, *, limit: int, cursor: str | None) -> Page[SessionSummary]:
        boundary = decode_cursor(cursor, kind="sessions", size=2)
        query: dict[str, Any] = {}
        if boundary is not None:
            updated_at = cursor_datetime(boundary[0])
            session_id = cursor_string(boundary[1])
            query = {
                "$or": [
                    {"updated_at": {"$lt": updated_at}},
                    {"updated_at": updated_at, "_id": {"$lt": session_id}},
                ]
            }
        try:
            mongo_cursor = (
                self._sessions.find(query)
                .sort([("updated_at", DESCENDING), ("_id", DESCENDING)])
                .limit(limit + 1)
            )
            documents = [document async for document in mongo_cursor]
            page_documents = documents[:limit]
            first_messages = await self._first_user_messages(
                [str(document["_id"]) for document in page_documents]
            )
            items = [
                _session_summary(
                    SessionDocument.model_validate(item),
                    first_messages.get(str(item["_id"]), "Untitled conversation"),
                )
                for item in page_documents
            ]
            return Page(
                items=items,
                next_cursor=(
                    encode_cursor(
                        "sessions", [items[-1].updated_at.isoformat(), items[-1].session_id]
                    )
                    if len(documents) > limit and items
                    else None
                ),
            )
        except (PyMongoError, ValidationError) as exc:
            raise DiagnosticReadError("Could not read diagnostic sessions") from exc

    async def session(self, session_id: str) -> SessionSummary | None:
        try:
            document = await self._sessions.find_one({"_id": session_id})
            first_messages = await self._first_user_messages([session_id])
            return (
                _session_summary(
                    SessionDocument.model_validate(document),
                    first_messages.get(session_id, "Untitled conversation"),
                )
                if document is not None
                else None
            )
        except (PyMongoError, ValidationError) as exc:
            raise DiagnosticReadError("Could not read diagnostic session") from exc

    async def turns(self, session_id: str, *, limit: int, cursor: str | None) -> Page[TurnSummary]:
        boundary = decode_cursor(cursor, kind="turns", size=1)
        query: dict[str, Any] = {"session_id": session_id}
        if boundary is not None:
            turn_number = cursor_integer(boundary[0])
            query["turn_number"] = {"$lt": turn_number}
        try:
            mongo_cursor = await self._turns.aggregate(_turn_summary_pipeline(query, limit))
            documents = [document async for document in mongo_cursor]
            items = [_turn_summary(validate_turn_document(item)) for item in documents[:limit]]
            return Page(
                items=items,
                next_cursor=(
                    encode_cursor("turns", [items[-1].turn_number])
                    if len(documents) > limit and items
                    else None
                ),
            )
        except (PyMongoError, ValueError) as exc:
            raise DiagnosticReadError("Could not read diagnostic turns") from exc

    async def session_attempts(
        self, session_id: str, *, limit: int, cursor: str | None
    ) -> Page[AttemptSummary]:
        boundary = decode_cursor(cursor, kind="session_attempts", size=3)
        identity_query: dict[str, Any] = {"session_id": session_id}
        query = _attempt_query(identity_query, boundary)
        return await self._attempt_page(query, limit=limit, kind="session_attempts")

    async def recent_attempts(self, *, limit: int, cursor: str | None) -> Page[AttemptSummary]:
        boundary = decode_cursor(cursor, kind="recent_attempts", size=2)
        query: dict[str, Any] = {}
        if boundary is not None:
            started_at = cursor_datetime(boundary[0])
            trace_id = cursor_string(boundary[1])
            query = {
                "$or": [
                    {"started_at": {"$lt": started_at}},
                    {"started_at": started_at, "trace_id": {"$lt": trace_id}},
                ]
            }
        return await self._attempt_page(query, limit=limit, kind="recent_attempts")

    async def trace(self, trace_id: str) -> TraceDetail | None:
        try:
            trace = await self._execution_store.get_trace(trace_id)
            return trace_detail(trace) if trace is not None else None
        except ExecutionTraceStorageError as exc:
            raise DiagnosticReadError("Could not read diagnostic trace") from exc

    async def spans(self, trace_id: str) -> list[SpanSummary]:
        try:
            spans = await self._execution_store.spans(trace_id)
            return [span_summary(span) for span in spans]
        except ExecutionTraceStorageError as exc:
            raise DiagnosticReadError("Could not read diagnostic spans") from exc

    async def span(self, trace_id: str, span_id: str) -> SpanDetail | None:
        try:
            span = await self._execution_store.span(trace_id, span_id)
            return span_detail(span) if span is not None else None
        except ExecutionTraceStorageError as exc:
            raise DiagnosticReadError("Could not read diagnostic span") from exc

    async def graph_definition(self, definition_id: str) -> GraphDefinition | None:
        try:
            return await self._execution_store.graph_definition(definition_id)
        except ExecutionTraceStorageError as exc:
            raise DiagnosticReadError("Could not read graph definition") from exc

    async def _first_user_messages(self, session_ids: list[str]) -> dict[str, str]:
        if not session_ids:
            return {}
        cursor = self._turns.find(
            {"session_id": {"$in": session_ids}, "turn_number": 1},
            {"session_id": 1, "input_message.content": 1},
        )
        messages: dict[str, str] = {}
        async for document in cursor:
            session_id = document.get("session_id")
            input_message = document.get("input_message")
            if not isinstance(session_id, str) or not isinstance(input_message, dict):
                continue
            content = input_message.get("content")
            if isinstance(content, (str, list)):
                messages[session_id] = conversation_title(text_content(content))
        return messages

    async def _attempt_page(
        self,
        query: dict[str, Any],
        *,
        limit: int,
        kind: Literal["session_attempts", "recent_attempts"],
    ) -> Page[AttemptSummary]:
        sort = (
            [
                ("attempted_turn_number", DESCENDING),
                ("started_at", DESCENDING),
                ("trace_id", DESCENDING),
            ]
            if kind == "session_attempts"
            else [("started_at", DESCENDING), ("trace_id", DESCENDING)]
        )
        try:
            mongo_cursor = self._traces.find(query).sort(sort).limit(limit + 1)
            documents = [document async for document in mongo_cursor]
            traces = [
                ExecutionTrace.model_validate(promote_observability_document(document))
                for document in documents[:limit]
            ]
            input_previews = await self._attempt_input_previews(traces)
            items = [
                attempt_summary(trace).model_copy(
                    update={"input_preview": input_previews.get(trace.trace_id)}
                )
                for trace in traces
            ]
            if len(documents) <= limit or not items:
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
        except (PyMongoError, ValidationError) as exc:
            raise DiagnosticReadError("Could not read diagnostic attempts") from exc

    async def _attempt_input_previews(self, traces: list[ExecutionTrace]) -> dict[str, str]:
        committed = [trace for trace in traces if trace.committed_turn_number is not None]
        if not committed:
            return {}
        cursor = self._turns.find(
            {
                "$or": [
                    {
                        "session_id": trace.session_id,
                        "turn_number": trace.committed_turn_number,
                    }
                    for trace in committed
                ]
            },
            {"trace_id": 1, "input_message.content": 1},
        )
        previews: dict[str, str] = {}
        async for document in cursor:
            trace_id = document.get("trace_id")
            input_message = document.get("input_message")
            if not isinstance(trace_id, str) or not isinstance(input_message, dict):
                continue
            content = input_message.get("content")
            if isinstance(content, (str, list)):
                previews[trace_id] = conversation_title(text_content(content))
        return previews


def _attempt_query(identity: dict[str, Any], boundary: list[str | int] | None) -> dict[str, Any]:
    if boundary is None:
        return identity
    attempted_turn = cursor_integer(boundary[0])
    started_at = cursor_datetime(boundary[1])
    trace_id = cursor_string(boundary[2])
    boundary_query = {
        "$or": [
            {"attempted_turn_number": {"$lt": attempted_turn}},
            {"attempted_turn_number": attempted_turn, "started_at": {"$lt": started_at}},
            {
                "attempted_turn_number": attempted_turn,
                "started_at": started_at,
                "trace_id": {"$lt": trace_id},
            },
        ]
    }
    return {"$and": [identity, boundary_query]}


def _turn_summary_pipeline(query: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    # This remains a valid TurnDocument, but deliberately projects generated_messages down to the
    # final response used by TurnSummary. Future TurnDocument validators that require complete
    # message history must either move below this read projection or receive a
    # summary-specific model.
    return [
        {"$match": query},
        {"$sort": {"turn_number": DESCENDING}},
        {"$limit": limit + 1},
        {
            "$project": {
                "_id": 1,
                "schema_version": 1,
                "session_id": 1,
                "turn_number": 1,
                "input_message": 1,
                "generated_messages": {
                    "$filter": {
                        "input": "$generated_messages",
                        "as": "message",
                        "cond": {"$eq": ["$$message.id", "$final_message_id"]},
                    }
                },
                "final_message_id": 1,
                "routing_outcome": 1,
                "selected_agent_id": 1,
                "trace_id": 1,
                "committed_at": 1,
            }
        },
    ]


def _session_summary(document: SessionDocument, display_title: str) -> SessionSummary:
    return SessionSummary(
        session_id=document.id,
        agent_id=document.agent_id,
        display_title=display_title,
        revision=document.revision,
        status=document.status,
        created_at=document.created_at,
        updated_at=document.updated_at,
    )


def _turn_summary(document: TurnDocument) -> TurnSummary:
    final = next(
        message
        for message in document.generated_messages
        if message.id == document.final_message_id
    )
    return TurnSummary(
        turn_number=document.turn_number,
        committed_at=document.committed_at,
        input=ConversationMessage(
            role="user", content=text_content(document.input_message.content)
        ),
        response=ConversationMessage(role="assistant", content=text_content(final.content)),
        routing_outcome=document.routing_outcome,
        selected_agent_id=document.selected_agent_id,
        trace_id=document.trace_id,
    )
