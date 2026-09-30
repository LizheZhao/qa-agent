"""External projection of durable session history."""

from datetime import UTC, datetime
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import BaseModel

from agentic_orchestration.api.dependencies import get_session_store
from agentic_orchestration.api.presentation import (
    PendingClarification,
    clarification_view,
    content_text,
)
from agentic_orchestration.sessions.contracts import (
    ClarificationAuditEntry,
    SessionStorageError,
    SessionStore,
    SessionTurn,
)
from agentic_orchestration.sessions.expiry import release_if_expired


class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class SessionHistoryResponse(BaseModel):
    """Committed conversation, plus the question the session is waiting on if there is one.

    `messages` holds committed turns only. An open clarification is not a committed turn
    and is projected separately, so a reader cannot mistake a question still being asked
    for part of the immutable history.
    """

    session_id: str
    storage: str
    durable: bool
    messages: list[HistoryMessage]
    pending_clarification: PendingClarification | None = None


router = APIRouter(prefix="/api/sessions", tags=["sessions"])


def _history_message(message: BaseMessage) -> HistoryMessage | None:
    if isinstance(message, HumanMessage):
        return HistoryMessage(role="user", content=content_text(message))
    if isinstance(message, AIMessage) and not message.tool_calls:
        return HistoryMessage(role="assistant", content=content_text(message))
    return None


def _exchange_messages(
    audit: tuple[ClarificationAuditEntry, ...], run_id: str
) -> list[HistoryMessage]:
    """Render one run's ordered clarification exchange from its audit entries.

    The audit is the record of what was actually asked and answered, including the
    questions a cancelled run displayed before it closed, which the committed turn's final
    message alone does not show.
    """

    rendered: list[HistoryMessage] = []
    for entry in sorted(
        (entry for entry in audit if entry.run_id == run_id), key=lambda entry: entry.sequence
    ):
        if entry.event == "published" and entry.question is not None:
            rendered.append(HistoryMessage(role="assistant", content=entry.question))
        elif entry.event == "answered" and entry.response_text is not None:
            rendered.append(HistoryMessage(role="user", content=entry.response_text))
    return rendered


def _turn_messages(
    turn: SessionTurn, audit: tuple[ClarificationAuditEntry, ...]
) -> list[HistoryMessage]:
    rendered = []
    if (opening := _history_message(turn.input_message)) is not None:
        rendered.append(opening)
    if turn.continuation is not None:
        rendered.extend(_exchange_messages(audit, turn.continuation.run_id))
        if turn.continuation.closed_by != "answered":
            # A cancelled or expired turn ends at the question it last displayed, which the
            # exchange above already rendered; repeating the final message would show it twice.
            return rendered
    final = next(
        (message for message in turn.generated_messages if message.id == turn.final_message_id),
        None,
    )
    if final is not None and (closing := _history_message(final)) is not None:
        rendered.append(closing)
    return rendered


@router.get("/{session_id}/history", response_model=SessionHistoryResponse)
async def history(
    session_id: str,
    session_store: Annotated[SessionStore, Depends(get_session_store)],
) -> SessionHistoryResponse:
    async with session_store.lock(session_id):
        try:
            snapshot = await session_store.load(session_id)
            pending = await release_if_expired(
                session_store,
                await session_store.active_pending_run(session_id),
                at=datetime.now(UTC),
            )
            if pending is None:
                snapshot = await session_store.load(session_id)
            audit = await session_store.clarification_audit(session_id)
        except SessionStorageError as exc:
            raise HTTPException(status_code=503, detail="Session storage is unavailable") from exc
        # A session whose first request paused has no committed turn yet. It is not
        # missing, and inventing a session document for it would break the guarantee that
        # a durable session always carries at least one committed turn.
        if snapshot is None and pending is None:
            raise HTTPException(status_code=404, detail="Session not found")

    conversation: list[HistoryMessage] = []
    for turn in snapshot.turns if snapshot is not None else ():
        conversation.extend(_turn_messages(turn, audit))
    return SessionHistoryResponse(
        session_id=session_id,
        storage=session_store.storage_name,
        durable=session_store.durable,
        messages=conversation,
        pending_clarification=(
            clarification_view(pending, pending.trace_id) if pending is not None else None
        ),
    )
