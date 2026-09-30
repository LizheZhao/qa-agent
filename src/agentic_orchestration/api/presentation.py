"""Renderer-neutral projection of a pause for HTTP clients.

What a client is shown is deliberately narrower than what the pending record holds. The
run ID, the child that produced the question, the graph it paused under, and the producer
freshness token are all omitted: a continuation resolves them from durable state, so
publishing them would only invite a client to send them back.
"""

from __future__ import annotations

from datetime import datetime

from langchain_core.messages import BaseMessage
from pydantic import BaseModel

from agentic_orchestration.sessions.contracts import PendingRun


class ClarificationOptionView(BaseModel):
    option_id: str
    label: str
    detail: str | None = None


class PendingClarification(BaseModel):
    """One question a session is waiting on, and how it may be answered.

    Free text and cancellation are always available, which is why they are stated here as
    fixed fields rather than left for a client to infer from the selection mode.
    """

    session_id: str
    clarification_id: str
    question: str
    reason_code: str
    selection_mode: str
    options: list[ClarificationOptionView]
    free_text_allowed: bool = True
    cancel_allowed: bool = True
    expires_at: datetime
    trace_id: str


class ClarificationError(BaseModel):
    code: str
    message: str


class ClarificationErrorResponse(BaseModel):
    error: ClarificationError
    session_id: str


def clarification_view(run: PendingRun, trace_id: str) -> PendingClarification:
    return PendingClarification(
        session_id=run.session_id,
        clarification_id=run.clarification_id,
        question=run.clarification.question,
        reason_code=run.clarification.reason_code,
        selection_mode=run.clarification.selection_mode,
        options=[
            ClarificationOptionView(
                option_id=option.option_id, label=option.label, detail=option.detail
            )
            for option in run.clarification.options
        ],
        expires_at=run.expires_at,
        trace_id=trace_id,
    )


def content_text(message: BaseMessage) -> str:
    if isinstance(message.content, str):
        return message.content
    return "\n".join(
        block if isinstance(block, str) else str(block.get("text", ""))
        for block in message.content
        if isinstance(block, (str, dict))
    )
