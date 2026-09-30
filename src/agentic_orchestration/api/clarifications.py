"""Typed continuation of one paused run, kept separate from ordinary chat.

Ordinary chat and continuation are different operations, and the split is an API
invariant rather than a convention held inside the handler: this route reaches the already
selected child directly and has no way to enter the router at all.

Nothing that decides which run is continued comes from the client. The body carries a
submission identifier and exactly one response; the run, the child to resume, the graph it
paused under, and the producer freshness token are all read from the durable pending-run
record. A client able to name those could aim an answer at a different run or a graph it
was never shown.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Annotated, Any, Literal
from uuid import uuid4

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from orchestration_core import (
    AgentInvocationError,
    ClarificationContractError,
    ClarificationResponse,
    ContinuationNotResumableError,
    ContinuationRequest,
    ConversationResult,
    ExecutionContext,
    InvocationContext,
    PausedConversation,
    RequestOrigin,
)
from pydantic import BaseModel, ConfigDict, Field

from agentic_orchestration.api.dependencies import (
    get_agent_invoker,
    get_entry_agent,
    get_executor,
    get_session_store,
)
from agentic_orchestration.api.presentation import (
    ClarificationError,
    ClarificationErrorResponse,
    PendingClarification,
    clarification_view,
    content_text,
)
from agentic_orchestration.execution.agent_invoker import RegistryAgentInvoker
from agentic_orchestration.execution.executor import Executor
from agentic_orchestration.sessions.contracts import (
    ClaimResponse,
    ClarificationAuditEntry,
    ClosePendingRun,
    CommitContinuedTurn,
    PendingRun,
    PendingRunConflictError,
    PendingRunExpiredError,
    PendingRunNotFoundError,
    SessionConflictError,
    SessionSnapshot,
    SessionStorageError,
    SessionStore,
    SubmissionConflictError,
)
from agentic_orchestration.sessions.pending import appended_attempt_trace, response_text

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/sessions", tags=["clarifications"])


class ContinuationBody(BaseModel):
    """A submission identifier and exactly one response form."""

    model_config = ConfigDict(extra="forbid")

    submission_id: str = Field(min_length=1, max_length=128)
    response: ClarificationResponse


class ContinuationCompleted(BaseModel):
    """The committed turn a continuation produced."""

    session_id: str
    trace_id: str
    turn_number: int
    closed_by: Literal["answered", "canceled"]
    message: str


def _error(status_code: int, code: str, message: str, session_id: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=ClarificationErrorResponse(
            error=ClarificationError(code=code, message=message), session_id=session_id
        ).model_dump(mode="json"),
    )


@router.post(
    "/{session_id}/clarifications/{clarification_id}/responses",
    response_model=ContinuationCompleted,
    responses={
        202: {"model": PendingClarification},
        404: {"model": ClarificationErrorResponse},
        409: {"model": ClarificationErrorResponse},
        410: {"model": ClarificationErrorResponse},
        422: {"model": ClarificationErrorResponse},
        500: {"model": ClarificationErrorResponse},
        503: {"model": ClarificationErrorResponse},
    },
)
async def respond(
    session_id: str,
    clarification_id: str,
    body: ContinuationBody,
    request: Request,
    session_store: Annotated[SessionStore, Depends(get_session_store)],
    invoker: Annotated[RegistryAgentInvoker, Depends(get_agent_invoker)],
    executor: Annotated[Executor, Depends(get_executor)],
    entry_agent: Annotated[str, Depends(get_entry_agent)],
) -> Any:
    attempt_trace_id = str(uuid4())
    async with session_store.lock(session_id):
        try:
            claim = await session_store.claim_response(
                ClaimResponse(
                    session_id=session_id,
                    clarification_id=clarification_id,
                    submission_id=body.submission_id,
                    response=body.response,
                    attempt_trace_id=attempt_trace_id,
                ),
                at=datetime.now(UTC),
            )
        except PendingRunExpiredError:
            return await _expire(session_store, session_id, clarification_id, attempt_trace_id)
        except PendingRunNotFoundError:
            return _error(
                404, "pending_run_not_found", "No open clarification to answer", session_id
            )
        except SubmissionConflictError:
            return _error(
                409,
                "submission_conflict",
                "Submission already used for a different answer",
                session_id,
            )
        except PendingRunConflictError:
            return _error(409, "pending_run_conflict", "The clarification changed", session_id)
        except SessionStorageError:
            logger.exception("clarification claim failed")
            return _error(
                503, "session_storage_unavailable", "Session storage is unavailable", session_id
            )
        except ClarificationContractError as exc:
            return _error(
                422, exc.reason_code, "The response does not fit the clarification", session_id
            )

        run = claim.pending_run
        try:
            run.clarification.accept(claim.response)
        except ClarificationContractError as exc:
            return _error(
                422, exc.reason_code, "The response does not fit the clarification", session_id
            )

        if run.status in {"consumed", "canceled"}:
            # A replay that arrived after its own outcome was already committed.
            return await _completed(session_store, run, attempt_trace_id)

        if claim.response.form == "cancel":
            return await _cancel(session_store, run, body.submission_id, attempt_trace_id)

        return await _continue(
            session_store,
            invoker,
            executor,
            run,
            claim.replayed,
            entry_agent,
            attempt_trace_id,
            request,
        )


class _TraceAttempt:
    """Owns the trace run for one continuation attempt, and finishes it exactly once.

    The invoker looks its run up by trace id before it touches the graph, and raises if nothing
    was begun under it. This route minted a trace id for the audit trail but never began a run,
    so every resume died on that lookup with a bare 500.

    Finishing is idempotent and never raises. An attempt that ends without a committed turn is
    cancelled rather than failed: nothing went wrong, the turn simply is not finished.
    """

    def __init__(self, executor: Executor, context: ExecutionContext) -> None:
        self._executor = executor
        self._context = context
        self._finished = False

    async def begin(self, entry_agent: str) -> None:
        await self._executor.begin(entry_agent, self._context)

    async def complete(self, committed_turn_number: int) -> None:
        await self._finish("complete", committed_turn_number=committed_turn_number)

    async def fail(self) -> None:
        await self._finish("fail")

    async def cancel(self) -> None:
        await self._finish("cancel")

    async def _finish(self, outcome: str, **kwargs: Any) -> None:
        if self._finished:
            return
        self._finished = True
        try:
            await getattr(self._executor, outcome)(self._context, **kwargs)
        except Exception as cleanup_error:
            # Observability must not decide whether the answer was accepted.
            logger.warning(
                "continuation trace finalization failed trace_id=%s outcome=%s type=%s",
                self._context.trace_id,
                outcome,
                type(cleanup_error).__name__,
            )


async def _continue(
    session_store: SessionStore,
    invoker: RegistryAgentInvoker,
    executor: Executor,
    run: PendingRun,
    replayed: bool,
    entry_agent: str,
    attempt_trace_id: str,
    request: Request,
) -> Any:
    context = InvocationContext(
        execution=ExecutionContext(
            trace_id=attempt_trace_id,
            root_span_id=str(uuid4()),
            session_id=run.session_id,
            attempted_turn_number=(run.expected_session_revision or 0) + 1,
            request_origin=RequestOrigin.API,
        ),
        caller_agent_id=entry_agent,
    )
    attempt = _TraceAttempt(executor, context.execution)
    await attempt.begin(entry_agent)
    try:
        return await _resume_and_settle(
            session_store, invoker, attempt, run, replayed, attempt_trace_id, context
        )
    finally:
        # Every path above finishes the attempt itself. This closes the one that raised, so a
        # failure cannot leave a run registered for the next request to trip over.
        await attempt.cancel()


async def _resume_and_settle(
    session_store: SessionStore,
    invoker: RegistryAgentInvoker,
    attempt: _TraceAttempt,
    run: PendingRun,
    replayed: bool,
    attempt_trace_id: str,
    context: InvocationContext,
) -> Any:
    outcome: ConversationResult | PausedConversation | None = None
    if replayed:
        # The graph may already have finished under this submission and died before its
        # turn was committed. Recovering the completed output re-runs nothing.
        outcome = await invoker.recover(run.selected_agent_id, run.run_id)
    if outcome is None:
        assert run.claimed_response is not None
        try:
            outcome = await invoker.resume(
                run.selected_agent_id,
                ContinuationRequest(
                    run_id=run.run_id,
                    clarification_id=run.clarification_id,
                    response=run.claimed_response,
                    freshness_token=run.clarification.freshness_token,
                    graph_definition_id=run.graph_definition_id,
                    continuation_contract_version=run.continuation_contract_version,
                ),
                context,
            )
        except ContinuationNotResumableError as exc:
            logger.warning("continuation refused reason=%s", exc.reason)
            await attempt.fail()
            return _error(
                409,
                f"restart_required:{exc.reason}",
                "This request must be restarted",
                run.session_id,
            )
        except ClarificationContractError as exc:
            await attempt.fail()
            return _error(
                422, exc.reason_code, "The response does not fit the clarification", run.session_id
            )
        except AgentInvocationError:
            # The claim stays, so the same submission may be retried; no turn is committed
            # and the session revision does not move.
            logger.exception("continuation execution failed")
            await attempt.fail()
            return _error(500, "agent_execution_error", "Agent execution failed", run.session_id)

    if isinstance(outcome, PausedConversation):
        try:
            republished = await session_store.republish_pending_run(
                run.session_id,
                run_id=run.run_id,
                submission_id=str(run.submission_id),
                clarification=outcome.clarification,
                attempt_trace_id=attempt_trace_id,
            )
        except SessionStorageError:
            logger.exception("republishing the next clarification failed")
            await attempt.fail()
            return _error(
                503, "session_storage_unavailable", "Session storage is unavailable", run.session_id
            )
        # Paused again, so no turn is committed: cancelled, the same way chat closes a pause.
        await attempt.cancel()
        return JSONResponse(
            status_code=202,
            content=clarification_view(republished, attempt_trace_id).model_dump(mode="json"),
        )

    return await _commit(session_store, run, outcome, attempt_trace_id, attempt)


async def _commit(
    session_store: SessionStore,
    run: PendingRun,
    outcome: ConversationResult,
    attempt_trace_id: str,
    attempt: _TraceAttempt,
) -> Any:
    final = next(
        (
            message
            for message in reversed(outcome.generated_messages)
            if isinstance(message, AIMessage) and not message.tool_calls
        ),
        None,
    )
    if final is None:
        await attempt.fail()
        return _error(
            500, "invalid_agent_result", "Agent returned an invalid result", run.session_id
        )
    if final.id is None:
        final.id = str(uuid4())
    try:
        audit = await session_store.clarification_audit(run.session_id)
        exchange = _exchange(audit, run.run_id)
        snapshot = await session_store.commit_continued_turn(
            CommitContinuedTurn(
                session_id=run.session_id,
                run_id=run.run_id,
                clarification_id=run.clarification_id,
                submission_id=str(run.submission_id),
                expected_revision=run.expected_session_revision,
                tenant_id=run.tenant_id,
                user_id=run.user_id,
                agent_id=run.agent_id,
                selected_agent_id=run.selected_agent_id,
                input_message=run.input_message,
                generated_messages=(*exchange, final),
                final_message_id=str(final.id),
                trace_id=run.trace_id,
                # The claim already recorded this attempt, and a run answered twice is claimed
                # twice. Appending unconditionally repeats an id the record requires to be
                # distinct, which only the Mongo store enforces.
                attempt_trace_ids=appended_attempt_trace(run.attempt_trace_ids, attempt_trace_id),
                clarification_ids=_clarification_ids(audit, run.run_id),
            )
        )
    # PendingRunConflictError subclasses SessionConflictError, so the narrower one is caught
    # first. The other order reports every pending-run conflict as a session conflict.
    except PendingRunConflictError:
        await attempt.fail()
        return _error(409, "pending_run_conflict", "The clarification changed", run.session_id)
    except SessionConflictError:
        await attempt.fail()
        return _error(409, "session_conflict", "Session changed; retry the request", run.session_id)
    except SessionStorageError:
        logger.exception("continued turn commit failed")
        await attempt.fail()
        return _error(
            503, "session_storage_unavailable", "Session storage is unavailable", run.session_id
        )
    await attempt.complete(snapshot.revision)
    return _completed_response(run, snapshot, "answered", content_text(final))


async def _cancel(
    session_store: SessionStore,
    run: PendingRun,
    submission_id: str,
    attempt_trace_id: str,
) -> Any:
    """Close the run at its last question. The graph is never resumed."""

    question = AIMessage(content=run.clarification.question, id=str(uuid4()))
    try:
        snapshot = await session_store.close_pending_run(
            ClosePendingRun(
                session_id=run.session_id,
                run_id=run.run_id,
                clarification_id=run.clarification_id,
                reason="canceled",
                submission_id=submission_id,
                final_message=question,
                trace_id=run.trace_id,
                attempt_trace_id=attempt_trace_id,
            )
        )
    except SessionConflictError:
        return _error(409, "session_conflict", "Session changed; retry the request", run.session_id)
    except SessionStorageError:
        logger.exception("cancelling the pending run failed")
        return _error(
            503, "session_storage_unavailable", "Session storage is unavailable", run.session_id
        )
    return _completed_response(run, snapshot, "canceled", content_text(question))


async def _expire(
    session_store: SessionStore,
    session_id: str,
    clarification_id: str,
    attempt_trace_id: str,
) -> JSONResponse:
    """Close an expired pause through the same durable transition as a cancellation."""

    run = await session_store.active_pending_run(session_id)
    if run is not None and run.clarification_id == clarification_id:
        try:
            await session_store.close_pending_run(
                ClosePendingRun(
                    session_id=session_id,
                    run_id=run.run_id,
                    clarification_id=clarification_id,
                    reason="expired",
                    submission_id=None,
                    final_message=AIMessage(content=run.clarification.question, id=str(uuid4())),
                    trace_id=run.trace_id,
                    attempt_trace_id=attempt_trace_id,
                )
            )
        except SessionStorageError:
            logger.exception("closing the expired pending run failed")
    return _error(410, "clarification_expired", "This clarification has expired", session_id)


async def _completed(session_store: SessionStore, run: PendingRun, attempt_trace_id: str) -> Any:
    """Return the outcome a replayed submission already produced."""

    snapshot = await session_store.load(run.session_id)
    if snapshot is None or not snapshot.turns:
        return _error(
            404, "pending_run_not_found", "No open clarification to answer", run.session_id
        )
    turn = snapshot.turns[-1]
    final = next(
        (message for message in turn.generated_messages if message.id == turn.final_message_id),
        None,
    )
    closed_by: Literal["answered", "canceled"] = (
        "canceled" if run.status == "canceled" else "answered"
    )
    return _completed_response(
        run, snapshot, closed_by, content_text(final) if final is not None else ""
    )


def _completed_response(
    run: PendingRun,
    snapshot: SessionSnapshot,
    closed_by: Literal["answered", "canceled"],
    message: str,
) -> ContinuationCompleted:
    return ContinuationCompleted(
        session_id=run.session_id,
        trace_id=run.trace_id,
        turn_number=snapshot.revision,
        closed_by=closed_by,
        message=message,
    )


def _exchange(audit: tuple[ClarificationAuditEntry, ...], run_id: str) -> tuple[BaseMessage, ...]:
    """Rebuild the ordered clarification exchange the committed turn keeps.

    Read from the audit rather than tracked across requests: each continuation is a
    separate request, and a run may have asked several questions before this one.
    """

    messages: list[BaseMessage] = []
    for entry in sorted(
        (entry for entry in audit if entry.run_id == run_id), key=lambda entry: entry.sequence
    ):
        if entry.event == "published" and entry.question is not None:
            messages.append(AIMessage(content=entry.question, id=str(uuid4())))
        elif entry.event == "answered" and entry.response_text is not None:
            messages.append(HumanMessage(content=entry.response_text, id=str(uuid4())))
    return tuple(messages)


def _clarification_ids(audit: tuple[ClarificationAuditEntry, ...], run_id: str) -> tuple[str, ...]:
    seen = {
        entry.clarification_id: None
        for entry in sorted(
            (entry for entry in audit if entry.run_id == run_id), key=lambda entry: entry.sequence
        )
    }
    return tuple(seen)


__all__ = ["ContinuationBody", "ContinuationCompleted", "response_text", "router"]
