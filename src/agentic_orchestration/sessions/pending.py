"""Transition rules for a session's single in-flight paused run.

These decide, for one durable record and one request, whether the request may proceed and
what the record becomes. They are pure so that the MongoDB adapter and the memory adapter
cannot drift on questions that decide correctness -- which retry is a replay, which is a
different choice under a used identifier, and who wins when a cancellation races a
response -- and so those questions can be tested without a database.

Both adapters apply them while holding the record: MongoDB inside its transaction under a
snapshot read, the memory adapter under its session lock. Reading and deciding outside
that would make each decision advisory rather than a compare-and-set.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime

from orchestration_core import ClarificationRequest, ClarificationResponse, is_clarification_expired

from agentic_orchestration.sessions.contracts import (
    ClaimedResponse,
    ClaimResponse,
    ClosePendingRun,
    CommitContinuedTurn,
    PendingRun,
    PendingRunConflictError,
    PendingRunExpiredError,
    PendingRunNotFoundError,
    SubmissionConflictError,
)

OPEN_STATUSES = frozenset({"awaiting_response", "claimed"})
CLOSED_STATUSES = frozenset({"consumed", "canceled", "expired"})


def decide_claim(
    run: PendingRun | None,
    command: ClaimResponse,
    *,
    at: datetime,
) -> ClaimedResponse:
    """Decide whether a submitted response may be claimed, or is a replay of one already.

    A replay returns the claim that already exists rather than re-applying it, so a client
    that retries after a dropped connection continues the same work instead of starting a
    second one.

    Raises `ClarificationContractError` for a response the clarification does not permit,
    before any of it is recorded.
    """

    if run is None or run.clarification_id != command.clarification_id:
        raise PendingRunNotFoundError("No open pause answers to this clarification")
    # Checked before anything is claimed or audited: a response naming an option the
    # producer never offered must leave no trace in the run's exchange record.
    run.clarification.accept(command.response)
    if run.submission_id == command.submission_id:
        if run.claimed_response != command.response:
            raise SubmissionConflictError("Submission identifier reused for different content")
        return ClaimedResponse(pending_run=run, response=command.response, replayed=True)
    if run.status in CLOSED_STATUSES:
        # Closed under some other submission: there is nothing left for this one to answer.
        raise PendingRunNotFoundError("The pause has already been closed")
    if run.status == "claimed":
        raise PendingRunConflictError("Another response is already being applied to this pause")
    if is_clarification_expired(run.expires_at, at=at):
        raise PendingRunExpiredError("The pause expired before the response was claimed")
    claimed = replace(
        run,
        status="claimed",
        submission_id=command.submission_id,
        claimed_response=command.response,
        attempt_trace_ids=appended_attempt_trace(run.attempt_trace_ids, command.attempt_trace_id),
    )
    return ClaimedResponse(pending_run=claimed, response=command.response, replayed=False)


def decide_republish(
    run: PendingRun | None,
    *,
    run_id: str,
    submission_id: str,
    clarification: ClarificationRequest,
    published_at: datetime,
) -> PendingRun:
    """Move a claimed run back to awaiting the next queued clarification.

    The run keeps its identity and its original request; only the question changes. The
    claim is cleared so the next answer is a new submission rather than a replay of the
    one that produced this question.
    """

    _require_claimed(run, run_id=run_id, submission_id=submission_id)
    assert run is not None
    return replace(
        run,
        status="awaiting_response",
        clarification=clarification,
        submission_id=None,
        claimed_response=None,
        created_at=published_at,
        expires_at=clarification.expires_at(published_at),
    )


def decide_commit(run: PendingRun | None, command: CommitContinuedTurn) -> PendingRun:
    """Consume a claimed run as its turn is committed."""

    _require_claimed(run, run_id=command.run_id, submission_id=command.submission_id)
    assert run is not None
    if run.clarification_id != command.clarification_id:
        raise PendingRunConflictError("The pause moved to a different clarification")
    return replace(run, status="consumed")


def decide_close(run: PendingRun | None, command: ClosePendingRun) -> PendingRun:
    """Close a run at the question it last displayed.

    Cancellation is itself a claimed response, so it closes a run this submission already
    holds. Expiry closes a run nobody has answered, which is what keeps it from cutting off
    a continuation that is already running.
    """

    if run is None or run.run_id != command.run_id:
        raise PendingRunNotFoundError("No pause answers to this run")
    if run.clarification_id != command.clarification_id:
        raise PendingRunConflictError("The pause moved to a different clarification")
    if run.status == command.reason and run.submission_id == command.submission_id:
        return run
    if command.reason == "canceled":
        _require_claimed(run, run_id=command.run_id, submission_id=command.submission_id)
        assert run is not None
        return replace(run, status="canceled")
    if run.status != "awaiting_response":
        raise PendingRunConflictError("A response was claimed before the pause expired")
    return replace(run, status="expired", submission_id=None, claimed_response=None)


def response_text(response: ClarificationResponse, clarification: ClarificationRequest) -> str:
    """Render an accepted response as the conversation message the committed turn keeps.

    Option IDs are rendered through their labels: the committed turn is read by people, and
    an identifier the producer chose says nothing about what was picked.
    """

    if response.form == "cancel":
        return "Cancelled this request."
    if response.form == "free_text":
        return response.text
    labels = {option.option_id: option.label for option in clarification.options}
    return ", ".join(labels[option_id] for option_id in response.option_ids)


def _require_claimed(run: PendingRun | None, *, run_id: str, submission_id: str | None) -> None:
    if run is None or run.run_id != run_id:
        raise PendingRunNotFoundError("No pause answers to this run")
    if run.status in CLOSED_STATUSES:
        raise PendingRunNotFoundError("The pause has already been closed")
    if run.status != "claimed" or run.submission_id != submission_id:
        raise PendingRunConflictError("This submission does not hold the pause")


def appended_attempt_trace(existing: tuple[str, ...], value: str) -> tuple[str, ...]:
    """Record one more attempt against a run, at most once.

    The durable record requires these to be distinct, and claiming a response already appends the
    attempt. A run that pauses a second time is claimed twice, which is where a caller that
    appends again at commit time first produces a duplicate.
    """

    return existing if value in existing else (*existing, value)
