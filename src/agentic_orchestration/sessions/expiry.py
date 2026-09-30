"""Closing paused runs that nobody came back to answer.

Expiry is checked when a response is claimed, which decides correctness: a pause past its
deadline is never answered. It does not, on its own, free anything. A run whose user simply
walked away stays `awaiting_response` forever, keeps the session's single in-flight slot,
and makes every later ordinary message a conflict. These two paths close it.

Request-time closure handles the session someone comes back to. The sweep handles the ones
nobody revisits, in small bounded batches.

Both reuse the same transactional close as a cancellation, because expiry owes the same
durable outcome: the logical turn committed at the question it last displayed, and an
`expired` audit entry. That is also why a MongoDB TTL deletion is the wrong tool here --
it would remove the record and leave no committed turn behind.

Only `awaiting_response` runs are ever candidates. A claimed run is in the middle of a
continuation, and closing it would discard work already under way; the compare-and-set
inside the close is what settles a race between two replicas reaching the same run.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from uuid import uuid4

from langchain_core.messages import AIMessage
from orchestration_core import is_clarification_expired

from agentic_orchestration.sessions.contracts import (
    ClosePendingRun,
    PendingRun,
    SessionStorageError,
    SessionStore,
)

logger = logging.getLogger(__name__)


def is_due(run: PendingRun, *, at: datetime) -> bool:
    """Whether this run may be closed for expiry now."""

    return run.status == "awaiting_response" and is_clarification_expired(run.expires_at, at=at)


async def close_expired_run(
    session_store: SessionStore,
    run: PendingRun,
    *,
    attempt_trace_id: str | None = None,
) -> bool:
    """Close one expired run, returning whether it is now closed.

    A run another replica already closed reports true: the close is idempotent, so the
    second caller commits no second turn and writes no second audit entry, and the slot is
    resolved either way. False means the run is still held -- almost always because a
    response was claimed in between, which legitimately beats expiry.
    """

    try:
        await session_store.close_pending_run(
            ClosePendingRun(
                session_id=run.session_id,
                run_id=run.run_id,
                clarification_id=run.clarification_id,
                reason="expired",
                submission_id=None,
                final_message=AIMessage(content=run.clarification.question, id=str(uuid4())),
                trace_id=run.trace_id,
                attempt_trace_id=attempt_trace_id,
            )
        )
    except SessionStorageError as exc:
        logger.info(
            "expired pending run was not closed by this caller run_id=%s type=%s",
            run.run_id,
            type(exc).__name__,
        )
        return False
    return True


async def release_if_expired(
    session_store: SessionStore,
    run: PendingRun | None,
    *,
    at: datetime,
    attempt_trace_id: str | None = None,
) -> PendingRun | None:
    """Return the run still holding this session's slot, closing it first if it has expired.

    Called before an ordinary chat decides whether the session is busy, so a message
    arriving after a pause lapsed starts a fresh turn instead of meeting a conflict that
    nothing would ever clear.
    """

    if run is None:
        return None
    if not is_due(run, at=at):
        return run
    await close_expired_run(session_store, run, attempt_trace_id=attempt_trace_id)
    return await session_store.active_pending_run(run.session_id)


async def sweep_expired_runs(
    session_store: SessionStore,
    *,
    at: datetime,
    limit: int,
) -> int:
    """Close up to `limit` runs nobody came back to, and report how many are now resolved.

    Bounded on purpose. This is lifecycle cleanup for a small number of abandoned pauses,
    not a scheduler: a batch that fills its limit simply leaves the rest to the next pass.
    """

    due = await session_store.due_pending_runs(at=at, limit=limit)
    closed = 0
    for run in due:
        if await close_expired_run(session_store, run):
            closed += 1
    return closed


async def run_expiry_sweeper(
    session_store: SessionStore,
    *,
    interval_seconds: float,
    limit: int,
) -> None:
    """Sweep expired pauses on an interval until cancelled.

    Deliberately not a scheduler: no queue, no persistence, no coordination between
    replicas. Each pass takes a bounded batch and lets the compare-and-set inside the close
    settle who actually closes what, so several replicas running this at once is correct
    rather than something to prevent.

    A failed pass is logged and the loop continues. Expiry correctness does not rest on
    this running at all -- a response arriving after the deadline is refused whether or not
    anything swept -- so an outage here delays cleanup instead of admitting a stale answer.
    """

    while True:
        await asyncio.sleep(interval_seconds)
        try:
            closed = await sweep_expired_runs(session_store, at=datetime.now(UTC), limit=limit)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("pending run expiry sweep failed type=%s", type(exc).__name__)
            continue
        if closed:
            logger.info("expired pending runs closed count=%d", closed)
