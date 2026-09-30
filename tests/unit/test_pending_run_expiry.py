"""Closing paused runs that nobody came back to answer."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest
from langchain_core.messages import HumanMessage
from orchestration_core import (
    ClarificationOption,
    ClarificationRequest,
    OptionSelectionResponse,
)

from agentic_orchestration.sessions import MemorySessionStore
from agentic_orchestration.sessions.contracts import (
    ClaimResponse,
    PendingRun,
    PublishPendingRun,
)
from agentic_orchestration.sessions.expiry import (
    close_expired_run,
    is_due,
    release_if_expired,
    sweep_expired_runs,
)

NOW = datetime(2026, 9, 15, 12, 0, tzinfo=UTC)
LATER = NOW + timedelta(minutes=11)


def clarification(run_id: str) -> ClarificationRequest:
    return ClarificationRequest(
        clarification_id=f"clarification-{run_id}",
        producer_agent_id="scripted_producer",
        task_id="task-1",
        question="Which market should the summary cover?",
        reason_code="ambiguous_market",
        selection_mode="single",
        options=(
            ClarificationOption(option_id="us", label="United States"),
            ClarificationOption(option_id="eu", label="Europe"),
        ),
        freshness_token="snapshot-1",
    )


async def publish(store: MemorySessionStore, session_id: str, run_id: str) -> PendingRun:
    return await store.publish_pending_run(
        PublishPendingRun(
            session_id=session_id,
            expected_revision=None,
            tenant_id=None,
            user_id=None,
            agent_id="router",
            selected_agent_id="scripted_producer",
            run_id=run_id,
            clarification=clarification(run_id),
            input_message=HumanMessage(content="How did spend perform?", id=f"input-{run_id}"),
            route_reason_code="ambiguous_market",
            trace_id=f"trace-{run_id}",
            graph_definition_id="definition-1",
            continuation_contract_version=1,
        )
    )


def age(store: MemorySessionStore, session_id: str, *, expires_at: datetime) -> None:
    """Move a published pause's deadline, as the wall clock would."""

    store._pending[session_id] = replace(
        store._pending[session_id],
        expires_at=expires_at,
    )


@pytest.mark.unit
async def test_only_an_unanswered_run_past_its_deadline_is_due() -> None:
    store = MemorySessionStore()
    run = await publish(store, "session-1", "run-1")

    assert is_due(run, at=NOW) is False
    assert is_due(replace(run, expires_at=NOW), at=NOW) is True
    # A claimed run is mid-continuation and is never a candidate.
    assert is_due(replace(run, expires_at=NOW, status="claimed"), at=NOW) is False


@pytest.mark.unit
async def test_closing_an_expired_run_commits_its_question_ending_turn() -> None:
    store = MemorySessionStore()
    await publish(store, "session-1", "run-1")
    age(store, "session-1", expires_at=NOW)

    expired = await store.active_pending_run("session-1")
    assert expired is not None
    closed = await close_expired_run(store, expired)

    assert closed is True
    snapshot = await store.load("session-1")
    assert snapshot is not None
    assert snapshot.revision == 1
    assert snapshot.turns[-1].continuation is not None
    assert snapshot.turns[-1].continuation.closed_by == "expired"
    audit = await store.clarification_audit("session-1")
    assert [entry.event for entry in audit] == ["published", "expired"]


@pytest.mark.unit
async def test_expiry_frees_the_session_run_slot() -> None:
    store = MemorySessionStore()
    await publish(store, "session-1", "run-1")
    age(store, "session-1", expires_at=NOW)

    released = await release_if_expired(
        store, await store.active_pending_run("session-1"), at=LATER
    )

    assert released is None
    assert await store.active_pending_run("session-1") is None


@pytest.mark.unit
async def test_a_live_pause_is_left_alone() -> None:
    store = MemorySessionStore()
    run = await publish(store, "session-1", "run-1")

    released = await release_if_expired(store, run, at=NOW)

    assert released is not None
    assert released.status == "awaiting_response"
    assert await store.load("session-1") is None


@pytest.mark.unit
async def test_release_handles_a_session_with_no_pause() -> None:
    store = MemorySessionStore()

    assert await release_if_expired(store, None, at=NOW) is None


@pytest.mark.unit
async def test_a_claimed_run_is_never_closed_by_expiry() -> None:
    """Closing one mid-continuation would discard work already under way."""

    store = MemorySessionStore()
    await publish(store, "session-1", "run-1")
    await store.claim_response(
        ClaimResponse(
            session_id="session-1",
            clarification_id="clarification-run-1",
            submission_id="submission-1",
            response=OptionSelectionResponse(option_ids=("us",)),
            attempt_trace_id="attempt-1",
        ),
        at=NOW,
    )
    age(store, "session-1", expires_at=NOW)

    assert await store.due_pending_runs(at=LATER, limit=10) == ()
    assert await sweep_expired_runs(store, at=LATER, limit=10) == 0
    run = await store.active_pending_run("session-1")
    assert run is not None
    assert run.status == "claimed"


@pytest.mark.unit
async def test_the_sweep_closes_runs_nobody_revisited() -> None:
    store = MemorySessionStore()
    for index in range(3):
        await publish(store, f"session-{index}", f"run-{index}")
        age(store, f"session-{index}", expires_at=NOW)

    closed = await sweep_expired_runs(store, at=LATER, limit=10)

    assert closed == 3
    for index in range(3):
        assert await store.active_pending_run(f"session-{index}") is None
        snapshot = await store.load(f"session-{index}")
        assert snapshot is not None
        assert snapshot.revision == 1


@pytest.mark.unit
async def test_the_sweep_is_bounded_and_leaves_the_rest_to_the_next_pass() -> None:
    store = MemorySessionStore()
    for index in range(5):
        await publish(store, f"session-{index}", f"run-{index}")
        age(store, f"session-{index}", expires_at=NOW + timedelta(seconds=index))

    first = await sweep_expired_runs(store, at=LATER, limit=2)
    remaining = await store.due_pending_runs(at=LATER, limit=10)

    assert first == 2
    assert len(remaining) == 3
    assert await sweep_expired_runs(store, at=LATER, limit=10) == 3


@pytest.mark.unit
async def test_the_sweep_takes_the_longest_expired_runs_first() -> None:
    store = MemorySessionStore()
    for index in range(3):
        await publish(store, f"session-{index}", f"run-{index}")
        age(store, f"session-{index}", expires_at=NOW + timedelta(seconds=10 - index))

    due = await store.due_pending_runs(at=LATER, limit=2)

    assert [run.session_id for run in due] == ["session-2", "session-1"]


@pytest.mark.unit
async def test_two_replicas_closing_the_same_run_commit_one_turn() -> None:
    """The close is idempotent, so a replica that lost the race changes nothing."""

    store = MemorySessionStore()
    await publish(store, "session-1", "run-1")
    age(store, "session-1", expires_at=NOW)
    run = await store.active_pending_run("session-1")
    assert run is not None

    assert await close_expired_run(store, run) is True
    assert await close_expired_run(store, run) is True

    snapshot = await store.load("session-1")
    assert snapshot is not None
    assert snapshot.revision == 1
    audit = await store.clarification_audit("session-1")
    assert [entry.event for entry in audit].count("expired") == 1


@pytest.mark.unit
async def test_a_response_claimed_in_between_beats_expiry() -> None:
    store = MemorySessionStore()
    await publish(store, "session-1", "run-1")
    age(store, "session-1", expires_at=NOW)
    run = await store.active_pending_run("session-1")
    assert run is not None
    await store.claim_response(
        ClaimResponse(
            session_id="session-1",
            clarification_id="clarification-run-1",
            submission_id="submission-1",
            response=OptionSelectionResponse(option_ids=("us",)),
            attempt_trace_id="attempt-1",
        ),
        at=NOW - timedelta(minutes=1),
    )

    assert await close_expired_run(store, run) is False
    still_held = await store.active_pending_run("session-1")
    assert still_held is not None
    assert still_held.status == "claimed"


@pytest.mark.unit
async def test_a_live_pause_is_not_swept() -> None:
    store = MemorySessionStore()
    await publish(store, "session-1", "run-1")

    assert await sweep_expired_runs(store, at=NOW, limit=10) == 0
    assert await store.active_pending_run("session-1") is not None


@pytest.mark.unit
async def test_sweeping_a_store_with_nothing_due_is_a_no_op() -> None:
    assert await sweep_expired_runs(MemorySessionStore(), at=NOW, limit=10) == 0
