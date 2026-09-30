"""Transition rules for a session's single in-flight paused run."""

from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from orchestration_core import (
    CLARIFICATION_PAUSE,
    CancelResponse,
    ClarificationOption,
    ClarificationRequest,
    FreeTextResponse,
    OptionSelectionResponse,
)

from agentic_orchestration.sessions import MemorySessionStore
from agentic_orchestration.sessions.contracts import (
    ClaimResponse,
    ClosePendingRun,
    CommitContinuedTurn,
    CommitTurn,
    PendingRunConflictError,
    PendingRunExpiredError,
    PendingRunNotFoundError,
    PublishPendingRun,
    SessionConflictError,
    SubmissionConflictError,
)
from agentic_orchestration.sessions.pending import appended_attempt_trace

# A store stamps a publication with the real clock, so a claim time is only meaningful
# against the run's own `expires_at`. NOW stands for a claim made inside the pause window;
# a claim made after it is taken from the published run rather than from a fixed date.
NOW = datetime(2026, 9, 15, 12, 0, tzinfo=UTC)


def clarification(task_id: str = "task-1") -> ClarificationRequest:
    return ClarificationRequest(
        clarification_id=f"clarification-{task_id}",
        producer_agent_id="scripted_producer",
        task_id=task_id,
        question=f"Which market should {task_id} cover?",
        reason_code="ambiguous_market",
        selection_mode="single",
        options=(
            ClarificationOption(option_id="us", label="United States"),
            ClarificationOption(option_id="eu", label="Europe"),
        ),
        freshness_token=f"snapshot-{task_id}",
    )


def publication(session_id: str = "session-1", *, expected_revision: int | None = None) -> Any:
    return PublishPendingRun(
        session_id=session_id,
        expected_revision=expected_revision,
        tenant_id=None,
        user_id=None,
        agent_id="router",
        selected_agent_id="scripted_producer",
        run_id="run-1",
        clarification=clarification(),
        input_message=HumanMessage(content="How did spend perform?", id="input-1"),
        route_reason_code="ambiguous_market",
        trace_id="trace-1",
        graph_definition_id="definition-1",
        continuation_contract_version=1,
    )


async def published_store() -> MemorySessionStore:
    store = MemorySessionStore()
    await store.publish_pending_run(publication())
    return store


def claim(submission_id: str = "submission-1", response: Any = None) -> ClaimResponse:
    return ClaimResponse(
        session_id="session-1",
        clarification_id="clarification-task-1",
        submission_id=submission_id,
        response=response or OptionSelectionResponse(option_ids=("us",)),
        attempt_trace_id="attempt-1",
    )


def commit(submission_id: str = "submission-1") -> CommitContinuedTurn:
    return CommitContinuedTurn(
        session_id="session-1",
        run_id="run-1",
        clarification_id="clarification-task-1",
        submission_id=submission_id,
        expected_revision=None,
        tenant_id=None,
        user_id=None,
        agent_id="router",
        selected_agent_id="scripted_producer",
        input_message=HumanMessage(content="How did spend perform?", id="input-1"),
        generated_messages=(AIMessage(content="Settled on the US", id="final-1"),),
        final_message_id="final-1",
        trace_id="trace-1",
        attempt_trace_ids=("trace-1", "attempt-1"),
        clarification_ids=("clarification-task-1",),
    )


def closure(reason: str, submission_id: str | None) -> ClosePendingRun:
    return ClosePendingRun(
        session_id="session-1",
        run_id="run-1",
        clarification_id="clarification-task-1",
        reason=reason,  # type: ignore[arg-type]
        submission_id=submission_id,
        final_message=AIMessage(content="Which market?", id="question-1"),
        trace_id="trace-1",
        attempt_trace_id="attempt-1",
    )


@pytest.mark.unit
async def test_publishing_a_pause_creates_no_session_document() -> None:
    """A pause on a session's first request must not make an empty committed session."""

    store = await published_store()

    assert await store.load("session-1") is None
    run = await store.active_pending_run("session-1")
    assert run is not None
    assert run.expected_session_revision is None
    assert run.expires_at - run.created_at == CLARIFICATION_PAUSE


@pytest.mark.unit
async def test_publication_is_audited_with_its_offered_options() -> None:
    store = await published_store()

    audit = await store.clarification_audit("session-1")

    assert [entry.event for entry in audit] == ["published"]
    assert audit[0].option_ids == ("us", "eu")
    assert audit[0].question == "Which market should task-1 cover?"
    assert audit[0].selection_mode == "single"


@pytest.mark.unit
async def test_a_session_holds_only_one_in_flight_run() -> None:
    store = await published_store()

    with pytest.raises(PendingRunConflictError):
        await store.publish_pending_run(publication())


@pytest.mark.unit
async def test_publication_is_guarded_by_the_expected_session_revision() -> None:
    store = MemorySessionStore()

    with pytest.raises(SessionConflictError):
        await store.publish_pending_run(publication(expected_revision=3))


@pytest.mark.unit
async def test_claiming_a_response_records_it_and_its_submission() -> None:
    store = await published_store()

    claimed = await store.claim_response(claim(), at=NOW)

    assert claimed.replayed is False
    assert claimed.pending_run.status == "claimed"
    assert claimed.pending_run.submission_id == "submission-1"
    audit = await store.clarification_audit("session-1")
    assert [entry.event for entry in audit] == ["published", "answered"]
    assert audit[1].response_text == "United States"
    assert audit[1].submission_id == "submission-1"


@pytest.mark.unit
async def test_replaying_the_same_submission_is_the_same_claim() -> None:
    store = await published_store()
    await store.claim_response(claim(), at=NOW)

    replay = await store.claim_response(claim(), at=NOW)

    assert replay.replayed is True
    # The audit records the answer once, not once per delivery.
    audit = await store.clarification_audit("session-1")
    assert [entry.event for entry in audit].count("answered") == 1


@pytest.mark.unit
async def test_reusing_a_submission_for_a_different_answer_is_refused() -> None:
    store = await published_store()
    await store.claim_response(claim(), at=NOW)

    with pytest.raises(SubmissionConflictError):
        await store.claim_response(
            claim(response=OptionSelectionResponse(option_ids=("eu",))), at=NOW
        )
    with pytest.raises(SubmissionConflictError):
        await store.claim_response(claim(response=FreeTextResponse(text="Canada")), at=NOW)


@pytest.mark.unit
async def test_a_second_submission_cannot_displace_one_being_applied() -> None:
    store = await published_store()
    await store.claim_response(claim(), at=NOW)

    with pytest.raises(PendingRunConflictError):
        await store.claim_response(claim("submission-2"), at=NOW)


@pytest.mark.unit
async def test_expiry_is_decided_when_the_response_is_claimed() -> None:
    store = await published_store()
    run = await store.active_pending_run("session-1")
    assert run is not None

    # Expiry is inclusive, so the deadline itself is already too late to answer.
    with pytest.raises(PendingRunExpiredError):
        await store.claim_response(claim(), at=run.expires_at)


@pytest.mark.unit
async def test_a_claim_made_before_expiry_is_not_cut_off_afterwards() -> None:
    """A continuation already running must not be killed by the clock."""

    store = await published_store()
    claimed = await store.claim_response(claim(), at=NOW)

    assert claimed.pending_run.status == "claimed"
    snapshot = await store.commit_continued_turn(commit())

    assert snapshot.revision == 1


@pytest.mark.unit
async def test_committing_consumes_the_run_and_records_its_lineage() -> None:
    store = await published_store()
    await store.claim_response(claim(), at=NOW)

    snapshot = await store.commit_continued_turn(commit())

    assert snapshot.revision == 1
    assert snapshot.turns[-1].routing_outcome == "delegate"
    assert snapshot.turns[-1].selected_agent_id == "scripted_producer"
    assert await store.active_pending_run("session-1") is None


@pytest.mark.unit
async def test_a_turn_cannot_be_committed_under_another_submission() -> None:
    store = await published_store()
    await store.claim_response(claim(), at=NOW)

    with pytest.raises(PendingRunConflictError):
        await store.commit_continued_turn(commit("submission-2"))


@pytest.mark.unit
async def test_cancelling_closes_the_turn_at_its_question() -> None:
    store = await published_store()
    await store.claim_response(claim(response=CancelResponse()), at=NOW)

    snapshot = await store.close_pending_run(closure("canceled", "submission-1"))

    assert snapshot.revision == 1
    assert snapshot.turns[-1].final_message_id == "question-1"
    assert await store.active_pending_run("session-1") is None
    audit = await store.clarification_audit("session-1")
    # A cancellation is audited once, as the closure, naming the submission that caused it.
    assert [entry.event for entry in audit] == ["published", "canceled"]
    assert audit[-1].submission_id == "submission-1"


@pytest.mark.unit
async def test_cancelling_is_idempotent_for_the_same_submission() -> None:
    store = await published_store()
    await store.claim_response(claim(response=CancelResponse()), at=NOW)
    await store.close_pending_run(closure("canceled", "submission-1"))

    snapshot = await store.close_pending_run(closure("canceled", "submission-1"))

    assert snapshot.revision == 1
    audit = await store.clarification_audit("session-1")
    assert [entry.event for entry in audit].count("canceled") == 1


@pytest.mark.unit
async def test_expiry_cannot_close_a_run_whose_response_is_already_claimed() -> None:
    """Concurrent expiry and response are decided by the record, not by who finishes first."""

    store = await published_store()
    await store.claim_response(claim(), at=NOW)

    with pytest.raises(PendingRunConflictError):
        await store.close_pending_run(closure("expired", None))


@pytest.mark.unit
async def test_expiry_closes_an_unanswered_run_at_its_question() -> None:
    store = await published_store()

    snapshot = await store.close_pending_run(closure("expired", None))

    assert snapshot.revision == 1
    audit = await store.clarification_audit("session-1")
    assert [entry.event for entry in audit] == ["published", "expired"]
    assert audit[-1].submission_id is None


@pytest.mark.unit
async def test_a_consumed_run_cannot_be_answered_by_another_submission() -> None:
    store = await published_store()
    await store.claim_response(claim(), at=NOW)
    await store.commit_continued_turn(commit())

    with pytest.raises(PendingRunNotFoundError):
        await store.claim_response(claim("submission-2"), at=NOW)


@pytest.mark.unit
async def test_a_replayed_submission_after_completion_is_recognised() -> None:
    store = await published_store()
    await store.claim_response(claim(), at=NOW)
    await store.commit_continued_turn(commit())

    replay = await store.claim_response(claim(), at=NOW)

    assert replay.replayed is True
    assert replay.pending_run.status == "consumed"


@pytest.mark.unit
async def test_an_unknown_clarification_has_nothing_to_answer() -> None:
    store = await published_store()

    with pytest.raises(PendingRunNotFoundError):
        await store.claim_response(
            ClaimResponse(
                session_id="session-1",
                clarification_id="clarification-absent",
                submission_id="submission-1",
                response=OptionSelectionResponse(option_ids=("us",)),
                attempt_trace_id="attempt-1",
            ),
            at=NOW,
        )


@pytest.mark.unit
async def test_the_next_queued_clarification_replaces_the_answered_one() -> None:
    store = await published_store()
    await store.claim_response(claim(), at=NOW)

    republished = await store.republish_pending_run(
        "session-1",
        run_id="run-1",
        submission_id="submission-1",
        clarification=clarification("task-2"),
        attempt_trace_id="attempt-1",
    )

    assert republished.status == "awaiting_response"
    assert republished.clarification_id == "clarification-task-2"
    assert republished.submission_id is None
    audit = await store.clarification_audit("session-1")
    assert [entry.event for entry in audit] == ["published", "answered", "published"]
    assert audit[-1].clarification_id == "clarification-task-2"


@pytest.mark.unit
async def test_answering_twice_records_each_attempt_once() -> None:
    """The durable record requires attempt_trace_ids to be distinct.

    Claiming a response already appends the attempt, so a caller that appends again at commit
    time repeats it. A run answered once is claimed once and the duplicate is invisible; a run
    that pauses a second time is claimed twice, and the commit is rejected outright. Only the
    Mongo schema enforces distinctness, so this stayed reachable while every memory-backed test
    passed. Hence asserting the invariant here, not just the shape.
    """

    store = await published_store()
    first = await store.claim_response(claim(), at=NOW)
    await store.republish_pending_run(
        "session-1",
        run_id="run-1",
        submission_id="submission-1",
        clarification=clarification("task-2"),
        attempt_trace_id=first.pending_run.attempt_trace_ids[-1],
    )
    second = await store.claim_response(
        replace(claim(), clarification_id="clarification-task-2", submission_id="submission-2"),
        at=NOW,
    )

    recorded = second.pending_run.attempt_trace_ids
    assert len(set(recorded)) == len(recorded), "an attempt was recorded twice"
    # And appending again at commit time must stay a no-op rather than duplicating it.
    assert appended_attempt_trace(recorded, recorded[-1]) == recorded


@pytest.mark.unit
async def test_a_paused_run_leaves_an_ordinary_turn_commit_alone() -> None:
    """Publishing a pause must not advance the session revision by itself."""

    store = MemorySessionStore()
    await store.commit_turn(
        CommitTurn(
            session_id="session-1",
            expected_revision=None,
            tenant_id=None,
            user_id=None,
            agent_id="router",
            input_message=HumanMessage(content="hello", id="input-0"),
            generated_messages=(AIMessage(content="hi", id="final-0"),),
            final_message_id="final-0",
            routing_outcome="answer",
            selected_agent_id=None,
            trace_id="trace-0",
        )
    )
    await store.publish_pending_run(publication(expected_revision=1))

    snapshot = await store.load("session-1")
    assert snapshot is not None
    assert snapshot.revision == 1
