"""Opt-in proof that pending-run transitions are transactional in MongoDB.

Uses the real collections with session identifiers of its own, and removes exactly the
documents it wrote. Requires a replica set, which the session store already depends on for
its commit transaction.
"""

import asyncio
import os
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from orchestration_core import (
    CancelResponse,
    ClarificationOption,
    ClarificationRequest,
    OptionSelectionResponse,
)

from agentic_orchestration.config import Settings
from agentic_orchestration.sessions import MongoSessionStore
from agentic_orchestration.sessions.contracts import (
    ClaimResponse,
    ClosePendingRun,
    CommitContinuedTurn,
    PendingRunConflictError,
    PendingRunNotFoundError,
    PublishPendingRun,
    SessionConflictError,
    SubmissionConflictError,
)
from agentic_orchestration.sessions.expiry import close_expired_run
from agentic_orchestration.sessions.schema import (
    CLARIFICATION_AUDIT_COLLECTION,
    PENDING_RUNS_COLLECTION,
    SESSION_TURNS_COLLECTION,
    SESSIONS_COLLECTION,
)
from integrations.mongodb import create_mongodb

pytestmark = [
    pytest.mark.integration,
    pytest.mark.mongo,
    pytest.mark.skipif(
        os.getenv("RUN_MONGO_CHECKPOINT_TESTS") != "1",
        reason="set RUN_MONGO_CHECKPOINT_TESTS=1 to exercise durable pending-run transitions",
    ),
    pytest.mark.skipif(
        Settings().mongodb_host is None, reason="MongoDB connection settings are absent"
    ),
]


class Fixture:
    def __init__(self, store: MongoSessionStore, session_id: str, run_id: str) -> None:
        self.store = store
        self.session_id = session_id
        self.run_id = run_id

    def clarification(self, task_id: str = "task-1") -> ClarificationRequest:
        return ClarificationRequest(
            clarification_id=f"clarification-{self.run_id}-{task_id}",
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

    def publication(self, **overrides: Any) -> PublishPendingRun:
        values: dict[str, Any] = {
            "session_id": self.session_id,
            "expected_revision": None,
            "tenant_id": None,
            "user_id": None,
            "agent_id": "router",
            "selected_agent_id": "scripted_producer",
            "run_id": self.run_id,
            "clarification": self.clarification(),
            "input_message": HumanMessage(content="How did spend perform?", id="input-1"),
            "route_reason_code": "ambiguous_market",
            "trace_id": f"trace-{self.run_id}",
            "graph_definition_id": "definition-1",
            "continuation_contract_version": 1,
        }
        values.update(overrides)
        return PublishPendingRun(**values)

    def claim(self, submission_id: str = "submission-1", response: Any = None) -> ClaimResponse:
        return ClaimResponse(
            session_id=self.session_id,
            clarification_id=self.clarification().clarification_id,
            submission_id=submission_id,
            response=response or OptionSelectionResponse(option_ids=("us",)),
            attempt_trace_id=f"attempt-{self.run_id}",
        )

    def commit(self, submission_id: str = "submission-1") -> CommitContinuedTurn:
        return CommitContinuedTurn(
            session_id=self.session_id,
            run_id=self.run_id,
            clarification_id=self.clarification().clarification_id,
            submission_id=submission_id,
            expected_revision=None,
            tenant_id=None,
            user_id=None,
            agent_id="router",
            selected_agent_id="scripted_producer",
            input_message=HumanMessage(content="How did spend perform?", id="input-1"),
            generated_messages=(AIMessage(content="US spend rose", id="final-1"),),
            final_message_id="final-1",
            trace_id=f"trace-{self.run_id}",
            attempt_trace_ids=(f"trace-{self.run_id}",),
            clarification_ids=(self.clarification().clarification_id,),
        )

    def closure(self, reason: str, submission_id: str | None) -> ClosePendingRun:
        return ClosePendingRun(
            session_id=self.session_id,
            run_id=self.run_id,
            clarification_id=self.clarification().clarification_id,
            reason=reason,  # type: ignore[arg-type]
            submission_id=submission_id,
            final_message=AIMessage(content="Which market?", id="question-1"),
            trace_id=f"trace-{self.run_id}",
            attempt_trace_id=f"attempt-{self.run_id}",
        )


@pytest.fixture
async def durable() -> AsyncIterator[Fixture]:
    settings = Settings()
    client, database = create_mongodb(
        host=settings.mongodb_host,
        port=settings.mongodb_port,
        user=settings.mongodb_user,
        password=(
            settings.mongodb_pwd.get_secret_value() if settings.mongodb_pwd is not None else None
        ),
        database=settings.mongodb_db,
    )
    session_id = f"test-session-{uuid4()}"
    run_id = f"test-run-{uuid4()}"
    try:
        store = MongoSessionStore(database)
        await store.initialize()
        yield Fixture(store, session_id, run_id)
    finally:
        for collection, query in (
            (SESSIONS_COLLECTION, {"_id": session_id}),
            (SESSION_TURNS_COLLECTION, {"session_id": session_id}),
            (PENDING_RUNS_COLLECTION, {"session_id": session_id}),
            (CLARIFICATION_AUDIT_COLLECTION, {"session_id": session_id}),
        ):
            await database[collection].delete_many(query)
        await client.close()


async def test_a_first_turn_pause_publishes_without_a_session_document(
    durable: Fixture,
) -> None:
    await durable.store.publish_pending_run(durable.publication())

    assert await durable.store.load(durable.session_id) is None
    run = await durable.store.active_pending_run(durable.session_id)
    assert run is not None
    assert run.status == "awaiting_response"
    assert run.expected_session_revision is None


async def test_the_unique_index_admits_one_in_flight_run_per_session(
    durable: Fixture,
) -> None:
    await durable.store.publish_pending_run(durable.publication())

    with pytest.raises(PendingRunConflictError):
        await durable.store.publish_pending_run(
            durable.publication(run_id=f"{durable.run_id}-second")
        )


async def test_publication_is_refused_against_an_unexpected_revision(
    durable: Fixture,
) -> None:
    with pytest.raises(SessionConflictError):
        await durable.store.publish_pending_run(durable.publication(expected_revision=4))

    assert await durable.store.active_pending_run(durable.session_id) is None


async def test_claiming_and_committing_advances_the_session_once(durable: Fixture) -> None:
    await durable.store.publish_pending_run(durable.publication())
    await durable.store.claim_response(durable.claim(), at=datetime.now(UTC))

    snapshot = await durable.store.commit_continued_turn(durable.commit())

    assert snapshot.revision == 1
    assert snapshot.turns[-1].routing_outcome == "delegate"
    assert await durable.store.active_pending_run(durable.session_id) is None
    audit = await durable.store.clarification_audit(durable.session_id)
    assert [entry.event for entry in audit] == ["published", "answered"]


async def test_the_committed_turn_retains_its_continuation_lineage(durable: Fixture) -> None:
    await durable.store.publish_pending_run(durable.publication())
    await durable.store.claim_response(durable.claim(), at=datetime.now(UTC))
    await durable.store.commit_continued_turn(durable.commit())

    reloaded = await durable.store.load(durable.session_id)

    assert reloaded is not None
    assert reloaded.turns[-1].selected_agent_id == "scripted_producer"


async def test_a_replayed_submission_is_recognised_after_the_turn_is_committed(
    durable: Fixture,
) -> None:
    await durable.store.publish_pending_run(durable.publication())
    await durable.store.claim_response(durable.claim(), at=datetime.now(UTC))
    await durable.store.commit_continued_turn(durable.commit())

    replay = await durable.store.claim_response(durable.claim(), at=datetime.now(UTC))

    assert replay.replayed is True
    assert replay.pending_run.status == "consumed"


async def test_a_different_submission_cannot_answer_a_consumed_run(durable: Fixture) -> None:
    await durable.store.publish_pending_run(durable.publication())
    await durable.store.claim_response(durable.claim(), at=datetime.now(UTC))
    await durable.store.commit_continued_turn(durable.commit())

    with pytest.raises(PendingRunNotFoundError):
        await durable.store.claim_response(durable.claim("submission-2"), at=datetime.now(UTC))


async def test_reusing_a_submission_for_different_content_is_refused(
    durable: Fixture,
) -> None:
    await durable.store.publish_pending_run(durable.publication())
    await durable.store.claim_response(durable.claim(), at=datetime.now(UTC))

    with pytest.raises(SubmissionConflictError):
        await durable.store.claim_response(
            durable.claim(response=OptionSelectionResponse(option_ids=("eu",))),
            at=datetime.now(UTC),
        )


async def test_cancelling_closes_the_turn_and_releases_the_session(durable: Fixture) -> None:
    await durable.store.publish_pending_run(durable.publication())
    await durable.store.claim_response(
        durable.claim(response=CancelResponse()), at=datetime.now(UTC)
    )

    snapshot = await durable.store.close_pending_run(durable.closure("canceled", "submission-1"))

    assert snapshot.revision == 1
    assert snapshot.turns[-1].final_message_id == "question-1"
    assert await durable.store.active_pending_run(durable.session_id) is None
    # The slot is free, so an ordinary turn can publish a new pause afterwards.
    await durable.store.publish_pending_run(
        durable.publication(run_id=f"{durable.run_id}-next", expected_revision=1)
    )


async def test_expiry_cannot_close_a_run_whose_response_is_claimed(durable: Fixture) -> None:
    await durable.store.publish_pending_run(durable.publication())
    await durable.store.claim_response(durable.claim(), at=datetime.now(UTC))

    with pytest.raises(PendingRunConflictError):
        await durable.store.close_pending_run(durable.closure("expired", None))

    run = await durable.store.active_pending_run(durable.session_id)
    assert run is not None
    assert run.status == "claimed"


async def test_a_failed_commit_leaves_the_claim_and_the_revision_alone(
    durable: Fixture,
) -> None:
    """A commit under the wrong submission changes nothing at all."""

    await durable.store.publish_pending_run(durable.publication())
    await durable.store.claim_response(durable.claim(), at=datetime.now(UTC))

    with pytest.raises(PendingRunConflictError):
        await durable.store.commit_continued_turn(durable.commit("submission-2"))

    assert await durable.store.load(durable.session_id) is None
    run = await durable.store.active_pending_run(durable.session_id)
    assert run is not None
    assert run.status == "claimed"
    assert run.submission_id == "submission-1"


async def test_the_next_queued_clarification_replaces_the_answered_one(
    durable: Fixture,
) -> None:
    await durable.store.publish_pending_run(durable.publication())
    await durable.store.claim_response(durable.claim(), at=datetime.now(UTC))

    republished = await durable.store.republish_pending_run(
        durable.session_id,
        run_id=durable.run_id,
        submission_id="submission-1",
        clarification=durable.clarification("task-2"),
        attempt_trace_id="attempt-2",
    )

    assert republished.status == "awaiting_response"
    assert republished.submission_id is None
    audit = await durable.store.clarification_audit(durable.session_id)
    assert [entry.event for entry in audit] == ["published", "answered", "published"]


async def test_only_unanswered_expired_runs_are_listed_as_due(durable: Fixture) -> None:
    await durable.store.publish_pending_run(durable.publication())
    run = await durable.store.active_pending_run(durable.session_id)
    assert run is not None

    live = await durable.store.due_pending_runs(at=run.created_at, limit=10)
    due = await durable.store.due_pending_runs(at=run.expires_at, limit=10)

    assert [candidate.run_id for candidate in live if candidate.run_id == durable.run_id] == []
    assert durable.run_id in [candidate.run_id for candidate in due]


async def test_a_claimed_run_is_never_listed_as_due(durable: Fixture) -> None:
    """The index leads on status, so a continuation under way is not even a candidate."""

    await durable.store.publish_pending_run(durable.publication())
    await durable.store.claim_response(durable.claim(), at=datetime.now(UTC))
    run = await durable.store.active_pending_run(durable.session_id)
    assert run is not None

    due = await durable.store.due_pending_runs(at=run.expires_at, limit=10)

    assert durable.run_id not in [candidate.run_id for candidate in due]


async def test_the_sweep_closes_an_abandoned_run_and_frees_its_session(
    durable: Fixture,
) -> None:
    await durable.store.publish_pending_run(durable.publication())
    run = await durable.store.active_pending_run(durable.session_id)
    assert run is not None

    closed = await close_expired_run(durable.store, run)

    assert closed is True
    assert await durable.store.active_pending_run(durable.session_id) is None
    snapshot = await durable.store.load(durable.session_id)
    assert snapshot is not None
    assert snapshot.revision == 1
    assert snapshot.turns[-1].continuation is not None
    assert snapshot.turns[-1].continuation.closed_by == "expired"
    audit = await durable.store.clarification_audit(durable.session_id)
    assert [entry.event for entry in audit] == ["published", "expired"]


async def test_two_replicas_sweeping_the_same_run_commit_one_turn(
    durable: Fixture,
) -> None:
    """The compare-and-set inside the close settles the race between replicas."""

    await durable.store.publish_pending_run(durable.publication())
    run = await durable.store.active_pending_run(durable.session_id)
    assert run is not None

    first, second = await asyncio.gather(
        close_expired_run(durable.store, run),
        close_expired_run(durable.store, run),
    )

    assert first is True and second is True
    snapshot = await durable.store.load(durable.session_id)
    assert snapshot is not None
    assert snapshot.revision == 1
    audit = await durable.store.clarification_audit(durable.session_id)
    assert [entry.event for entry in audit].count("expired") == 1


async def test_a_committed_turn_retains_its_closure_reason(durable: Fixture) -> None:
    await durable.store.publish_pending_run(durable.publication())
    await durable.store.claim_response(durable.claim(), at=datetime.now(UTC))
    await durable.store.commit_continued_turn(durable.commit())

    reloaded = await durable.store.load(durable.session_id)

    assert reloaded is not None
    continuation = reloaded.turns[-1].continuation
    assert continuation is not None
    assert continuation.closed_by == "answered"
    assert continuation.run_id == durable.run_id
