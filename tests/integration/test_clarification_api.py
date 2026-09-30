"""The chat and continuation endpoints as one workflow, with a scripted child boundary."""

from collections.abc import AsyncIterator
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from langchain_core.messages import AIMessage
from orchestration_core import (
    ClarificationOption,
    ClarificationRequest,
    ContinuationNotResumableError,
    ConversationResult,
    PausedConversation,
)

from agentic_orchestration.api.dependencies import (
    get_agent_invoker,
    get_entry_agent,
    get_executor,
    get_graph_definition_ids,
    get_session_store,
)
from agentic_orchestration.main import create_app
from agentic_orchestration.sessions import MemorySessionStore

pytestmark = pytest.mark.integration

ROOT = Path(__file__).parents[2]


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


def paused(task_id: str = "task-1", run_id: str = "run-1") -> PausedConversation:
    return PausedConversation(
        agent_id="scripted_producer",
        run_id=run_id,
        clarification=clarification(task_id),
        graph_definition_id="definition-1",
    )


class ScriptedExecutor:
    """Stands in for the entry graph so the publication path can be driven directly."""

    def __init__(self, results: list[dict[str, Any]]) -> None:
        self.results = results
        self.traces: list[str] = []
        # Trace ids begun but not yet finished, so a test can ask what the invoker would find.
        self.open_traces: set[str] = set()

    async def begin(self, agent_name: str, context: Any) -> None:
        self.traces.append("begin")
        self.open_traces.add(context.trace_id)

    async def complete(self, context: Any, *, committed_turn_number: int) -> None:
        self.traces.append("complete")
        self.open_traces.discard(context.trace_id)

    async def fail(self, context: Any) -> None:
        self.traces.append("fail")
        self.open_traces.discard(context.trace_id)

    async def cancel(self, context: Any) -> None:
        self.traces.append("cancel")
        self.open_traces.discard(context.trace_id)

    async def invoke(self, agent_name: str, state: dict[str, Any], *, context: Any) -> Any:
        result = dict(self.results.pop(0))
        answer = result.pop("answer", None)
        messages = list(state["messages"])
        if answer is not None:
            messages.append(AIMessage(content=answer, id=str(uuid4())))
        return {"messages": messages, **result}


class ScriptedInvoker:
    """Stands in for the restricted child seam; records what a continuation asked of it."""

    def __init__(self, outcomes: list[Any], executor: Any = None) -> None:
        self.outcomes = outcomes
        self.executor = executor
        self.trace_open_on_resume: list[bool] = []
        self.resumed: list[Any] = []
        self.recovered: list[str] = []
        self.recovery: ConversationResult | None = None

    async def resume(self, agent_id: str, continuation: Any, context: Any) -> Any:
        self.resumed.append(continuation)
        # The real invoker looks its trace run up right here and raises if none was begun.
        if self.executor is not None:
            self.trace_open_on_resume.append(
                context.execution.trace_id in self.executor.open_traces
            )
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    async def recover(self, agent_id: str, run_id: str) -> ConversationResult | None:
        self.recovered.append(run_id)
        return self.recovery


class Harness:
    def __init__(self, app: FastAPI, client: AsyncClient, store: MemorySessionStore) -> None:
        self.app = app
        self.client = client
        self.store = store
        self.executor = ScriptedExecutor([])
        self.invoker = ScriptedInvoker([], self.executor)

    async def chat(self, message: str = "How did spend perform?", **body: Any) -> Any:
        return await self.client.post("/api/chat", json={"message": message, **body})

    async def respond(
        self, session_id: str, clarification_id: str, submission_id: str, response: dict[str, Any]
    ) -> Any:
        return await self.client.post(
            f"/api/sessions/{session_id}/clarifications/{clarification_id}/responses",
            json={"submission_id": submission_id, "response": response},
        )


@pytest.fixture
async def harness() -> AsyncIterator[Harness]:
    app = create_app()
    store = MemorySessionStore()
    harness: Harness | None = None
    app.dependency_overrides[get_session_store] = lambda: store
    app.dependency_overrides[get_entry_agent] = lambda: "router"
    app.dependency_overrides[get_graph_definition_ids] = lambda: {
        "scripted_producer": "definition-1"
    }
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://clarify") as client:
        harness = Harness(app, client, store)
        app.dependency_overrides[get_executor] = lambda: harness.executor  # type: ignore[union-attr]
        app.dependency_overrides[get_agent_invoker] = lambda: harness.invoker  # type: ignore[union-attr]
        yield harness


def expire_pending_run(store: MemorySessionStore, session_id: str, run: Any) -> None:
    """Move a published pause into the past, as the wall clock would."""

    store._pending[session_id] = replace(run, expires_at=datetime.now(UTC) - timedelta(seconds=1))


async def start_pause(harness: Harness, task_id: str = "task-1") -> Any:
    harness.executor.results = [{"paused_run": paused(task_id)}]
    return await harness.chat()


async def test_a_paused_child_is_published_and_returned_as_a_clarification(
    harness: Harness,
) -> None:
    response = await start_pause(harness)

    assert response.status_code == 202
    body = response.json()
    assert body["clarification_id"] == "clarification-task-1"
    assert body["question"] == "Which market should task-1 cover?"
    assert body["selection_mode"] == "single"
    assert [option["option_id"] for option in body["options"]] == ["us", "eu"]
    assert body["free_text_allowed"] is True
    assert body["cancel_allowed"] is True


async def test_the_clarification_never_exposes_what_a_client_must_not_send(
    harness: Harness,
) -> None:
    body = (await start_pause(harness)).json()

    for hidden in ("run_id", "agent_id", "freshness_token", "graph_definition_id"):
        assert hidden not in body


async def test_a_pause_is_durable_before_its_question_is_returned(harness: Harness) -> None:
    body = (await start_pause(harness)).json()

    run = await harness.store.active_pending_run(body["session_id"])
    audit = await harness.store.clarification_audit(body["session_id"])

    assert run is not None
    assert run.clarification_id == body["clarification_id"]
    assert [entry.event for entry in audit] == ["published"]


async def test_a_paused_turn_is_not_committed(harness: Harness) -> None:
    body = (await start_pause(harness)).json()

    assert await harness.store.load(body["session_id"]) is None


async def test_ordinary_chat_is_refused_while_a_clarification_is_open(
    harness: Harness,
) -> None:
    body = (await start_pause(harness)).json()
    harness.executor.results = [{"routing_outcome": "answer", "answer": "Noted."}]

    conflict = await harness.chat("something else", session_id=body["session_id"])

    assert conflict.status_code == 409
    # The unrelated message did not become an implicit answer.
    run = await harness.store.active_pending_run(body["session_id"])
    assert run is not None
    assert run.status == "awaiting_response"


async def test_answering_completes_the_turn(harness: Harness) -> None:
    body = (await start_pause(harness)).json()
    harness.invoker.outcomes = [
        ConversationResult(generated_messages=(AIMessage(content="US spend rose", id="final-1"),))
    ]

    completed = await harness.respond(
        body["session_id"],
        body["clarification_id"],
        "submission-1",
        {"form": "options", "option_ids": ["us"]},
    )

    assert completed.status_code == 200
    assert completed.json()["closed_by"] == "answered"
    assert completed.json()["message"] == "US spend rose"
    snapshot = await harness.store.load(body["session_id"])
    assert snapshot is not None
    assert snapshot.revision == 1


async def test_the_resume_runs_inside_a_trace_it_began(harness: Harness) -> None:
    """A resume is a traced execution, and the invoker looks its run up before touching the graph.

    This route minted a trace id for the audit trail but never began a run under it, so every
    resume raised `trace ... is not active`. That is not an AgentInvocationError, so it escaped
    all three handlers and returned a bare 500 with nothing recorded. Found the first time a
    clarification was answered end to end; the stubs here never reached that lookup.
    """

    body = (await start_pause(harness)).json()
    harness.invoker.outcomes = [
        ConversationResult(generated_messages=(AIMessage(content="US spend rose", id="final-1"),))
    ]
    harness.executor.traces.clear()

    completed = await harness.respond(
        body["session_id"],
        body["clarification_id"],
        "submission-1",
        {"form": "options", "option_ids": ["us"]},
    )

    assert completed.status_code == 200
    # Begun before the child was resumed...
    assert harness.invoker.trace_open_on_resume == [True]
    # ...and closed against the committed turn, so it cannot leak into the next request.
    assert harness.executor.traces == ["begin", "complete"]
    assert harness.executor.open_traces == set()


async def test_a_resume_that_pauses_again_still_closes_its_trace(harness: Harness) -> None:
    """Pausing is not failing, so the attempt is cancelled rather than failed. It must still be
    closed, or the next attempt inherits a run nobody finished."""

    body = (await start_pause(harness)).json()
    harness.invoker.outcomes = [paused("task-2")]
    harness.executor.traces.clear()

    again = await harness.respond(
        body["session_id"],
        body["clarification_id"],
        "submission-1",
        {"form": "options", "option_ids": ["us"]},
    )

    assert again.status_code == 202
    assert harness.invoker.trace_open_on_resume == [True]
    assert harness.executor.traces == ["begin", "cancel"]
    assert harness.executor.open_traces == set()


async def test_the_committed_turn_keeps_the_ordered_exchange(harness: Harness) -> None:
    body = (await start_pause(harness)).json()
    harness.invoker.outcomes = [
        ConversationResult(generated_messages=(AIMessage(content="US spend rose", id="final-1"),))
    ]
    await harness.respond(
        body["session_id"],
        body["clarification_id"],
        "submission-1",
        {"form": "options", "option_ids": ["us"]},
    )

    history = await harness.client.get(f"/api/sessions/{body['session_id']}/history")

    assert [message["content"] for message in history.json()["messages"]] == [
        "How did spend perform?",
        "Which market should task-1 cover?",
        "United States",
        "US spend rose",
    ]
    assert history.json()["pending_clarification"] is None
    snapshot = await harness.store.load(body["session_id"])
    assert snapshot is not None
    exchange = [message.content for message in snapshot.turns[-1].generated_messages]
    assert exchange == ["Which market should task-1 cover?", "United States", "US spend rose"]


async def test_the_continuation_resolves_the_run_from_durable_state(harness: Harness) -> None:
    body = (await start_pause(harness)).json()
    harness.invoker.outcomes = [
        ConversationResult(generated_messages=(AIMessage(content="US spend rose", id="final-1"),))
    ]
    await harness.respond(
        body["session_id"],
        body["clarification_id"],
        "submission-1",
        {"form": "options", "option_ids": ["us"]},
    )

    continuation = harness.invoker.resumed[0]
    assert continuation.run_id == "run-1"
    assert continuation.freshness_token == "snapshot-task-1"
    assert continuation.graph_definition_id == "definition-1"


async def test_another_queued_clarification_is_returned_as_a_pause(harness: Harness) -> None:
    body = (await start_pause(harness)).json()
    harness.invoker.outcomes = [paused("task-2")]

    second = await harness.respond(
        body["session_id"],
        body["clarification_id"],
        "submission-1",
        {"form": "options", "option_ids": ["us"]},
    )

    assert second.status_code == 202
    assert second.json()["clarification_id"] == "clarification-task-2"
    assert await harness.store.load(body["session_id"]) is None


async def test_replaying_the_same_submission_and_payload_is_idempotent(
    harness: Harness,
) -> None:
    body = (await start_pause(harness)).json()
    harness.invoker.outcomes = [
        ConversationResult(generated_messages=(AIMessage(content="US spend rose", id="final-1"),))
    ]
    payload = {"form": "options", "option_ids": ["us"]}
    first = await harness.respond(
        body["session_id"], body["clarification_id"], "submission-1", payload
    )

    replay = await harness.respond(
        body["session_id"], body["clarification_id"], "submission-1", payload
    )

    assert first.status_code == 200
    assert replay.status_code == 200
    assert replay.json() == first.json()
    snapshot = await harness.store.load(body["session_id"])
    assert snapshot is not None
    # One turn, not two: the retry did not run the graph again.
    assert snapshot.revision == 1
    assert len(harness.invoker.resumed) == 1


async def test_reusing_a_submission_for_different_content_is_refused(
    harness: Harness,
) -> None:
    body = (await start_pause(harness)).json()
    harness.invoker.outcomes = [
        ConversationResult(generated_messages=(AIMessage(content="US spend rose", id="final-1"),))
    ]
    await harness.respond(
        body["session_id"],
        body["clarification_id"],
        "submission-1",
        {"form": "options", "option_ids": ["us"]},
    )

    conflict = await harness.respond(
        body["session_id"],
        body["clarification_id"],
        "submission-1",
        {"form": "options", "option_ids": ["eu"]},
    )

    assert conflict.status_code == 409
    assert conflict.json()["error"]["code"] == "submission_conflict"


async def test_cancelling_closes_the_turn_without_resuming_the_graph(
    harness: Harness,
) -> None:
    body = (await start_pause(harness)).json()

    canceled = await harness.respond(
        body["session_id"], body["clarification_id"], "submission-1", {"form": "cancel"}
    )

    assert canceled.status_code == 200
    assert canceled.json()["closed_by"] == "canceled"
    assert canceled.json()["message"] == "Which market should task-1 cover?"
    assert harness.invoker.resumed == []
    snapshot = await harness.store.load(body["session_id"])
    assert snapshot is not None
    assert snapshot.revision == 1


async def test_a_new_message_after_cancelling_is_a_separate_request(
    harness: Harness,
) -> None:
    body = (await start_pause(harness)).json()
    await harness.respond(
        body["session_id"], body["clarification_id"], "submission-1", {"form": "cancel"}
    )
    harness.executor.results = [{"routing_outcome": "answer", "answer": "Noted."}]

    followup = await harness.chat("Just the US then", session_id=body["session_id"])

    assert followup.status_code == 200
    snapshot = await harness.store.load(body["session_id"])
    assert snapshot is not None
    assert snapshot.revision == 2


async def test_an_expired_clarification_is_closed_and_refused(harness: Harness) -> None:
    body = (await start_pause(harness)).json()
    run = await harness.store.active_pending_run(body["session_id"])
    assert run is not None
    expire_pending_run(harness.store, body["session_id"], run)

    expired = await harness.respond(
        body["session_id"],
        body["clarification_id"],
        "submission-1",
        {"form": "options", "option_ids": ["us"]},
    )

    assert expired.status_code == 410
    assert expired.json()["error"]["code"] == "clarification_expired"
    assert harness.invoker.resumed == []
    audit = await harness.store.clarification_audit(body["session_id"])
    assert [entry.event for entry in audit] == ["published", "expired"]


async def test_a_restart_required_outcome_is_distinct(harness: Harness) -> None:
    body = (await start_pause(harness)).json()
    harness.invoker.outcomes = [ContinuationNotResumableError("graph_version", "the graph changed")]

    refused = await harness.respond(
        body["session_id"],
        body["clarification_id"],
        "submission-1",
        {"form": "options", "option_ids": ["us"]},
    )

    assert refused.status_code == 409
    assert refused.json()["error"]["code"] == "restart_required:graph_version"
    assert await harness.store.load(body["session_id"]) is None


async def test_an_unknown_clarification_is_not_found(harness: Harness) -> None:
    body = (await start_pause(harness)).json()

    missing = await harness.respond(
        body["session_id"], "clarification-absent", "submission-1", {"form": "cancel"}
    )

    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "pending_run_not_found"


async def test_an_option_that_was_never_offered_is_refused(harness: Harness) -> None:
    body = (await start_pause(harness)).json()

    refused = await harness.respond(
        body["session_id"],
        body["clarification_id"],
        "submission-1",
        {"form": "options", "option_ids": ["mx"]},
    )

    assert refused.status_code == 422
    assert refused.json()["error"]["code"] == "unknown_option"


async def test_a_mixed_response_form_is_rejected_by_the_contract(harness: Harness) -> None:
    body = (await start_pause(harness)).json()

    refused = await harness.respond(
        body["session_id"],
        body["clarification_id"],
        "submission-1",
        {"form": "options", "option_ids": ["us"], "text": "and Canada"},
    )

    assert refused.status_code == 422


async def test_a_failed_continuation_leaves_the_submission_retryable(
    harness: Harness,
) -> None:
    """Graph failure commits nothing and does not let a different choice through."""

    from orchestration_core import AgentExecutionError

    body = (await start_pause(harness)).json()
    harness.invoker.outcomes = [
        AgentExecutionError("boom"),
        ConversationResult(generated_messages=(AIMessage(content="US spend rose", id="final-1"),)),
    ]
    payload = {"form": "options", "option_ids": ["us"]}
    failed = await harness.respond(
        body["session_id"], body["clarification_id"], "submission-1", payload
    )

    assert failed.status_code == 500
    assert await harness.store.load(body["session_id"]) is None

    other = await harness.respond(
        body["session_id"],
        body["clarification_id"],
        "submission-1",
        {"form": "options", "option_ids": ["eu"]},
    )
    assert other.status_code == 409

    retry = await harness.respond(
        body["session_id"], body["clarification_id"], "submission-1", payload
    )
    assert retry.status_code == 200


async def test_a_retry_recovers_a_run_that_finished_before_its_turn_was_committed(
    harness: Harness,
) -> None:
    """Crash recovery: the answer is read from the checkpoint, the graph is not resumed."""

    from orchestration_core import AgentExecutionError

    body = (await start_pause(harness)).json()
    harness.invoker.outcomes = [AgentExecutionError("died after finishing")]
    payload = {"form": "options", "option_ids": ["us"]}
    await harness.respond(body["session_id"], body["clarification_id"], "submission-1", payload)
    harness.invoker.recovery = ConversationResult(
        generated_messages=(AIMessage(content="US spend rose", id="final-1"),)
    )

    recovered = await harness.respond(
        body["session_id"], body["clarification_id"], "submission-1", payload
    )

    assert recovered.status_code == 200
    assert recovered.json()["message"] == "US spend rose"
    assert harness.invoker.recovered == ["run-1"]
    # Only the first attempt reached the graph; recovery ran nothing.
    assert len(harness.invoker.resumed) == 1


async def test_free_text_continues_the_same_run(harness: Harness) -> None:
    body = (await start_pause(harness)).json()
    harness.invoker.outcomes = [
        ConversationResult(generated_messages=(AIMessage(content="Canada spend rose", id="f"),))
    ]

    completed = await harness.respond(
        body["session_id"],
        body["clarification_id"],
        "submission-1",
        {"form": "free_text", "text": "Canada please"},
    )

    assert completed.status_code == 200
    snapshot = await harness.store.load(body["session_id"])
    assert snapshot is not None
    exchange = [message.content for message in snapshot.turns[-1].generated_messages]
    assert "Canada please" in exchange


async def test_an_ordinary_turn_still_returns_the_existing_response(harness: Harness) -> None:
    harness.executor.results = [{"routing_outcome": "answer", "answer": "Noted."}]

    response = await harness.chat("hello")

    assert response.status_code == 200
    assert "message" in response.json()
    assert "clarification_id" not in response.json()


async def test_history_of_a_first_turn_pause_has_no_committed_messages(
    harness: Harness,
) -> None:
    """A paused first request is a real session, with nothing committed yet."""

    body = (await start_pause(harness)).json()

    history = await harness.client.get(f"/api/sessions/{body['session_id']}/history")

    assert history.status_code == 200
    assert history.json()["messages"] == []
    pending = history.json()["pending_clarification"]
    assert pending["clarification_id"] == "clarification-task-1"
    assert pending["question"] == "Which market should task-1 cover?"
    assert [option["option_id"] for option in pending["options"]] == ["us", "eu"]


async def test_history_of_an_unknown_session_is_still_not_found(harness: Harness) -> None:
    missing = await harness.client.get(f"/api/sessions/{uuid4()}/history")

    assert missing.status_code == 404


async def test_a_paused_session_is_not_given_a_fabricated_session_document(
    harness: Harness,
) -> None:
    body = (await start_pause(harness)).json()
    await harness.client.get(f"/api/sessions/{body['session_id']}/history")

    assert await harness.store.load(body["session_id"]) is None


async def test_the_open_question_is_projected_apart_from_committed_messages(
    harness: Harness,
) -> None:
    """A second pause after a committed turn keeps the two clearly separated."""

    harness.executor.results = [{"routing_outcome": "answer", "answer": "Noted."}]
    first = await harness.chat("hello")
    session_id = first.json()["session_id"]
    harness.executor.results = [{"paused_run": paused("task-1", run_id="run-2")}]
    await harness.chat("How did spend perform?", session_id=session_id)

    history = (await harness.client.get(f"/api/sessions/{session_id}/history")).json()

    assert [message["content"] for message in history["messages"]] == ["hello", "Noted."]
    assert history["pending_clarification"]["clarification_id"] == "clarification-task-1"


async def test_a_cancelled_run_renders_its_exchange_after_closure(harness: Harness) -> None:
    body = (await start_pause(harness)).json()
    harness.invoker.outcomes = [paused("task-2")]
    await harness.respond(
        body["session_id"],
        body["clarification_id"],
        "submission-1",
        {"form": "options", "option_ids": ["us"]},
    )
    await harness.respond(
        body["session_id"], "clarification-task-2", "submission-2", {"form": "cancel"}
    )

    history = (await harness.client.get(f"/api/sessions/{body['session_id']}/history")).json()

    assert [message["content"] for message in history["messages"]] == [
        "How did spend perform?",
        "Which market should task-1 cover?",
        "United States",
        "Which market should task-2 cover?",
    ]
    assert history["pending_clarification"] is None


async def test_an_expired_pause_releases_the_session_for_ordinary_chat(
    harness: Harness,
) -> None:
    """Otherwise an abandoned pause holds the slot and every later message is a conflict."""

    body = (await start_pause(harness)).json()
    run = await harness.store.active_pending_run(body["session_id"])
    assert run is not None
    expire_pending_run(harness.store, body["session_id"], run)
    harness.executor.results = [{"routing_outcome": "answer", "answer": "Starting over."}]

    followup = await harness.chat("Just the US then", session_id=body["session_id"])

    assert followup.status_code == 200
    audit = await harness.store.clarification_audit(body["session_id"])
    assert [entry.event for entry in audit] == ["published", "expired"]
    snapshot = await harness.store.load(body["session_id"])
    assert snapshot is not None
    # The lapsed question was committed as its own closed turn before the new one.
    assert snapshot.revision == 2


async def test_history_closes_a_lapsed_pause_it_finds(harness: Harness) -> None:
    body = (await start_pause(harness)).json()
    run = await harness.store.active_pending_run(body["session_id"])
    assert run is not None
    expire_pending_run(harness.store, body["session_id"], run)

    history = (await harness.client.get(f"/api/sessions/{body['session_id']}/history")).json()

    assert history["pending_clarification"] is None
    assert [message["content"] for message in history["messages"]] == [
        "How did spend perform?",
        "Which market should task-1 cover?",
    ]


async def test_a_live_pause_still_blocks_ordinary_chat(harness: Harness) -> None:
    body = (await start_pause(harness)).json()

    conflict = await harness.chat("something else", session_id=body["session_id"])

    assert conflict.status_code == 409
    assert await harness.store.active_pending_run(body["session_id"]) is not None


async def test_sanitized_failures_carry_no_internal_detail(harness: Harness) -> None:
    body = (await start_pause(harness)).json()
    harness.invoker.outcomes = [
        ContinuationNotResumableError("stale_producer", "producer snapshot moved on")
    ]

    refused = await harness.respond(
        body["session_id"],
        body["clarification_id"],
        "submission-1",
        {"form": "options", "option_ids": ["us"]},
    )

    rendered = refused.text
    assert refused.json()["error"]["code"] == "restart_required:stale_producer"
    for leaked in ("run-1", "snapshot-task-1", "definition-1", "Traceback", "scripted_producer"):
        assert leaked not in rendered
