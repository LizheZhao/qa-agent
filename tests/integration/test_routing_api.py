"""Integration tests for router-owned chat and delegated specialist turns."""

import asyncio
from pathlib import Path

import pytest
from enterprise_llm import GatewayTransportError
from httpx import ASGITransport, AsyncClient
from langchain_core.messages import HumanMessage, SystemMessage
from orchestration_core import ExecutionStatus, SpanKind

from agentic_orchestration.main import create_app
from agentic_orchestration.sessions import MemorySessionStore
from agentic_orchestration.sessions.contracts import (
    CommitTurn,
    SessionConflictError,
    SessionSnapshot,
    SessionStorageError,
)
from tests.fakes import ScriptedChatModel

ROOT = Path(__file__).parents[2]


class DelayedCommitReturnStore(MemorySessionStore):
    def __init__(self) -> None:
        super().__init__()
        self.committed = asyncio.Event()
        self.session_id: str | None = None

    async def commit_turn(self, command: CommitTurn) -> SessionSnapshot:
        snapshot = await super().commit_turn(command)
        self.session_id = command.session_id
        self.committed.set()
        await asyncio.sleep(0.05)
        return snapshot


def configure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEPLOYMENT_PATH", str(ROOT / "deployments/local.yaml"))
    monkeypatch.setenv("DEPLOYMENT_SCHEMA_PATH", str(ROOT / "schemas/deployment.schema.json"))


def queue_route(
    fake: ScriptedChatModel,
    outcome: str,
    *,
    response: str | None = None,
    target_agent: str | None = None,
) -> ScriptedChatModel:
    return fake.queue_tool_call(
        "RouteDecision",
        {
            "outcome": outcome,
            "reason_code": outcome,
            "response": response,
            "target_agent": target_agent,
        },
    )


@pytest.mark.integration
async def test_startup_chat_multiturn_history_isolation_and_tool_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure(monkeypatch)
    fake = ScriptedChatModel()
    queue_route(fake, "answer", response="Hello")
    queue_route(fake, "answer", response="I remember")
    queue_route(fake, "delegate", target_agent="marketing_science")
    fake.queue_tool_call("calculate_sum", {"a": 17, "b": 25}, call_id="api-sum")
    fake.queue_text("The sum is 42")
    queue_route(fake, "answer", response="Separate")
    app = create_app(
        model_factory=lambda settings: fake,
        session_store_factory=lambda settings: MemorySessionStore(),
    )
    async with (
        app.router.lifespan_context(app),
        AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client,
    ):
        assert app.state.entry_agent == "router"
        blank = await client.post("/api/chat", json={"message": "   "})
        first = await client.post("/api/chat", json={"message": "My name is Morgan."})
        session_id = first.json()["session_id"]
        second = await client.post(
            "/api/chat", json={"session_id": session_id, "message": "What is my name?"}
        )
        tool = await client.post(
            "/api/chat",
            json={"session_id": session_id, "message": "Add 17 and 25."},
        )
        separate = await client.post("/api/chat", json={"message": "Hello"})
        history = await client.get(f"/api/sessions/{session_id}/history")
        unknown = await client.get("/api/sessions/unknown/history")
        tool_spans = await app.state.execution_store.spans(tool.json()["trace_id"])

    assert blank.status_code == 422
    assert (
        first.status_code == second.status_code == tool.status_code == separate.status_code == 200
    )
    assert session_id
    assert any(
        isinstance(message, HumanMessage) and message.content == "My name is Morgan."
        for message in fake.calls[1]["messages"]
    )
    assert tool.json()["tool_calls"] == [
        {
            "name": "calculate_sum",
            "arguments": {"a": 17, "b": 25},
            "status": "completed",
        }
    ]
    assert [message["content"] for message in history.json()["messages"]] == [
        "My name is Morgan.",
        "Hello",
        "What is my name?",
        "I remember",
        "Add 17 and 25.",
        "The sum is 42",
    ]
    assert [message["role"] for message in history.json()["messages"]] == [
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert history.json()["storage"] == "process_memory"
    assert history.json()["durable"] is False
    assert all(message.content != "My name is Morgan." for message in fake.calls[5]["messages"])
    assert unknown.status_code == 404
    node_spans = [span for span in tool_spans if span.kind == SpanKind.NODE]
    model_spans = [span for span in tool_spans if span.kind == SpanKind.MODEL]
    tool_call_spans = [span for span in tool_spans if span.kind == SpanKind.TOOL]
    marketing_model_nodes = [
        span
        for span in node_spans
        if span.agent_id == "marketing_science" and span.definition_node_id == "model"
    ]
    assert [(span.superstep, span.iteration) for span in marketing_model_nodes] == [
        (1, 1),
        (3, 2),
    ]
    assert {span.agent_id for span in model_spans} == {"router", "marketing_science"}
    assert len(tool_call_spans) == 1
    assert tool_call_spans[0].attributes["tool_id"] == "calculate_sum"
    assert all(span.graph_definition_id for span in node_spans)


@pytest.mark.integration
async def test_gateway_failure_is_sanitized_and_does_not_mutate_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure(monkeypatch)
    secret = "do-not-return-this-token"
    fake = ScriptedChatModel()
    queue_route(fake, "answer", response="saved")
    queue_route(fake, "delegate", target_agent="marketing_science")
    fake.queue_error(GatewayTransportError(f"transport exposed {secret}"))
    app = create_app(
        model_factory=lambda settings: fake,
        session_store_factory=lambda settings: MemorySessionStore(),
    )
    async with (
        app.router.lifespan_context(app),
        AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client,
    ):
        first = await client.post("/api/chat", json={"message": "one"})
        session_id = first.json()["session_id"]
        before = await client.get(f"/api/sessions/{session_id}/history")
        failed = await client.post("/api/chat", json={"session_id": session_id, "message": secret})
        after = await client.get(f"/api/sessions/{session_id}/history")
        traces = await app.state.execution_store.traces()
        failure_spans = await app.state.execution_store.spans(traces[1].trace_id)

    assert first.status_code == 200
    assert failed.status_code == 502
    failed_body = failed.json()
    assert failed_body["error"] == {
        "code": "gateway_error",
        "message": "Enterprise gateway request failed",
    }
    assert failed_body["session_id"] == session_id
    assert failed_body["trace_id"] == traces[1].trace_id
    assert secret not in failed.text
    assert after.json() == before.json()
    assert [trace.status for trace in traces] == [
        ExecutionStatus.COMPLETED,
        ExecutionStatus.FAILED,
    ]
    assert traces[1].committed_turn_number is None
    persisted_failure_details = [
        span.failure_detail.model_dump_json()
        for span in failure_spans
        if span.failure_detail is not None
    ]
    assert persisted_failure_details
    assert all(secret not in detail for detail in persisted_failure_details)
    assert any("[REDACTED EXTERNAL ERROR]" in detail for detail in persisted_failure_details)


@pytest.mark.integration
async def test_same_session_concurrent_requests_do_not_lose_turns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure(monkeypatch)
    fake = ScriptedChatModel()
    queue_route(fake, "answer", response="initial answer")
    queue_route(fake, "answer", response="first answer")
    queue_route(fake, "answer", response="second answer")
    app = create_app(
        model_factory=lambda settings: fake,
        session_store_factory=lambda settings: MemorySessionStore(),
    )
    async with (
        app.router.lifespan_context(app),
        AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client,
    ):
        initial = await client.post("/api/chat", json={"message": "initial"})
        session_id = initial.json()["session_id"]
        responses = await asyncio.gather(
            client.post("/api/chat", json={"session_id": session_id, "message": "first"}),
            client.post("/api/chat", json={"session_id": session_id, "message": "second"}),
        )
        history = await client.get(f"/api/sessions/{session_id}/history")

    assert all(response.status_code == 200 for response in responses)
    assert len(history.json()["messages"]) == 6
    assert (
        len(
            [
                message
                for message in fake.calls[1]["messages"]
                if not isinstance(message, SystemMessage)
            ]
        )
        == 3
    )


@pytest.mark.integration
async def test_delegated_turn_persists_affinity_for_next_router_decision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure(monkeypatch)
    fake = ScriptedChatModel()
    queue_route(fake, "delegate", target_agent="marketing_science")
    fake.queue_text("Which period?")
    queue_route(fake, "delegate", target_agent="marketing_science")
    fake.queue_text("Here is the scoped answer.")
    store = MemorySessionStore()
    app = create_app(
        model_factory=lambda settings: fake,
        session_store_factory=lambda settings: store,
    )

    async with (
        app.router.lifespan_context(app),
        AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client,
    ):
        first = await client.post("/api/chat", json={"message": "Explain an experiment."})
        session_id = first.json()["session_id"]
        second = await client.post(
            "/api/chat", json={"session_id": session_id, "message": "Last quarter."}
        )

    snapshot = await store.load(session_id)
    assert first.status_code == second.status_code == 200
    assert snapshot is not None
    assert snapshot.agent_id == "router"
    assert snapshot.active_agent_id == "marketing_science"
    assert [turn.selected_agent_id for turn in snapshot.turns] == [
        "marketing_science",
        "marketing_science",
    ]
    router_system = fake.calls[2]["messages"][0]
    assert isinstance(router_system, SystemMessage)
    assert "previous selected child was 'marketing_science'" in str(router_system.content)


@pytest.mark.integration
async def test_delegated_trace_preserves_router_to_child_hierarchy_and_sequence_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure(monkeypatch)
    fake = ScriptedChatModel()
    queue_route(fake, "delegate", target_agent="marketing_science")
    fake.queue_text("A specialist answer.")
    app = create_app(
        model_factory=lambda settings: fake,
        session_store_factory=lambda settings: MemorySessionStore(),
    )

    async with (
        app.router.lifespan_context(app),
        AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client,
    ):
        response = await client.post("/api/chat", json={"message": "Explain lift."})
        body = response.json()
        trace = await app.state.execution_store.get_trace(body["trace_id"])
        spans = await app.state.execution_store.spans(body["trace_id"])
        router_definition_id = next(
            span.graph_definition_id
            for span in spans
            if span.kind == SpanKind.GRAPH and span.agent_id == "router"
        )
        assert router_definition_id is not None
        router_definition = await app.state.execution_store.graph_definition(router_definition_id)

    assert response.status_code == 200
    assert trace is not None
    assert trace.status == ExecutionStatus.COMPLETED
    assert trace.committed_turn_number == 1
    assert trace.session_id == body["session_id"]
    assert body["trace_id"] == trace.trace_id
    assert router_definition is not None
    root = next(span for span in spans if span.kind == SpanKind.TURN)
    router_graph = next(
        span for span in spans if span.kind == SpanKind.GRAPH and span.agent_id == "router"
    )
    child = next(span for span in spans if span.kind == SpanKind.CHILD_AGENT)
    delegate_node = next(
        span
        for span in spans
        if span.kind == SpanKind.NODE
        and span.agent_id == "router"
        and span.definition_node_id == "delegate"
    )
    child_graph = next(
        span
        for span in spans
        if span.kind == SpanKind.GRAPH and span.agent_id == "marketing_science"
    )
    router_model = next(
        span for span in spans if span.kind == SpanKind.MODEL and span.agent_id == "router"
    )
    router_node_definition = next(
        node for node in router_definition.nodes if node.node_id == router_model.definition_node_id
    )
    assert router_graph.parent_span_id == root.span_id
    assert delegate_node.parent_span_id == router_graph.span_id
    assert child.parent_span_id == delegate_node.span_id
    assert child_graph.parent_span_id == child.span_id
    assert router_model.prompt_template_id == "router.route_decision.system"
    assert router_node_definition.prompt_template is not None
    assert router_node_definition.prompt_template.template_id == router_model.prompt_template_id
    lifecycle_sequences = sorted(
        [trace.sequence_started, trace.sequence_completed]
        + [span.sequence_started for span in spans]
        + [span.sequence_completed for span in spans]
    )
    assert lifecycle_sequences == list(range(1, len(lifecycle_sequences) + 1))


@pytest.mark.integration
async def test_cancellation_after_turn_commit_finalizes_trace_as_completed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure(monkeypatch)
    fake = ScriptedChatModel()
    queue_route(fake, "answer", response="Committed answer")
    store = DelayedCommitReturnStore()
    app = create_app(
        model_factory=lambda settings: fake,
        session_store_factory=lambda settings: store,
    )

    async with (
        app.router.lifespan_context(app),
        AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client,
    ):
        request = asyncio.create_task(client.post("/api/chat", json={"message": "Commit me"}))
        await store.committed.wait()
        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request
        assert store.session_id is not None
        snapshot = await store.load(store.session_id)
        traces = await app.state.execution_store.traces()

    assert snapshot is not None
    assert snapshot.revision == 1
    assert len(traces) == 1
    assert traces[0].status == ExecutionStatus.COMPLETED
    assert traces[0].committed_turn_number == 1


class FailingCommitStore(MemorySessionStore):
    def __init__(self, error: Exception) -> None:
        super().__init__()
        self._error = error

    async def commit_turn(self, command: CommitTurn) -> SessionSnapshot:
        raise self._error


@pytest.mark.parametrize(
    ("error", "expected_status"),
    [
        (SessionConflictError("session changed"), 409),
        (SessionStorageError("storage down"), 503),
    ],
)
@pytest.mark.integration
async def test_commit_failure_finalizes_the_trace(
    monkeypatch: pytest.MonkeyPatch, error: Exception, expected_status: int
) -> None:
    configure(monkeypatch)
    fake = ScriptedChatModel()
    queue_route(fake, "answer", response="Never committed")
    app = create_app(
        model_factory=lambda settings: fake,
        session_store_factory=lambda settings: FailingCommitStore(error),
    )

    async with (
        app.router.lifespan_context(app),
        AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client,
    ):
        response = await client.post("/api/chat", json={"message": "Commit me"})
        traces = await app.state.execution_store.traces()

    assert response.status_code == expected_status
    assert len(traces) == 1
    assert traces[0].status == ExecutionStatus.FAILED
    assert traces[0].committed_turn_number is None
