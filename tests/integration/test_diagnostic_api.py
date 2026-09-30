"""In-process coverage for the diagnostic read workflow."""

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest
from enterprise_llm import GatewayTransportError
from httpx import ASGITransport, AsyncClient

from agentic_orchestration.diagnostics.store import DiagnosticReadError
from agentic_orchestration.main import create_app
from agentic_orchestration.sessions import MemorySessionStore
from tests.fakes import ScriptedChatModel

ROOT = Path(__file__).parents[2]


def configure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEPLOYMENT_PATH", str(ROOT / "deployments/local.yaml"))
    monkeypatch.setenv("DEPLOYMENT_SCHEMA_PATH", str(ROOT / "schemas/deployment.schema.json"))


def queue_answer(model: ScriptedChatModel, response: str) -> None:
    model.queue_tool_call(
        "RouteDecision",
        {
            "outcome": "answer",
            "reason_code": "general_marketing_science",
            "response": response,
            "target_agent": None,
        },
    )


class FailingDiagnosticReadStore:
    async def sessions(self, *, limit: int, cursor: str | None) -> None:
        del limit, cursor
        raise DiagnosticReadError("private database detail")


@pytest.mark.integration
async def test_diagnostic_session_trace_and_span_hydration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure(monkeypatch)
    model = ScriptedChatModel()
    queue_answer(model, "Incrementality estimates causal lift.")
    app = create_app(
        model_factory=lambda settings: model,
        session_store_factory=lambda settings: MemorySessionStore(),
    )

    async with (
        app.router.lifespan_context(app),
        AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client,
    ):
        chat = await client.post(
            "/api/chat",
            headers={"X-Orchestration-Origin": "diagnostic_ui"},
            json={"message": "What is incrementality?"},
        )
        body = chat.json()
        sessions = await client.get("/api/diagnostics/sessions")
        session = await client.get(f"/api/diagnostics/sessions/{body['session_id']}")
        turns = await client.get(f"/api/diagnostics/sessions/{body['session_id']}/turns")
        attempts = await client.get(f"/api/diagnostics/sessions/{body['session_id']}/traces")
        trace = await client.get(f"/api/diagnostics/traces/{body['trace_id']}")
        spans = await client.get(f"/api/diagnostics/traces/{body['trace_id']}/spans")
        span_id = spans.json()[0]["span_id"]
        span = await client.get(f"/api/diagnostics/traces/{body['trace_id']}/spans/{span_id}")
        deployment = await client.get("/api/diagnostics/deployment")
        definition_id = deployment.json()["agents"][0]["graph_definition_id"]
        definition = await client.get(f"/api/diagnostics/graph-definitions/{definition_id}")

    assert chat.status_code == 200
    assert sessions.json()["items"][0]["session_id"] == body["session_id"]
    assert sessions.json()["items"][0]["display_title"] == "What is incrementality?"
    assert session.json()["revision"] == 1
    assert session.json()["display_title"] == "What is incrementality?"
    assert turns.json()["items"][0]["response"]["content"] == body["message"]["content"]
    assert attempts.json()["items"][0]["trace_id"] == body["trace_id"]
    assert attempts.json()["items"][0]["input_preview"] == "What is incrementality?"
    assert trace.json()["request_origin"] == "diagnostic_ui"
    assert spans.status_code == 200
    assert "output" not in spans.json()[0]
    assert span.json()["span_id"] == span_id
    assert deployment.json()["entry_agent"] == "router"
    assert {agent["agent_id"] for agent in deployment.json()["agents"]} == {
        "router",
        "marketing_science",
        "ask_genome",
    }
    assert definition.status_code == 200
    assert definition.json()["definition_id"] == definition_id


@pytest.mark.integration
async def test_session_pages_are_ten_items_with_stable_cursor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure(monkeypatch)
    model = ScriptedChatModel()
    for index in range(11):
        queue_answer(model, f"Answer {index}")
    store = MemorySessionStore()
    app = create_app(
        model_factory=lambda settings: model,
        session_store_factory=lambda settings: store,
    )

    async with (
        app.router.lifespan_context(app),
        AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client,
    ):
        for index in range(11):
            response = await client.post("/api/chat", json={"message": f"Question {index}"})
            assert response.status_code == 200
        identical_update_time = datetime(2026, 8, 18, tzinfo=UTC)
        store._sessions = {
            session_id: replace(snapshot, updated_at=identical_update_time)
            for session_id, snapshot in store._sessions.items()
        }
        first = await client.get("/api/diagnostics/sessions")
        second = await client.get(
            "/api/diagnostics/sessions",
            params={"cursor": first.json()["next_cursor"]},
        )

    assert len(first.json()["items"]) == 10
    assert first.json()["next_cursor"] is not None
    assert len(second.json()["items"]) == 1
    assert second.json()["next_cursor"] is None
    assert {item["session_id"] for item in first.json()["items"]}.isdisjoint(
        {item["session_id"] for item in second.json()["items"]}
    )


@pytest.mark.integration
async def test_turn_pages_preserve_cursor_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure(monkeypatch)
    model = ScriptedChatModel()
    for index in range(26):
        queue_answer(model, f"Answer {index + 1}")
    app = create_app(
        model_factory=lambda settings: model,
        session_store_factory=lambda settings: MemorySessionStore(),
    )

    async with (
        app.router.lifespan_context(app),
        AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client,
    ):
        session_id = None
        for index in range(26):
            payload = {"message": f"Question {index + 1}"}
            if session_id is not None:
                payload["session_id"] = session_id
            response = await client.post("/api/chat", json=payload)
            assert response.status_code == 200
            session_id = response.json()["session_id"]
        first = await client.get(f"/api/diagnostics/sessions/{session_id}/turns")
        second = await client.get(
            f"/api/diagnostics/sessions/{session_id}/turns",
            params={"cursor": first.json()["next_cursor"]},
        )

    assert [item["turn_number"] for item in first.json()["items"]] == list(range(26, 1, -1))
    assert [item["turn_number"] for item in second.json()["items"]] == [1]
    assert second.json()["next_cursor"] is None


@pytest.mark.integration
async def test_failed_first_attempt_is_discoverable_without_a_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure(monkeypatch)
    model = ScriptedChatModel().queue_error(GatewayTransportError("private transport detail"))
    app = create_app(
        model_factory=lambda settings: model,
        session_store_factory=lambda settings: MemorySessionStore(),
    )

    async with (
        app.router.lifespan_context(app),
        AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client,
    ):
        failed = await client.post(
            "/api/chat",
            headers={"X-Orchestration-Origin": "diagnostic_ui"},
            json={"message": "Run a failing diagnostic turn"},
        )
        body = failed.json()
        recent = await client.get("/api/diagnostics/traces")
        missing_session = await client.get(f"/api/diagnostics/sessions/{body['session_id']}")
        trace = await client.get(f"/api/diagnostics/traces/{body['trace_id']}")
        spans = await client.get(f"/api/diagnostics/traces/{body['trace_id']}/spans")
        failed_span = next(
            item for item in spans.json() if item["status"] == "failed" and item["kind"] == "model"
        )
        span = await client.get(
            f"/api/diagnostics/traces/{body['trace_id']}/spans/{failed_span['span_id']}"
        )
        invalid_cursor = await client.get("/api/diagnostics/traces?cursor=not-a-cursor")

    assert failed.status_code == 502
    assert body["error"]["code"] == "gateway_error"
    assert recent.json()["items"][0]["trace_id"] == body["trace_id"]
    assert missing_session.status_code == 404
    assert trace.json()["status"] == "failed"
    assert trace.json()["committed_turn_number"] is None
    assert "failure_detail" not in failed_span
    assert span.json()["failure_detail"]["exception_chain"][0]["exception_type"]
    assert "private transport detail" not in span.text
    assert invalid_cursor.status_code == 422


@pytest.mark.integration
async def test_session_attempts_include_failed_retry_of_same_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure(monkeypatch)
    model = ScriptedChatModel()
    queue_answer(model, "First answer")
    model.queue_error(GatewayTransportError("private transport detail"))
    queue_answer(model, "Retried answer")
    app = create_app(
        model_factory=lambda settings: model,
        session_store_factory=lambda settings: MemorySessionStore(),
    )

    async with (
        app.router.lifespan_context(app),
        AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client,
    ):
        first = await client.post("/api/chat", json={"message": "First turn"})
        session_id = first.json()["session_id"]
        failed = await client.post(
            "/api/chat", json={"session_id": session_id, "message": "Second turn"}
        )
        retried = await client.post(
            "/api/chat", json={"session_id": session_id, "message": "Second turn retry"}
        )
        attempts = await client.get(f"/api/diagnostics/sessions/{session_id}/traces")

    assert first.status_code == 200
    assert failed.status_code == 502
    assert retried.status_code == 200
    assert [item["attempted_turn_number"] for item in attempts.json()["items"]] == [2, 2, 1]
    assert [item["status"] for item in attempts.json()["items"]] == [
        "completed",
        "failed",
        "completed",
    ]


@pytest.mark.integration
async def test_diagnostic_storage_failure_is_sanitized_as_503(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure(monkeypatch)
    app = create_app(
        model_factory=lambda settings: ScriptedChatModel(),
        session_store_factory=lambda settings: MemorySessionStore(),
    )

    async with (
        app.router.lifespan_context(app),
        AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client,
    ):
        app.state.diagnostic_read_store = FailingDiagnosticReadStore()
        response = await client.get("/api/diagnostics/sessions")

    assert response.status_code == 503
    assert response.json() == {"detail": "Diagnostic storage is unavailable"}
    assert "private database detail" not in response.text


@pytest.mark.integration
async def test_unknown_descriptive_origin_is_recorded_without_rejecting_chat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure(monkeypatch)
    model = ScriptedChatModel()
    queue_answer(model, "Answer")
    app = create_app(
        model_factory=lambda settings: model,
        session_store_factory=lambda settings: MemorySessionStore(),
    )

    async with (
        app.router.lifespan_context(app),
        AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client,
    ):
        response = await client.post(
            "/api/chat",
            headers={"X-Orchestration-Origin": "future_internal_harness"},
            json={"message": "Question"},
        )
        trace = await client.get(f"/api/diagnostics/traces/{response.json()['trace_id']}")

    assert response.status_code == 200
    assert trace.json()["request_origin"] == "unknown"
