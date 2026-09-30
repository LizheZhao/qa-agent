"""Explicitly observed, retained MongoDB execution trace test."""

import os

import pytest
from enterprise_llm import GatewayTransportError
from httpx import ASGITransport, AsyncClient
from orchestration_core import ExecutionStatus, SpanKind

from agentic_orchestration.main import create_app
from tests.fakes import ScriptedChatModel


@pytest.mark.integration
@pytest.mark.mongo
@pytest.mark.skipif(
    os.getenv("RUN_MONGO_OBSERVABILITY_TESTS") != "1",
    reason="set RUN_MONGO_OBSERVABILITY_TESTS=1 to write and inspect a retained Mongo trace",
)
async def test_observe_complete_router_marketing_science_turn_in_mongodb() -> None:
    model = (
        ScriptedChatModel()
        .queue_tool_call(
            "RouteDecision",
            {
                "outcome": "delegate",
                "reason_code": "marketing_science",
                "response": None,
                "target_agent": "marketing_science",
            },
        )
        .queue_tool_call("calculate_sum", {"a": 17, "b": 25}, call_id="observed-sum")
        .queue_text("The combined value is 42.")
    )
    app = create_app(model_factory=lambda settings: model)

    async with (
        app.router.lifespan_context(app),
        AsyncClient(transport=ASGITransport(app=app), base_url="http://observed") as client,
    ):
        response = await client.post(
            "/api/chat",
            json={"message": "Use the marketing-science specialist to add 17 and 25."},
        )
        response.raise_for_status()
        body = response.json()
        trace = await app.state.execution_store.get_trace(body["trace_id"])
        spans = await app.state.execution_store.spans(body["trace_id"])
        definition_ids = sorted(
            {span.graph_definition_id for span in spans if span.graph_definition_id is not None}
        )
        definitions = [
            await app.state.execution_store.graph_definition(definition_id)
            for definition_id in definition_ids
        ]
        sessions = await client.get("/api/diagnostics/sessions")
        turns = await client.get(f"/api/diagnostics/sessions/{body['session_id']}/turns")
        attempts = await client.get(f"/api/diagnostics/sessions/{body['session_id']}/traces")
        diagnostic_trace = await client.get(f"/api/diagnostics/traces/{body['trace_id']}")
        diagnostic_spans = await client.get(f"/api/diagnostics/traces/{body['trace_id']}/spans")

    assert trace is not None
    print("CHAT RESPONSE")
    print(response.text)
    print("EXECUTION TRACE")
    print(trace.model_dump_json(indent=2))
    print("EXECUTION SPANS")
    for span in spans:
        print(span.model_dump_json(indent=2))
    print("GRAPH DEFINITIONS")
    for definition in definitions:
        assert definition is not None
        print(definition.model_dump_json(indent=2))
    assert sessions.json()["items"][0]["session_id"] == body["session_id"]
    assert turns.json()["items"][0]["trace_id"] == body["trace_id"]
    assert attempts.json()["items"][0]["trace_id"] == body["trace_id"]
    assert attempts.json()["items"][0]["input_preview"].startswith("Use the marketing-science")
    assert diagnostic_trace.json()["trace_id"] == body["trace_id"]
    assert diagnostic_spans.json()


@pytest.mark.integration
@pytest.mark.mongo
@pytest.mark.skipif(
    os.getenv("RUN_MONGO_OBSERVABILITY_TESTS") != "1",
    reason="set RUN_MONGO_OBSERVABILITY_TESTS=1 to write and inspect a retained Mongo trace",
)
async def test_observe_failed_span_detail_in_mongodb() -> None:
    secret = "provider-response-content-must-not-persist"
    app = create_app(
        model_factory=lambda settings: ScriptedChatModel().queue_error(
            GatewayTransportError(secret)
        )
    )

    async with (
        app.router.lifespan_context(app),
        AsyncClient(transport=ASGITransport(app=app), base_url="http://observed") as client,
    ):
        response = await client.post(
            "/api/chat",
            json={"message": "Retain a failed diagnostic trace."},
        )
        trace_id = response.json()["trace_id"]
        spans = await app.state.execution_store.spans(trace_id)
        failed_model = next(
            span
            for span in spans
            if span.kind == SpanKind.MODEL and span.status == ExecutionStatus.FAILED
        )
        detail = await client.get(
            f"/api/diagnostics/traces/{trace_id}/spans/{failed_model.span_id}"
        )

    assert response.status_code == 502
    assert failed_model.schema_version == 4
    assert failed_model.failure_detail is not None
    assert failed_model.failure_detail.redacted is True
    assert secret not in failed_model.model_dump_json()
    assert detail.json()["failure_detail"]["fingerprint"]
    assert secret not in detail.text


@pytest.mark.integration
@pytest.mark.mongo
@pytest.mark.live
@pytest.mark.skipif(
    os.getenv("RUN_MONGO_OBSERVABILITY_TESTS") != "1" or os.getenv("RUN_LIVE_GATEWAY_TESTS") != "1",
    reason="set both live and Mongo opt-in flags to retain a live gateway trace",
)
async def test_observe_live_gateway_router_marketing_turn_in_mongodb() -> None:
    app = create_app()

    async with (
        app.router.lifespan_context(app),
        AsyncClient(transport=ASGITransport(app=app), base_url="http://live-observed") as client,
    ):
        response = await client.post(
            "/api/chat",
            json={
                "message": (
                    "For a marketing experiment, treatment produced 17 conversions and control "
                    "produced 25 conversions. Delegate this measurement request to the "
                    "marketing-science specialist, have the specialist call calculate_sum with "
                    "a=17 and b=25, and report the combined observed conversions."
                )
            },
        )
        response.raise_for_status()
        body = response.json()
        trace = await app.state.execution_store.get_trace(body["trace_id"])
        spans = await app.state.execution_store.spans(body["trace_id"])

    assert trace is not None
    assert trace.status == ExecutionStatus.COMPLETED
    assert any(span.kind == SpanKind.CHILD_AGENT for span in spans)
    assert any(span.kind == SpanKind.TOOL for span in spans)
    assert body["tool_calls"] == [
        {
            "name": "calculate_sum",
            "arguments": {"a": 17, "b": 25},
            "status": "completed",
        }
    ]
    print("LIVE CHAT RESPONSE")
    print(response.text)
    print("LIVE EXECUTION TRACE")
    print(trace.model_dump_json(indent=2))
    print("LIVE SPAN SUMMARY")
    for span in spans:
        print(
            span.sequence_started,
            span.sequence_completed,
            span.kind,
            span.agent_id,
            span.definition_node_id,
            span.status,
        )
