"""Validation and serialization tests for durable observability contracts."""

from datetime import UTC, datetime, timedelta

import pytest
from orchestration_core import (
    CapturedContent,
    ContentCaptureMode,
    EventKind,
    ExceptionObservation,
    ExecutionEvent,
    ExecutionSpan,
    ExecutionStatus,
    ExecutionTrace,
    FailureDetail,
    GraphDefinition,
    GraphEdgeDefinition,
    GraphEdgeKind,
    GraphNodeDefinition,
    GraphNodeKind,
    SanitizedExecutionError,
    SpanKind,
    TracebackFrame,
)
from pydantic import ValidationError

NOW = datetime(2026, 8, 11, 12, 0, tzinfo=UTC)
HASH = "a" * 64


def graph_definition() -> GraphDefinition:
    return GraphDefinition(
        definition_id="router:0.1.0:abc",
        definition_hash=HASH,
        agent_id="router",
        agent_version="0.1.0",
        entrypoint="router_agent.graph:create_graph",
        nodes=(
            GraphNodeDefinition(
                node_id="__start__", display_name="START", kind=GraphNodeKind.START
            ),
            GraphNodeDefinition(node_id="decide", display_name="Decide", kind=GraphNodeKind.ROUTER),
            GraphNodeDefinition(node_id="__end__", display_name="END", kind=GraphNodeKind.END),
        ),
        edges=(
            GraphEdgeDefinition(source_node_id="__start__", target_node_id="decide"),
            GraphEdgeDefinition(
                source_node_id="decide",
                target_node_id="__end__",
                kind=GraphEdgeKind.CONDITIONAL,
                label="answer",
            ),
        ),
        created_at=NOW,
    )


def test_graph_definition_round_trips_through_json() -> None:
    definition = graph_definition()

    assert GraphDefinition.model_validate_json(definition.model_dump_json()) == definition


@pytest.mark.parametrize(
    ("nodes", "edges", "message"),
    [
        (
            (
                GraphNodeDefinition(node_id="same", display_name="START", kind=GraphNodeKind.START),
                GraphNodeDefinition(node_id="same", display_name="END", kind=GraphNodeKind.END),
            ),
            (),
            "node IDs must be unique",
        ),
        (
            (
                GraphNodeDefinition(
                    node_id="start", display_name="START", kind=GraphNodeKind.START
                ),
                GraphNodeDefinition(node_id="end", display_name="END", kind=GraphNodeKind.END),
            ),
            (GraphEdgeDefinition(source_node_id="start", target_node_id="missing"),),
            "edges must reference known nodes",
        ),
    ],
)
def test_graph_definition_rejects_invalid_topology(
    nodes: tuple[GraphNodeDefinition, ...],
    edges: tuple[GraphEdgeDefinition, ...],
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        GraphDefinition(
            definition_id="invalid",
            definition_hash=HASH,
            agent_id="router",
            agent_version="0.1.0",
            entrypoint="entrypoint",
            nodes=nodes,
            edges=edges,
            created_at=NOW,
        )


def test_capture_modes_are_unambiguous() -> None:
    inline = CapturedContent(
        mode=ContentCaptureMode.INLINE, content_type="application/json", value={"x": 1}
    )
    reference = CapturedContent(
        mode=ContentCaptureMode.REFERENCE,
        content_type="application/x.dataframe",
        reference_id="artifact-1",
        byte_size=1024,
    )
    omitted = CapturedContent(
        mode=ContentCaptureMode.OMITTED, content_type="text/plain", reason="secret"
    )

    assert inline.mode == ContentCaptureMode.INLINE
    assert reference.reference_id == "artifact-1"
    assert omitted.reason == "secret"

    with pytest.raises(ValidationError, match="requires only a reference_id"):
        CapturedContent(
            mode=ContentCaptureMode.REFERENCE, content_type="text/plain", value="leaked"
        )


def test_terminal_span_requires_ordering_timing_and_failed_error() -> None:
    with pytest.raises(ValidationError, match="sequence_completed"):
        ExecutionSpan(
            trace_id="trace-1",
            span_id="span-1",
            session_id="session-1",
            attempted_turn_number=1,
            agent_id="router",
            agent_version="0.1.0",
            kind=SpanKind.NODE,
            status=ExecutionStatus.COMPLETED,
            sequence_started=1,
            snapshot_sequence=1,
            started_at=NOW,
            completed_at=NOW + timedelta(milliseconds=5),
            duration_ms=5,
        )

    error = SanitizedExecutionError(code="model_timeout", message="Model request timed out")
    failure_detail = FailureDetail(
        fingerprint=HASH,
        exception_chain=(
            ExceptionObservation(
                exception_type="TimeoutError",
                exception_module="builtins",
                message="request timed out",
                frames=(TracebackFrame(filename="worker.py", function="run", line_number=10),),
            ),
        ),
    )
    span = ExecutionSpan(
        trace_id="trace-1",
        span_id="span-1",
        session_id="session-1",
        attempted_turn_number=1,
        agent_id="router",
        agent_version="0.1.0",
        kind=SpanKind.MODEL,
        status=ExecutionStatus.FAILED,
        sequence_started=2,
        sequence_completed=3,
        snapshot_sequence=3,
        started_at=NOW,
        completed_at=NOW + timedelta(milliseconds=5),
        duration_ms=5,
        error=error,
        failure_detail=failure_detail,
    )

    assert ExecutionSpan.model_validate_json(span.model_dump_json()) == span

    with pytest.raises(ValidationError, match="allowed only on failed spans"):
        ExecutionSpan(
            trace_id="trace-1",
            span_id="span-2",
            session_id="session-1",
            attempted_turn_number=1,
            agent_id="router",
            agent_version="0.1.0",
            kind=SpanKind.NODE,
            status=ExecutionStatus.COMPLETED,
            sequence_started=2,
            sequence_completed=3,
            snapshot_sequence=3,
            started_at=NOW,
            completed_at=NOW + timedelta(milliseconds=5),
            duration_ms=5,
            failure_detail=failure_detail,
        )


def test_trace_and_event_round_trip_with_attempt_identity() -> None:
    trace = ExecutionTrace(
        trace_id="trace-1",
        root_span_id="span-root",
        session_id="prospective-1",
        attempted_turn_number=1,
        status=ExecutionStatus.RUNNING,
        sequence_started=1,
        snapshot_sequence=1,
        started_at=NOW,
        deployment_version="dev",
        application_version="0.2.0",
        core_version="0.2.0",
        agent_versions={"router": "0.1.0"},
    )
    event = ExecutionEvent(
        trace_id=trace.trace_id,
        sequence=1,
        kind=EventKind.TRACE_STARTED,
        occurred_at=NOW,
    )

    assert ExecutionTrace.model_validate_json(trace.model_dump_json()) == trace
    assert ExecutionEvent.model_validate_json(event.model_dump_json()) == event

    with pytest.raises(ValidationError, match="trace events must omit"):
        ExecutionEvent(
            trace_id="trace-1",
            span_id="span-1",
            sequence=2,
            kind=EventKind.TRACE_COMPLETED,
            occurred_at=NOW,
        )


def test_datetimes_must_be_timezone_aware() -> None:
    payload = graph_definition().model_dump(mode="python")
    payload["created_at"] = datetime(2026, 8, 11)

    with pytest.raises(ValidationError, match="created_at must be timezone-aware"):
        GraphDefinition.model_validate(payload)


def test_direct_edges_cannot_carry_conditional_labels() -> None:
    with pytest.raises(ValidationError, match="direct graph edges"):
        GraphEdgeDefinition(
            source_node_id="start",
            target_node_id="end",
            kind=GraphEdgeKind.DIRECT,
            label="always",
        )
