"""Recorder and process-memory trace store behavior."""

import asyncio
from collections.abc import Sequence
from typing import Any

import pytest
from orchestration_core import (
    EventKind,
    ExecutionContext,
    ExecutionStatus,
    ObservabilityStatus,
    SanitizedExecutionError,
    SpanKind,
)

from agentic_orchestration.execution.agent_registry import AgentRegistry
from agentic_orchestration.execution.executor import Executor
from agentic_orchestration.observability import (
    MemoryExecutionEventPublisher,
    MemoryExecutionTraceStore,
    TraceRecorder,
)
from agentic_orchestration.observability.store import ExecutionSnapshot


class FailingExecutionTraceStore(MemoryExecutionTraceStore):
    async def save_snapshots(self, snapshots: Sequence[ExecutionSnapshot]) -> None:
        del snapshots
        raise RuntimeError("simulated observability outage")


class BlockingGraph:
    def __init__(self) -> None:
        self.entered = asyncio.Event()

    async def ainvoke(self, state: dict[str, Any], config: Any = None) -> dict[str, Any]:
        del state, config
        self.entered.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


def recorder(
    store: MemoryExecutionTraceStore,
    publisher: MemoryExecutionEventPublisher | None = None,
) -> TraceRecorder:
    return TraceRecorder(
        store,
        deployment_version="dev",
        application_version="0.2.0",
        core_version="0.2.0",
        agent_versions={"router": "0.1.0", "child": "0.1.0"},
        event_publisher=publisher,
    )


def context() -> ExecutionContext:
    return ExecutionContext(
        trace_id="trace-1",
        root_span_id="root",
        session_id="session-1",
        attempted_turn_number=2,
    )


async def test_recorder_persists_hierarchy_and_monotonic_events() -> None:
    store = MemoryExecutionTraceStore()
    publisher = MemoryExecutionEventPublisher()
    trace_recorder = recorder(store, publisher)
    run = await trace_recorder.begin(context(), "router")
    graph = await run.start_span(agent_id="router", kind=SpanKind.GRAPH, parent_span_id="root")
    first = await run.start_span(agent_id="child", kind=SpanKind.NODE, parent_span_id=graph.span_id)
    second = await run.start_span(
        agent_id="child", kind=SpanKind.NODE, parent_span_id=graph.span_id
    )

    await run.complete_span(first.span_id)
    await run.complete_span(second.span_id)
    await run.complete_span(graph.span_id)
    await run.finish(committed_turn_number=2)
    await trace_recorder.release(context().trace_id)

    trace = await store.get_trace("trace-1")
    spans = await store.spans("trace-1")
    events = publisher.events("trace-1")
    assert trace is not None
    assert trace.status == ExecutionStatus.COMPLETED
    assert trace.observability_status == ObservabilityStatus.COMPLETE
    assert trace.sequence_started == 1
    assert trace.sequence_completed == len(events)
    assert trace.snapshot_sequence == len(events)
    assert [event.sequence for event in events] == list(range(1, len(events) + 1))
    assert events[0].kind == EventKind.TRACE_STARTED
    assert events[-1].kind == EventKind.TRACE_COMPLETED
    parallel_parent_ids = {
        span.parent_span_id for span in spans if span.span_id in {first.span_id, second.span_id}
    }
    assert parallel_parent_ids == {graph.span_id}


async def test_failed_attempt_remains_queryable_without_a_successful_turn() -> None:
    store = MemoryExecutionTraceStore()
    trace_recorder = recorder(store)
    execution_context = context()
    run = await trace_recorder.begin(execution_context, "router")
    graph = await run.start_span(agent_id="router", kind=SpanKind.GRAPH, parent_span_id="root")
    error = SanitizedExecutionError(code="failed", message="Execution failed")

    await run.complete_span(graph.span_id, error=error)
    await run.finish(error=error)
    await trace_recorder.release(execution_context.trace_id)

    trace = await store.get_trace("trace-1")
    assert trace is not None
    assert trace.status == ExecutionStatus.FAILED
    assert trace.committed_turn_number is None
    assert trace.session_id == "session-1"
    assert trace.error == error


async def test_observability_outage_degrades_trace_without_failing_execution() -> None:
    store = FailingExecutionTraceStore()
    trace_recorder = TraceRecorder(
        store,
        deployment_version="dev",
        application_version="0.6.0",
        core_version="0.5.0",
        agent_versions={"router": "0.1.0"},
        flush_timeout_seconds=0.01,
    )
    run = await trace_recorder.begin(context(), "router")

    trace = await run.finish(committed_turn_number=2)
    await trace_recorder.release(context().trace_id)

    assert trace.status == ExecutionStatus.COMPLETED
    assert trace.observability_status == ObservabilityStatus.DEGRADED
    assert await store.get_trace(trace.trace_id) is None


async def test_cancelled_execution_terminalizes_trace_spans_and_writer() -> None:
    store = MemoryExecutionTraceStore()
    trace_recorder = recorder(store)
    graph = BlockingGraph()
    executor = Executor(
        AgentRegistry({"router": graph}),
        trace_recorder,
        {"router": "router-definition"},
    )
    execution_context = context()

    await executor.begin("router", execution_context)
    invocation = asyncio.create_task(executor.invoke("router", {}, context=execution_context))
    await graph.entered.wait()
    invocation.cancel()
    with pytest.raises(asyncio.CancelledError):
        await invocation
    await executor.cancel(execution_context)

    trace = await store.get_trace(execution_context.trace_id)
    spans = await store.spans(execution_context.trace_id)
    assert trace is not None
    assert trace.status == ExecutionStatus.CANCELLED
    assert all(span.status == ExecutionStatus.CANCELLED for span in spans)
    with pytest.raises(RuntimeError, match="not active"):
        await trace_recorder.active(execution_context.trace_id)
