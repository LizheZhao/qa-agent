"""Capture, compiled topology, and callback instrumentation behavior."""

from typing import Annotated, TypedDict

import pytest
from langgraph.graph import END, START, StateGraph
from orchestration_core import (
    ContentCaptureMode,
    ExecutionContext,
    ExecutionStatus,
    GraphNodeKind,
    SpanKind,
)

from agentic_orchestration.execution.agent_registry import AgentRegistry
from agentic_orchestration.execution.executor import Executor
from agentic_orchestration.observability import MemoryExecutionTraceStore, TraceRecorder
from agentic_orchestration.observability.callbacks import GraphTraceCallback
from agentic_orchestration.observability.capture import REDACTED, capture_content
from agentic_orchestration.observability.definitions import (
    build_graph_definition,
    graph_predecessors,
)


class ParallelState(TypedDict):
    values: Annotated[list[str], list.__add__]


def parallel_graph():  # type: ignore[no-untyped-def]
    async def first(state: ParallelState) -> dict[str, list[str]]:
        del state
        return {"values": ["first"]}

    async def second(state: ParallelState) -> dict[str, list[str]]:
        del state
        return {"values": ["second"]}

    async def join(state: ParallelState) -> dict[str, list[str]]:
        return {"values": [f"joined:{len(state['values'])}"]}

    builder = StateGraph(ParallelState)
    builder.add_node("first", first)
    builder.add_node("second", second)
    builder.add_node("join", join)
    builder.add_edge(START, "first")
    builder.add_edge(START, "second")
    builder.add_edge("first", "join")
    builder.add_edge("second", "join")
    builder.add_edge("join", END)
    return builder.compile()


def split_graph():  # type: ignore[no-untyped-def]
    async def a(state: ParallelState) -> dict[str, list[str]]:
        del state
        return {"values": ["a"]}

    async def b(state: ParallelState) -> dict[str, list[str]]:
        del state
        return {"values": ["b"]}

    async def c(state: ParallelState) -> dict[str, list[str]]:
        del state
        return {"values": ["c"]}

    async def d(state: ParallelState) -> dict[str, list[str]]:
        del state
        return {"values": ["d"]}

    builder = StateGraph(ParallelState)
    builder.add_node("a", a)
    builder.add_node("b", b)
    builder.add_node("c", c)
    builder.add_node("d", d)
    builder.add_edge(START, "a")
    builder.add_edge(START, "b")
    builder.add_edge("a", "c")
    builder.add_edge("b", "d")
    builder.add_edge("c", END)
    builder.add_edge("d", END)
    return builder.compile()


def failing_graph():  # type: ignore[no-untyped-def]
    async def fail(state: ParallelState) -> dict[str, list[str]]:
        del state
        raise KeyError("ASK_GENOME_CLIENT_CODE")

    builder = StateGraph(ParallelState)
    builder.add_node("fail", fail)
    builder.add_edge(START, "fail")
    builder.add_edge("fail", END)
    return builder.compile()


def definition(graph):  # type: ignore[no-untyped-def]
    return build_graph_definition(
        graph,
        agent_id="parallel",
        agent_version="1.0.0",
        entrypoint="tests:parallel_graph",
    )


def test_capture_redacts_secrets_and_omits_oversized_values() -> None:
    captured = capture_content(
        {
            "authorization": "Bearer secret",
            "nested": {"client_secret": "secret", "password": "secret", "safe": 2},
        }
    )
    oversized = capture_content("large", max_inline_bytes=2)

    assert captured.value == {
        "authorization": REDACTED,
        "nested": {"client_secret": REDACTED, "password": REDACTED, "safe": 2},
    }
    assert oversized.reason == "inline_size_limit_exceeded"
    assert oversized.value is None


def test_capture_previews_oversized_values_that_shorten_enough() -> None:
    readout = "line one\nline two\n" + "x" * 5_000
    captured = capture_content(
        {"readout": readout, "records": list(range(20))}, max_inline_bytes=2_000
    )

    assert captured.mode == ContentCaptureMode.INLINE
    assert captured.value["readout"].startswith("line one\nline two\n")
    assert captured.value["readout"].endswith(f"... [truncated {len(readout) - 1_000} chars]")
    assert captured.value["records"] == [0, 1, 2, 3, 4, "[truncated 15 more items]"]
    assert captured.metadata["preview"] is True
    assert captured.metadata["original_byte_size"] > 2_000


def test_observability_callback_errors_do_not_fail_graph_execution() -> None:
    assert GraphTraceCallback.raise_error is False


def test_compiled_definition_is_stable_and_contains_start_end_and_edges() -> None:
    graph = parallel_graph()
    first = definition(graph)
    second = definition(graph)

    assert first.definition_id == second.definition_id
    assert first.definition_hash == second.definition_hash
    assert {node.kind for node in first.nodes} >= {GraphNodeKind.START, GraphNodeKind.END}
    assert {(edge.source_node_id, edge.target_node_id) for edge in first.edges} >= {
        ("__start__", "first"),
        ("__start__", "second"),
        ("first", "join"),
        ("second", "join"),
    }


async def test_parallel_callbacks_share_superstep_and_preserve_fan_in_causality() -> None:
    graph = parallel_graph()
    graph_definition = definition(graph)
    store = MemoryExecutionTraceStore()
    recorder = TraceRecorder(
        store,
        deployment_version="test",
        application_version="1.0.0",
        core_version="1.0.0",
        agent_versions={"parallel": "1.0.0"},
    )
    executor = Executor(
        AgentRegistry({"parallel": graph}),
        recorder,
        {"parallel": graph_definition.definition_id},
        {"parallel": dict(graph_predecessors(graph_definition))},
    )
    context = ExecutionContext(
        trace_id="parallel-trace",
        root_span_id="parallel-root",
        session_id="parallel-session",
        attempted_turn_number=1,
    )

    await executor.begin("parallel", context)
    await executor.invoke("parallel", {"values": []}, context=context)
    await executor.complete(context, committed_turn_number=1)

    spans = await store.spans(context.trace_id)
    nodes = {span.definition_node_id: span for span in spans if span.kind == SpanKind.NODE}
    assert nodes["first"].superstep == nodes["second"].superstep == 1
    assert set(nodes["join"].caused_by_span_ids) == {
        nodes["first"].span_id,
        nodes["second"].span_id,
    }
    assert nodes["join"].state_delta is not None


async def test_parallel_branches_preserve_distinct_direct_causes() -> None:
    graph = split_graph()
    graph_definition = definition(graph)
    store = MemoryExecutionTraceStore()
    recorder = TraceRecorder(
        store,
        deployment_version="test",
        application_version="1.0.0",
        core_version="1.0.0",
        agent_versions={"parallel": "1.0.0"},
    )
    executor = Executor(
        AgentRegistry({"parallel": graph}),
        recorder,
        {"parallel": graph_definition.definition_id},
        {"parallel": dict(graph_predecessors(graph_definition))},
    )
    context = ExecutionContext(
        trace_id="split-trace",
        root_span_id="split-root",
        session_id="split-session",
        attempted_turn_number=1,
    )

    await executor.begin("parallel", context)
    await executor.invoke("parallel", {"values": []}, context=context)
    await executor.complete(context, committed_turn_number=1)

    spans = await store.spans(context.trace_id)
    nodes = {span.definition_node_id: span for span in spans if span.kind == SpanKind.NODE}
    assert nodes["c"].caused_by_span_ids == (nodes["a"].span_id,)
    assert nodes["d"].caused_by_span_ids == (nodes["b"].span_id,)


async def test_failed_node_persists_bounded_exception_detail() -> None:
    graph = failing_graph()
    graph_definition = definition(graph)
    store = MemoryExecutionTraceStore()
    recorder = TraceRecorder(
        store,
        deployment_version="test",
        application_version="1.0.0",
        core_version="1.0.0",
        agent_versions={"parallel": "1.0.0"},
    )
    executor = Executor(
        AgentRegistry({"parallel": graph}),
        recorder,
        {"parallel": graph_definition.definition_id},
        {"parallel": dict(graph_predecessors(graph_definition))},
    )
    context = ExecutionContext(
        trace_id="failed-trace",
        root_span_id="failed-root",
        session_id="failed-session",
        attempted_turn_number=1,
    )

    await executor.begin("parallel", context)
    with pytest.raises(KeyError, match="ASK_GENOME_CLIENT_CODE"):
        await executor.invoke("parallel", {"values": []}, context=context)
    await executor.fail(context)

    spans = await store.spans(context.trace_id)
    failed_node = next(span for span in spans if span.definition_node_id == "fail")
    assert failed_node.status == ExecutionStatus.FAILED
    assert failed_node.failure_detail is not None
    assert failed_node.failure_detail.exception_chain[0].exception_type == "KeyError"
    assert failed_node.failure_detail.exception_chain[0].message == "'ASK_GENOME_CLIENT_CODE'"
