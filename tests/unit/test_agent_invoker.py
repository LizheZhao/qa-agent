from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from orchestration_core import (
    AgentInvocationDepthError,
    ConversationRequest,
    ExecutionContext,
    ForbiddenAgentError,
    InvocationContext,
    ObservabilityStatus,
    UnknownAgentError,
)

from agentic_orchestration.execution.agent_invoker import RegistryAgentInvoker
from agentic_orchestration.execution.agent_registry import GRAPH_RECURSION_LIMIT, AgentRegistry
from agentic_orchestration.observability import MemoryExecutionTraceStore, TraceRecorder


class FixtureGraph:
    def __init__(self) -> None:
        self.configs: list[dict[str, Any]] = []

    async def ainvoke(
        self, state: dict[str, Any], config: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        # Every child run carries a config now, even one with no thread of its own, because the
        # recursion limit applies to all of them. A stub that rejected it hid that.
        self.configs.append(config or {})
        return {"messages": [*state["messages"], AIMessage(content="child answer")]}


def invoker() -> RegistryAgentInvoker:
    return RegistryAgentInvoker(
        AgentRegistry({"child": FixtureGraph(), "forbidden": FixtureGraph()}),
        caller_agent_id="router",
        allowed_agent_ids=("child",),
        timeout_seconds=1,
    )


def context(*, caller: str = "router", depth: int = 0) -> InvocationContext:
    return InvocationContext(
        execution=ExecutionContext(
            trace_id="trace",
            root_span_id="root",
            session_id="session",
            attempted_turn_number=1,
        ),
        caller_agent_id=caller,
        depth=depth,
    )


@pytest.mark.unit
async def test_invoker_returns_only_child_generated_messages() -> None:
    request = ConversationRequest(messages=(HumanMessage(content="hello"),))

    result = await invoker().invoke("child", request, context())

    assert [message.content for message in result.generated_messages] == ["child answer"]


@pytest.mark.unit
async def test_every_child_run_carries_the_raised_recursion_limit() -> None:
    """LangGraph's default of 25 counts supersteps, not nodes, so it caps an agent that loops
    rather than one that is merely large. ask_genome walks its subqueries through one cursor and a
    two-subquery question already exceeded it, which surfaced as GraphRecursionError rather than
    as anything naming the real cause."""

    graph = FixtureGraph()
    registry = AgentRegistry({"child": graph})
    invoker = RegistryAgentInvoker(
        registry, caller_agent_id="router", allowed_agent_ids=("child",), timeout_seconds=1
    )

    request = ConversationRequest(messages=(HumanMessage(content="hello"),))
    await invoker.invoke("child", request, context())

    assert graph.configs[0]["recursion_limit"] == GRAPH_RECURSION_LIMIT
    assert GRAPH_RECURSION_LIMIT > 25


@pytest.mark.unit
async def test_invoker_enforces_target_caller_and_depth() -> None:
    request = ConversationRequest(messages=(HumanMessage(content="hello"),))

    with pytest.raises(UnknownAgentError):
        await invoker().invoke("missing", request, context())
    with pytest.raises(ForbiddenAgentError):
        await invoker().invoke("forbidden", request, context())
    with pytest.raises(ForbiddenAgentError):
        await invoker().invoke("router", request, context())
    with pytest.raises(ForbiddenAgentError):
        await invoker().invoke("child", request, context(caller="other"))
    with pytest.raises(AgentInvocationDepthError):
        await invoker().invoke("child", request, context(depth=1))


@pytest.mark.unit
async def test_missing_graph_definition_degrades_observability_without_failing_child() -> None:
    store = MemoryExecutionTraceStore()
    recorder = TraceRecorder(
        store,
        deployment_version="test",
        application_version="test",
        core_version="test",
        agent_versions={"router": "1", "child": "1"},
    )
    invocation_context = context()
    await recorder.begin(invocation_context.execution, "router")
    child_invoker = RegistryAgentInvoker(
        AgentRegistry({"child": FixtureGraph()}),
        caller_agent_id="router",
        allowed_agent_ids=("child",),
        timeout_seconds=1,
        recorder=recorder,
        graph_definition_ids={},
    )

    result = await child_invoker.invoke(
        "child",
        ConversationRequest(messages=(HumanMessage(content="hello"),)),
        invocation_context,
    )
    run = await recorder.active(invocation_context.execution.trace_id)
    trace = await run.finish(committed_turn_number=1)
    await recorder.release(invocation_context.execution.trace_id)

    assert [message.content for message in result.generated_messages] == ["child answer"]
    assert trace.observability_status == ObservabilityStatus.DEGRADED
