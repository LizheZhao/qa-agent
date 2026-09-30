from dataclasses import dataclass, field

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from orchestration_core import (
    AgentDependencies,
    AgentDescriptor,
    ClarificationOption,
    ClarificationRequest,
    ConversationRequest,
    ConversationResult,
    ExecutionContext,
    InvocationContext,
    PausedConversation,
)
from router_agent.graph import create_graph

from tests.fakes import ScriptedChatModel

EXECUTION_CONTEXT = ExecutionContext(
    trace_id="trace",
    root_span_id="root",
    session_id="session",
    attempted_turn_number=1,
)


@dataclass
class RecordingInvoker:
    calls: list[tuple[str, ConversationRequest, InvocationContext]] = field(default_factory=list)

    async def invoke(
        self,
        agent_id: str,
        request: ConversationRequest,
        context: InvocationContext,
    ) -> ConversationResult:
        self.calls.append((agent_id, request, context))
        return ConversationResult((AIMessage(content="specialist answer"),))


def compile_router(fake: ScriptedChatModel, invoker: RecordingInvoker):  # type: ignore[no-untyped-def]
    return create_graph(
        AgentDependencies(
            model=fake,
            tools=(),
            agent_invoker=invoker,
            agent_catalog=(
                AgentDescriptor(
                    agent_id="marketing_science",
                    description="Explain marketing science.",
                    capabilities=("marketing_science_explanation",),
                ),
            ),
        )
    ).compile()


@pytest.mark.unit
async def test_router_answers_directly_without_persisting_decision_message() -> None:
    fake = ScriptedChatModel().queue_tool_call(
        "RouteDecision",
        {
            "outcome": "answer",
            "reason_code": "greeting",
            "response": "Hello",
            "target_agent": None,
        },
    )
    invoker = RecordingInvoker()

    result = await compile_router(fake, invoker).ainvoke(
        {"messages": [HumanMessage(content="hello")], "execution_context": EXECUTION_CONTEXT}
    )

    assert result["messages"][-1].content == "Hello"
    assert result["routing_outcome"] == "answer"
    assert result["selected_agent_id"] is None
    assert invoker.calls == []
    assert isinstance(fake.calls[0]["messages"][0], SystemMessage)
    assert len(result["messages"]) == 2


@pytest.mark.unit
async def test_router_delegates_once_and_returns_child_messages() -> None:
    fake = ScriptedChatModel().queue_tool_call(
        "RouteDecision",
        {
            "outcome": "delegate",
            "reason_code": "marketing_science",
            "response": None,
            "target_agent": "marketing_science",
        },
    )
    invoker = RecordingInvoker()
    message = HumanMessage(content="What is incrementality?")

    result = await compile_router(fake, invoker).ainvoke(
        {
            "messages": [message],
            "active_agent_id": "marketing_science",
            "execution_context": EXECUTION_CONTEXT,
        }
    )

    assert result["messages"][-1].content == "specialist answer"
    assert result["routing_outcome"] == "delegate"
    assert result["selected_agent_id"] == "marketing_science"
    assert len(invoker.calls) == 1
    assert invoker.calls[0][0] == "marketing_science"
    assert invoker.calls[0][1].messages == (message,)
    assert invoker.calls[0][2].execution is EXECUTION_CONTEXT
    assert "previous selected child" in str(fake.calls[0]["messages"][0].content)
    assert "Never infer, police, or make claims about a child's tool inventory" in str(
        fake.calls[0]["messages"][0].content
    )


@pytest.mark.unit
async def test_router_rejects_model_selected_ineligible_agent() -> None:
    fake = ScriptedChatModel().queue_tool_call(
        "RouteDecision",
        {
            "outcome": "delegate",
            "reason_code": "invalid",
            "response": None,
            "target_agent": "missing",
        },
    )

    with pytest.raises(ValueError, match="ineligible"):
        await compile_router(fake, RecordingInvoker()).ainvoke(
            {
                "messages": [HumanMessage(content="route me")],
                "execution_context": EXECUTION_CONTEXT,
            }
        )


@dataclass
class PausingInvoker:
    """A child that reports a clarification instead of a final assistant message."""

    paused: PausedConversation
    calls: list[str] = field(default_factory=list)

    async def invoke(
        self,
        agent_id: str,
        request: ConversationRequest,
        context: InvocationContext,
    ) -> PausedConversation:
        self.calls.append(agent_id)
        return self.paused


def paused_conversation() -> PausedConversation:
    return PausedConversation(
        agent_id="marketing_science",
        run_id="run-1",
        clarification=ClarificationRequest(
            clarification_id="clarification-1",
            producer_agent_id="marketing_science",
            task_id="task-1",
            question="Which market should the summary cover?",
            reason_code="ambiguous_market",
            selection_mode="single",
            options=(
                ClarificationOption(option_id="us", label="United States"),
                ClarificationOption(option_id="eu", label="Europe"),
            ),
            freshness_token="snapshot-1",
        ),
        queued_clarification_ids=("clarification-2",),
        graph_definition_id="definition-1",
    )


@pytest.mark.unit
async def test_router_surfaces_a_paused_child_with_its_route_affinity() -> None:
    invoker = PausingInvoker(paused=paused_conversation())
    fake = ScriptedChatModel().queue_tool_call(
        "RouteDecision",
        {
            "outcome": "delegate",
            "reason_code": "marketing_science",
            "response": None,
            "target_agent": "marketing_science",
        },
    )
    graph = compile_router(fake, invoker)

    result = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="How did spend perform?")],
            "execution_context": EXECUTION_CONTEXT,
        }
    )

    assert result["paused_run"] is invoker.paused
    assert result["selected_agent_id"] == "marketing_science"
    assert invoker.calls == ["marketing_science"]


@pytest.mark.unit
async def test_router_reports_no_routing_outcome_for_a_turn_that_is_not_finished() -> None:
    """A paused child has produced no turn, so a caller commits nothing rather than the question."""

    invoker = PausingInvoker(paused=paused_conversation())
    fake = ScriptedChatModel().queue_tool_call(
        "RouteDecision",
        {
            "outcome": "delegate",
            "reason_code": "marketing_science",
            "response": None,
            "target_agent": "marketing_science",
        },
    )
    graph = compile_router(fake, invoker)

    result = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="How did spend perform?")],
            "execution_context": EXECUTION_CONTEXT,
        }
    )

    assert result.get("routing_outcome") is None
    # The question is not appended as if it were the turn's final answer.
    assert [message.content for message in result["messages"]] == ["How did spend perform?"]


@pytest.mark.unit
async def test_router_clears_any_pause_when_a_child_answers_normally() -> None:
    invoker = RecordingInvoker()
    fake = ScriptedChatModel().queue_tool_call(
        "RouteDecision",
        {
            "outcome": "delegate",
            "reason_code": "marketing_science",
            "response": None,
            "target_agent": "marketing_science",
        },
    )
    graph = compile_router(fake, invoker)

    result = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="How did spend perform?")],
            "execution_context": EXECUTION_CONTEXT,
        }
    )

    assert result["paused_run"] is None
    assert result["routing_outcome"] == "delegate"
