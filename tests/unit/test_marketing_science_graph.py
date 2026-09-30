"""Unit tests for the marketing-science agent graph."""

import pytest
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from marketing_science_agent.graph import create_graph
from orchestration_core import AgentDependencies, ExecutionContext
from orchestration_tools.arithmetic import calculate_sum_tool

from tests.fakes import ScriptedChatModel

EXECUTION_CONTEXT = ExecutionContext(
    trace_id="trace",
    root_span_id="root",
    session_id="session",
    attempted_turn_number=1,
)


def compile_graph(fake: ScriptedChatModel):  # type: ignore[no-untyped-def]
    return create_graph(AgentDependencies(model=fake, tools=(calculate_sum_tool,))).compile()


@pytest.mark.unit
async def test_plain_response_uses_marketing_science_system_prompt_without_persisting_it() -> None:
    fake = ScriptedChatModel().queue_text("hello")
    result = await compile_graph(fake).ainvoke(
        {"messages": [HumanMessage(content="hi")], "execution_context": EXECUTION_CONTEXT}
    )

    assert result["messages"][-1].content == "hello"
    assert len(fake.calls) == 1
    assert isinstance(fake.calls[0]["messages"][0], SystemMessage)
    assert not any(isinstance(message, SystemMessage) for message in result["messages"])


@pytest.mark.unit
async def test_tool_call_uses_toolnode_and_matching_result_reaches_model() -> None:
    fake = (
        ScriptedChatModel()
        .queue_tool_call("calculate_sum", {"a": 17, "b": 25}, call_id="sum-call")
        .queue_text("42")
    )
    result = await compile_graph(fake).ainvoke(
        {
            "messages": [HumanMessage(content="add 17 and 25")],
            "execution_context": EXECUTION_CONTEXT,
        }
    )

    assert len(fake.calls) == 2
    tool_message = next(
        message for message in fake.calls[1]["messages"] if isinstance(message, ToolMessage)
    )
    assert tool_message.tool_call_id == "sum-call"
    assert tool_message.content == "42.0"
    assert result["messages"][-1].content == "42"
    assert all(isinstance(call["messages"][0], SystemMessage) for call in fake.calls)
