import pytest
from langchain_core.messages import HumanMessage, ToolMessage

from tests.fakes import ScriptedChatModel
from tests.fixtures.graph import create_test_graph


@pytest.mark.integration
async def test_standard_tool_calls_drive_toolnode_and_tools_condition() -> None:
    fake = (
        ScriptedChatModel()
        .queue_tool_call("test_add_numbers", {"left": 20, "right": 22}, call_id="call-add")
        .queue_text("finished")
    )
    result = await create_test_graph(fake).ainvoke({"messages": [HumanMessage("add them")]})
    tool_messages = [message for message in result["messages"] if isinstance(message, ToolMessage)]
    assert len(tool_messages) == 1
    assert tool_messages[0].content == "42"
    assert result["messages"][-1].content == "finished"
    assert len(fake.calls) == 2
