"""Test-only graph proving model -> ToolNode -> model behavior."""

from typing import Any

from langchain_core.messages import BaseMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from orchestration_core import AgentManifest

from tests.fakes import ScriptedChatModel
from tests.fixtures.tools import add_tool

FIXTURE_AGENT_MANIFEST = AgentManifest(
    agent_id="fixture",
    version="0.1.0",
    description="Deployment-loader test fixture.",
    capabilities=("fixture",),
)


def create_test_graph(model: ScriptedChatModel) -> Any:
    bound = model.bind_tools([add_tool])

    async def call_model(state: MessagesState) -> dict[str, list[BaseMessage]]:
        response = await bound.ainvoke(state["messages"])
        return {"messages": [response]}

    async def route_after_model(state: MessagesState) -> str:
        return tools_condition(state)

    builder = StateGraph(MessagesState)
    builder.add_node("model", call_model)
    builder.add_node("tools", ToolNode([add_tool]))
    builder.add_edge(START, "model")
    builder.add_conditional_edges(
        "model", route_after_model, path_map={"tools": "tools", "__end__": END}
    )
    builder.add_edge("tools", "model")
    return builder.compile()
