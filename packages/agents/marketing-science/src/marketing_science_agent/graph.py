"""Marketing-science model and tool loop."""

from typing import Any, cast

from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from orchestration_core import (
    NODE_KIND_METADATA_KEY,
    PROMPT_TEMPLATE_ID_METADATA_KEY,
    PROMPT_TEMPLATE_METADATA_KEY,
    AgentDependencies,
    GraphNodeKind,
)

from marketing_science_agent.prompts import SYSTEM_PROMPT
from marketing_science_agent.state import MarketingScienceState


def create_graph(dependencies: AgentDependencies) -> StateGraph[MarketingScienceState]:
    """Build an uncompiled graph using only this agent's declared dependencies."""

    tools = list(dependencies.tools)
    model = dependencies.model.bind_tools(tools)

    async def call_model(state: MarketingScienceState) -> dict[str, list[BaseMessage]]:
        response = await model.ainvoke([SystemMessage(content=SYSTEM_PROMPT), *state["messages"]])
        return {"messages": [response]}

    async def route_after_model(state: MarketingScienceState) -> str:
        return tools_condition(cast(dict[str, Any], state))

    builder = StateGraph(MarketingScienceState)
    builder.add_node(
        "model",
        call_model,
        metadata={
            NODE_KIND_METADATA_KEY: GraphNodeKind.MODEL.value,
            PROMPT_TEMPLATE_ID_METADATA_KEY: "marketing_science.system",
            PROMPT_TEMPLATE_METADATA_KEY: SYSTEM_PROMPT,
        },
    )
    builder.add_node(
        "tools",
        ToolNode(tools, handle_tool_errors=False),
        metadata={NODE_KIND_METADATA_KEY: GraphNodeKind.TOOL.value},
    )
    builder.add_edge(START, "model")
    builder.add_conditional_edges(
        "model", route_after_model, path_map={"tools": "tools", "__end__": END}
    )
    builder.add_edge("tools", "model")
    return builder
