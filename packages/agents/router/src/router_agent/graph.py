"""Structured single-child routing graph."""

from langchain_core.messages import AIMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from orchestration_core import (
    NODE_KIND_METADATA_KEY,
    PROMPT_TEMPLATE_ID_METADATA_KEY,
    PROMPT_TEMPLATE_METADATA_KEY,
    AgentDependencies,
    ConversationRequest,
    GraphNodeKind,
    InvocationContext,
    PausedConversation,
)

from router_agent.contracts import RouteDecision
from router_agent.prompts import SYSTEM_PROMPT_TEMPLATE, system_prompt
from router_agent.state import RouterState


def create_graph(dependencies: AgentDependencies) -> StateGraph[RouterState]:
    """Build an uncompiled router using explicitly injected runtime capabilities."""

    if dependencies.agent_invoker is None:
        raise ValueError("router requires an AgentInvoker")
    eligible = {agent.agent_id for agent in dependencies.agent_catalog}
    router_model = dependencies.model.bind_tools([RouteDecision], tool_choice="RouteDecision")
    invoker = dependencies.agent_invoker

    async def decide(state: RouterState) -> dict[str, RouteDecision]:
        active = state.get("active_agent_id")
        response = await router_model.ainvoke(
            [
                SystemMessage(content=system_prompt(dependencies.agent_catalog, active)),
                *state["messages"],
            ]
        )
        if not isinstance(response, AIMessage) or len(response.tool_calls) != 1:
            raise TypeError("router model must return exactly one structured decision")
        call = response.tool_calls[0]
        if call["name"] != RouteDecision.__name__:
            raise TypeError("router model returned an unexpected structured decision")
        decision = RouteDecision.model_validate(call["args"])
        if decision.outcome == "delegate" and decision.target_agent not in eligible:
            raise ValueError("router selected an ineligible agent")
        return {"decision": decision}

    async def after_decision(state: RouterState) -> str:
        return "delegate" if state["decision"].outcome == "delegate" else "respond"

    async def respond(state: RouterState) -> dict[str, object]:
        decision = state["decision"]
        assert decision.response is not None
        return {
            "messages": [AIMessage(content=decision.response)],
            "routing_outcome": decision.outcome,
            "selected_agent_id": None,
        }

    async def delegate(state: RouterState) -> dict[str, object]:
        decision = state["decision"]
        assert decision.target_agent is not None
        outcome = await invoker.invoke(
            decision.target_agent,
            ConversationRequest(messages=tuple(state["messages"])),
            InvocationContext(
                execution=state["execution_context"],
                caller_agent_id="router",
                depth=0,
            ),
        )
        if isinstance(outcome, PausedConversation):
            # A paused child has not produced a turn, so no routing outcome is reported.
            # Publishing the pending run and presenting the question belongs to the
            # application boundary; until that exists, a caller that reads this state as
            # an ordinary result fails its own validation and commits nothing, which is
            # the intended failure direction for a turn that is not finished.
            return {
                "paused_run": outcome,
                "selected_agent_id": decision.target_agent,
            }
        return {
            "messages": list(outcome.generated_messages),
            "routing_outcome": "delegate",
            "selected_agent_id": decision.target_agent,
            "paused_run": None,
        }

    builder = StateGraph(RouterState)
    builder.add_node(
        "decide",
        decide,
        metadata={
            NODE_KIND_METADATA_KEY: GraphNodeKind.ROUTER.value,
            PROMPT_TEMPLATE_ID_METADATA_KEY: "router.route_decision.system",
            PROMPT_TEMPLATE_METADATA_KEY: SYSTEM_PROMPT_TEMPLATE,
        },
    )
    builder.add_node(
        "respond",
        respond,
        metadata={NODE_KIND_METADATA_KEY: GraphNodeKind.DETERMINISTIC.value},
    )
    builder.add_node(
        "delegate",
        delegate,
        metadata={NODE_KIND_METADATA_KEY: GraphNodeKind.CHILD_AGENT.value},
    )
    builder.add_edge(START, "decide")
    builder.add_conditional_edges(
        "decide", after_decision, path_map={"respond": "respond", "delegate": "delegate"}
    )
    builder.add_edge("respond", END)
    builder.add_edge("delegate", END)
    return builder
