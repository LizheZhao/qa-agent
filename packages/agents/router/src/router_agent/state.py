"""Private state for one router turn."""

from typing import NotRequired

from langgraph.graph import MessagesState
from orchestration_core import ExecutionContext, PausedConversation

from router_agent.contracts import RouteDecision, RoutingOutcome


class RouterState(MessagesState):
    active_agent_id: NotRequired[str | None]
    execution_context: ExecutionContext
    decision: NotRequired[RouteDecision]
    routing_outcome: NotRequired[RoutingOutcome]
    selected_agent_id: NotRequired[str | None]
    paused_run: NotRequired[PausedConversation | None]
