"""Private marketing-science conversation state."""

from langgraph.graph import MessagesState
from orchestration_core import ExecutionContext


class MarketingScienceState(MessagesState):
    """Standard reduced message state owned by the agent package."""

    execution_context: ExecutionContext
