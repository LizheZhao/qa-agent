"""Required LangGraph factory entrypoint; intentionally non-runnable."""

from agent_template.state import AgentState
from langgraph.graph import StateGraph
from orchestration_core import AgentDependencies


def create_graph(dependencies: AgentDependencies) -> StateGraph[AgentState]:
    """TODO: build and return an uncompiled graph; the worker compiles it once."""

    del dependencies
    raise NotImplementedError("Replace the agent template before deployment")
