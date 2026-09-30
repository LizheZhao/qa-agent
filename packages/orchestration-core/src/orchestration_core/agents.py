"""Agent graph factory protocol."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph

from orchestration_core.invocation import AgentDescriptor, AgentInvoker


@dataclass(frozen=True, slots=True)
class AgentDependencies:
    """Stable runtime dependencies injected into an agent graph factory."""

    model: BaseChatModel
    tools: Sequence[BaseTool]
    agent_invoker: AgentInvoker | None = None
    agent_catalog: Sequence[AgentDescriptor] = ()


class AgentGraphFactory(Protocol):
    """The entrypoint contract implemented by agent packages."""

    def __call__(self, dependencies: Any) -> StateGraph[Any]: ...
