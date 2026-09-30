"""Immutable registry for explicitly declared, startup-compiled agent graphs."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver

# Supersteps one graph invocation may take before LangGraph calls it a runaway loop.
#
# Raised from LangGraph's default of 25, which assumes a graph whose node count is its step count.
# An agent that loops over a list does not work that way: ask_genome walks its subqueries through
# one cursor, so its step count is roughly nodes x subqueries, and a two-subquery question already
# exceeded 25. The limit is still a real backstop against a cycle that never converges, just sized
# for the loop-shaped graphs this deployment actually runs.
GRAPH_RECURSION_LIMIT = 150


class AgentRegistry:
    def __init__(self, graphs: Mapping[str, Any]) -> None:
        self._graphs = MappingProxyType(dict(graphs))

    @classmethod
    def compile(
        cls,
        entrypoints: Mapping[str, Any],
        dependencies: Mapping[str, Any] | None = None,
        savers: Mapping[str, BaseCheckpointSaver[Any]] | None = None,
    ) -> AgentRegistry:
        """Compile each declared graph, checkpointing only the agents named in `savers`.

        Pause capability is per agent rather than per deployment: the router and a child
        are separate invocations, so a child that can pause is compiled with a saver while
        every other graph, the router included, compiles without one and keeps its
        existing stateless behaviour.
        """

        dependencies = dependencies or {}
        savers = savers or {}
        unknown = set(savers) - set(entrypoints)
        if unknown:
            raise KeyError(f"Checkpointed agents are not in this registry: {sorted(unknown)}")
        compiled: dict[str, Any] = {}
        for name, factory in entrypoints.items():
            graph = factory(dependencies.get(name))
            compiled[name] = graph.compile(checkpointer=savers.get(name))
        return cls(compiled)

    @property
    def graphs(self) -> Mapping[str, Any]:
        return self._graphs

    def get(self, name: str) -> Any:
        try:
            return self._graphs[name]
        except KeyError as exc:
            raise KeyError(f"Agent {name!r} is not registered in this deployment") from exc
