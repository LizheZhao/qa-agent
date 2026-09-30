"""Immutable registry that resolves agent-declared tool identifiers."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from types import MappingProxyType

from langchain_core.tools import BaseTool


class ToolRegistryError(RuntimeError):
    """Registered tools or an agent tool declaration are invalid."""


class ToolRegistry:
    def __init__(self, tools: Iterable[BaseTool]) -> None:
        registered: dict[str, BaseTool] = {}
        for tool in tools:
            if tool.name in registered:
                raise ToolRegistryError(f"Tool {tool.name!r} is registered more than once")
            registered[tool.name] = tool
        self._tools = MappingProxyType(registered)

    def resolve(self, tool_ids: Sequence[str]) -> tuple[BaseTool, ...]:
        missing = [tool_id for tool_id in tool_ids if tool_id not in self._tools]
        if missing:
            raise ToolRegistryError("Unknown declared tools: " + ", ".join(sorted(missing)))
        return tuple(self._tools[tool_id] for tool_id in tool_ids)
