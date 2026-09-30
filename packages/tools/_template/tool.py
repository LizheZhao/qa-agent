"""Async tool template with explicit dependencies and normalized outcomes."""

from dataclasses import dataclass
from typing import Any

from langchain_core.tools import StructuredTool
from orchestration_core import ToolError, ToolResult

from .manifest import TOOL_DESCRIPTION, TOOL_NAME
from .schemas import ToolInput


@dataclass(frozen=True, slots=True)
class ToolDependencies:
    api_client: Any


async def execute(value: str, *, dependencies: ToolDependencies) -> ToolResult:
    """TODO: call an API client; never access a domain database directly."""

    del value, dependencies
    return ToolResult(error=ToolError(code="not_implemented", message="Replace this template"))


def as_langchain_tool(dependencies: ToolDependencies) -> StructuredTool:
    async def bound_execute(value: str) -> ToolResult:
        return await execute(value, dependencies=dependencies)

    return StructuredTool.from_function(
        coroutine=bound_execute,
        name=TOOL_NAME,
        description=TOOL_DESCRIPTION,
        args_schema=ToolInput,
    )
