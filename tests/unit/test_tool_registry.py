import pytest
from langchain_core.tools import tool

from agentic_orchestration.execution.tool_registry import ToolRegistry, ToolRegistryError


@tool
def fixture_tool(value: str) -> str:
    """Return the supplied value."""

    return value


@pytest.mark.unit
def test_tool_registry_resolves_only_declared_tools() -> None:
    registry = ToolRegistry((fixture_tool,))

    assert registry.resolve(()) == ()
    assert registry.resolve(("fixture_tool",)) == (fixture_tool,)
    with pytest.raises(ToolRegistryError, match="missing"):
        registry.resolve(("missing",))


@pytest.mark.unit
def test_tool_registry_rejects_duplicate_names() -> None:
    with pytest.raises(ToolRegistryError, match="more than once"):
        ToolRegistry((fixture_tool, fixture_tool))
