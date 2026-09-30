"""Harmless test-only tools used to prove standard LangGraph integration."""

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


class AddInput(BaseModel):
    left: int = Field(description="First integer")
    right: int = Field(description="Second integer")


async def add_numbers(left: int, right: int) -> str:
    """Return the sum of two integers as text."""

    return str(left + right)


def add_numbers_sync(left: int, right: int) -> str:
    """Synchronous test seam for LangGraph's synchronous execution path."""

    return str(left + right)


add_tool = StructuredTool.from_function(
    func=add_numbers_sync,
    coroutine=add_numbers,
    name="test_add_numbers",
    description="Add two integers for an integration test.",
    args_schema=AddInput,
)
