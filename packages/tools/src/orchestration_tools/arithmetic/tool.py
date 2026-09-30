"""Deterministic arithmetic tool available to agents that explicitly declare it."""

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict, Field


class CalculateSumInput(BaseModel):
    """Validated arguments exposed to the model."""

    model_config = ConfigDict(extra="forbid")

    a: float = Field(description="The first numeric value")
    b: float = Field(description="The second numeric value")


def calculate_sum(a: float, b: float) -> float:
    """Add two numeric values and return the result."""

    return a + b


async def _calculate_sum_async(a: float, b: float) -> float:
    return calculate_sum(a, b)


calculate_sum_tool = StructuredTool.from_function(
    func=calculate_sum,
    coroutine=_calculate_sum_async,
    name="calculate_sum",
    description="Add two numeric values and return the result.",
    args_schema=CalculateSumInput,
)
