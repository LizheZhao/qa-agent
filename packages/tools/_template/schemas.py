"""Typed inputs for the tool template."""

from pydantic import BaseModel, Field


class ToolInput(BaseModel):
    value: str = Field(description="TODO: describe this input for the model")
