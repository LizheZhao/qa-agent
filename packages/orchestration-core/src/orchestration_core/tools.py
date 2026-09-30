"""Normalized tool outcomes shared at the framework boundary."""

from typing import Any

from pydantic import BaseModel, ConfigDict


class ToolError(BaseModel):
    model_config = ConfigDict(frozen=True)

    code: str
    message: str
    retryable: bool = False


class ToolResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    value: Any | None = None
    error: ToolError | None = None
