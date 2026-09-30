"""Versioned request, result, and manifest contracts."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class AgentManifest(BaseModel):
    """Static metadata exported by an independently versioned agent package."""

    model_config = ConfigDict(frozen=True)

    agent_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    version: str
    description: str
    capabilities: tuple[str, ...] = ()
    tool_ids: tuple[str, ...] = ()

    @field_validator("capabilities", "tool_ids")
    @classmethod
    def values_must_be_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("values must be unique")
        return value


class AgentRequest(BaseModel):
    """Minimal generic envelope; real agents should expose narrower models."""

    model_config = ConfigDict(frozen=True)

    input: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentResult(BaseModel):
    """Minimal execution result envelope."""

    model_config = ConfigDict(frozen=True)

    output: str
    metadata: dict[str, Any] = Field(default_factory=dict)
