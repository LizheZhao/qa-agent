"""Controlled outcomes emitted by the router model."""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

RoutingOutcome = Literal["answer", "delegate", "clarify", "reject"]


class RouteDecision(BaseModel):
    """Choose one routing outcome without exposing free-form reasoning."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    outcome: RoutingOutcome
    target_agent: str | None = Field(
        default=None,
        description="Eligible agent ID for delegate; null for every other outcome.",
    )
    reason_code: str = Field(
        min_length=1,
        description="Short operational category, not hidden reasoning.",
    )
    response: str | None = Field(
        default=None,
        description="User-facing text for answer, clarify, or reject; null for delegate.",
    )

    @model_validator(mode="after")
    def fields_must_match_outcome(self) -> Self:
        if self.outcome == "delegate":
            if self.target_agent is None or self.response is not None:
                raise ValueError("delegate requires target_agent and forbids response")
        elif self.target_agent is not None or self.response is None or not self.response.strip():
            raise ValueError(f"{self.outcome} requires response and forbids target_agent")
        return self
