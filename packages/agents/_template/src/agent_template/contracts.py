"""TODO: define narrow, agent-specific request and result models here."""

from pydantic import BaseModel


class AgentInput(BaseModel):
    request: str


class AgentOutput(BaseModel):
    result: str
