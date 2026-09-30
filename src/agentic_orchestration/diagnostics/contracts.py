"""Typed, storage-neutral responses for diagnostic APIs."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from orchestration_core import (
    CapturedContent,
    ExecutionStatus,
    FailureDetail,
    ObservabilityStatus,
    RequestOrigin,
    SanitizedExecutionError,
    SpanKind,
)
from pydantic import BaseModel, ConfigDict, Field


class DiagnosticModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Page[ItemT](DiagnosticModel):
    items: list[ItemT]
    next_cursor: str | None = None


class SessionSummary(DiagnosticModel):
    session_id: str
    agent_id: str
    display_title: str = Field(min_length=1, max_length=160)
    revision: int = Field(ge=1)
    status: Literal["active", "archived"]
    created_at: datetime
    updated_at: datetime


class ConversationMessage(DiagnosticModel):
    role: Literal["user", "assistant"]
    content: str


class TurnSummary(DiagnosticModel):
    turn_number: int = Field(ge=1)
    committed_at: datetime
    input: ConversationMessage
    response: ConversationMessage
    routing_outcome: Literal["answer", "delegate", "clarify", "reject"]
    selected_agent_id: str | None = None
    trace_id: str | None = None


class AttemptSummary(DiagnosticModel):
    trace_id: str
    session_id: str
    request_origin: RequestOrigin
    attempted_turn_number: int = Field(ge=1)
    committed_turn_number: int | None = Field(default=None, ge=1)
    status: ExecutionStatus
    observability_status: ObservabilityStatus
    started_at: datetime
    completed_at: datetime | None = None
    error: SanitizedExecutionError | None = None
    input_preview: str | None = Field(default=None, min_length=1, max_length=160)


class TraceDetail(AttemptSummary):
    root_span_id: str
    sequence_started: int = Field(ge=1)
    sequence_completed: int | None = Field(default=None, ge=1)
    deployment_version: str
    application_version: str
    core_version: str
    agent_versions: dict[str, str]


class CapturePresence(DiagnosticModel):
    input: bool = False
    resolved_messages: bool = False
    output: bool = False
    state_delta: bool = False


class SpanSummary(DiagnosticModel):
    span_id: str
    parent_span_id: str | None = None
    caused_by_span_ids: list[str]
    agent_id: str
    agent_version: str
    graph_definition_id: str | None = None
    definition_node_id: str | None = None
    kind: SpanKind
    status: ExecutionStatus
    sequence_started: int = Field(ge=1)
    sequence_completed: int | None = Field(default=None, ge=1)
    superstep: int = Field(ge=0)
    iteration: int = Field(ge=1)
    started_at: datetime
    completed_at: datetime | None = None
    duration_ms: float | None = Field(default=None, ge=0)
    prompt_template_id: str | None = None
    captures: CapturePresence
    error: SanitizedExecutionError | None = None


class UsageObservation(DiagnosticModel):
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)


class ModelObservation(DiagnosticModel):
    identity: str | None = None
    provider: str | None = None
    usage: UsageObservation | None = None


class ToolObservation(DiagnosticModel):
    tool_id: str | None = None


class SpanDetail(SpanSummary):
    input: CapturedContent | None = None
    resolved_messages: CapturedContent | None = None
    output: CapturedContent | None = None
    state_delta: CapturedContent | None = None
    model: ModelObservation | None = None
    tool: ToolObservation | None = None
    scheduler_triggers: list[str] = Field(default_factory=list)
    failure_detail: FailureDetail | None = None


class AgentDeployment(DiagnosticModel):
    agent_id: str
    version: str
    routable: bool
    graph_definition_id: str | None = None


class DeploymentDiagnostic(DiagnosticModel):
    name: str
    revision: str
    entry_agent: str
    agents: list[AgentDeployment]


def text_content(value: str | list[str | dict[str, Any]]) -> str:
    if isinstance(value, str):
        return value
    return "\n".join(part if isinstance(part, str) else str(part.get("text", "")) for part in value)


def conversation_title(value: str) -> str:
    normalized = " ".join(value.split()) or "Untitled conversation"
    return normalized if len(normalized) <= 160 else f"{normalized[:159].rstrip()}…"
