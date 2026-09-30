"""Versioned, storage-neutral contracts for execution observability."""

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

OBSERVABILITY_SCHEMA_VERSION: Literal[4] = 4
NODE_KIND_METADATA_KEY = "orchestration_node_kind"
PROMPT_TEMPLATE_ID_METADATA_KEY = "orchestration_prompt_template_id"
PROMPT_TEMPLATE_METADATA_KEY = "orchestration_prompt_template"


class ObservabilityModel(BaseModel):
    """Common behavior for durable observability records."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ExecutionStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"


class ObservabilityStatus(StrEnum):
    """Durability state independent from the outcome of the application turn."""

    RECORDING = "recording"
    COMPLETE = "complete"
    DEGRADED = "degraded"


class RequestOrigin(StrEnum):
    """Descriptive request source; never an authorization signal."""

    API = "api"
    DIAGNOSTIC_UI = "diagnostic_ui"
    UNKNOWN = "unknown"


class SpanKind(StrEnum):
    TURN = "turn"
    GRAPH = "graph"
    NODE = "node"
    MODEL = "model"
    TOOL = "tool"
    CHILD_AGENT = "child_agent"


class GraphNodeKind(StrEnum):
    START = "start"
    END = "end"
    DETERMINISTIC = "deterministic"
    ROUTER = "router"
    MODEL = "model"
    TOOL = "tool"
    CHILD_AGENT = "child_agent"
    SUBGRAPH = "subgraph"


class GraphEdgeKind(StrEnum):
    DIRECT = "direct"
    CONDITIONAL = "conditional"


class ContentCaptureMode(StrEnum):
    INLINE = "inline"
    REFERENCE = "reference"
    OMITTED = "omitted"


class EventKind(StrEnum):
    TRACE_STARTED = "trace_started"
    TRACE_COMPLETED = "trace_completed"
    TRACE_FAILED = "trace_failed"
    TRACE_CANCELLED = "trace_cancelled"
    SPAN_STARTED = "span_started"
    SPAN_COMPLETED = "span_completed"
    SPAN_FAILED = "span_failed"
    SPAN_CANCELLED = "span_cancelled"
    SPAN_SKIPPED = "span_skipped"


class ExecutionContext(ObservabilityModel):
    """Root attempt identity propagated unchanged through an execution tree."""

    trace_id: str
    root_span_id: str
    session_id: str
    attempted_turn_number: int = Field(ge=1)
    request_origin: RequestOrigin = RequestOrigin.API
    parent_span_id: str | None = None


class CapturedContent(ObservabilityModel):
    """A controlled inline value, authorized artifact reference, or omission marker."""

    mode: ContentCaptureMode
    content_type: str
    value: Any | None = None
    reference_id: str | None = None
    byte_size: int | None = Field(default=None, ge=0)
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def capture_shape_matches_mode(self) -> Self:
        if self.mode == ContentCaptureMode.INLINE:
            if self.reference_id is not None or self.reason is not None:
                raise ValueError("inline content cannot have a reference_id or omission reason")
        elif self.mode == ContentCaptureMode.REFERENCE:
            if self.reference_id is None or self.value is not None or self.reason is not None:
                raise ValueError("referenced content requires only a reference_id")
        elif self.value is not None or self.reference_id is not None or self.reason is None:
            raise ValueError("omitted content requires only a reason")
        return self


class PromptTemplateDefinition(ObservabilityModel):
    template_id: str
    template_text: str
    version: str | None = None


class GraphNodeDefinition(ObservabilityModel):
    node_id: str
    display_name: str
    kind: GraphNodeKind
    owner_path: tuple[str, ...] = ()
    prompt_template: PromptTemplateDefinition | None = None
    destination_agent_id: str | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)


class GraphEdgeDefinition(ObservabilityModel):
    source_node_id: str
    target_node_id: str
    kind: GraphEdgeKind = GraphEdgeKind.DIRECT
    label: str | None = None

    @model_validator(mode="after")
    def conditional_edges_are_labelled(self) -> Self:
        if self.kind == GraphEdgeKind.CONDITIONAL and not self.label:
            raise ValueError("conditional graph edges require a label")
        if self.kind == GraphEdgeKind.DIRECT and self.label is not None:
            raise ValueError("direct graph edges cannot have a conditional label")
        return self


class GraphDefinition(ObservabilityModel):
    schema_version: Literal[4] = OBSERVABILITY_SCHEMA_VERSION
    definition_id: str
    definition_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    agent_id: str
    agent_version: str
    entrypoint: str
    nodes: tuple[GraphNodeDefinition, ...]
    edges: tuple[GraphEdgeDefinition, ...] = ()
    created_at: datetime

    @field_validator("created_at")
    @classmethod
    def created_at_is_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware")
        return value

    @model_validator(mode="after")
    def topology_is_well_formed(self) -> Self:
        node_ids = [node.node_id for node in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("graph definition node IDs must be unique")
        known_nodes = set(node_ids)
        if sum(node.kind == GraphNodeKind.START for node in self.nodes) != 1:
            raise ValueError("graph definition must contain exactly one START node")
        if sum(node.kind == GraphNodeKind.END for node in self.nodes) != 1:
            raise ValueError("graph definition must contain exactly one END node")
        for edge in self.edges:
            if edge.source_node_id not in known_nodes or edge.target_node_id not in known_nodes:
                raise ValueError("graph definition edges must reference known nodes")
        return self


class SanitizedExecutionError(ObservabilityModel):
    code: str
    message: str
    category: str | None = None
    retryable: bool = False


class TracebackFrame(ObservabilityModel):
    filename: str = Field(min_length=1, max_length=512)
    function: str = Field(min_length=1, max_length=256)
    line_number: int = Field(ge=1)


class ExceptionObservation(ObservabilityModel):
    exception_type: str = Field(min_length=1, max_length=256)
    exception_module: str = Field(min_length=1, max_length=256)
    message: str = Field(max_length=2048)
    frames: tuple[TracebackFrame, ...] = Field(default=(), max_length=32)


class FailureDetail(ObservabilityModel):
    """Bounded, redacted internal detail for one failed execution span."""

    fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    exception_chain: tuple[ExceptionObservation, ...] = Field(min_length=1, max_length=5)
    redacted: bool = False
    truncated: bool = False


class ExecutionTrace(ObservabilityModel):
    schema_version: Literal[4] = OBSERVABILITY_SCHEMA_VERSION
    trace_id: str
    root_span_id: str
    session_id: str
    request_origin: RequestOrigin = RequestOrigin.API
    attempted_turn_number: int = Field(ge=1)
    committed_turn_number: int | None = Field(default=None, ge=1)
    status: ExecutionStatus
    observability_status: ObservabilityStatus = ObservabilityStatus.RECORDING
    sequence_started: int = Field(ge=1)
    sequence_completed: int | None = Field(default=None, ge=1)
    snapshot_sequence: int = Field(ge=1)
    started_at: datetime
    completed_at: datetime | None = None
    deployment_version: str
    application_version: str
    core_version: str
    agent_versions: dict[str, str] = Field(default_factory=dict)
    error: SanitizedExecutionError | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def trace_state_is_consistent(self) -> Self:
        _validate_temporal_state(self.status, self.started_at, self.completed_at, self.error)
        terminal = self.status not in {ExecutionStatus.PENDING, ExecutionStatus.RUNNING}
        if terminal != (self.sequence_completed is not None):
            raise ValueError("terminal traces require sequence_completed")
        if terminal and self.observability_status == ObservabilityStatus.RECORDING:
            raise ValueError("terminal traces cannot have recording observability status")
        if not terminal and self.observability_status != ObservabilityStatus.RECORDING:
            raise ValueError("running traces require recording observability status")
        expected_snapshot_sequence = self.sequence_completed or self.sequence_started
        if self.snapshot_sequence != expected_snapshot_sequence:
            raise ValueError("snapshot_sequence must identify the latest trace lifecycle change")
        return self


class ExecutionSpan(ObservabilityModel):
    schema_version: Literal[4] = OBSERVABILITY_SCHEMA_VERSION
    trace_id: str
    span_id: str
    parent_span_id: str | None = None
    caused_by_span_ids: tuple[str, ...] = ()
    session_id: str
    attempted_turn_number: int = Field(ge=1)
    agent_id: str
    agent_version: str
    graph_definition_id: str | None = None
    definition_node_id: str | None = None
    kind: SpanKind
    status: ExecutionStatus
    sequence_started: int = Field(ge=1)
    sequence_completed: int | None = Field(default=None, ge=1)
    snapshot_sequence: int = Field(ge=1)
    superstep: int = Field(default=0, ge=0)
    iteration: int = Field(default=1, ge=1)
    started_at: datetime
    completed_at: datetime | None = None
    duration_ms: float | None = Field(default=None, ge=0)
    input: CapturedContent | None = None
    prompt_template_id: str | None = None
    resolved_messages: CapturedContent | None = None
    output: CapturedContent | None = None
    state_delta: CapturedContent | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)
    error: SanitizedExecutionError | None = None
    failure_detail: FailureDetail | None = None

    @model_validator(mode="after")
    def span_state_is_consistent(self) -> Self:
        _validate_temporal_state(self.status, self.started_at, self.completed_at, self.error)
        terminal = self.status not in {ExecutionStatus.PENDING, ExecutionStatus.RUNNING}
        if terminal != (self.sequence_completed is not None):
            raise ValueError("terminal spans require sequence_completed")
        if self.sequence_completed is not None and self.sequence_completed < self.sequence_started:
            raise ValueError("sequence_completed cannot precede sequence_started")
        expected_snapshot_sequence = self.sequence_completed or self.sequence_started
        if self.snapshot_sequence != expected_snapshot_sequence:
            raise ValueError("snapshot_sequence must identify the latest span lifecycle change")
        if terminal != (self.duration_ms is not None):
            raise ValueError("terminal spans require duration_ms")
        if len(self.caused_by_span_ids) != len(set(self.caused_by_span_ids)):
            raise ValueError("caused_by_span_ids must be unique")
        if self.span_id in self.caused_by_span_ids:
            raise ValueError("a span cannot cause itself")
        if self.failure_detail is not None and self.status != ExecutionStatus.FAILED:
            raise ValueError("failure_detail is allowed only on failed spans")
        return self


class ExecutionEvent(ObservabilityModel):
    """An ephemeral live-delivery envelope ordered within one trace."""

    schema_version: Literal[4] = OBSERVABILITY_SCHEMA_VERSION
    trace_id: str
    sequence: int = Field(ge=1)
    kind: EventKind
    occurred_at: datetime
    span_id: str | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)

    @field_validator("occurred_at")
    @classmethod
    def occurred_at_is_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("occurred_at must be timezone-aware")
        return value

    @model_validator(mode="after")
    def event_target_matches_kind(self) -> Self:
        is_span_event = self.kind.value.startswith("span_")
        if is_span_event != (self.span_id is not None):
            raise ValueError("span events require span_id; trace events must omit it")
        return self


def _validate_temporal_state(
    status: ExecutionStatus,
    started_at: datetime,
    completed_at: datetime | None,
    error: SanitizedExecutionError | None,
) -> None:
    for field_name, value in (("started_at", started_at), ("completed_at", completed_at)):
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError(f"{field_name} must be timezone-aware")
    terminal = status not in {ExecutionStatus.PENDING, ExecutionStatus.RUNNING}
    if terminal != (completed_at is not None):
        raise ValueError("terminal records require completed_at")
    if completed_at is not None and completed_at < started_at:
        raise ValueError("completed_at cannot precede started_at")
    if (status == ExecutionStatus.FAILED) != (error is not None):
        raise ValueError("error must be set exactly when status is failed")
