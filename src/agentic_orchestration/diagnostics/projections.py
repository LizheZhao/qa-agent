"""Controlled projections from durable execution records."""

from __future__ import annotations

from typing import Any

from orchestration_core import ExecutionSpan, ExecutionTrace, SpanKind

from agentic_orchestration.diagnostics.contracts import (
    AttemptSummary,
    CapturePresence,
    ModelObservation,
    SpanDetail,
    SpanSummary,
    ToolObservation,
    TraceDetail,
    UsageObservation,
)


def attempt_summary(trace: ExecutionTrace) -> AttemptSummary:
    return AttemptSummary.model_validate(
        trace.model_dump(
            include={
                "trace_id",
                "session_id",
                "request_origin",
                "attempted_turn_number",
                "committed_turn_number",
                "status",
                "observability_status",
                "started_at",
                "completed_at",
                "error",
            }
        )
    )


def trace_detail(trace: ExecutionTrace) -> TraceDetail:
    return TraceDetail.model_validate(
        {
            **attempt_summary(trace).model_dump(),
            "root_span_id": trace.root_span_id,
            "sequence_started": trace.sequence_started,
            "sequence_completed": trace.sequence_completed,
            "deployment_version": trace.deployment_version,
            "application_version": trace.application_version,
            "core_version": trace.core_version,
            "agent_versions": trace.agent_versions,
        }
    )


def span_summary(span: ExecutionSpan) -> SpanSummary:
    return SpanSummary(
        span_id=span.span_id,
        parent_span_id=span.parent_span_id,
        caused_by_span_ids=list(span.caused_by_span_ids),
        agent_id=span.agent_id,
        agent_version=span.agent_version,
        graph_definition_id=span.graph_definition_id,
        definition_node_id=span.definition_node_id,
        kind=span.kind,
        status=span.status,
        sequence_started=span.sequence_started,
        sequence_completed=span.sequence_completed,
        superstep=span.superstep,
        iteration=span.iteration,
        started_at=span.started_at,
        completed_at=span.completed_at,
        duration_ms=span.duration_ms,
        prompt_template_id=span.prompt_template_id,
        captures=CapturePresence(
            input=span.input is not None,
            resolved_messages=span.resolved_messages is not None,
            output=span.output is not None,
            state_delta=span.state_delta is not None,
        ),
        error=span.error,
    )


def span_detail(span: ExecutionSpan) -> SpanDetail:
    attributes = span.attributes
    model = _model_observation(attributes) if span.kind == SpanKind.MODEL else None
    tool = (
        ToolObservation(tool_id=_string(attributes.get("tool_id")))
        if span.kind == SpanKind.TOOL
        else None
    )
    triggers = attributes.get("langgraph_triggers", [])
    return SpanDetail(
        **span_summary(span).model_dump(),
        input=span.input,
        resolved_messages=span.resolved_messages,
        output=span.output,
        state_delta=span.state_delta,
        model=model,
        tool=tool,
        scheduler_triggers=[str(item) for item in triggers] if isinstance(triggers, list) else [],
        failure_detail=span.failure_detail,
    )


def _model_observation(attributes: dict[str, Any]) -> ModelObservation:
    identity_value = attributes.get("model_identity")
    identity: str | None
    if isinstance(identity_value, list):
        identity = ".".join(str(item) for item in identity_value)
    else:
        identity = _string(identity_value)
    return ModelObservation(
        identity=identity,
        provider=_string(attributes.get("provider")),
        usage=_usage(attributes.get("usage")),
    )


def _usage(value: Any) -> UsageObservation | None:
    if not isinstance(value, dict):
        return None
    token_usage = value.get("token_usage")
    candidates: dict[str, Any] = token_usage if isinstance(token_usage, dict) else value
    input_tokens = _integer(candidates.get("input_tokens", candidates.get("prompt_tokens")))
    output_tokens = _integer(candidates.get("output_tokens", candidates.get("completion_tokens")))
    total_tokens = _integer(candidates.get("total_tokens"))
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens
    if input_tokens is None and output_tokens is None and total_tokens is None:
        return None
    return UsageObservation(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
    )


def _string(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _integer(value: Any) -> int | None:
    return value if isinstance(value, int) and value >= 0 else None
