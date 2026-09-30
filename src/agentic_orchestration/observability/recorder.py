"""Ordered trace lifecycle recording independent of graph implementation details."""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from time import monotonic
from typing import Any
from uuid import uuid4

from orchestration_core import (
    CapturedContent,
    EventKind,
    ExecutionContext,
    ExecutionEvent,
    ExecutionSpan,
    ExecutionStatus,
    ExecutionTrace,
    FailureDetail,
    ObservabilityStatus,
    SanitizedExecutionError,
    SpanKind,
)

from agentic_orchestration.observability.publisher import (
    ExecutionEventPublisher,
    NullExecutionEventPublisher,
)
from agentic_orchestration.observability.store import ExecutionTraceStore
from agentic_orchestration.observability.writer import BufferedExecutionWriter

logger = logging.getLogger(__name__)


class TraceRecorder:
    """Own active trace runs and allocate monotonic event sequences."""

    def __init__(
        self,
        store: ExecutionTraceStore,
        *,
        deployment_version: str,
        application_version: str,
        core_version: str,
        agent_versions: dict[str, str],
        event_publisher: ExecutionEventPublisher | None = None,
        flush_timeout_seconds: float = 2.0,
        max_pending_snapshots: int = 256,
    ) -> None:
        self._store = store
        self._deployment_version = deployment_version
        self._application_version = application_version
        self._core_version = core_version
        self._agent_versions = dict(agent_versions)
        self._event_publisher = event_publisher or NullExecutionEventPublisher()
        self._flush_timeout_seconds = flush_timeout_seconds
        self._max_pending_snapshots = max_pending_snapshots
        self._runs: dict[str, TraceRun] = {}
        self._lock = asyncio.Lock()

    async def begin(self, context: ExecutionContext, entry_agent_id: str) -> TraceRun:
        run = TraceRun(
            store=self._store,
            context=context,
            deployment_version=self._deployment_version,
            application_version=self._application_version,
            core_version=self._core_version,
            agent_versions=self._agent_versions,
            event_publisher=self._event_publisher,
            flush_timeout_seconds=self._flush_timeout_seconds,
            max_pending_snapshots=self._max_pending_snapshots,
        )
        async with self._lock:
            if context.trace_id in self._runs:
                raise ValueError(f"trace {context.trace_id!r} is already active")
            self._runs[context.trace_id] = run
        try:
            await run.begin(entry_agent_id)
        except BaseException as exc:
            cleanup = asyncio.create_task(
                run.finish(
                    error=None
                    if isinstance(exc, asyncio.CancelledError)
                    else sanitized_execution_error(),
                    cancelled=isinstance(exc, asyncio.CancelledError),
                )
            )
            try:
                await asyncio.shield(cleanup)
            except BaseException:
                logger.exception(
                    "trace cleanup failed after begin failure trace_id=%s", context.trace_id
                )
            async with self._lock:
                self._runs.pop(context.trace_id, None)
            raise
        return run

    async def active(self, trace_id: str) -> TraceRun:
        async with self._lock:
            try:
                return self._runs[trace_id]
            except KeyError as exc:
                raise RuntimeError(f"trace {trace_id!r} is not active") from exc

    async def release(self, trace_id: str) -> None:
        async with self._lock:
            self._runs.pop(trace_id, None)


class TraceRun:
    """Mutable sequencing coordinator for one trace; persisted records remain immutable."""

    def __init__(
        self,
        *,
        store: ExecutionTraceStore,
        context: ExecutionContext,
        deployment_version: str,
        application_version: str,
        core_version: str,
        agent_versions: dict[str, str],
        event_publisher: ExecutionEventPublisher,
        flush_timeout_seconds: float,
        max_pending_snapshots: int,
    ) -> None:
        self.context = context
        self._deployment_version = deployment_version
        self._application_version = application_version
        self._core_version = core_version
        self._agent_versions = agent_versions
        self._event_publisher = event_publisher
        self._flush_timeout_seconds = flush_timeout_seconds
        self._sequence = 0
        self._lock = asyncio.Lock()
        self._spans: dict[str, ExecutionSpan] = {}
        self._started_monotonic: dict[str, float] = {}
        self._node_spans_by_checkpoint_namespace: dict[str, str] = {}
        self._trace: ExecutionTrace | None = None
        self._diagnostics_degraded = False
        self._writer = BufferedExecutionWriter(
            store,
            trace_id=context.trace_id,
            max_pending_snapshots=max_pending_snapshots,
        )

    async def begin(self, entry_agent_id: str) -> None:
        started_at = datetime.now(UTC)
        async with self._lock:
            sequence = self._next_sequence()
            trace = ExecutionTrace(
                trace_id=self.context.trace_id,
                root_span_id=self.context.root_span_id,
                session_id=self.context.session_id,
                request_origin=self.context.request_origin,
                attempted_turn_number=self.context.attempted_turn_number,
                status=ExecutionStatus.RUNNING,
                sequence_started=sequence,
                snapshot_sequence=sequence,
                started_at=started_at,
                deployment_version=self._deployment_version,
                application_version=self._application_version,
                core_version=self._core_version,
                agent_versions=self._agent_versions,
            )
            self._trace = trace
            self._writer.enqueue(trace)
            self._publish_event_locked(
                ExecutionEvent(
                    trace_id=self.context.trace_id,
                    sequence=sequence,
                    kind=EventKind.TRACE_STARTED,
                    occurred_at=started_at,
                )
            )
        await self.start_span(
            agent_id=entry_agent_id,
            kind=SpanKind.TURN,
            span_id=self.context.root_span_id,
            parent_span_id=None,
        )

    def register_node_span(self, checkpoint_namespace: str, span_id: str) -> None:
        self._node_spans_by_checkpoint_namespace[checkpoint_namespace] = span_id

    def node_span_id(self, checkpoint_namespace: str | None) -> str | None:
        if checkpoint_namespace is None:
            return None
        return self._node_spans_by_checkpoint_namespace.get(checkpoint_namespace)

    def mark_degraded(self) -> None:
        """Record loss of diagnostic detail without affecting application execution."""

        self._diagnostics_degraded = True

    async def start_span(
        self,
        *,
        agent_id: str,
        kind: SpanKind,
        parent_span_id: str | None,
        span_id: str | None = None,
        caused_by_span_ids: tuple[str, ...] = (),
        graph_definition_id: str | None = None,
        definition_node_id: str | None = None,
        superstep: int = 0,
        iteration: int = 1,
        input: CapturedContent | None = None,
        prompt_template_id: str | None = None,
        resolved_messages: CapturedContent | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> ExecutionSpan:
        actual_span_id = span_id or str(uuid4())
        started_at = datetime.now(UTC)
        async with self._lock:
            sequence = self._next_sequence()
            agent_version = self._agent_versions.get(agent_id)
            if agent_version is None:
                agent_version = "unknown"
                self._diagnostics_degraded = True
            span = ExecutionSpan(
                trace_id=self.context.trace_id,
                span_id=actual_span_id,
                parent_span_id=parent_span_id,
                caused_by_span_ids=caused_by_span_ids,
                session_id=self.context.session_id,
                attempted_turn_number=self.context.attempted_turn_number,
                agent_id=agent_id,
                agent_version=agent_version,
                graph_definition_id=graph_definition_id,
                definition_node_id=definition_node_id,
                kind=kind,
                status=ExecutionStatus.RUNNING,
                sequence_started=sequence,
                snapshot_sequence=sequence,
                superstep=superstep,
                iteration=iteration,
                started_at=started_at,
                input=input,
                prompt_template_id=prompt_template_id,
                resolved_messages=resolved_messages,
                attributes=attributes or {},
            )
            self._spans[actual_span_id] = span
            self._started_monotonic[actual_span_id] = monotonic()
            self._writer.enqueue(span)
            self._publish_event_locked(
                ExecutionEvent(
                    trace_id=self.context.trace_id,
                    span_id=actual_span_id,
                    sequence=sequence,
                    kind=EventKind.SPAN_STARTED,
                    occurred_at=started_at,
                )
            )
        return span

    async def complete_span(
        self,
        span_id: str,
        *,
        error: SanitizedExecutionError | None = None,
        failure_detail: FailureDetail | None = None,
        output: CapturedContent | None = None,
        state_delta: CapturedContent | None = None,
        attributes: dict[str, Any] | None = None,
        status: ExecutionStatus | None = None,
    ) -> ExecutionSpan:
        completed_at = datetime.now(UTC)
        async with self._lock:
            current = self._spans[span_id]
            if current.status not in {ExecutionStatus.PENDING, ExecutionStatus.RUNNING}:
                return current
            sequence = self._next_sequence()
            terminal_status = status or (
                ExecutionStatus.FAILED if error is not None else ExecutionStatus.COMPLETED
            )
            if terminal_status not in {
                ExecutionStatus.COMPLETED,
                ExecutionStatus.FAILED,
                ExecutionStatus.CANCELLED,
                ExecutionStatus.SKIPPED,
            }:
                raise ValueError("span completion requires a terminal status")
            if (terminal_status == ExecutionStatus.FAILED) != (error is not None):
                raise ValueError("span errors must be set exactly for failed status")
            span = ExecutionSpan.model_validate(
                {
                    **current.model_dump(mode="python"),
                    "status": terminal_status,
                    "sequence_completed": sequence,
                    "snapshot_sequence": sequence,
                    "completed_at": completed_at,
                    "duration_ms": (monotonic() - self._started_monotonic[span_id]) * 1000,
                    "output": output,
                    "state_delta": state_delta,
                    "attributes": {**current.attributes, **(attributes or {})},
                    "error": error,
                    "failure_detail": failure_detail,
                }
            )
            self._spans[span_id] = span
            self._writer.enqueue(span)
            self._publish_event_locked(
                ExecutionEvent(
                    trace_id=self.context.trace_id,
                    span_id=span_id,
                    sequence=sequence,
                    kind={
                        ExecutionStatus.COMPLETED: EventKind.SPAN_COMPLETED,
                        ExecutionStatus.FAILED: EventKind.SPAN_FAILED,
                        ExecutionStatus.CANCELLED: EventKind.SPAN_CANCELLED,
                        ExecutionStatus.SKIPPED: EventKind.SPAN_SKIPPED,
                    }[terminal_status],
                    occurred_at=completed_at,
                )
            )
        return span

    async def finish(
        self,
        *,
        error: SanitizedExecutionError | None = None,
        committed_turn_number: int | None = None,
        cancelled: bool = False,
    ) -> ExecutionTrace:
        if self._trace is None:
            await self._writer.close(timeout_seconds=self._flush_timeout_seconds)
            raise RuntimeError("trace has not started")
        if self._trace.status not in {ExecutionStatus.PENDING, ExecutionStatus.RUNNING}:
            return self._trace
        if cancelled and error is not None:
            raise ValueError("cancelled traces cannot contain an execution error")
        terminal_status = (
            ExecutionStatus.CANCELLED
            if cancelled
            else ExecutionStatus.FAILED
            if error is not None
            else ExecutionStatus.COMPLETED
        )
        open_spans = sorted(
            (
                span
                for span in self._spans.values()
                if span.status in {ExecutionStatus.PENDING, ExecutionStatus.RUNNING}
            ),
            key=lambda span: span.sequence_started,
            reverse=True,
        )
        for span in open_spans:
            span_error = error if terminal_status == ExecutionStatus.FAILED else None
            await self.complete_span(
                span.span_id,
                error=span_error,
                status=terminal_status,
            )
        completed_at = datetime.now(UTC)
        kind = {
            ExecutionStatus.COMPLETED: EventKind.TRACE_COMPLETED,
            ExecutionStatus.FAILED: EventKind.TRACE_FAILED,
            ExecutionStatus.CANCELLED: EventKind.TRACE_CANCELLED,
        }[terminal_status]
        async with self._lock:
            sequence = self._next_sequence()
            trace = ExecutionTrace.model_validate(
                {
                    **self._trace.model_dump(mode="python"),
                    "status": terminal_status,
                    "observability_status": (
                        ObservabilityStatus.DEGRADED
                        if self._writer.degraded or self._diagnostics_degraded
                        else ObservabilityStatus.COMPLETE
                    ),
                    "sequence_completed": sequence,
                    "snapshot_sequence": sequence,
                    "completed_at": completed_at,
                    "error": error,
                    "committed_turn_number": committed_turn_number,
                }
            )
            self._trace = trace
            self._writer.enqueue(trace)
            self._publish_event_locked(
                ExecutionEvent(
                    trace_id=self.context.trace_id,
                    sequence=sequence,
                    kind=kind,
                    occurred_at=completed_at,
                )
            )
        durable = await self._writer.close(timeout_seconds=self._flush_timeout_seconds)
        if not durable:
            self._trace = ExecutionTrace.model_validate(
                {
                    **self._trace.model_dump(mode="python"),
                    "observability_status": ObservabilityStatus.DEGRADED,
                }
            )
        return self._trace

    def _publish_event_locked(self, event: ExecutionEvent) -> None:
        try:
            self._event_publisher.publish(event)
        except Exception as exc:
            logger.warning(
                "live execution event publish failed trace_id=%s type=%s",
                self.context.trace_id,
                type(exc).__name__,
            )

    def _next_sequence(self) -> int:
        self._sequence += 1
        return self._sequence


def sanitized_execution_error() -> SanitizedExecutionError:
    """Return a stable public diagnostic error without exception text or provider payloads."""

    return SanitizedExecutionError(
        code="agent_execution_failed",
        message="Agent execution failed",
        category="execution",
        retryable=False,
    )
