"""In-process graph execution seam, intentionally separate from API routes."""

import asyncio
from typing import Any

from orchestration_core import ExecutionContext, ExecutionStatus, SpanKind

from agentic_orchestration.execution.agent_registry import (
    GRAPH_RECURSION_LIMIT,
    AgentRegistry,
)
from agentic_orchestration.observability.callbacks import GraphTraceCallback
from agentic_orchestration.observability.failures import capture_failure_detail
from agentic_orchestration.observability.recorder import TraceRecorder, sanitized_execution_error


class Executor:
    def __init__(
        self,
        registry: AgentRegistry,
        recorder: TraceRecorder,
        graph_definition_ids: dict[str, str],
        graph_predecessors: dict[str, dict[str, tuple[str, ...]]] | None = None,
    ) -> None:
        self._registry = registry
        self._recorder = recorder
        self._graph_definition_ids = dict(graph_definition_ids)
        self._graph_predecessors = dict(graph_predecessors or {})

    async def begin(self, agent_name: str, context: ExecutionContext) -> None:
        await self._recorder.begin(context, agent_name)

    async def complete(self, context: ExecutionContext, *, committed_turn_number: int) -> None:
        run = await self._recorder.active(context.trace_id)
        try:
            await run.finish(committed_turn_number=committed_turn_number)
        finally:
            await self._recorder.release(context.trace_id)

    async def fail(self, context: ExecutionContext) -> None:
        run = await self._recorder.active(context.trace_id)
        try:
            await run.finish(error=sanitized_execution_error())
        finally:
            await self._recorder.release(context.trace_id)

    async def cancel(self, context: ExecutionContext) -> None:
        run = await self._recorder.active(context.trace_id)
        try:
            await run.finish(cancelled=True)
        finally:
            await self._recorder.release(context.trace_id)

    async def invoke(
        self,
        agent_name: str,
        state: dict[str, Any],
        *,
        context: ExecutionContext,
    ) -> dict[str, Any]:
        if "execution_context" in state:
            raise ValueError("execution_context is supplied separately from agent state")
        run = await self._recorder.active(context.trace_id)
        definition_id = self._graph_definition_ids.get(agent_name)
        predecessors = self._graph_predecessors.get(agent_name)
        if definition_id is None or predecessors is None:
            run.mark_degraded()
        graph_span = await run.start_span(
            agent_id=agent_name,
            kind=SpanKind.GRAPH,
            parent_span_id=context.root_span_id,
            graph_definition_id=definition_id,
        )
        graph_context = context.model_copy(update={"parent_span_id": graph_span.span_id})
        callback = (
            GraphTraceCallback(
                run,
                agent_id=agent_name,
                graph_span_id=graph_span.span_id,
                graph_definition_id=definition_id,
                graph_predecessors=predecessors,
            )
            if definition_id is not None
            else None
        )
        try:
            invocation_state = {**state, "execution_context": graph_context}
            graph = self._registry.get(agent_name)
            # thread_id is required once a graph is compiled with a checkpointer (see
            # AgentRegistry.compile) -- it's how LangGraph knows which run's state to checkpoint,
            # e.g. for an interrupt() pause to later resume into the same run. Scoped to trace_id
            # so each turn attempt gets its own checkpoint space.
            invoke_config: dict[str, Any] = {
                "configurable": {"thread_id": context.trace_id},
                "recursion_limit": GRAPH_RECURSION_LIMIT,
            }
            if callback is not None:
                invoke_config["callbacks"] = [callback]
            result: dict[str, Any] = await graph.ainvoke(invocation_state, config=invoke_config)
        except BaseException as exc:
            cancelled = isinstance(exc, asyncio.CancelledError)
            await run.complete_span(
                graph_span.span_id,
                error=None if cancelled else sanitized_execution_error(),
                failure_detail=None if cancelled else capture_failure_detail(exc),
                status=ExecutionStatus.CANCELLED if cancelled else ExecutionStatus.FAILED,
            )
            raise
        else:
            await run.complete_span(graph_span.span_id)
            return result
