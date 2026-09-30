"""Translate LangGraph and LangChain callbacks into normalized execution spans."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Collection, Mapping
from typing import Any
from uuid import UUID

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult
from orchestration_core import ExecutionSpan, ExecutionStatus, SpanKind

from agentic_orchestration.observability.capture import capture_content
from agentic_orchestration.observability.failures import capture_failure_detail
from agentic_orchestration.observability.recorder import TraceRun, sanitized_execution_error

_PROMPT_ID = "orchestration_prompt_template_id"


class GraphTraceCallback(AsyncCallbackHandler):
    """Record one compiled graph invocation and its nested model and tool calls."""

    # Diagnostics cannot become part of the graph's availability boundary.
    raise_error = False
    run_inline = True

    def __init__(
        self,
        run: TraceRun,
        *,
        agent_id: str,
        graph_span_id: str,
        graph_definition_id: str,
        graph_predecessors: Mapping[str, Collection[str]] | None = None,
    ) -> None:
        super().__init__()
        self._run = run
        self._agent_id = agent_id
        self._graph_span_id = graph_span_id
        self._graph_definition_id = graph_definition_id
        self._graph_predecessors = {
            node_id: tuple(source_ids) for node_id, source_ids in (graph_predecessors or {}).items()
        }
        self._callback_spans: dict[UUID, ExecutionSpan] = {}
        self._node_iterations: defaultdict[str, int] = defaultdict(int)
        self._node_spans_by_step: defaultdict[int, list[str]] = defaultdict(list)
        self._last_span_by_node: dict[str, str] = {}
        self._last_step_by_node: dict[str, int] = {}

    async def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        del serialized, parent_run_id, kwargs
        node_metadata = metadata or {}
        node_id = node_metadata.get("langgraph_node")
        step = node_metadata.get("langgraph_step")
        if not isinstance(node_id, str) or not isinstance(step, int):
            return
        if f"graph:step:{step}" not in (tags or []):
            return
        self._node_iterations[node_id] += 1
        span = await self._run.start_span(
            agent_id=self._agent_id,
            kind=SpanKind.NODE,
            parent_span_id=self._graph_span_id,
            caused_by_span_ids=self._causal_predecessors(node_metadata, step),
            graph_definition_id=self._graph_definition_id,
            definition_node_id=node_id,
            superstep=step,
            iteration=self._node_iterations[node_id],
            input=capture_content(inputs, content_type="application/vnd.orchestration.state+json"),
            attributes={"langgraph_triggers": list(node_metadata.get("langgraph_triggers", ()))},
        )
        self._callback_spans[run_id] = span
        checkpoint_namespace = node_metadata.get("langgraph_checkpoint_ns")
        if isinstance(checkpoint_namespace, str):
            self._run.register_node_span(checkpoint_namespace, span.span_id)
        self._node_spans_by_step[step].append(span.span_id)
        self._last_span_by_node[node_id] = span.span_id
        self._last_step_by_node[node_id] = step

    async def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        del parent_run_id, kwargs
        span = self._callback_spans.pop(run_id, None)
        if span is None or span.kind != SpanKind.NODE:
            return
        captured = capture_content(
            outputs, content_type="application/vnd.orchestration.state-delta+json"
        )
        await self._run.complete_span(span.span_id, state_delta=captured)

    async def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        del parent_run_id, kwargs
        await self._fail_callback_span(run_id, error)

    async def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        del tags, kwargs
        runtime_metadata = metadata or {}
        node_id = runtime_metadata.get("langgraph_node")
        step = runtime_metadata.get("langgraph_step", 0)
        parent_span = self._callback_spans.get(parent_run_id) if parent_run_id is not None else None
        prompt_template_id = runtime_metadata.get(_PROMPT_ID)
        span = await self._run.start_span(
            agent_id=self._agent_id,
            kind=SpanKind.MODEL,
            parent_span_id=parent_span.span_id if parent_span else self._graph_span_id,
            graph_definition_id=self._graph_definition_id,
            definition_node_id=node_id if isinstance(node_id, str) else None,
            superstep=step if isinstance(step, int) else 0,
            iteration=parent_span.iteration if parent_span else 1,
            prompt_template_id=(
                prompt_template_id if isinstance(prompt_template_id, str) else None
            ),
            resolved_messages=capture_content(
                messages[0] if messages else [],
                content_type="application/vnd.orchestration.messages+json",
            ),
            attributes={
                "model_identity": serialized.get("id", []),
                "provider": runtime_metadata.get("ls_provider"),
            },
        )
        self._callback_spans[run_id] = span

    async def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        del parent_run_id, kwargs
        span = self._callback_spans.pop(run_id, None)
        if span is None:
            return
        messages = [
            generation.message
            for generations in response.generations
            for generation in generations
            if hasattr(generation, "message")
        ]
        await self._run.complete_span(
            span.span_id,
            output=capture_content(
                messages, content_type="application/vnd.orchestration.messages+json"
            ),
            attributes={"usage": response.llm_output or {}},
        )

    async def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        del parent_run_id, kwargs
        await self._fail_callback_span(run_id, error)

    async def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        del input_str, tags, kwargs
        runtime_metadata = metadata or {}
        node_id = runtime_metadata.get("langgraph_node")
        step = runtime_metadata.get("langgraph_step", 0)
        parent_span = self._callback_spans.get(parent_run_id) if parent_run_id is not None else None
        span = await self._run.start_span(
            agent_id=self._agent_id,
            kind=SpanKind.TOOL,
            parent_span_id=parent_span.span_id if parent_span else self._graph_span_id,
            graph_definition_id=self._graph_definition_id,
            definition_node_id=node_id if isinstance(node_id, str) else None,
            superstep=step if isinstance(step, int) else 0,
            iteration=parent_span.iteration if parent_span else 1,
            input=capture_content(inputs or {}, content_type="application/json"),
            attributes={"tool_id": serialized.get("name")},
        )
        self._callback_spans[run_id] = span

    async def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        del parent_run_id, kwargs
        span = self._callback_spans.pop(run_id, None)
        if span is not None:
            await self._run.complete_span(
                span.span_id,
                output=capture_content(output, content_type="application/json"),
            )

    async def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        del parent_run_id, kwargs
        await self._fail_callback_span(run_id, error)

    def _causal_predecessors(self, metadata: dict[str, Any], step: int) -> tuple[str, ...]:
        node_id = metadata.get("langgraph_node")
        if isinstance(node_id, str):
            candidates = [
                (self._last_step_by_node[source_id], self._last_span_by_node[source_id])
                for source_id in self._graph_predecessors.get(node_id, ())
                if source_id in self._last_span_by_node
                and self._last_step_by_node[source_id] < step
            ]
            if candidates:
                latest_step = max(candidate_step for candidate_step, _ in candidates)
                return tuple(
                    span_id
                    for candidate_step, span_id in candidates
                    if candidate_step == latest_step
                )
        previous_steps = [candidate for candidate in self._node_spans_by_step if candidate < step]
        if not previous_steps:
            return ()
        return tuple(self._node_spans_by_step[max(previous_steps)])

    async def _fail_callback_span(self, run_id: UUID, error: BaseException) -> None:
        span = self._callback_spans.pop(run_id, None)
        if span is not None:
            cancelled = isinstance(error, asyncio.CancelledError)
            await self._run.complete_span(
                span.span_id,
                error=None if cancelled else sanitized_execution_error(),
                failure_detail=None if cancelled else capture_failure_detail(error),
                status=ExecutionStatus.CANCELLED if cancelled else ExecutionStatus.FAILED,
            )
