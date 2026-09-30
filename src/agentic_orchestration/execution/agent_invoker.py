"""Restricted adapter from typed child requests to startup-compiled graphs."""

from __future__ import annotations

import asyncio
from collections.abc import Collection
from typing import Any
from uuid import uuid4

from langchain_core.messages import AIMessage, BaseMessage
from langgraph.config import get_config
from langgraph.types import Command
from orchestration_core import (
    CONTINUATION_CONTRACT_VERSION,
    QUEUED_CLARIFICATIONS_CHANNEL,
    AgentExecutionError,
    AgentInvocationDepthError,
    AgentInvocationTimeoutError,
    ChildOutcome,
    ClarificationRequest,
    ContinuationNotResumableError,
    ContinuationRequest,
    ConversationRequest,
    ConversationResult,
    ForbiddenAgentError,
    InvalidAgentResultError,
    InvocationContext,
    PausedConversation,
    SpanKind,
    UnknownAgentError,
)
from pydantic import ValidationError

from agentic_orchestration.execution.agent_registry import (
    GRAPH_RECURSION_LIMIT,
    AgentRegistry,
)
from agentic_orchestration.observability.callbacks import GraphTraceCallback
from agentic_orchestration.observability.failures import capture_failure_detail
from agentic_orchestration.observability.recorder import TraceRecorder, sanitized_execution_error


class RegistryAgentInvoker:
    """Invoke only the child graphs explicitly granted to one caller."""

    def __init__(
        self,
        registry: AgentRegistry,
        *,
        caller_agent_id: str,
        allowed_agent_ids: Collection[str],
        timeout_seconds: float,
        max_depth: int = 1,
        recorder: TraceRecorder | None = None,
        graph_definition_ids: dict[str, str] | None = None,
        graph_predecessors: dict[str, dict[str, tuple[str, ...]]] | None = None,
    ) -> None:
        self._registry = registry
        self._caller_agent_id = caller_agent_id
        self._allowed_agent_ids = frozenset(allowed_agent_ids)
        self._timeout_seconds = timeout_seconds
        self._max_depth = max_depth
        self._recorder = recorder
        self._graph_definition_ids = dict(graph_definition_ids or {})
        self._graph_predecessors = dict(graph_predecessors or {})

    def _authorize(self, agent_id: str, context: InvocationContext) -> None:
        """Apply the same eligibility rules to a first invocation and to a continuation.

        A resume reaches a child directly, without the router choosing it again, so it has
        to re-check what routing would have checked: that the agent is still installed and
        still eligible for this caller under the deployment in force now.
        """

        if context.caller_agent_id != self._caller_agent_id:
            raise ForbiddenAgentError("Invocation context caller does not match invoker owner")
        if agent_id == context.caller_agent_id:
            raise ForbiddenAgentError("An agent cannot invoke itself")
        if agent_id not in self._registry.graphs:
            raise UnknownAgentError(f"Agent {agent_id!r} is not installed")
        if agent_id not in self._allowed_agent_ids:
            raise ForbiddenAgentError(f"Agent {agent_id!r} is not eligible for this caller")
        if context.depth >= self._max_depth:
            raise AgentInvocationDepthError("Agent invocation depth exceeded")

    def _pause_capable(self, agent_id: str) -> bool:
        """Whether this child was compiled with a checkpointer and can therefore pause.

        Pause capability is not a separate declaration to keep in step with reality: a
        graph can only interrupt if a saver was attached when it was compiled, so the
        compiled graph is the one honest source for it.
        """

        return getattr(self._registry.get(agent_id), "checkpointer", None) is not None

    async def invoke(
        self,
        agent_id: str,
        request: ConversationRequest,
        context: InvocationContext,
    ) -> ChildOutcome:
        self._authorize(agent_id, context)
        # A pause-capable child needs a thread of its own before it starts, because the
        # checkpoint that a later continuation resumes from is written under it. The run ID
        # is not the session ID: a session may run many child invocations over its life.
        run_id = str(uuid4()) if self._pause_capable(agent_id) else None
        result = await self._traced_run(
            agent_id,
            context,
            {"messages": list(request.messages), "execution_context": context.execution},
            run_id=run_id,
        )
        if run_id is not None and (paused := self._paused(agent_id, result, run_id)) is not None:
            return paused
        return ConversationResult(
            generated_messages=self._generated(agent_id, result, len(request.messages))
        )

    async def resume(
        self,
        agent_id: str,
        continuation: ContinuationRequest,
        context: InvocationContext,
    ) -> ChildOutcome:
        """Continue one paused child run directly, without re-entering the router.

        The router already chose this child and that choice was recorded; asking it again
        would re-run a model for a decision that is settled and could pick differently.
        What this does re-check is everything routing would have enforced anyway, plus the
        gates that only matter to a continuation: that the paused run still exists, that
        the deployed graph is still the one it paused against, and that the answer belongs
        to the clarification the producer actually stored.
        """

        self._authorize(agent_id, context)
        if not self._pause_capable(agent_id):
            raise ContinuationNotResumableError(
                "missing_checkpoint", f"Agent {agent_id!r} does not retain paused runs"
            )
        graph = self._registry.get(agent_id)
        config = {"configurable": {"thread_id": continuation.run_id}}
        snapshot = await graph.aget_state(config)
        active = self._resumable_clarification(agent_id, snapshot, continuation)
        # Validated against the clarification held in the checkpoint, never against the
        # caller's copy of it, so a submitted option must be one the producer offered.
        accepted = active.accept(continuation.response)
        baseline = len(snapshot.values.get("messages", ()) if snapshot.values else ())
        result = await self._traced_run(
            agent_id,
            context,
            Command(resume=accepted.model_dump(mode="json")),
            run_id=continuation.run_id,
        )
        if (paused := self._paused(agent_id, result, continuation.run_id)) is not None:
            return paused
        return ConversationResult(generated_messages=self._generated(agent_id, result, baseline))

    async def recover(self, agent_id: str, run_id: str) -> ConversationResult | None:
        """Read a finished run's answer from its checkpoint, running nothing.

        For the window between a graph completing and its turn being committed. If the
        process dies there, the work is done and its result is durable, but the session
        still shows a pause; resuming again would restart the interrupted node and repeat
        whatever the producer did after the answer. This reads the completed output
        instead, so recovery performs no model or tool side effects at all.

        Returns None when the run is not finished, which is the ordinary case: the caller
        then resumes as normal.
        """

        if not self._pause_capable(agent_id):
            return None
        snapshot = await self._registry.get(agent_id).aget_state(
            {"configurable": {"thread_id": run_id}}
        )
        if snapshot is None or snapshot.next or snapshot.interrupts:
            return None
        messages = (snapshot.values or {}).get("messages") or []
        final = next(
            (
                message
                for message in reversed(messages)
                if isinstance(message, AIMessage) and not message.tool_calls
            ),
            None,
        )
        if final is None:
            return None
        return ConversationResult(generated_messages=(final,))

    def _resumable_clarification(
        self,
        agent_id: str,
        snapshot: Any,
        continuation: ContinuationRequest,
    ) -> ClarificationRequest:
        """Decide whether this paused run may be continued, and with which clarification.

        Every refusal is a restart, never a best-effort answer: a run whose graph or
        contract has moved on cannot be reasoned about from state that described the old
        one, and a producer whose snapshot has gone stale would apply the answer to work
        that no longer exists.
        """

        if snapshot is None or not snapshot.config.get("configurable", {}).get("checkpoint_id"):
            raise ContinuationNotResumableError(
                "missing_checkpoint", "The paused run has no checkpoint to continue from"
            )
        interrupts = tuple(snapshot.interrupts)
        if not interrupts:
            raise ContinuationNotResumableError(
                "not_paused", "The run is not waiting on a clarification"
            )
        if continuation.continuation_contract_version != CONTINUATION_CONTRACT_VERSION:
            raise ContinuationNotResumableError(
                "contract_version", "The run paused under a different continuation contract"
            )
        installed_definition_id = self._graph_definition_ids.get(agent_id)
        if (
            continuation.graph_definition_id is not None
            and installed_definition_id is not None
            and continuation.graph_definition_id != installed_definition_id
        ):
            raise ContinuationNotResumableError(
                "graph_version", "The deployed graph changed while the run was paused"
            )
        active = self._one_clarification(agent_id, interrupts)
        if active.clarification_id != continuation.clarification_id:
            raise ContinuationNotResumableError(
                "unknown_clarification", "The answer does not address the active clarification"
            )
        if active.freshness_token != continuation.freshness_token:
            raise ContinuationNotResumableError(
                "stale_producer", "The producer snapshot behind the question has moved on"
            )
        return active

    def _one_clarification(
        self, agent_id: str, interrupts: tuple[Any, ...]
    ) -> ClarificationRequest:
        """Read the single clarification a paused child is presenting.

        More than one is a contract violation rather than something to arbitrate here: a
        producer with several ambiguities orders them with `ClarificationQueue` and asks
        one, so two live interrupts mean the queue was bypassed and the user would be shown
        a question whose position depends on which branch finished first.
        """

        if len(interrupts) != 1:
            raise InvalidAgentResultError(
                f"Agent {agent_id!r} must present exactly one clarification at a time"
            )
        try:
            return ClarificationRequest.model_validate(interrupts[0].value)
        except ValidationError as exc:
            raise InvalidAgentResultError(
                f"Agent {agent_id!r} paused with an invalid clarification request"
            ) from exc

    def _paused(self, agent_id: str, result: Any, run_id: str) -> PausedConversation | None:
        """Read a pause out of a graph result, or return None if the child finished."""

        interrupts = result.get("__interrupt__") if isinstance(result, dict) else None
        if not interrupts:
            return None
        active = self._one_clarification(agent_id, tuple(interrupts))
        queued = result.get(QUEUED_CLARIFICATIONS_CHANNEL) or ()
        # The channel keeps the active clarification at its head, so it is dropped here
        # rather than reported as also waiting behind itself.
        queued_ids = tuple(
            dict.fromkeys(
                ClarificationRequest.model_validate(request).clarification_id for request in queued
            )
        )
        return PausedConversation(
            agent_id=agent_id,
            run_id=run_id,
            clarification=active,
            queued_clarification_ids=tuple(
                identifier for identifier in queued_ids if identifier != active.clarification_id
            ),
            graph_definition_id=self._graph_definition_ids.get(agent_id),
        )

    def _generated(self, agent_id: str, result: Any, baseline: int) -> tuple[BaseMessage, ...]:
        """Validate a completed child result and return only what this call produced."""

        raw_messages = result.get("messages") if isinstance(result, dict) else None
        if not isinstance(raw_messages, list) or not all(
            isinstance(message, BaseMessage) for message in raw_messages
        ):
            raise InvalidAgentResultError(f"Agent {agent_id!r} returned invalid messages")
        generated = tuple(raw_messages[baseline:])
        final = next(
            (message for message in reversed(generated) if isinstance(message, AIMessage)), None
        )
        if final is None or final.tool_calls:
            raise InvalidAgentResultError(f"Agent {agent_id!r} returned no final assistant message")
        return generated

    async def _traced_run(
        self,
        agent_id: str,
        context: InvocationContext,
        graph_input: Any,
        *,
        run_id: str | None,
    ) -> Any:
        """Run one child graph inside its spans, timeout, and failure reporting.

        Shared by a first invocation and by a continuation so that a resumed run is
        observed exactly like the call that paused it, and so the two cannot drift in how
        they report failures.
        """

        run = await self._recorder.active(context.execution.trace_id) if self._recorder else None
        definition_id = self._graph_definition_ids.get(agent_id)
        predecessors = self._graph_predecessors.get(agent_id)
        if run is not None and (definition_id is None or predecessors is None):
            run.mark_degraded()
        child_span = (
            await run.start_span(
                agent_id=agent_id,
                kind=SpanKind.CHILD_AGENT,
                parent_span_id=(
                    run.node_span_id(_current_checkpoint_namespace())
                    or context.execution.parent_span_id
                ),
                attributes={"caller_agent_id": context.caller_agent_id, "depth": context.depth},
            )
            if run is not None
            else None
        )
        graph_span = (
            await run.start_span(
                agent_id=agent_id,
                kind=SpanKind.GRAPH,
                parent_span_id=child_span.span_id,
                graph_definition_id=definition_id,
            )
            if run is not None and child_span is not None
            else None
        )
        child_context = context.execution.model_copy(
            update={
                "parent_span_id": (
                    graph_span.span_id
                    if graph_span is not None
                    else context.execution.parent_span_id
                )
            }
        )
        if isinstance(graph_input, dict) and "execution_context" in graph_input:
            graph_input = {**graph_input, "execution_context": child_context}
        config: dict[str, Any] = {"recursion_limit": GRAPH_RECURSION_LIMIT}
        if run_id is not None:
            config["configurable"] = {"thread_id": run_id}
        try:
            async with asyncio.timeout(self._timeout_seconds):
                callback = (
                    GraphTraceCallback(
                        run,
                        agent_id=agent_id,
                        graph_span_id=graph_span.span_id,
                        graph_definition_id=definition_id,
                        graph_predecessors=predecessors,
                    )
                    if run is not None and graph_span is not None and definition_id is not None
                    else None
                )
                if callback is not None:
                    config["callbacks"] = [callback]
                graph = self._registry.get(agent_id)
                result = await (
                    graph.ainvoke(graph_input, config=config)
                    if config
                    else graph.ainvoke(graph_input)
                )
        except TimeoutError as exc:
            await self._fail_spans(run, child_span, graph_span, exc)
            raise AgentInvocationTimeoutError(
                f"Agent {agent_id!r} exceeded its invocation timeout"
            ) from exc
        except Exception as exc:
            await self._fail_spans(run, child_span, graph_span, exc)
            raise AgentExecutionError(f"Agent {agent_id!r} execution failed") from exc
        if run is not None and graph_span is not None and child_span is not None:
            await run.complete_span(graph_span.span_id)
            await run.complete_span(child_span.span_id)
        return result

    async def _fail_spans(
        self, run: Any, child_span: Any, graph_span: Any, exc: BaseException
    ) -> None:
        if run is None or graph_span is None or child_span is None:
            return
        error = sanitized_execution_error()
        failure_detail = capture_failure_detail(exc)
        await run.complete_span(graph_span.span_id, error=error, failure_detail=failure_detail)
        await run.complete_span(child_span.span_id, error=error, failure_detail=failure_detail)


def _current_checkpoint_namespace() -> str | None:
    try:
        metadata = get_config().get("metadata", {})
    except RuntimeError:
        return None
    value = metadata.get("langgraph_checkpoint_ns")
    return value if isinstance(value, str) else None
