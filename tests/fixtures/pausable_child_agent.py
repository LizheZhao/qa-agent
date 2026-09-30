"""Test-only pause-capable child agent; generic, not a deployment agent and not BI/BR shaped.

It stands in for the data-access worker that does not exist yet, and proves the three
things the graph boundary owes: a child can pause instead of answering, a continuation
reaches it directly without the router deciding anything again, and work finished before
the pause is restored from the checkpoint rather than repeated.

The coordinator is the part worth reading. Parallel analysers report their ambiguities in
whatever order they happen to finish; `collect` orders them by the plan, and `coordinate`
asks exactly one, keeping the rest in checkpointed state. Answering the head promotes the
next before any further work is dispatched.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import interrupt
from orchestration_core import (
    QUEUED_CLARIFICATIONS_CHANNEL,
    ActionReceipt,
    ArtifactReference,
    ClarificationOption,
    ClarificationQueue,
    ClarificationRequest,
    ExecutionContext,
    order_clarifications,
)

AGENT_ID = "scripted_producer"
ARTIFACT_STORE = "scripted-artifacts"
ROW_COUNT = 3


class CallLog:
    """A durable count of node executions, shared across processes by file."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def record(self, node: str) -> None:
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(f"{node}\n")

    def calls(self) -> tuple[str, ...]:
        if not self._path.exists():
            return ()
        return tuple(self._path.read_text(encoding="utf-8").split())


def append(left: tuple[Any, ...], right: tuple[Any, ...]) -> tuple[Any, ...]:
    return tuple(left) + tuple(right)


class ChildState(TypedDict):
    """Channels for one child run; `messages` is the conversation the invoker exchanges."""

    messages: Annotated[list[Any], add_messages]
    execution_context: ExecutionContext
    plan_task_ids: tuple[str, ...]
    reported: Annotated[tuple[ClarificationRequest, ...], append]
    queued_clarifications: tuple[ClarificationRequest, ...]
    receipts: Annotated[tuple[ActionReceipt, ...], append]
    answers: Annotated[tuple[tuple[str, str], ...], append]


def clarification_for(task_id: str, *, freshness_token: str | None = None) -> ClarificationRequest:
    return ClarificationRequest(
        clarification_id=f"clarification-{task_id}",
        producer_agent_id=AGENT_ID,
        task_id=task_id,
        question=f"Which market should {task_id} cover?",
        reason_code="ambiguous_market",
        selection_mode="single",
        options=(
            ClarificationOption(option_id="us", label="United States"),
            ClarificationOption(option_id="eu", label="Europe"),
        ),
        freshness_token=freshness_token or f"snapshot-{task_id}",
    )


def create_pausable_child_agent(
    call_log: CallLog,
    *,
    task_ids: tuple[str, ...] = ("task-1",),
) -> StateGraph[Any]:
    """Build a child that pauses once per task in `task_ids`, asking one question at a time."""

    async def plan(state: ChildState) -> dict[str, Any]:
        call_log.record("plan")
        return {"plan_task_ids": task_ids}

    def analyser(task_id: str) -> Any:
        async def analyse(state: ChildState) -> dict[str, Any]:
            # Recorded so a test can prove the checkpoint restored this instead of rerunning it.
            call_log.record(f"analyse_{task_id}")
            artifact = ArtifactReference(
                store=ARTIFACT_STORE,
                artifact_id=f"rows-{task_id}",
                media_type="application/vnd.apache.arrow.file",
                row_count=ROW_COUNT,
            )
            return {
                "receipts": (
                    ActionReceipt(
                        receipt_id=f"receipt-{task_id}",
                        task_id=task_id,
                        action="market_lookup",
                        status="succeeded",
                        completed_at=datetime.now(UTC),
                        summary=f"{ROW_COUNT} markets available",
                        artifacts=(artifact,),
                    ),
                ),
                "reported": (clarification_for(task_id),),
            }

        return analyse

    async def collect(state: ChildState) -> dict[str, Any]:
        call_log.record("collect")
        # Ordered by the plan, never by which analyser finished first, so the same run
        # always asks the same question first.
        ordered = order_clarifications(state["reported"], plan_task_ids=state["plan_task_ids"])
        return {QUEUED_CLARIFICATIONS_CHANNEL: ordered}

    async def coordinate(state: ChildState) -> dict[str, Any]:
        queue = ClarificationQueue(pending=tuple(state[QUEUED_CLARIFICATIONS_CHANNEL]))
        active = queue.active
        assert active is not None
        # Everything above this line reruns when the node restarts on resume, so it holds
        # no side effects; the recorded call sits after the interrupt for that reason.
        reply = interrupt(active.model_dump(mode="json"))
        call_log.record(f"answered_{active.task_id}")
        remaining = queue.answered(active.clarification_id)
        return {
            QUEUED_CLARIFICATIONS_CHANNEL: remaining.pending,
            "answers": ((active.task_id, _answer_text(reply)),),
        }

    async def after_coordinate(state: ChildState) -> str:
        return "coordinate" if state[QUEUED_CLARIFICATIONS_CHANNEL] else "answer"

    async def answer(state: ChildState) -> dict[str, Any]:
        call_log.record("answer")
        settled = ", ".join(f"{task}={choice}" for task, choice in state["answers"])
        rows = sum(receipt.artifacts[0].row_count or 0 for receipt in state["receipts"])
        return {"messages": [AIMessage(content=f"Settled {settled} over {rows} rows")]}

    builder: StateGraph[Any] = StateGraph(ChildState)
    builder.add_node("plan", plan)
    for task_id in task_ids:
        builder.add_node(f"analyse_{task_id}", analyser(task_id))
        builder.add_edge("plan", f"analyse_{task_id}")
        builder.add_edge(f"analyse_{task_id}", "collect")
    builder.add_node("collect", collect)
    builder.add_node("coordinate", coordinate)
    builder.add_node("answer", answer)
    builder.add_edge(START, "plan")
    builder.add_conditional_edges(
        "coordinate", after_coordinate, path_map={"coordinate": "coordinate", "answer": "answer"}
    )
    builder.add_edge("collect", "coordinate")
    builder.add_edge("answer", END)
    return builder


def _answer_text(reply: Any) -> str:
    """Read the choice out of the validated response the boundary resumed with."""

    if isinstance(reply, dict):
        if reply.get("form") == "options":
            return ",".join(reply["option_ids"])
        if reply.get("form") == "free_text":
            return str(reply["text"])
        return str(reply.get("form"))
    return str(reply)
