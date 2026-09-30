"""Test-only pause-capable child graph; not a deployment agent.

The graph is deliberately generic rather than BI/BR shaped. It proves the primitive the
production data-access worker will need: do some lookup work, pause for a bounded choice,
then finish using the work that already completed. Its `lookup` node records every call in
a shared counter so a test can assert the completed lookup is not repeated when the run
resumes -- including after the process that started it is gone.

State is the bounded contract from `orchestration_core.checkpoint_state`, so what the
checkpointer persists is task context, receipts, compact results, and artifact references.
The row data the lookup "produced" never enters the state; only its artifact reference and
row count do.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from orchestration_core import (
    ActionReceipt,
    ArtifactReference,
    CheckpointedTaskState,
    ClarificationOption,
    ClarificationRequest,
    CompactResult,
    TaskContext,
)

LOOKUP_ACTION = "market_lookup"
CHANNEL_LOOKUP_ACTION = "channel_lookup"
TASK_ID = "task-1"
CLARIFICATION_ID = "clarification-1"
FRESHNESS_TOKEN = "snapshot-1"
PRODUCER_AGENT_ID = "scripted_producer"
ARTIFACT_STORE = "scripted-artifacts"
ROW_COUNT = 3


def merge_receipts(
    left: tuple[ActionReceipt, ...], right: tuple[ActionReceipt, ...]
) -> tuple[ActionReceipt, ...]:
    """Append receipts, keeping the first record of each action."""

    seen = {(receipt.task_id, receipt.action) for receipt in left}
    return left + tuple(
        receipt for receipt in right if (receipt.task_id, receipt.action) not in seen
    )


def merge_artifacts(
    left: tuple[ArtifactReference, ...], right: tuple[ArtifactReference, ...]
) -> tuple[ArtifactReference, ...]:
    """Append artifact references, keeping the first reference to each artifact."""

    seen = {(artifact.store, artifact.artifact_id) for artifact in left}
    return left + tuple(
        artifact for artifact in right if (artifact.store, artifact.artifact_id) not in seen
    )


class PausableState(TypedDict):
    """The graph channels; every value is drawn from the bounded checkpoint contract."""

    run_id: str
    tasks: tuple[TaskContext, ...]
    receipts: Annotated[tuple[ActionReceipt, ...], merge_receipts]
    results: tuple[CompactResult, ...]
    artifacts: Annotated[tuple[ArtifactReference, ...], merge_artifacts]
    answer: str


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


def initial_state(run_id: str) -> PausableState:
    return {
        "run_id": run_id,
        "tasks": (TaskContext(task_id=TASK_ID, objective="Summarise spend by market"),),
        "receipts": (),
        "results": (),
        "artifacts": (),
        "answer": "",
    }


def bounded_state(state: PausableState) -> CheckpointedTaskState:
    """Validate the live channels against the bounded checkpoint contract."""

    return CheckpointedTaskState(
        run_id=state["run_id"],
        tasks=state["tasks"],
        receipts=state["receipts"],
        results=state["results"],
        artifacts=state["artifacts"],
    )


def clarification_request() -> ClarificationRequest:
    return ClarificationRequest(
        clarification_id=CLARIFICATION_ID,
        producer_agent_id=PRODUCER_AGENT_ID,
        task_id=TASK_ID,
        question="Which market should the summary cover?",
        reason_code="ambiguous_market",
        selection_mode="single",
        options=(
            ClarificationOption(option_id="us", label="United States"),
            ClarificationOption(option_id="eu", label="Europe"),
        ),
        freshness_token=FRESHNESS_TOKEN,
    )


def create_parallel_pausable_graph(call_log: CallLog) -> StateGraph[Any]:
    """A fan-out variant: two independent lookups complete before the single pause.

    Proves the property that matters for a real worker -- work finished in parallel
    branches before the interrupt is restored from the checkpoint rather than redone.
    """

    async def lookup_markets(state: PausableState) -> dict[str, Any]:
        return _lookup(state, call_log, LOOKUP_ACTION, "spend-by-market", "receipt-1")

    async def lookup_channels(state: PausableState) -> dict[str, Any]:
        return _lookup(state, call_log, CHANNEL_LOOKUP_ACTION, "spend-by-channel", "receipt-2")

    async def ask(state: PausableState) -> dict[str, Any]:
        call_log.record("ask")
        selection = interrupt(clarification_request().model_dump(mode="json"))
        return {
            "results": (
                CompactResult(
                    result_id="result-1",
                    task_id=TASK_ID,
                    headline=f"Resolved market {selection}",
                    metrics={"receipts": float(len(state["receipts"]))},
                    artifacts=state["artifacts"],
                ),
            ),
        }

    async def summarise(state: PausableState) -> dict[str, Any]:
        call_log.record("summarise")
        bounded = bounded_state(state)
        actions = sorted(receipt.action for receipt in bounded.receipts)
        return {"answer": f"{state['results'][0].headline} using {', '.join(actions)}"}

    builder: StateGraph[Any] = StateGraph(PausableState)
    builder.add_node("lookup_markets", lookup_markets)
    builder.add_node("lookup_channels", lookup_channels)
    builder.add_node("ask", ask)
    builder.add_node("summarise", summarise)
    builder.add_edge(START, "lookup_markets")
    builder.add_edge(START, "lookup_channels")
    builder.add_edge("lookup_markets", "ask")
    builder.add_edge("lookup_channels", "ask")
    builder.add_edge("ask", "summarise")
    builder.add_edge("summarise", END)
    return builder


def _lookup(
    state: PausableState, call_log: CallLog, action: str, artifact_id: str, receipt_id: str
) -> dict[str, Any]:
    if bounded_state(state).receipt_for(TASK_ID, action) is not None:
        return {}
    call_log.record(action)
    artifact = ArtifactReference(
        store=ARTIFACT_STORE,
        artifact_id=artifact_id,
        media_type="application/vnd.apache.arrow.file",
        row_count=ROW_COUNT,
    )
    return {
        "receipts": (
            ActionReceipt(
                receipt_id=receipt_id,
                task_id=TASK_ID,
                action=action,
                status="succeeded",
                completed_at=datetime.now(UTC),
                summary=f"{ROW_COUNT} markets available",
                artifacts=(artifact,),
            ),
        ),
        "artifacts": (artifact,),
    }


def create_pausable_graph(call_log: CallLog) -> StateGraph[Any]:
    """Build a lookup -> pause -> summarise graph that records each node execution."""

    async def lookup(state: PausableState) -> dict[str, Any]:
        # A resumed graph restarts the interrupted node from its beginning, so work that
        # must not repeat lives behind its receipt rather than before the interrupt.
        if bounded_state(state).receipt_for(TASK_ID, LOOKUP_ACTION) is not None:
            return {}
        call_log.record("lookup")
        artifact = ArtifactReference(
            store=ARTIFACT_STORE,
            artifact_id="spend-by-market",
            media_type="application/vnd.apache.arrow.file",
            row_count=ROW_COUNT,
        )
        return {
            "receipts": (
                ActionReceipt(
                    receipt_id="receipt-1",
                    task_id=TASK_ID,
                    action=LOOKUP_ACTION,
                    status="succeeded",
                    completed_at=datetime.now(UTC),
                    summary=f"{ROW_COUNT} markets available",
                    artifacts=(artifact,),
                ),
            ),
            "artifacts": (artifact,),
        }

    async def ask(state: PausableState) -> dict[str, Any]:
        call_log.record("ask")
        selection = interrupt(clarification_request().model_dump(mode="json"))
        return {
            "results": (
                CompactResult(
                    result_id="result-1",
                    task_id=TASK_ID,
                    headline=f"Resolved market {selection}",
                    metrics={"rows": float(ROW_COUNT)},
                    artifacts=state["artifacts"],
                ),
            ),
        }

    async def summarise(state: PausableState) -> dict[str, Any]:
        call_log.record("summarise")
        receipt = bounded_state(state).receipt_for(TASK_ID, LOOKUP_ACTION)
        assert receipt is not None
        return {"answer": f"{state['results'][0].headline} across {receipt.summary}"}

    builder: StateGraph[Any] = StateGraph(PausableState)
    builder.add_node("lookup", lookup)
    builder.add_node("ask", ask)
    builder.add_node("summarise", summarise)
    builder.add_edge(START, "lookup")
    builder.add_edge("lookup", "ask")
    builder.add_edge("ask", "summarise")
    builder.add_edge("summarise", END)
    return builder
