"""Pause and resume of a scripted pause-capable child graph."""

from pathlib import Path
from typing import Any

import pytest
from langgraph.types import Command
from orchestration_core import CLARIFICATION_RESPONSE_ADAPTER, ClarificationRequest

from agentic_orchestration.execution.agent_registry import AgentRegistry
from agentic_orchestration.execution.checkpointing import (
    MemoryGraphCheckpointer,
    graph_checkpointer_scope,
)
from tests.fixtures.pausable_graph import (
    LOOKUP_ACTION,
    TASK_ID,
    CallLog,
    bounded_state,
    create_pausable_graph,
    initial_state,
)


def config(run_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": run_id}}


@pytest.mark.integration
async def test_child_graph_pauses_with_a_typed_clarification_request(tmp_path: Path) -> None:
    call_log = CallLog(tmp_path / "calls.txt")
    async with graph_checkpointer_scope(MemoryGraphCheckpointer()) as checkpointer:
        graph = create_pausable_graph(call_log).compile(checkpointer=checkpointer.saver)
        result = await graph.ainvoke(initial_state("run-1"), config("run-1"))

    interrupts = result["__interrupt__"]
    assert len(interrupts) == 1
    request = ClarificationRequest.model_validate(interrupts[0].value)
    assert request.selection_mode == "single"
    assert request.option_ids == ("us", "eu")
    assert call_log.calls() == ("lookup", "ask")


@pytest.mark.integration
async def test_resume_completes_without_repeating_the_finished_lookup(tmp_path: Path) -> None:
    call_log = CallLog(tmp_path / "calls.txt")
    async with graph_checkpointer_scope(MemoryGraphCheckpointer()) as checkpointer:
        graph = create_pausable_graph(call_log).compile(checkpointer=checkpointer.saver)
        paused = await graph.ainvoke(initial_state("run-2"), config("run-2"))
        request = ClarificationRequest.model_validate(paused["__interrupt__"][0].value)
        response = request.accept(
            CLARIFICATION_RESPONSE_ADAPTER.validate_python(
                {"form": "options", "option_ids": ["us"]}
            )
        )
        assert response.form == "options"
        resumed = await graph.ainvoke(Command(resume=response.option_ids[0]), config("run-2"))

    assert resumed["answer"] == "Resolved market us across 3 markets available"
    # `ask` restarts from its beginning on resume; the completed lookup does not.
    assert call_log.calls().count("lookup") == 1
    assert call_log.calls() == ("lookup", "ask", "ask", "summarise")


@pytest.mark.integration
async def test_paused_state_recovers_from_the_checkpoint_alone(tmp_path: Path) -> None:
    """A separately compiled graph resumes the run using only the saver's state."""

    call_log = CallLog(tmp_path / "calls.txt")
    async with graph_checkpointer_scope(MemoryGraphCheckpointer()) as checkpointer:
        starting = create_pausable_graph(call_log).compile(checkpointer=checkpointer.saver)
        await starting.ainvoke(initial_state("run-3"), config("run-3"))

        continuing = create_pausable_graph(call_log).compile(checkpointer=checkpointer.saver)
        recovered = await continuing.aget_state(config("run-3"))
        assert recovered.next == ("ask",)
        resumed = await continuing.ainvoke(Command(resume="eu"), config("run-3"))

    assert resumed["answer"].startswith("Resolved market eu")
    assert call_log.calls().count("lookup") == 1


@pytest.mark.integration
async def test_checkpointed_state_carries_only_receipts_and_artifact_references(
    tmp_path: Path,
) -> None:
    call_log = CallLog(tmp_path / "calls.txt")
    async with graph_checkpointer_scope(MemoryGraphCheckpointer()) as checkpointer:
        graph = create_pausable_graph(call_log).compile(checkpointer=checkpointer.saver)
        await graph.ainvoke(initial_state("run-4"), config("run-4"))
        snapshot = await graph.aget_state(config("run-4"))

    state = bounded_state(snapshot.values)
    receipt = state.receipt_for(TASK_ID, LOOKUP_ACTION)
    assert receipt is not None
    assert [artifact.artifact_id for artifact in receipt.artifacts] == ["spend-by-market"]
    assert receipt.artifacts[0].row_count == 3
    serialized = state.model_dump_json()
    assert "spend-by-market" in serialized
    # The rows the lookup produced stay in the artifact store; only their identity and
    # count are checkpointed, so the pause window cannot retain a result set.
    assert "row_count" in serialized
    assert state.model_dump().get("rows") is None


@pytest.mark.integration
async def test_registry_checkpoints_only_the_declared_pause_capable_child(tmp_path: Path) -> None:
    call_log = CallLog(tmp_path / "calls.txt")
    async with graph_checkpointer_scope(MemoryGraphCheckpointer()) as checkpointer:
        registry = AgentRegistry.compile(
            {
                "producer": lambda _dependencies: create_pausable_graph(call_log),
                "stateless": lambda _dependencies: create_pausable_graph(call_log),
            },
            savers={"producer": checkpointer.saver},
        )
        assert registry.get("producer").checkpointer is checkpointer.saver
        assert registry.get("stateless").checkpointer is None


@pytest.mark.integration
async def test_registry_rejects_a_checkpointer_for_an_unregistered_agent(tmp_path: Path) -> None:
    call_log = CallLog(tmp_path / "calls.txt")
    async with graph_checkpointer_scope(MemoryGraphCheckpointer()) as checkpointer:
        with pytest.raises(KeyError, match="not in this registry"):
            AgentRegistry.compile(
                {"producer": lambda _dependencies: create_pausable_graph(call_log)},
                savers={"absent": checkpointer.saver},
            )
