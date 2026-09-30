"""Ownership and durability rules for the pause-capable graph checkpointer."""

import pytest
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver

from agentic_orchestration.execution.checkpointing import (
    DurableCheckpointerUnavailableError,
    GraphCheckpointer,
    GraphCheckpointerError,
    MemoryGraphCheckpointer,
    graph_checkpointer_scope,
    require_durable_checkpointer,
)


class DurableStub:
    """A stand-in for a durable saver, recording the lifecycle calls it received."""

    def __init__(self) -> None:
        self._saver = InMemorySaver()
        self.calls: list[str] = []

    @property
    def storage_name(self) -> str:
        return "durable_stub"

    @property
    def durable(self) -> bool:
        return True

    @property
    def saver(self) -> BaseCheckpointSaver[object]:
        self.calls.append("saver")
        return self._saver

    async def initialize(self) -> None:
        self.calls.append("initialize")

    async def close(self) -> None:
        self.calls.append("close")


@pytest.mark.unit
def test_memory_checkpointing_reports_itself_as_non_durable() -> None:
    checkpointer = MemoryGraphCheckpointer()

    assert checkpointer.storage_name == "process_memory"
    assert checkpointer.durable is False
    assert isinstance(checkpointer, GraphCheckpointer)


@pytest.mark.unit
def test_memory_checkpointing_is_refused_as_a_runtime_checkpointer() -> None:
    with pytest.raises(DurableCheckpointerUnavailableError, match="not durable"):
        require_durable_checkpointer(MemoryGraphCheckpointer())


@pytest.mark.unit
def test_absent_checkpointer_fails_instead_of_falling_back_to_memory() -> None:
    with pytest.raises(DurableCheckpointerUnavailableError, match="No durable graph checkpointer"):
        require_durable_checkpointer(None)


@pytest.mark.unit
def test_durable_checkpointer_is_accepted() -> None:
    durable = DurableStub()

    assert require_durable_checkpointer(durable) is durable


@pytest.mark.unit
async def test_scope_initializes_before_use_and_closes_on_exit() -> None:
    durable = DurableStub()

    async with graph_checkpointer_scope(durable) as checkpointer:
        assert checkpointer.saver is not None

    assert durable.calls == ["initialize", "saver", "close"]


@pytest.mark.unit
async def test_scope_closes_the_checkpointer_when_a_later_startup_step_fails() -> None:
    durable = DurableStub()

    with pytest.raises(RuntimeError, match="registry assembly failed"):
        async with graph_checkpointer_scope(durable):
            raise RuntimeError("registry assembly failed")

    assert durable.calls == ["initialize", "close"]


@pytest.mark.unit
async def test_saver_is_unavailable_before_initialization_and_after_close() -> None:
    checkpointer = MemoryGraphCheckpointer()

    with pytest.raises(GraphCheckpointerError, match="not initialized"):
        _ = checkpointer.saver

    async with graph_checkpointer_scope(checkpointer):
        assert isinstance(checkpointer.saver, InMemorySaver)

    with pytest.raises(GraphCheckpointerError, match="closed"):
        _ = checkpointer.saver
