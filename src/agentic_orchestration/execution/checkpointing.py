"""Ownership of the durable LangGraph checkpointer used by pause-capable graphs.

A pause-capable child graph keeps its in-flight state in a checkpointer rather than in the
request that started it, so the checkpointer is an external runtime resource with the same
ownership rules as the session store: constructed during startup, cleaned up immediately
after successful construction, and never silently downgraded.

Two boundaries matter here and are deliberately kept apart.

Durability is not optional. MongoDB is the runtime durability boundary, so a process-local
checkpointer is an injected test adapter. `require_durable_checkpointer` exists so that a
missing or misconfigured durable saver fails startup instead of quietly producing a
deployment whose pauses evaporate on restart.

Checkpoint writes are not part of the session-turn transaction. LangGraph writes the
interrupted checkpoint through the saver on its own; publishing the pending run is a
separate short session-store transaction. Nothing in this module makes those two writes
atomic, and callers must not assume they are.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Protocol, runtime_checkable

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver


class GraphCheckpointerError(RuntimeError):
    """A graph checkpointer could not be provided for a pause-capable graph."""


class DurableCheckpointerUnavailableError(GraphCheckpointerError):
    """No durable checkpointer is configured, and memory checkpointing is test-only."""


@runtime_checkable
class GraphCheckpointer(Protocol):
    """A saver plus the lifecycle the runtime needs in order to own it."""

    @property
    def storage_name(self) -> str: ...

    @property
    def durable(self) -> bool: ...

    @property
    def saver(self) -> BaseCheckpointSaver[Any]:
        """The saver to pass to `StateGraph.compile(checkpointer=...)`."""

    async def initialize(self) -> None:
        """Prepare storage before any graph compiles against this saver."""

    async def close(self) -> None:
        """Release only what this checkpointer constructed.

        A checkpointer handed an externally owned connection must leave it open: the
        lifespan resource stack that built the connection also closes it, and closing it
        twice would tear down the session store's client along with the saver.
        """


class MemoryGraphCheckpointer:
    """Process-local checkpointer for deterministic tests.

    Not a runtime fallback. It is reported as non-durable so `require_durable_checkpointer`
    rejects it, and it is only ever reachable through explicit test injection.
    """

    def __init__(self) -> None:
        self._saver = InMemorySaver()
        self._initialized = False
        self._closed = False

    @property
    def storage_name(self) -> str:
        return "process_memory"

    @property
    def durable(self) -> bool:
        return False

    @property
    def saver(self) -> BaseCheckpointSaver[Any]:
        if not self._initialized:
            raise GraphCheckpointerError("Graph checkpointer was not initialized")
        if self._closed:
            raise GraphCheckpointerError("Graph checkpointer is closed")
        return self._saver

    async def initialize(self) -> None:
        self._initialized = True

    async def close(self) -> None:
        self._closed = True


def require_durable_checkpointer(checkpointer: GraphCheckpointer | None) -> GraphCheckpointer:
    """Return a checkpointer only if it can outlive the process that wrote to it.

    Startup calls this before compiling any pause-capable graph so that an absent or
    non-durable saver is a startup failure, not a deployment that loses every pause.
    """

    if checkpointer is None:
        raise DurableCheckpointerUnavailableError(
            "No durable graph checkpointer is configured for pause-capable graphs"
        )
    if not checkpointer.durable:
        raise DurableCheckpointerUnavailableError(
            f"Graph checkpointer {checkpointer.storage_name!r} is not durable"
        )
    return checkpointer


@asynccontextmanager
async def graph_checkpointer_scope(
    checkpointer: GraphCheckpointer,
) -> AsyncIterator[GraphCheckpointer]:
    """Initialize a checkpointer and close it on every exit from the scope.

    Entered during startup so that a failure in a later startup step still closes the
    saver, matching how the lifespan stack treats every other external resource.
    """

    await checkpointer.initialize()
    try:
        yield checkpointer
    finally:
        await checkpointer.close()
