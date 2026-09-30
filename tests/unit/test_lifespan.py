"""Application lifespan resource cleanup behavior."""

import asyncio
import importlib
from contextlib import suppress
from typing import Any

import pytest
from fastapi import FastAPI

from agentic_orchestration.sessions import MemorySessionStore

lifespan_module = importlib.import_module("agentic_orchestration.lifespan")
expiry_module = importlib.import_module("agentic_orchestration.sessions.expiry")


class FakeAdmin:
    async def command(self, name: str) -> None:
        assert name == "ping"


class FakeMongoClient:
    def __init__(self) -> None:
        self.admin = FakeAdmin()
        self.close_calls = 0

    async def close(self) -> None:
        self.close_calls += 1


class FakeDatabase:
    def __getitem__(self, name: str) -> object:
        del name
        return object()


class FakeSessionStore:
    def __init__(self, database: Any) -> None:
        del database

    async def initialize(self) -> None:
        return None


class FakeExecutionStore:
    def __init__(self, database: Any) -> None:
        del database

    async def initialize(self) -> None:
        return None


class FakeGraphCheckpointer:
    """Stands in for MongoGraphCheckpointer: durable enough to pass the startup requirement,
    with no storage behind it."""

    def __init__(self, database: Any) -> None:
        del database
        self.close_calls = 0

    @property
    def storage_name(self) -> str:
        return "fake"

    @property
    def durable(self) -> bool:
        return True

    @property
    def saver(self) -> object:
        return object()

    async def initialize(self) -> None:
        return None

    async def close(self) -> None:
        self.close_calls += 1


class UnsupportedInjectedSessionStore:
    pass


@pytest.mark.unit
async def test_injected_session_store_never_falls_back_to_empty_memory_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(lifespan_module, "load_and_validate_deployment", lambda *args: object())
    app = FastAPI()
    app.state.session_store_factory = lambda settings: UnsupportedInjectedSessionStore()

    with pytest.raises(TypeError, match="compatible diagnostic read adapter"):
        async with lifespan_module.lifespan(app):
            raise AssertionError("startup unexpectedly succeeded")


@pytest.mark.unit
async def test_memory_session_store_remains_supported_test_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = MemorySessionStore()
    monkeypatch.setattr(lifespan_module, "load_and_validate_deployment", lambda *args: object())
    monkeypatch.setattr(lifespan_module, "_assemble_runtime", _noop_runtime_assembly)
    app = FastAPI()
    app.state.session_store_factory = lambda settings: store

    async with lifespan_module.lifespan(app):
        assert app.state.diagnostic_read_store._session_store is store


async def _noop_runtime_assembly(*args: Any) -> None:
    return None


@pytest.mark.unit
async def test_mongo_client_closes_when_runtime_assembly_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeMongoClient()
    checkpointers: list[FakeGraphCheckpointer] = []
    monkeypatch.setattr(
        lifespan_module,
        "create_mongodb",
        lambda **kwargs: (client, FakeDatabase()),
    )
    monkeypatch.setattr(lifespan_module, "MongoSessionStore", FakeSessionStore)
    monkeypatch.setattr(lifespan_module, "MongoExecutionTraceStore", FakeExecutionStore)

    def _build_checkpointer(database: Any) -> FakeGraphCheckpointer:
        checkpointers.append(FakeGraphCheckpointer(database))
        return checkpointers[-1]

    monkeypatch.setattr(lifespan_module, "MongoGraphCheckpointer", _build_checkpointer)
    app = FastAPI()

    def fail_model_factory(settings: Any) -> None:
        del settings
        raise RuntimeError("model assembly failed")

    app.state.model_factory = fail_model_factory

    with pytest.raises(RuntimeError, match="model assembly failed"):
        async with lifespan_module.lifespan(app):
            raise AssertionError("startup unexpectedly succeeded")

    assert client.close_calls == 1
    # The checkpointer is on the same resource stack, so a paused graph's saver is released too.
    assert [c.close_calls for c in checkpointers] == [1]


@pytest.mark.unit
async def test_expiry_sweeper_runs_during_lifespan_and_stops_with_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The sweep is owned by the lifespan stack, so shutdown leaves no task running."""

    swept = asyncio.Event()
    observed: list[tuple[Any, int]] = []

    async def record(store: Any, *, at: Any, limit: int) -> int:
        observed.append((store, limit))
        swept.set()
        return 0

    monkeypatch.setenv("PENDING_RUN_SWEEP_INTERVAL_SECONDS", "0.01")
    monkeypatch.setenv("PENDING_RUN_SWEEP_LIMIT", "7")
    monkeypatch.setattr(lifespan_module, "load_and_validate_deployment", lambda *args: object())
    monkeypatch.setattr(lifespan_module, "_assemble_runtime", _noop_runtime_assembly)
    monkeypatch.setattr(expiry_module, "sweep_expired_runs", record)
    store = MemorySessionStore()
    app = FastAPI()
    app.state.session_store_factory = lambda settings: store

    async with lifespan_module.lifespan(app):
        await asyncio.wait_for(swept.wait(), timeout=2)
        running = {task.get_name() for task in asyncio.all_tasks()}
        assert "pending-run-expiry-sweep" in running

    assert observed[0] == (store, 7)
    assert "pending-run-expiry-sweep" not in {task.get_name() for task in asyncio.all_tasks()}


@pytest.mark.unit
async def test_expiry_sweeper_can_be_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PENDING_RUN_SWEEP_INTERVAL_SECONDS", "0")
    monkeypatch.setattr(lifespan_module, "load_and_validate_deployment", lambda *args: object())
    monkeypatch.setattr(lifespan_module, "_assemble_runtime", _noop_runtime_assembly)
    app = FastAPI()
    app.state.session_store_factory = lambda settings: MemorySessionStore()

    async with lifespan_module.lifespan(app):
        assert "pending-run-expiry-sweep" not in {task.get_name() for task in asyncio.all_tasks()}


@pytest.mark.unit
async def test_a_failing_sweep_does_not_stop_later_passes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Expiry correctness never rests on the sweep, so an outage only delays cleanup."""

    attempts: list[int] = []
    recovered = asyncio.Event()

    async def flaky(store: Any, *, at: Any, limit: int) -> int:
        attempts.append(len(attempts))
        if len(attempts) == 1:
            raise RuntimeError("storage blipped")
        recovered.set()
        return 0

    monkeypatch.setattr(expiry_module, "sweep_expired_runs", flaky)
    sweeper = asyncio.create_task(
        expiry_module.run_expiry_sweeper(MemorySessionStore(), interval_seconds=0.01, limit=5)
    )
    try:
        await asyncio.wait_for(recovered.wait(), timeout=2)
    finally:
        sweeper.cancel()
        with suppress(asyncio.CancelledError):
            await sweeper

    assert len(attempts) >= 2
