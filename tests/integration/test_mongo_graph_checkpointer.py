"""Opt-in coverage of the repository-owned durable graph checkpointer.

Requires a reachable MongoDB, per the repository's opt-in `mongo` test policy. Every test
works in collections of its own and removes them afterwards, so a run leaves nothing behind.
"""

import os
import subprocess
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from langgraph.checkpoint.base import Checkpoint, CheckpointTuple
from langgraph.checkpoint.serde.types import ERROR, RESUME
from langgraph.types import Command
from pymongo.errors import PyMongoError

from agentic_orchestration.config import Settings
from agentic_orchestration.execution.checkpointing import graph_checkpointer_scope
from agentic_orchestration.execution.mongodb import MongoCheckpointSaver, MongoGraphCheckpointer
from integrations.mongodb import create_mongodb
from tests.fixtures.checkpoint_restart_phase import (
    CHECKPOINT_COLLECTION,
    CHECKPOINT_WRITES_COLLECTION,
)
from tests.fixtures.pausable_graph import CallLog

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

pytestmark = [
    pytest.mark.integration,
    pytest.mark.mongo,
    pytest.mark.skipif(
        os.getenv("RUN_MONGO_CHECKPOINT_TESTS") != "1",
        reason="set RUN_MONGO_CHECKPOINT_TESTS=1 to exercise the durable graph checkpointer",
    ),
    pytest.mark.skipif(
        Settings().mongodb_host is None,
        reason="MongoDB connection settings are absent",
    ),
]


def thread_config(run_id: str, checkpoint_id: str | None = None) -> Any:
    configurable: dict[str, Any] = {"thread_id": run_id, "checkpoint_ns": ""}
    if checkpoint_id is not None:
        configurable["checkpoint_id"] = checkpoint_id
    return {"configurable": configurable}


def checkpoint(checkpoint_id: str, step: int) -> Checkpoint:
    return Checkpoint(
        v=1,
        id=checkpoint_id,
        ts=f"2026-09-15T12:00:{step:02d}+00:00",
        channel_values={"task": f"step-{step}"},
        channel_versions={"task": step + 1},
        versions_seen={},
        updated_channels=None,
    )


@pytest.fixture
async def checkpointer() -> AsyncIterator[MongoGraphCheckpointer]:
    """A checkpointer in collections unique to one test, dropped when it finishes."""

    settings = Settings()
    client, database = create_mongodb(
        host=settings.mongodb_host,
        port=settings.mongodb_port,
        user=settings.mongodb_user,
        password=(
            settings.mongodb_pwd.get_secret_value() if settings.mongodb_pwd is not None else None
        ),
        database=settings.mongodb_db,
    )
    suffix = uuid4().hex[:12]
    checkpoints = f"test_checkpoints_{suffix}"
    writes = f"test_checkpoint_writes_{suffix}"
    try:
        async with graph_checkpointer_scope(
            MongoGraphCheckpointer(
                database, checkpoints_collection=checkpoints, writes_collection=writes
            )
        ) as scoped:
            assert isinstance(scoped, MongoGraphCheckpointer)
            yield scoped
        await database[checkpoints].drop()
        await database[writes].drop()
    finally:
        await client.close()


async def test_checkpointer_reports_itself_durable(
    checkpointer: MongoGraphCheckpointer,
) -> None:
    assert checkpointer.durable is True
    assert checkpointer.storage_name == "mongodb"


async def test_latest_checkpoint_is_the_highest_checkpoint_id(
    checkpointer: MongoGraphCheckpointer,
) -> None:
    saver = checkpointer.saver
    run_id = f"run-{uuid4()}"
    await saver.aput(thread_config(run_id), checkpoint("checkpoint-1", 1), {"step": 1}, {})
    await saver.aput(
        thread_config(run_id, "checkpoint-1"), checkpoint("checkpoint-2", 2), {"step": 2}, {}
    )

    latest = await saver.aget_tuple(thread_config(run_id))

    assert latest is not None
    assert latest.checkpoint["id"] == "checkpoint-2"
    assert latest.metadata["step"] == 2
    assert latest.parent_config is not None
    assert latest.parent_config["configurable"]["checkpoint_id"] == "checkpoint-1"


async def test_an_exact_checkpoint_id_is_retrievable(
    checkpointer: MongoGraphCheckpointer,
) -> None:
    saver = checkpointer.saver
    run_id = f"run-{uuid4()}"
    await saver.aput(thread_config(run_id), checkpoint("checkpoint-1", 1), {"step": 1}, {})
    await saver.aput(
        thread_config(run_id, "checkpoint-1"), checkpoint("checkpoint-2", 2), {"step": 2}, {}
    )

    earlier = await saver.aget_tuple(thread_config(run_id, "checkpoint-1"))

    assert earlier is not None
    assert earlier.checkpoint["id"] == "checkpoint-1"
    assert earlier.parent_config is None


async def test_unknown_thread_has_no_checkpoint(checkpointer: MongoGraphCheckpointer) -> None:
    assert await checkpointer.saver.aget_tuple(thread_config(f"absent-{uuid4()}")) is None


async def test_list_orders_newest_first_and_honours_before_and_limit(
    checkpointer: MongoGraphCheckpointer,
) -> None:
    saver = checkpointer.saver
    run_id = f"run-{uuid4()}"
    for step, checkpoint_id in enumerate(("checkpoint-1", "checkpoint-2", "checkpoint-3"), start=1):
        await saver.aput(thread_config(run_id), checkpoint(checkpoint_id, step), {"step": step}, {})

    listed = [tuple_ async for tuple_ in saver.alist(thread_config(run_id))]
    limited = [tuple_ async for tuple_ in saver.alist(thread_config(run_id), limit=2)]
    before = [
        tuple_
        async for tuple_ in saver.alist(
            thread_config(run_id), before=thread_config(run_id, "checkpoint-3")
        )
    ]

    assert [tuple_.checkpoint["id"] for tuple_ in listed] == [
        "checkpoint-3",
        "checkpoint-2",
        "checkpoint-1",
    ]
    assert [tuple_.checkpoint["id"] for tuple_ in limited] == ["checkpoint-3", "checkpoint-2"]
    assert [tuple_.checkpoint["id"] for tuple_ in before] == ["checkpoint-2", "checkpoint-1"]


async def test_list_filters_on_stored_metadata(checkpointer: MongoGraphCheckpointer) -> None:
    saver = checkpointer.saver
    run_id = f"run-{uuid4()}"
    await saver.aput(
        thread_config(run_id), checkpoint("checkpoint-1", 1), {"source": "input", "step": 1}, {}
    )
    await saver.aput(
        thread_config(run_id), checkpoint("checkpoint-2", 2), {"source": "loop", "step": 2}, {}
    )

    matched = [
        tuple_ async for tuple_ in saver.alist(thread_config(run_id), filter={"source": "loop"})
    ]

    assert [tuple_.checkpoint["id"] for tuple_ in matched] == ["checkpoint-2"]


async def test_a_duplicated_ordinary_write_keeps_the_value_already_stored(
    checkpointer: MongoGraphCheckpointer,
) -> None:
    """A retried task re-sends writes it already persisted; the first value must win."""

    saver = checkpointer.saver
    run_id = f"run-{uuid4()}"
    await saver.aput(thread_config(run_id), checkpoint("checkpoint-1", 1), {"step": 1}, {})
    config = thread_config(run_id, "checkpoint-1")

    await saver.aput_writes(config, [("task", "first")], "task-1")
    await saver.aput_writes(config, [("task", "second")], "task-1")

    stored = await saver.aget_tuple(config)
    assert stored is not None
    assert stored.pending_writes == [("task-1", "task", "first")]


async def test_a_retried_task_may_replace_its_error_and_resume_writes(
    checkpointer: MongoGraphCheckpointer,
) -> None:
    saver = checkpointer.saver
    run_id = f"run-{uuid4()}"
    await saver.aput(thread_config(run_id), checkpoint("checkpoint-1", 1), {"step": 1}, {})
    config = thread_config(run_id, "checkpoint-1")

    await saver.aput_writes(config, [(ERROR, "first failure")], "task-1")
    await saver.aput_writes(config, [(ERROR, "second failure")], "task-1")
    await saver.aput_writes(config, [(RESUME, "answer")], "task-1")

    stored = await saver.aget_tuple(config)
    assert stored is not None
    values = {channel: value for _task, channel, value in stored.pending_writes or []}
    assert values[ERROR] == "second failure"
    assert values[RESUME] == "answer"


async def test_parallel_tasks_each_retain_their_own_writes(
    checkpointer: MongoGraphCheckpointer,
) -> None:
    saver = checkpointer.saver
    run_id = f"run-{uuid4()}"
    await saver.aput(thread_config(run_id), checkpoint("checkpoint-1", 1), {"step": 1}, {})
    config = thread_config(run_id, "checkpoint-1")

    await saver.aput_writes(config, [("markets", "us"), ("markets", "eu")], "task-1")
    await saver.aput_writes(config, [("channels", "paid")], "task-2")

    stored = await saver.aget_tuple(config)
    assert stored is not None
    # Deterministic across replicas: ordered by task, then by the index within that task.
    assert stored.pending_writes == [
        ("task-1", "markets", "us"),
        ("task-1", "markets", "eu"),
        ("task-2", "channels", "paid"),
    ]


async def test_a_repeated_write_with_a_different_task_path_does_not_duplicate(
    checkpointer: MongoGraphCheckpointer,
) -> None:
    """Task path is payload, not identity; keying on it would admit a second copy."""

    saver = checkpointer.saver
    run_id = f"run-{uuid4()}"
    await saver.aput(thread_config(run_id), checkpoint("checkpoint-1", 1), {"step": 1}, {})
    config = thread_config(run_id, "checkpoint-1")

    await saver.aput_writes(config, [("task", "value")], "task-1", task_path="~start:lookup")
    await saver.aput_writes(config, [("task", "value")], "task-1", task_path="~retry:lookup")

    stored = await saver.aget_tuple(config)
    assert stored is not None
    assert len(stored.pending_writes or []) == 1


async def test_writes_belong_to_the_checkpoint_that_stored_them(
    checkpointer: MongoGraphCheckpointer,
) -> None:
    saver = checkpointer.saver
    run_id = f"run-{uuid4()}"
    await saver.aput(thread_config(run_id), checkpoint("checkpoint-1", 1), {"step": 1}, {})
    await saver.aput(
        thread_config(run_id, "checkpoint-1"), checkpoint("checkpoint-2", 2), {"step": 2}, {}
    )
    await saver.aput_writes(thread_config(run_id, "checkpoint-1"), [("task", "early")], "task-1")

    latest = await saver.aget_tuple(thread_config(run_id))
    earlier = await saver.aget_tuple(thread_config(run_id, "checkpoint-1"))

    assert latest is not None and latest.pending_writes == []
    assert earlier is not None and earlier.pending_writes == [("task-1", "task", "early")]


async def test_deleting_a_thread_removes_its_checkpoints_and_writes(
    checkpointer: MongoGraphCheckpointer,
) -> None:
    saver = checkpointer.saver
    kept_id = f"run-{uuid4()}"
    removed_id = f"run-{uuid4()}"
    for run_id in (kept_id, removed_id):
        await saver.aput(thread_config(run_id), checkpoint("checkpoint-1", 1), {"step": 1}, {})
        await saver.aput_writes(thread_config(run_id, "checkpoint-1"), [("task", "v")], "task-1")

    await saver.adelete_thread(removed_id)

    assert await saver.aget_tuple(thread_config(removed_id)) is None
    remaining = await saver.aget_tuple(thread_config(kept_id))
    assert remaining is not None
    assert remaining.pending_writes == [("task-1", "task", "v")]


async def test_closing_the_checkpointer_leaves_its_connection_to_the_owner() -> None:
    """The lifespan stack owns the client; the checkpointer must not close it."""

    settings = Settings()
    client, database = create_mongodb(
        host=settings.mongodb_host,
        port=settings.mongodb_port,
        user=settings.mongodb_user,
        password=(
            settings.mongodb_pwd.get_secret_value() if settings.mongodb_pwd is not None else None
        ),
        database=settings.mongodb_db,
    )
    suffix = uuid4().hex[:12]
    checkpoints = f"test_checkpoints_{suffix}"
    writes = f"test_checkpoint_writes_{suffix}"
    try:
        async with graph_checkpointer_scope(
            MongoGraphCheckpointer(
                database, checkpoints_collection=checkpoints, writes_collection=writes
            )
        ):
            pass

        # The client that outlived the checkpointer is still usable.
        assert (await client.admin.command("ping"))["ok"] == 1.0
        await database[checkpoints].drop()
        await database[writes].drop()
    finally:
        await client.close()
    with pytest.raises(PyMongoError):
        await client.admin.command("ping")


async def test_startup_can_initialize_the_same_collections_twice() -> None:
    """A second replica starting against existing collections must not fail startup."""

    settings = Settings()
    client, database = create_mongodb(
        host=settings.mongodb_host,
        port=settings.mongodb_port,
        user=settings.mongodb_user,
        password=(
            settings.mongodb_pwd.get_secret_value() if settings.mongodb_pwd is not None else None
        ),
        database=settings.mongodb_db,
    )
    suffix = uuid4().hex[:12]
    checkpoints = f"test_checkpoints_{suffix}"
    writes = f"test_checkpoint_writes_{suffix}"
    try:
        for _replica in range(2):
            saver = MongoCheckpointSaver(
                database, checkpoints_collection=checkpoints, writes_collection=writes
            )
            await saver.initialize()
        await database[checkpoints].drop()
        await database[writes].drop()
    finally:
        await client.close()


def run_phase(phase: str, run_id: str, call_log: Path, variant: str = "linear") -> list[str]:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "tests.fixtures.checkpoint_restart_phase",
            phase,
            run_id,
            str(call_log),
            variant,
        ],
        cwd=REPOSITORY_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPOSITORY_ROOT)},
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout.split()


async def drop_restart_collections() -> None:
    settings = Settings()
    client, database = create_mongodb(
        host=settings.mongodb_host,
        port=settings.mongodb_port,
        user=settings.mongodb_user,
        password=(
            settings.mongodb_pwd.get_secret_value() if settings.mongodb_pwd is not None else None
        ),
        database=settings.mongodb_db,
    )
    try:
        await database[CHECKPOINT_COLLECTION].drop()
        await database[CHECKPOINT_WRITES_COLLECTION].drop()
    finally:
        await client.close()


async def test_paused_run_resumes_in_a_new_process_without_repeating_the_lookup(
    tmp_path: Path,
) -> None:
    run_id = f"run-{uuid4()}"
    call_log = tmp_path / "calls.txt"
    try:
        paused = run_phase("pause", run_id, call_log)
        assert paused == ["clarification-1"]
        assert CallLog(call_log).calls() == ("lookup", "ask")

        resumed = run_phase("resume", run_id, call_log)
        assert resumed[0] == "ask"
        assert " ".join(resumed[1:]) == "Resolved market us across 3 markets available"
        # The two processes share only the checkpoint, and the completed lookup did
        # not run again.
        assert CallLog(call_log).calls().count("lookup") == 1
    finally:
        await drop_restart_collections()


async def test_work_finished_in_parallel_branches_survives_the_restart(
    tmp_path: Path,
) -> None:
    run_id = f"run-{uuid4()}"
    call_log = tmp_path / "calls.txt"
    try:
        run_phase("pause", run_id, call_log, "parallel")
        assert sorted(CallLog(call_log).calls()) == ["ask", "channel_lookup", "market_lookup"]

        resumed = run_phase("resume", run_id, call_log, "parallel")
        assert " ".join(resumed[1:]) == ("Resolved market us using channel_lookup, market_lookup")
        calls = CallLog(call_log).calls()
        assert calls.count("market_lookup") == 1
        assert calls.count("channel_lookup") == 1
    finally:
        await drop_restart_collections()


async def test_a_child_paused_in_one_process_is_continued_from_another(
    tmp_path: Path,
) -> None:
    """The whole boundary end to end: pause, process death, direct continuation, FIFO order.

    The second process is told only the run ID the first one reported. Everything else --
    which clarification is active, what it offered, the work already finished -- comes back
    from the checkpoint.
    """

    call_log = tmp_path / "calls.txt"
    try:
        paused = run_phase("pause", "unused", call_log, "child")
        run_id, active, queued = paused[0], paused[1], paused[2]
        assert active == "clarification-task-1"
        assert queued == "clarification-task-2"
        assert sorted(CallLog(call_log).calls()) == [
            "analyse_task-1",
            "analyse_task-2",
            "collect",
            "plan",
        ]

        resumed = run_phase("resume", run_id, call_log, "child")
        assert resumed[:2] == ["clarification-task-1", "clarification-task-2"]
        assert " ".join(resumed[2:]) == "Settled task-1=us, task-2=eu over 6 rows"
        calls = CallLog(call_log).calls()
        assert calls.count("analyse_task-1") == 1
        assert calls.count("analyse_task-2") == 1
        assert calls.count("plan") == 1
    finally:
        await drop_restart_collections()


async def test_a_full_pause_and_resume_never_needs_the_blocking_interface(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Guards the async-only decision: a sync call in this path would block the loop."""

    def refuse(*_args: Any, **_kwargs: Any) -> CheckpointTuple:
        raise AssertionError("the async execution path called a blocking saver method")

    for name in ("get_tuple", "list", "put", "put_writes", "delete_thread"):
        monkeypatch.setattr(MongoCheckpointSaver, name, refuse)

    run_id = f"run-{uuid4()}"
    call_log = tmp_path / "calls.txt"
    settings = Settings()
    client, database = create_mongodb(
        host=settings.mongodb_host,
        port=settings.mongodb_port,
        user=settings.mongodb_user,
        password=(
            settings.mongodb_pwd.get_secret_value() if settings.mongodb_pwd is not None else None
        ),
        database=settings.mongodb_db,
    )
    suffix = uuid4().hex[:12]
    checkpoints = f"test_checkpoints_{suffix}"
    writes = f"test_checkpoint_writes_{suffix}"
    try:
        from tests.fixtures.pausable_graph import create_pausable_graph, initial_state

        async with graph_checkpointer_scope(
            MongoGraphCheckpointer(
                database, checkpoints_collection=checkpoints, writes_collection=writes
            )
        ) as scoped:
            graph = create_pausable_graph(CallLog(call_log)).compile(checkpointer=scoped.saver)
            config = thread_config(run_id)
            await graph.ainvoke(initial_state(run_id), config)
            resumed = await graph.ainvoke(Command(resume="us"), config)

        assert resumed["answer"] == "Resolved market us across 3 markets available"
        await database[checkpoints].drop()
        await database[writes].drop()
    finally:
        await client.close()
