"""Test-only subprocess phase proving a paused run outlives the process that started it.

Run as `python -m tests.fixtures.checkpoint_restart_phase <phase> <run_id> <call_log> [variant]`.
In the `child` variant the invocation boundary chooses the run ID itself, so the pause
phase ignores the argument and prints the ID the resume phase must be given.
Each phase is a separate operating-system process, which is the only way to show that the
pause survived a restart rather than a stale object still holding the state in memory. The
two processes share nothing but MongoDB.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any
from uuid import uuid4

from langchain_core.messages import HumanMessage
from langgraph.types import Command
from orchestration_core import (
    ContinuationRequest,
    ConversationRequest,
    ConversationResult,
    ExecutionContext,
    InvocationContext,
    OptionSelectionResponse,
    PausedConversation,
    RequestOrigin,
)

from agentic_orchestration.config import Settings
from agentic_orchestration.execution.agent_invoker import RegistryAgentInvoker
from agentic_orchestration.execution.agent_registry import AgentRegistry
from agentic_orchestration.execution.checkpointing import graph_checkpointer_scope
from agentic_orchestration.execution.mongodb import MongoGraphCheckpointer
from integrations.mongodb import create_mongodb
from tests.fixtures.pausable_child_agent import AGENT_ID, create_pausable_child_agent
from tests.fixtures.pausable_child_agent import CallLog as ChildCallLog
from tests.fixtures.pausable_graph import (
    CallLog,
    create_parallel_pausable_graph,
    create_pausable_graph,
    initial_state,
)

CHILD_GRAPH_DEFINITION_ID = "definition-1"

CHECKPOINT_COLLECTION = "test_graph_checkpoints"
CHECKPOINT_WRITES_COLLECTION = "test_graph_checkpoint_writes"


async def run_phase(phase: str, run_id: str, call_log_path: Path, variant: str) -> None:
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
    if variant == "child":
        try:
            await _run_child_phase(phase, run_id, call_log_path, database)
        finally:
            await client.close()
        return
    build = create_parallel_pausable_graph if variant == "parallel" else create_pausable_graph
    config = {"configurable": {"thread_id": run_id}}
    try:
        async with graph_checkpointer_scope(
            MongoGraphCheckpointer(
                database,
                checkpoints_collection=CHECKPOINT_COLLECTION,
                writes_collection=CHECKPOINT_WRITES_COLLECTION,
            )
        ) as checkpointer:
            graph = build(CallLog(call_log_path)).compile(checkpointer=checkpointer.saver)
            if phase == "pause":
                result = await graph.ainvoke(initial_state(run_id), config)
                print(result["__interrupt__"][0].value["clarification_id"])
            elif phase == "resume":
                recovered = await graph.aget_state(config)
                print(",".join(recovered.next))
                resumed = await graph.ainvoke(Command(resume="us"), config)
                print(resumed["answer"])
            else:
                raise SystemExit(f"unknown phase {phase!r}")
    finally:
        await client.close()


async def _run_child_phase(phase: str, run_id: str, call_log_path: Path, database: Any) -> None:
    """Drive a pause-capable child through the restricted invocation seam.

    The production shape rather than a bare graph call: the boundary chooses the run ID,
    reports the pause, and later resumes the child directly. `run_id` is reused across the
    two processes as the identity a continuation names.
    """

    call_log = ChildCallLog(call_log_path)
    async with graph_checkpointer_scope(
        MongoGraphCheckpointer(
            database,
            checkpoints_collection=CHECKPOINT_COLLECTION,
            writes_collection=CHECKPOINT_WRITES_COLLECTION,
        )
    ) as checkpointer:
        registry = AgentRegistry.compile(
            {
                AGENT_ID: lambda _dependencies: create_pausable_child_agent(
                    call_log, task_ids=("task-1", "task-2")
                )
            },
            savers={AGENT_ID: checkpointer.saver},
        )
        invoker = RegistryAgentInvoker(
            registry,
            caller_agent_id="router",
            allowed_agent_ids=(AGENT_ID,),
            timeout_seconds=120,
            graph_definition_ids={AGENT_ID: CHILD_GRAPH_DEFINITION_ID},
        )
        context = InvocationContext(
            execution=ExecutionContext(
                trace_id=str(uuid4()),
                root_span_id=str(uuid4()),
                session_id="session-1",
                attempted_turn_number=1,
                request_origin=RequestOrigin.API,
            ),
            caller_agent_id="router",
        )
        if phase == "pause":
            outcome = await invoker.invoke(
                AGENT_ID,
                ConversationRequest(messages=(HumanMessage(content="How did spend perform?"),)),
                context,
            )
            assert isinstance(outcome, PausedConversation)
            print(outcome.run_id)
            print(outcome.clarification.clarification_id)
            print(",".join(outcome.queued_clarification_ids))
            return
        if phase != "resume":
            raise SystemExit(f"unknown phase {phase!r}")
        # A fresh process knows only the run ID; everything else comes from the checkpoint.
        for choice in ("us", "eu"):
            snapshot = await registry.get(AGENT_ID).aget_state(
                {"configurable": {"thread_id": run_id}}
            )
            from orchestration_core import ClarificationRequest

            active = ClarificationRequest.model_validate(snapshot.interrupts[0].value)
            outcome = await invoker.resume(
                AGENT_ID,
                ContinuationRequest(
                    run_id=run_id,
                    clarification_id=active.clarification_id,
                    response=OptionSelectionResponse(option_ids=(choice,)),
                    freshness_token=active.freshness_token,
                    graph_definition_id=CHILD_GRAPH_DEFINITION_ID,
                ),
                context,
            )
            print(active.clarification_id)
            if isinstance(outcome, ConversationResult):
                print(str(outcome.generated_messages[-1].content))
                return


if __name__ == "__main__":
    asyncio.run(
        run_phase(
            sys.argv[1],
            sys.argv[2],
            Path(sys.argv[3]),
            sys.argv[4] if len(sys.argv) > 4 else "linear",
        )
    )
