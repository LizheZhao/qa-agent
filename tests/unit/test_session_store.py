import asyncio
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agentic_orchestration.sessions import MemorySessionStore
from agentic_orchestration.sessions.contracts import CommitTurn, SessionConflictError


def command(
    session_id: str,
    text: str,
    *,
    expected_revision: int | None,
) -> CommitTurn:
    human = HumanMessage(content=text, id=str(uuid4()))
    assistant = AIMessage(content=f"answer {text}", id=str(uuid4()))
    assert assistant.id is not None
    return CommitTurn(
        session_id=session_id,
        expected_revision=expected_revision,
        tenant_id=None,
        user_id=None,
        agent_id="router",
        input_message=human,
        generated_messages=(assistant,),
        final_message_id=assistant.id,
        routing_outcome="delegate",
        selected_agent_id="marketing_science",
        trace_id=f"trace-{session_id}-{text}",
    )


@pytest.mark.unit
async def test_session_store_commits_successful_turns_and_isolates_sessions() -> None:
    store = MemorySessionStore()
    first = await store.commit_turn(command("one", "one", expected_revision=None))
    second = await store.commit_turn(command("one", "two", expected_revision=first.revision))

    assert second.revision == 2
    assert second.active_agent_id == "marketing_science"
    assert [turn.trace_id for turn in second.turns] == ["trace-one-one", "trace-one-two"]
    assert [message.content for message in second.messages] == [
        "one",
        "answer one",
        "two",
        "answer two",
    ]
    assert await store.load("two") is None


@pytest.mark.unit
async def test_session_store_rejects_stale_revision() -> None:
    store = MemorySessionStore()
    await store.commit_turn(command("same", "one", expected_revision=None))

    with pytest.raises(SessionConflictError):
        await store.commit_turn(command("same", "stale", expected_revision=0))


@pytest.mark.unit
async def test_session_lock_serializes_load_and_commit() -> None:
    store = MemorySessionStore()
    await store.commit_turn(command("same", "initial", expected_revision=None))

    async def append(value: str) -> None:
        async with store.lock("same"):
            snapshot = await store.load("same")
            assert snapshot is not None
            await asyncio.sleep(0)
            await store.commit_turn(command("same", value, expected_revision=snapshot.revision))

    await asyncio.gather(append("one"), append("two"))
    snapshot = await store.load("same")
    assert snapshot is not None
    assert snapshot.revision == 3
