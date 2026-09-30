"""The graph boundary: a child that pauses, and a continuation that reaches it directly."""

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from langchain_core.messages import HumanMessage
from orchestration_core import (
    CancelResponse,
    ContinuationNotResumableError,
    ContinuationRequest,
    ConversationRequest,
    ConversationResult,
    ExecutionContext,
    ForbiddenAgentError,
    FreeTextResponse,
    InvalidAgentResultError,
    InvocationContext,
    OptionSelectionResponse,
    PausedConversation,
    RequestOrigin,
)

from agentic_orchestration.execution.agent_invoker import RegistryAgentInvoker
from agentic_orchestration.execution.agent_registry import AgentRegistry
from agentic_orchestration.execution.checkpointing import (
    MemoryGraphCheckpointer,
    graph_checkpointer_scope,
)
from tests.fixtures.pausable_child_agent import (
    AGENT_ID,
    CallLog,
    create_pausable_child_agent,
)

pytestmark = pytest.mark.integration

GRAPH_DEFINITION_ID = "definition-1"


class Boundary:
    """One compiled pause-capable child behind the restricted invocation seam."""

    def __init__(
        self, invoker: RegistryAgentInvoker, registry: AgentRegistry, call_log: CallLog
    ) -> None:
        self.invoker = invoker
        self.registry = registry
        self.calls = call_log

    def context(self) -> InvocationContext:
        return InvocationContext(
            execution=ExecutionContext(
                trace_id=str(uuid4()),
                root_span_id=str(uuid4()),
                session_id="session-1",
                attempted_turn_number=1,
                request_origin=RequestOrigin.API,
            ),
            caller_agent_id="router",
        )

    async def start(self, message: str = "How did spend perform?") -> Any:
        return await self.invoker.invoke(
            AGENT_ID,
            ConversationRequest(messages=(HumanMessage(content=message),)),
            self.context(),
        )

    async def answer(self, paused: PausedConversation, response: Any, **overrides: Any) -> Any:
        fields: dict[str, Any] = {
            "run_id": paused.run_id,
            "clarification_id": paused.clarification.clarification_id,
            "response": response,
            "freshness_token": paused.clarification.freshness_token,
            "graph_definition_id": paused.graph_definition_id,
        }
        fields.update(overrides)
        return await self.invoker.resume(AGENT_ID, ContinuationRequest(**fields), self.context())


async def boundary(
    tmp_path: Path, *, task_ids: tuple[str, ...] = ("task-1",)
) -> AsyncIterator[Boundary]:
    call_log = CallLog(tmp_path / "calls.txt")
    async with graph_checkpointer_scope(MemoryGraphCheckpointer()) as checkpointer:
        registry = AgentRegistry.compile(
            {
                AGENT_ID: lambda _dependencies: create_pausable_child_agent(
                    call_log, task_ids=task_ids
                )
            },
            savers={AGENT_ID: checkpointer.saver},
        )
        yield Boundary(
            RegistryAgentInvoker(
                registry,
                caller_agent_id="router",
                allowed_agent_ids=(AGENT_ID,),
                timeout_seconds=30,
                graph_definition_ids={AGENT_ID: GRAPH_DEFINITION_ID},
            ),
            registry,
            call_log,
        )


@pytest.fixture
async def single(tmp_path: Path) -> AsyncIterator[Boundary]:
    async for value in boundary(tmp_path):
        yield value


@pytest.fixture
async def parallel(tmp_path: Path) -> AsyncIterator[Boundary]:
    async for value in boundary(tmp_path, task_ids=("task-1", "task-2")):
        yield value


async def test_a_child_may_pause_instead_of_answering(single: Boundary) -> None:
    paused = await single.start()

    assert isinstance(paused, PausedConversation)
    assert paused.agent_id == AGENT_ID
    assert paused.clarification.question.startswith("Which market")
    assert paused.clarification.option_ids == ("us", "eu")
    assert paused.graph_definition_id == GRAPH_DEFINITION_ID


async def test_the_run_id_is_its_own_identity_not_the_session(single: Boundary) -> None:
    paused = await single.start()

    assert isinstance(paused, PausedConversation)
    assert paused.run_id != "session-1"
    assert paused.run_id


async def test_a_continuation_reaches_the_child_without_the_router(single: Boundary) -> None:
    paused = await single.start()
    assert isinstance(paused, PausedConversation)

    finished = await single.answer(paused, OptionSelectionResponse(option_ids=("us",)))

    assert isinstance(finished, ConversationResult)
    assert "task-1=us" in str(finished.generated_messages[-1].content)
    # The analysis that finished before the pause was restored, not repeated.
    assert single.calls.calls().count("analyse_task-1") == 1


async def test_free_text_and_cancel_continue_the_same_run(single: Boundary) -> None:
    paused = await single.start()
    assert isinstance(paused, PausedConversation)

    finished = await single.answer(paused, FreeTextResponse(text="Canada, please"))

    assert isinstance(finished, ConversationResult)
    assert "Canada, please" in str(finished.generated_messages[-1].content)


async def test_cancel_is_carried_to_the_producer(single: Boundary) -> None:
    paused = await single.start()
    assert isinstance(paused, PausedConversation)

    finished = await single.answer(paused, CancelResponse())

    assert isinstance(finished, ConversationResult)
    assert "cancel" in str(finished.generated_messages[-1].content)


async def test_only_the_offered_options_are_accepted(single: Boundary) -> None:
    from orchestration_core import ClarificationContractError

    paused = await single.start()
    assert isinstance(paused, PausedConversation)

    with pytest.raises(ClarificationContractError) as raised:
        await single.answer(paused, OptionSelectionResponse(option_ids=("mx",)))

    assert raised.value.reason_code == "unknown_option"


async def test_parallel_ambiguities_are_asked_one_at_a_time_in_plan_order(
    parallel: Boundary,
) -> None:
    first = await parallel.start()
    assert isinstance(first, PausedConversation)
    assert first.clarification.task_id == "task-1"
    assert first.queued_clarification_ids == ("clarification-task-2",)

    second = await parallel.answer(first, OptionSelectionResponse(option_ids=("us",)))
    assert isinstance(second, PausedConversation)
    assert second.clarification.task_id == "task-2"
    assert second.queued_clarification_ids == ()
    assert second.run_id == first.run_id

    finished = await parallel.answer(second, OptionSelectionResponse(option_ids=("eu",)))
    assert isinstance(finished, ConversationResult)
    assert "task-1=us, task-2=eu" in str(finished.generated_messages[-1].content)


async def test_neither_parallel_branch_repeats_its_completed_work(parallel: Boundary) -> None:
    first = await parallel.start()
    assert isinstance(first, PausedConversation)
    second = await parallel.answer(first, OptionSelectionResponse(option_ids=("us",)))
    assert isinstance(second, PausedConversation)
    await parallel.answer(second, OptionSelectionResponse(option_ids=("eu",)))

    calls = parallel.calls.calls()
    assert calls.count("analyse_task-1") == 1
    assert calls.count("analyse_task-2") == 1
    assert calls.count("plan") == 1


async def test_answering_the_queued_clarification_first_is_refused(parallel: Boundary) -> None:
    first = await parallel.start()
    assert isinstance(first, PausedConversation)

    with pytest.raises(ContinuationNotResumableError) as raised:
        await parallel.answer(
            first,
            OptionSelectionResponse(option_ids=("us",)),
            clarification_id="clarification-task-2",
        )

    assert raised.value.reason == "unknown_clarification"


async def test_a_graph_that_changed_while_paused_requires_a_restart(single: Boundary) -> None:
    paused = await single.start()
    assert isinstance(paused, PausedConversation)

    with pytest.raises(ContinuationNotResumableError) as raised:
        await single.answer(
            paused, OptionSelectionResponse(option_ids=("us",)), graph_definition_id="definition-2"
        )

    assert raised.value.reason == "graph_version"


async def test_a_run_paused_under_another_contract_requires_a_restart(single: Boundary) -> None:
    paused = await single.start()
    assert isinstance(paused, PausedConversation)

    with pytest.raises(ContinuationNotResumableError) as raised:
        await single.answer(
            paused, OptionSelectionResponse(option_ids=("us",)), continuation_contract_version=2
        )

    assert raised.value.reason == "contract_version"


async def test_a_stale_producer_snapshot_requires_a_restart(single: Boundary) -> None:
    paused = await single.start()
    assert isinstance(paused, PausedConversation)

    with pytest.raises(ContinuationNotResumableError) as raised:
        await single.answer(
            paused, OptionSelectionResponse(option_ids=("us",)), freshness_token="snapshot-stale"
        )

    assert raised.value.reason == "stale_producer"


async def test_an_unknown_run_has_no_checkpoint_to_continue(single: Boundary) -> None:
    paused = await single.start()
    assert isinstance(paused, PausedConversation)

    with pytest.raises(ContinuationNotResumableError) as raised:
        await single.answer(
            paused, OptionSelectionResponse(option_ids=("us",)), run_id=f"run-{uuid4()}"
        )

    assert raised.value.reason == "missing_checkpoint"


async def test_a_finished_run_cannot_be_answered_again(single: Boundary) -> None:
    """The graph is no longer waiting, so a replayed answer is refused rather than reapplied."""

    paused = await single.start()
    assert isinstance(paused, PausedConversation)
    await single.answer(paused, OptionSelectionResponse(option_ids=("us",)))

    with pytest.raises(ContinuationNotResumableError) as raised:
        await single.answer(paused, OptionSelectionResponse(option_ids=("eu",)))

    assert raised.value.reason == "not_paused"
    assert single.calls.calls().count("answer") == 1


async def test_a_continuation_obeys_the_same_eligibility_as_routing(single: Boundary) -> None:
    paused = await single.start()
    assert isinstance(paused, PausedConversation)
    restricted = RegistryAgentInvoker(
        single.registry,
        caller_agent_id="router",
        allowed_agent_ids=(),
        timeout_seconds=30,
        graph_definition_ids={AGENT_ID: GRAPH_DEFINITION_ID},
    )

    with pytest.raises(ForbiddenAgentError):
        await restricted.resume(
            AGENT_ID,
            ContinuationRequest(
                run_id=paused.run_id,
                clarification_id=paused.clarification.clarification_id,
                response=OptionSelectionResponse(option_ids=("us",)),
                freshness_token=paused.clarification.freshness_token,
            ),
            single.context(),
        )


async def test_an_agent_without_a_checkpointer_cannot_be_continued(tmp_path: Path) -> None:
    call_log = CallLog(tmp_path / "calls.txt")
    registry = AgentRegistry.compile(
        {AGENT_ID: lambda _dependencies: create_pausable_child_agent(call_log)},
    )
    invoker = RegistryAgentInvoker(
        registry,
        caller_agent_id="router",
        allowed_agent_ids=(AGENT_ID,),
        timeout_seconds=30,
    )

    with pytest.raises(ContinuationNotResumableError) as raised:
        await invoker.resume(
            AGENT_ID,
            ContinuationRequest(
                run_id="run-1",
                clarification_id="clarification-task-1",
                response=OptionSelectionResponse(option_ids=("us",)),
                freshness_token="snapshot-task-1",
            ),
            InvocationContext(
                execution=ExecutionContext(
                    trace_id="trace-1",
                    root_span_id="root-1",
                    session_id="session-1",
                    attempted_turn_number=1,
                ),
                caller_agent_id="router",
            ),
        )

    assert raised.value.reason == "missing_checkpoint"


async def test_a_pause_that_is_not_a_clarification_is_an_invalid_result(tmp_path: Path) -> None:
    """The pause payload is the clarification contract, not whatever a producer passes."""

    from langgraph.graph import END, START, StateGraph
    from langgraph.types import interrupt

    async def ask(_state: dict[str, Any]) -> dict[str, Any]:
        interrupt({"question": "which one?"})
        return {}

    def build(_dependencies: Any) -> Any:
        builder: StateGraph[Any] = StateGraph(dict)
        builder.add_node("ask", ask)
        builder.add_edge(START, "ask")
        builder.add_edge("ask", END)
        return builder

    async with graph_checkpointer_scope(MemoryGraphCheckpointer()) as checkpointer:
        registry = AgentRegistry.compile({AGENT_ID: build}, savers={AGENT_ID: checkpointer.saver})
        invoker = RegistryAgentInvoker(
            registry,
            caller_agent_id="router",
            allowed_agent_ids=(AGENT_ID,),
            timeout_seconds=30,
        )

        with pytest.raises(InvalidAgentResultError, match="invalid clarification request"):
            await invoker.invoke(
                AGENT_ID,
                ConversationRequest(messages=(HumanMessage(content="spend?"),)),
                InvocationContext(
                    execution=ExecutionContext(
                        trace_id="trace-1",
                        root_span_id="root-1",
                        session_id="session-1",
                        attempted_turn_number=1,
                    ),
                    caller_agent_id="router",
                ),
            )
