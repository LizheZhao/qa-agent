"""Deterministic ordering of clarifications reported by parallel producers."""

import pytest
from orchestration_core import (
    ClarificationContractError,
    ClarificationOption,
    ClarificationQueue,
    ClarificationRequest,
    order_clarifications,
)


def request(task_id: str, clarification_id: str | None = None) -> ClarificationRequest:
    return ClarificationRequest(
        clarification_id=clarification_id or f"clarification-{task_id}",
        producer_agent_id="scripted_producer",
        task_id=task_id,
        question=f"Which market should {task_id} cover?",
        reason_code="ambiguous_market",
        selection_mode="single",
        options=(
            ClarificationOption(option_id="us", label="United States"),
            ClarificationOption(option_id="eu", label="Europe"),
        ),
        freshness_token=f"snapshot-{task_id}",
    )


@pytest.mark.unit
def test_plan_order_decides_which_question_is_asked_first() -> None:
    plan = ("task-b", "task-a")

    ordered = order_clarifications((request("task-a"), request("task-b")), plan_task_ids=plan)

    assert [entry.task_id for entry in ordered] == ["task-b", "task-a"]


@pytest.mark.unit
def test_ordering_ignores_the_order_producers_finished_in() -> None:
    plan = ("task-a", "task-b", "task-c")
    arrivals = (request("task-c"), request("task-a"), request("task-b"))

    first = order_clarifications(arrivals, plan_task_ids=plan)
    second = order_clarifications(tuple(reversed(arrivals)), plan_task_ids=plan)

    assert [entry.task_id for entry in first] == ["task-a", "task-b", "task-c"]
    assert first == second


@pytest.mark.unit
def test_tasks_outside_the_plan_follow_it_in_task_order() -> None:
    ordered = order_clarifications(
        (request("task-z"), request("task-planned"), request("task-y")),
        plan_task_ids=("task-planned",),
    )

    assert [entry.task_id for entry in ordered] == ["task-planned", "task-y", "task-z"]


@pytest.mark.unit
def test_two_clarifications_for_one_task_are_ordered_by_clarification_id() -> None:
    ordered = order_clarifications(
        (request("task-a", "clarification-2"), request("task-a", "clarification-1")),
        plan_task_ids=("task-a",),
    )

    assert [entry.clarification_id for entry in ordered] == ["clarification-1", "clarification-2"]


@pytest.mark.unit
def test_without_a_plan_ordering_still_does_not_depend_on_arrival() -> None:
    ordered = order_clarifications((request("task-b"), request("task-a")))

    assert [entry.task_id for entry in ordered] == ["task-a", "task-b"]


@pytest.mark.unit
def test_an_empty_queue_has_nothing_active() -> None:
    queue = ClarificationQueue()

    assert queue.active is None
    assert queue.queued_ids == ()


@pytest.mark.unit
def test_only_the_head_is_active_and_the_rest_are_queued() -> None:
    queue = ClarificationQueue(pending=(request("task-a"), request("task-b")))

    assert queue.active is not None
    assert queue.active.task_id == "task-a"
    assert queue.queued_ids == ("clarification-task-b",)


@pytest.mark.unit
def test_answering_the_head_promotes_the_next_clarification() -> None:
    queue = ClarificationQueue(pending=(request("task-a"), request("task-b")))

    remaining = queue.answered("clarification-task-a")

    assert remaining.active is not None
    assert remaining.active.task_id == "task-b"
    assert remaining.queued_ids == ()
    assert remaining.answered("clarification-task-b").active is None


@pytest.mark.unit
def test_only_the_active_clarification_can_be_answered() -> None:
    queue = ClarificationQueue(pending=(request("task-a"), request("task-b")))

    with pytest.raises(ClarificationContractError) as raised:
        queue.answered("clarification-task-b")

    assert raised.value.reason_code == "unknown_clarification"


@pytest.mark.unit
def test_answering_an_empty_queue_is_refused() -> None:
    with pytest.raises(ClarificationContractError):
        ClarificationQueue().answered("clarification-task-a")


@pytest.mark.unit
def test_a_later_arrival_cannot_displace_the_question_already_shown() -> None:
    """The head has been put in front of a user; reordering it would change the question."""

    queue = ClarificationQueue(pending=(request("task-z"),))

    extended = queue.extend((request("task-a"),), plan_task_ids=("task-a", "task-z"))

    assert extended.active is not None
    assert extended.active.task_id == "task-z"
    assert extended.queued_ids == ("clarification-task-a",)


@pytest.mark.unit
def test_arrivals_are_ordered_against_the_clarifications_not_yet_asked() -> None:
    queue = ClarificationQueue(pending=(request("task-a"), request("task-d")))

    extended = queue.extend(
        (request("task-c"), request("task-b")),
        plan_task_ids=("task-a", "task-b", "task-c", "task-d"),
    )

    assert extended.queued_ids == (
        "clarification-task-b",
        "clarification-task-c",
        "clarification-task-d",
    )


@pytest.mark.unit
def test_a_clarification_already_queued_is_not_added_twice() -> None:
    queue = ClarificationQueue(pending=(request("task-a"),))

    extended = queue.extend((request("task-a"),))

    assert extended.pending == queue.pending


@pytest.mark.unit
def test_a_queue_cannot_hold_the_same_clarification_twice() -> None:
    with pytest.raises(ValueError, match="same clarification twice"):
        ClarificationQueue(pending=(request("task-a"), request("task-a")))
