"""The bounded state a paused graph may retain across a checkpoint."""

from datetime import UTC, datetime

import pytest
from orchestration_core import (
    MAX_CHECKPOINT_STATE_BYTES,
    ActionReceipt,
    ArtifactReference,
    CheckpointedTaskState,
    CompactResult,
    TaskContext,
    UnboundedCheckpointStateError,
)
from pydantic import ValidationError

COMPLETED_AT = datetime(2026, 9, 15, 12, 0, tzinfo=UTC)


def artifact(artifact_id: str = "spend-by-market") -> ArtifactReference:
    return ArtifactReference(
        store="scripted-artifacts",
        artifact_id=artifact_id,
        media_type="application/vnd.apache.arrow.file",
        row_count=3,
    )


def receipt(receipt_id: str = "receipt-1", action: str = "market_lookup") -> ActionReceipt:
    return ActionReceipt(
        receipt_id=receipt_id,
        task_id="task-1",
        action=action,
        status="succeeded",
        completed_at=COMPLETED_AT,
        summary="3 markets available",
        artifacts=(artifact(),),
    )


def state(**overrides: object) -> CheckpointedTaskState:
    values: dict[str, object] = {
        "run_id": "run-1",
        "tasks": (TaskContext(task_id="task-1", objective="Summarise spend by market"),),
        "receipts": (receipt(),),
        "results": (
            CompactResult(result_id="result-1", task_id="task-1", headline="United States"),
        ),
        "artifacts": (artifact(),),
    }
    values.update(overrides)
    return CheckpointedTaskState.model_validate(values)


@pytest.mark.unit
def test_state_keeps_task_context_receipts_results_and_artifact_references() -> None:
    bounded = state()

    assert bounded.continuation_contract_version == 1
    assert bounded.receipt_for("task-1", "market_lookup") is not None
    assert bounded.artifacts[0].artifact_id == "spend-by-market"
    assert bounded.artifacts[0].row_count == 3


@pytest.mark.unit
def test_receipt_lookup_distinguishes_actions_and_tasks() -> None:
    bounded = state(receipts=(receipt(), receipt("receipt-2", "market_summary")))

    assert bounded.receipt_for("task-1", "market_summary") is not None
    assert bounded.receipt_for("task-1", "unrun_action") is None
    assert bounded.receipt_for("task-2", "market_lookup") is None


@pytest.mark.unit
@pytest.mark.parametrize(
    "rejected",
    [
        {"rows": [["us", 1], ["eu", 2]]},
        {"credentials": {"authorization": "Bearer token"}},
        {"provider_payload": {"choices": [{"message": {"content": "raw"}}]}},
    ],
    ids=["raw_table", "credentials", "provider_payload"],
)
def test_state_has_nowhere_to_hold_bulk_or_secret_content(rejected: dict[str, object]) -> None:
    with pytest.raises(ValidationError) as raised:
        state(**rejected)

    assert raised.value.errors()[0]["type"] == "extra_forbidden"


@pytest.mark.unit
def test_receipt_has_no_field_for_the_rows_it_describes() -> None:
    assert "rows" not in ActionReceipt.model_fields
    assert "artifacts" in ActionReceipt.model_fields

    with pytest.raises(ValidationError) as raised:
        ActionReceipt.model_validate(
            {**receipt().model_dump(mode="json"), "rows": [["us", 1], ["eu", 2]]}
        )

    assert raised.value.errors()[0]["type"] == "extra_forbidden"


@pytest.mark.unit
def test_compact_result_metrics_cannot_carry_untyped_payloads() -> None:
    with pytest.raises(ValidationError):
        CompactResult(
            result_id="result-1",
            task_id="task-1",
            headline="United States",
            metrics={"raw": {"nested": "payload"}},  # type: ignore[dict-item]
        )


@pytest.mark.unit
def test_state_above_its_byte_bound_is_refused() -> None:
    oversized = tuple(
        ActionReceipt(
            receipt_id=f"receipt-{index}",
            task_id="task-1",
            action="market_lookup",
            status="succeeded",
            completed_at=COMPLETED_AT,
            summary="x" * 1000,
            artifacts=tuple(artifact(f"artifact-{index}-{inner}") for inner in range(20)),
        )
        for index in range(200)
    )

    with pytest.raises(UnboundedCheckpointStateError, match="above the"):
        state(receipts=oversized)

    assert len(state().model_dump_json().encode("utf-8")) < MAX_CHECKPOINT_STATE_BYTES


@pytest.mark.unit
def test_duplicate_identities_within_one_state_are_refused() -> None:
    with pytest.raises(ValidationError, match="receipts must have distinct identities"):
        state(receipts=(receipt(), receipt()))

    with pytest.raises(ValidationError, match="tasks must have distinct identities"):
        state(
            tasks=(
                TaskContext(task_id="task-1", objective="first"),
                TaskContext(task_id="task-1", objective="second"),
            )
        )


@pytest.mark.unit
def test_task_dependencies_must_be_distinct_and_not_self_referential() -> None:
    with pytest.raises(ValidationError, match="depend on itself"):
        TaskContext(task_id="task-1", objective="Summarise", depends_on=("task-1",))

    with pytest.raises(ValidationError, match="depends_on must be distinct"):
        TaskContext(task_id="task-1", objective="Summarise", depends_on=("task-2", "task-2"))


@pytest.mark.unit
def test_timestamps_must_be_absolute() -> None:
    with pytest.raises(ValidationError, match="timezone aware"):
        ActionReceipt(
            receipt_id="receipt-1",
            task_id="task-1",
            action="market_lookup",
            status="succeeded",
            completed_at=COMPLETED_AT.replace(tzinfo=None),
        )

    with pytest.raises(ValidationError, match="timezone aware"):
        ArtifactReference(
            store="scripted-artifacts",
            artifact_id="spend-by-market",
            media_type="text/csv",
            retained_until=COMPLETED_AT.replace(tzinfo=None),
        )
