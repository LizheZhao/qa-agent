"""The bounded state a pause-capable graph may carry across a durable checkpoint.

A checkpoint outlives the request that produced it and is written to shared storage, so
what a paused graph keeps is a security and cost boundary rather than a convenience. This
contract is closed by construction: every model forbids unknown fields, and none of them
declares a field able to hold a result set, an authentication header, or a provider
response. Bulk data stays in its authorized artifact or tool store and is referenced here
by identity only.

An artifact a paused run depends on must remain readable for at least the pause window, so
`ArtifactReference.retained_until` is the producer's assertion about that, not a hint.
"""

from __future__ import annotations

from datetime import datetime
from typing import Final, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from orchestration_core.clarification import (
    CONTINUATION_CONTRACT_VERSION,
    OPAQUE_ID_PATTERN,
    REASON_CODE_PATTERN,
    ClarificationRequest,
)

MAX_CHECKPOINT_STATE_BYTES: Final = 256 * 1024
MAX_TASKS: Final = 50
MAX_RECEIPTS: Final = 200
MAX_RESULTS: Final = 50
MAX_ARTIFACTS: Final = 100
MAX_METRICS: Final = 50
MAX_SUMMARY_CHARACTERS: Final = 1000

ReceiptStatus = Literal["succeeded", "failed", "skipped"]


class UnboundedCheckpointStateError(RuntimeError):
    """A candidate checkpoint state exceeds the size a paused run may retain.

    Not a `ValueError`, so it propagates out of model validation with its own type
    instead of collapsing into a field-shape `ValidationError`: exceeding the retention
    bound is a policy breach a caller may want to handle on its own, and the distinction
    keeps it from being reported as a malformed field.
    """


class ArtifactReference(BaseModel):
    """Identity of a result held in an authorized store, never its contents."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    store: str = Field(pattern=OPAQUE_ID_PATTERN)
    artifact_id: str = Field(pattern=OPAQUE_ID_PATTERN)
    media_type: str = Field(min_length=1, max_length=128)
    row_count: int | None = Field(default=None, ge=0)
    retained_until: datetime | None = None

    @field_validator("retained_until")
    @classmethod
    def retention_must_be_absolute(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.tzinfo is None:
            raise ValueError("retained_until must be timezone aware")
        return value


class ActionReceipt(BaseModel):
    """Evidence that one action already happened, so a resume does not repeat it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    receipt_id: str = Field(pattern=OPAQUE_ID_PATTERN)
    task_id: str = Field(pattern=OPAQUE_ID_PATTERN)
    action: str = Field(pattern=REASON_CODE_PATTERN)
    status: ReceiptStatus
    completed_at: datetime
    summary: str = Field(default="", max_length=MAX_SUMMARY_CHARACTERS)
    artifacts: tuple[ArtifactReference, ...] = Field(default=(), max_length=MAX_ARTIFACTS)

    @field_validator("completed_at")
    @classmethod
    def completion_must_be_absolute(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("completed_at must be timezone aware")
        return value


class CompactResult(BaseModel):
    """A compacted answer to one task: headline text, numeric metrics, artifact identity.

    `metrics` values are numeric so the mapping cannot become an untyped payload hole.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    result_id: str = Field(pattern=OPAQUE_ID_PATTERN)
    task_id: str = Field(pattern=OPAQUE_ID_PATTERN)
    headline: str = Field(min_length=1, max_length=MAX_SUMMARY_CHARACTERS)
    metrics: dict[str, float] = Field(default_factory=dict, max_length=MAX_METRICS)
    artifacts: tuple[ArtifactReference, ...] = Field(default=(), max_length=MAX_ARTIFACTS)


class TaskContext(BaseModel):
    """One planned unit of work a paused run still owns."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str = Field(pattern=OPAQUE_ID_PATTERN)
    objective: str = Field(min_length=1, max_length=MAX_SUMMARY_CHARACTERS)
    depends_on: tuple[str, ...] = Field(default=(), max_length=MAX_TASKS)

    @model_validator(mode="after")
    def dependencies_must_be_distinct_and_acyclic(self) -> Self:
        if len(set(self.depends_on)) != len(self.depends_on):
            raise ValueError("depends_on must be distinct")
        if self.task_id in self.depends_on:
            raise ValueError("a task must not depend on itself")
        return self


class CheckpointedTaskState(BaseModel):
    """Everything a pause-capable child graph may carry across a pause.

    Deliberately not a place for conversation messages: the router owns conversation
    history in the session store, and a checkpoint that duplicated it would have two
    sources of truth for the committed turn.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    continuation_contract_version: Literal[1] = CONTINUATION_CONTRACT_VERSION
    run_id: str = Field(pattern=OPAQUE_ID_PATTERN)
    tasks: tuple[TaskContext, ...] = Field(default=(), max_length=MAX_TASKS)
    receipts: tuple[ActionReceipt, ...] = Field(default=(), max_length=MAX_RECEIPTS)
    results: tuple[CompactResult, ...] = Field(default=(), max_length=MAX_RESULTS)
    artifacts: tuple[ArtifactReference, ...] = Field(default=(), max_length=MAX_ARTIFACTS)
    queued_clarifications: tuple[ClarificationRequest, ...] = Field(
        default=(), max_length=MAX_TASKS
    )

    @model_validator(mode="after")
    def identities_must_be_distinct(self) -> Self:
        for field, values in (
            ("tasks", [task.task_id for task in self.tasks]),
            ("receipts", [receipt.receipt_id for receipt in self.receipts]),
            ("results", [result.result_id for result in self.results]),
            (
                "queued_clarifications",
                [request.clarification_id for request in self.queued_clarifications],
            ),
        ):
            if len(set(values)) != len(values):
                raise ValueError(f"{field} must have distinct identities")
        return self

    @model_validator(mode="after")
    def state_must_stay_within_its_bound(self) -> Self:
        size = len(self.model_dump_json().encode("utf-8"))
        if size > MAX_CHECKPOINT_STATE_BYTES:
            raise UnboundedCheckpointStateError(
                f"checkpoint state is {size} bytes, above the {MAX_CHECKPOINT_STATE_BYTES} bound"
            )
        return self

    def receipt_for(self, task_id: str, action: str) -> ActionReceipt | None:
        """The receipt proving `action` already ran for `task_id`, if it did.

        A resumed graph restarts the interrupted node from its beginning, so a node that
        performs work consults its receipt rather than assuming it has not run.
        """

        return next(
            (
                receipt
                for receipt in self.receipts
                if receipt.task_id == task_id and receipt.action == action
            ),
            None,
        )
