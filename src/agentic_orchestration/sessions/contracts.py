"""Application-owned contracts for durable conversation history."""

from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Protocol

from langchain_core.messages import BaseMessage
from orchestration_core import ClarificationRequest, ClarificationResponse

RoutingOutcome = Literal["answer", "delegate", "clarify", "reject"]
PendingRunStatus = Literal["awaiting_response", "claimed", "consumed", "canceled", "expired"]
ContinuationClosure = Literal["answered", "canceled", "expired"]
ClarificationEvent = Literal["published", "answered", "canceled", "expired"]


class SessionStorageError(RuntimeError):
    """A session store operation could not be completed."""


class SessionConflictError(SessionStorageError):
    """The durable session changed after the caller loaded its snapshot."""


class PendingRunConflictError(SessionConflictError):
    """The session already holds an in-flight run, or the one addressed has moved on."""


class PendingRunNotFoundError(SessionStorageError):
    """No open pending run answers to this session and clarification.

    Raised for a run that never existed and for one already consumed, cancelled, or
    expired. The two are the same to a caller holding a stale clarification ID: there is
    nothing left to answer either way.
    """


class PendingRunExpiredError(SessionStorageError):
    """The pause is past its expiry and can no longer be answered."""


class SubmissionConflictError(SessionStorageError):
    """A submission identifier was reused for different content.

    A retry may repeat its own submission while the outcome is unresolved; it may not
    substitute a different choice under an identifier the user has already committed to.
    """


@dataclass(frozen=True)
class TurnContinuation:
    """How a committed turn relates to the paused run it closed."""

    run_id: str
    clarification_ids: tuple[str, ...]
    closed_by: ContinuationClosure


@dataclass(frozen=True)
class SessionTurn:
    turn_number: int
    input_message: BaseMessage
    generated_messages: tuple[BaseMessage, ...]
    final_message_id: str
    routing_outcome: RoutingOutcome
    selected_agent_id: str | None
    trace_id: str | None = None
    committed_at: datetime | None = None
    continuation: TurnContinuation | None = None

    def __post_init__(self) -> None:
        if (self.routing_outcome == "delegate") != (self.selected_agent_id is not None):
            raise ValueError("selected_agent_id must be set exactly for delegated turns")

    @property
    def messages(self) -> tuple[BaseMessage, ...]:
        return (self.input_message, *self.generated_messages)


@dataclass(frozen=True)
class SessionSnapshot:
    session_id: str
    revision: int
    tenant_id: str | None
    user_id: str | None
    agent_id: str
    turns: tuple[SessionTurn, ...]
    status: Literal["active", "archived"] = "active"
    created_at: datetime | None = None
    updated_at: datetime | None = None

    @property
    def messages(self) -> tuple[BaseMessage, ...]:
        return tuple(message for turn in self.turns for message in turn.messages)

    @property
    def active_agent_id(self) -> str | None:
        if not self.turns:
            return None
        return self.turns[-1].selected_agent_id


@dataclass(frozen=True)
class CommitTurn:
    session_id: str
    expected_revision: int | None
    tenant_id: str | None
    user_id: str | None
    agent_id: str
    input_message: BaseMessage
    generated_messages: tuple[BaseMessage, ...]
    final_message_id: str
    routing_outcome: RoutingOutcome
    selected_agent_id: str | None
    trace_id: str

    def __post_init__(self) -> None:
        if (self.routing_outcome == "delegate") != (self.selected_agent_id is not None):
            raise ValueError("selected_agent_id must be set exactly for delegated turns")


@dataclass(frozen=True)
class PendingRun:
    """One paused run a session is holding, as it stands durably.

    Everything a continuation needs is resolved from here rather than supplied by the
    caller: which child to resume, the thread it paused on, the graph and contract it
    paused under, and the producer freshness token behind the question. A client that
    could name those could aim a continuation at a different run or a different graph.
    """

    session_id: str
    run_id: str
    clarification: ClarificationRequest
    status: PendingRunStatus
    agent_id: str
    selected_agent_id: str
    expected_session_revision: int | None
    tenant_id: str | None
    user_id: str | None
    input_message: BaseMessage
    route_reason_code: str
    trace_id: str
    graph_definition_id: str | None
    continuation_contract_version: int
    submission_id: str | None
    claimed_response: ClarificationResponse | None
    created_at: datetime
    expires_at: datetime
    attempt_trace_ids: tuple[str, ...] = ()

    @property
    def clarification_id(self) -> str:
        return self.clarification.clarification_id


@dataclass(frozen=True)
class PublishPendingRun:
    """Publish a paused run and its question as one durable, revision-guarded record."""

    session_id: str
    expected_revision: int | None
    tenant_id: str | None
    user_id: str | None
    agent_id: str
    selected_agent_id: str
    run_id: str
    clarification: ClarificationRequest
    input_message: BaseMessage
    route_reason_code: str
    trace_id: str
    graph_definition_id: str | None
    continuation_contract_version: int


@dataclass(frozen=True)
class ClaimResponse:
    """Claim one submitted answer against the active clarification of a paused run."""

    session_id: str
    clarification_id: str
    submission_id: str
    response: ClarificationResponse
    attempt_trace_id: str


@dataclass(frozen=True)
class ClaimedResponse:
    """The claim a continuation may act on.

    `replayed` marks a submission that was already claimed with identical content, so the
    caller continues the same work rather than treating the retry as a new answer.
    """

    pending_run: PendingRun
    response: ClarificationResponse
    replayed: bool


@dataclass(frozen=True)
class CommitContinuedTurn:
    """Commit the logical turn a paused run produced and consume the run, together."""

    session_id: str
    run_id: str
    clarification_id: str
    submission_id: str
    expected_revision: int | None
    tenant_id: str | None
    user_id: str | None
    agent_id: str
    selected_agent_id: str
    input_message: BaseMessage
    generated_messages: tuple[BaseMessage, ...]
    final_message_id: str
    trace_id: str
    attempt_trace_ids: tuple[str, ...]
    clarification_ids: tuple[str, ...]


@dataclass(frozen=True)
class ClosePendingRun:
    """Close a paused run at the question it last displayed, without resuming its graph."""

    session_id: str
    run_id: str
    clarification_id: str
    reason: Literal["canceled", "expired"]
    submission_id: str | None
    final_message: BaseMessage
    trace_id: str
    attempt_trace_id: str | None


@dataclass(frozen=True)
class ClarificationAuditEntry:
    """One entry in a run's ordered, append-only exchange record."""

    session_id: str
    run_id: str
    clarification_id: str
    sequence: int
    event: ClarificationEvent
    recorded_at: datetime
    selection_mode: str | None = None
    question: str | None = None
    option_ids: tuple[str, ...] = ()
    response: ClarificationResponse | None = None
    response_text: str | None = None
    submission_id: str | None = None
    attempt_trace_id: str | None = None


class SessionStore(Protocol):
    @property
    def storage_name(self) -> str: ...

    @property
    def durable(self) -> bool: ...

    def lock(self, session_id: str) -> AbstractAsyncContextManager[None]: ...

    async def load(self, session_id: str) -> SessionSnapshot | None: ...

    async def commit_turn(self, command: CommitTurn) -> SessionSnapshot: ...

    async def publish_pending_run(self, command: PublishPendingRun) -> PendingRun: ...

    async def active_pending_run(self, session_id: str) -> PendingRun | None: ...

    async def due_pending_runs(self, *, at: datetime, limit: int) -> tuple[PendingRun, ...]: ...

    async def claim_response(self, command: ClaimResponse, *, at: datetime) -> ClaimedResponse: ...

    async def republish_pending_run(
        self,
        session_id: str,
        *,
        run_id: str,
        submission_id: str,
        clarification: ClarificationRequest,
        attempt_trace_id: str,
    ) -> PendingRun: ...

    async def commit_continued_turn(self, command: CommitContinuedTurn) -> SessionSnapshot: ...

    async def close_pending_run(self, command: ClosePendingRun) -> SessionSnapshot: ...

    async def clarification_audit(self, session_id: str) -> tuple[ClarificationAuditEntry, ...]: ...


SessionLockIterator = AsyncIterator[None]
