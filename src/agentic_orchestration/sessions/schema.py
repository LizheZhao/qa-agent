"""Controlled MongoDB document shapes and collection definitions."""

from datetime import datetime
from typing import Any, Final, Literal, Self

from orchestration_core import (
    CONTINUATION_CONTRACT_VERSION,
    ClarificationOption,
    ClarificationRequest,
    ClarificationResponse,
    ContinuationIdentity,
    SelectionMode,
)
from pydantic import BaseModel, ConfigDict, Field, model_validator

MESSAGE_SCHEMA_VERSION: Final = 1
SESSION_SCHEMA_VERSION: Final = 1
TURN_SCHEMA_VERSION: Final = 3
PENDING_RUN_SCHEMA_VERSION: Final = 1
CLARIFICATION_AUDIT_SCHEMA_VERSION: Final = 1
SESSIONS_COLLECTION = "sessions"
SESSION_TURNS_COLLECTION = "session_turns"
PENDING_RUNS_COLLECTION = "pending_runs"
CLARIFICATION_AUDIT_COLLECTION = "clarification_audit"
SESSION_TURN_INDEX = "session_turn_unique"
PENDING_RUN_ACTIVE_INDEX = "pending_run_active_session_unique"
PENDING_RUN_DUE_INDEX = "pending_run_due"
CLARIFICATION_AUDIT_INDEX = "clarification_audit_sequence_unique"

PendingRunStatus = Literal["awaiting_response", "claimed", "consumed", "canceled", "expired"]
ClarificationEvent = Literal["published", "answered", "canceled", "expired"]

ACTIVE_PENDING_RUN_STATUSES: Final[frozenset[str]] = frozenset({"awaiting_response", "claimed"})
"""Statuses that hold the session's single in-flight run slot."""


class StoredMessage(BaseModel):
    """Application-owned, BSON-safe representation of a LangChain message."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = MESSAGE_SCHEMA_VERSION
    id: str
    type: Literal["system", "human", "ai", "tool"]
    content: str | list[str | dict[str, Any]]
    name: str | None = None
    example: bool | None = None
    additional_kwargs: dict[str, Any] = Field(default_factory=dict)
    response_metadata: dict[str, Any] = Field(default_factory=dict)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    invalid_tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    usage_metadata: dict[str, Any] | None = None
    tool_call_id: str | None = None
    artifact: Any | None = None
    tool_status: str | None = None


class SessionDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    id: str = Field(alias="_id")
    schema_version: Literal[1] = SESSION_SCHEMA_VERSION
    tenant_id: str | None = None
    user_id: str | None = None
    agent_id: str
    revision: int = Field(ge=1)
    status: Literal["active", "archived"] = "active"
    created_at: datetime
    updated_at: datetime
    expires_at: datetime | None = None


class StoredContinuation(BaseModel):
    """Lineage of the paused run a continued turn closed.

    The turn keeps one canonical `trace_id`; `attempt_trace_ids` retains the separate
    traces each continuation attempt created so the audit does not lose them.
    """

    model_config = ConfigDict(extra="forbid")

    run_id: str
    clarification_ids: list[str] = Field(min_length=1)
    continuation_contract_version: int = Field(ge=1)
    attempt_trace_ids: list[str] = Field(default_factory=list)
    closed_by: Literal["answered", "canceled", "expired"]

    @model_validator(mode="after")
    def lineage_must_be_distinct(self) -> Self:
        for field, values in (
            ("clarification_ids", self.clarification_ids),
            ("attempt_trace_ids", self.attempt_trace_ids),
        ):
            if len(set(values)) != len(values):
                raise ValueError(f"{field} must be distinct")
        return self


class TurnDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    id: str = Field(alias="_id")
    schema_version: Literal[3] = TURN_SCHEMA_VERSION
    session_id: str
    turn_number: int = Field(ge=1)
    input_message: StoredMessage
    generated_messages: list[StoredMessage]
    final_message_id: str
    routing_outcome: Literal["answer", "delegate", "clarify", "reject"]
    selected_agent_id: str | None = None
    trace_id: str | None = None
    continuation: StoredContinuation | None = None
    committed_at: datetime

    @model_validator(mode="after")
    def final_message_must_identify_generated_assistant(self) -> Self:
        final = next(
            (message for message in self.generated_messages if message.id == self.final_message_id),
            None,
        )
        if final is None or final.type != "ai" or final.tool_calls:
            raise ValueError("final_message_id must identify a final generated AI message")
        if (self.routing_outcome == "delegate") != (self.selected_agent_id is not None):
            raise ValueError("selected_agent_id must be set exactly for delegated turns")
        # Only pause-capable child graphs are checkpointed in this contract, so a turn
        # closed from a paused run is always a delegated one. Router-only text
        # clarification keeps its existing turn-ending behaviour and carries no lineage.
        if self.continuation is not None and self.routing_outcome != "delegate":
            raise ValueError("continuation lineage belongs to delegated turns")
        return self


def validate_turn_document(raw_turn: dict[str, Any]) -> TurnDocument:
    """Promote stored turns written before continuation lineage existed.

    Router-era v1 and v2 turns are read-compatible: both already carry
    `routing_outcome`, and `continuation` is absent for every turn committed without a
    paused run. Pre-router v1 documents have no routing outcome and stay incompatible.
    """

    stored_version = raw_turn.get("schema_version")
    if stored_version in {1, 2} and "routing_outcome" in raw_turn:
        raw_turn = {**raw_turn, "schema_version": TURN_SCHEMA_VERSION}
    return TurnDocument.model_validate(raw_turn)


def validate_session_document(raw_session: dict[str, Any]) -> SessionDocument:
    """Read a stored session document.

    Publishing a pause deliberately leaves this shape alone: the pending run lives in its
    own collection, so a first-turn pause creates no session document and `revision >= 1`
    keeps meaning "at least one committed turn".
    """

    return SessionDocument.model_validate(raw_session)


class PendingRunDocument(BaseModel):
    """The single in-flight paused run a session may hold.

    Published in its own collection so a pause on a session's first request needs no
    session document, which keeps `revision >= 1` and "no empty durable session" intact.
    `active_session_key` carries the session ID only while the run still holds the
    session's run slot, so one unique sparse index on it enforces at most one active
    pending run per session without a partial-filter expression.

    The embedded `clarification` is the producer contract verbatim rather than a second
    copy of its rules; `continuation_contract_version` is what a resume checks before
    trusting it.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    id: str = Field(alias="_id")
    schema_version: Literal[1] = PENDING_RUN_SCHEMA_VERSION
    continuation_contract_version: Literal[1] = CONTINUATION_CONTRACT_VERSION
    session_id: str
    run_id: str
    clarification_id: str
    active_session_key: str | None = None
    tenant_id: str | None = None
    user_id: str | None = None
    agent_id: str
    selected_agent_id: str
    expected_session_revision: int | None = Field(default=None, ge=1)
    graph_definition_id: str | None = None
    original_input_message: StoredMessage
    routing_outcome: Literal["delegate"] = "delegate"
    route_reason_code: str
    clarification: ClarificationRequest
    status: PendingRunStatus = "awaiting_response"
    submission_id: str | None = None
    claimed_response: ClarificationResponse | None = None
    trace_id: str
    attempt_trace_ids: list[str] = Field(default_factory=list)
    created_at: datetime
    expires_at: datetime

    @model_validator(mode="after")
    def record_must_be_self_consistent(self) -> Self:
        if self.run_id != self.id:
            raise ValueError("_id must be the run ID")
        ContinuationIdentity(
            session_id=self.session_id,
            run_id=self.run_id,
            clarification_id=self.clarification_id,
        )
        if self.clarification.clarification_id != self.clarification_id:
            raise ValueError("clarification_id must match the embedded request")
        if self.selected_agent_id != self.clarification.producer_agent_id:
            raise ValueError("selected_agent_id must be the producing agent")
        if self.original_input_message.type != "human":
            raise ValueError("original_input_message must be the user request")
        if self.expires_at <= self.created_at:
            raise ValueError("expires_at must follow created_at")
        active = self.status in ACTIVE_PENDING_RUN_STATUSES
        if active and self.active_session_key != self.session_id:
            raise ValueError("an active pending run must key its session")
        if not active and self.active_session_key is not None:
            raise ValueError("a closed pending run must release its session key")
        # A cancellation is itself a submitted response, so a cancelled run records the
        # submission that closed it; only expiry closes a run with nothing submitted.
        # Retaining the claimed response is what lets a replayed submission be told apart
        # from the same identifier reused for a different choice.
        claimed = self.status in {"claimed", "consumed", "canceled"}
        if (self.submission_id is not None) != claimed:
            raise ValueError("submission_id must be set exactly for a submitted response")
        if (self.claimed_response is not None) != claimed:
            raise ValueError("claimed_response must be retained exactly for a submitted response")
        if len(set(self.attempt_trace_ids)) != len(self.attempt_trace_ids):
            raise ValueError("attempt_trace_ids must be distinct")
        return self


class ClarificationAuditDocument(BaseModel):
    """One append-only entry in a paused run's ordered exchange record.

    Only outcomes that preserved a user choice are recorded: a published question, an
    accepted response, a cancellation, an expiry. Rejected and duplicate attempts keep
    only sanitized bounded diagnostic traces, so they are deliberately not events here.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    id: str = Field(alias="_id")
    schema_version: Literal[1] = CLARIFICATION_AUDIT_SCHEMA_VERSION
    session_id: str
    run_id: str
    clarification_id: str
    sequence: int = Field(ge=1)
    event: ClarificationEvent
    selection_mode: SelectionMode | None = None
    question: str | None = None
    options: list[ClarificationOption] | None = None
    response: ClarificationResponse | None = None
    response_text: str | None = None
    submission_id: str | None = None
    attempt_trace_id: str | None = None
    recorded_at: datetime

    @model_validator(mode="after")
    def entry_must_match_its_event(self) -> Self:
        if self.id != f"{self.run_id}:{self.sequence}":
            raise ValueError("_id must be the run ID and sequence")
        published = self.event == "published"
        if published != (self.options is not None and self.selection_mode is not None):
            raise ValueError("a published event records exactly the offered options")
        # The question and the rendered choice are kept so the committed turn's ordered
        # exchange can be rebuilt from the audit alone, without reopening a consumed run.
        if published != (self.question is not None):
            raise ValueError("a published event records exactly its question")
        if (self.event == "answered") != (self.response_text is not None):
            raise ValueError("a rendered response is recorded exactly for an answered event")
        if self.options is not None:
            option_ids = [option.option_id for option in self.options]
            if not option_ids:
                raise ValueError("a published event must offer at least one option")
            if len(set(option_ids)) != len(option_ids):
                raise ValueError("offered options must be distinct")
        answered = self.event == "answered"
        if answered != (self.response is not None):
            raise ValueError("a response is recorded exactly for an answered event")
        # A cancellation is audited as the closure rather than as a response, but it still
        # names the submission that caused it, so a replayed cancel can be recognised.
        # Expiry is the one closure nobody submitted.
        submitted = self.event in {"answered", "canceled"}
        if submitted != (self.submission_id is not None):
            raise ValueError("a submission ID is recorded exactly for a submitted event")
        return self
