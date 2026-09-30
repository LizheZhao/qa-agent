"""Strict clarification-pause contracts shared by pause-capable agent graphs.

A pause-capable graph offers the user a bounded, auditable choice and then waits. The
models here are the only shape that crosses that boundary: they carry the producing
agent's question, its offered options in display order, and the freshness token that
proves the answer applies to the snapshot the question was asked from. They deliberately
have no field able to hold a raw provider payload, a credential, or arbitrary graph state.

Free text and cancellation are always available responses. They are invariants of the
contract rather than per-request flags, so a producer cannot offer a question the user is
unable to decline. `selection_mode` constrains only how many of the offered option IDs a
typed selection may name.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Annotated, Final, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator, model_validator

CONTINUATION_CONTRACT_VERSION: Final = 1
"""Stamped on every paused run so a resume can refuse an incompatible contract."""

CLARIFICATION_PAUSE: Final = timedelta(minutes=10)
"""How long a published pause stays answerable. Checked when a response is claimed."""

MAX_CLARIFICATION_OPTIONS: Final = 10
MAX_FREE_TEXT_CHARACTERS: Final = 2000
MAX_OPTION_LABEL_CHARACTERS: Final = 200
MAX_OPTION_DETAIL_CHARACTERS: Final = 500
MAX_QUESTION_CHARACTERS: Final = 1000

QUEUED_CLARIFICATIONS_CHANNEL: Final = "queued_clarifications"
"""State channel holding the clarifications a run still owes, the active one first.

Named here rather than agreed informally, because the invocation boundary reads it to
report what is queued behind the question a user is looking at. The active clarification
stays in the channel while it is being asked, so the producer has one list to maintain
rather than an active slot and a tail that can disagree.
"""

OPAQUE_ID_PATTERN: Final = r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$"
AGENT_ID_PATTERN: Final = r"^[a-z][a-z0-9_]*$"
REASON_CODE_PATTERN: Final = r"^[a-z][a-z0-9_]{0,63}$"

SelectionMode = Literal["single", "multiple"]
ResponseForm = Literal["options", "free_text", "cancel"]


class ClarificationContractError(ValueError):
    """A clarification response does not satisfy the request that offered it.

    `reason_code` is a short bounded category safe to retain in a sanitized rejection
    trace; the message is operator-facing and never echoes the submitted content.
    """

    def __init__(self, reason_code: str, message: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code


class ClarificationOption(BaseModel):
    """One offered choice. Its position in the request is its display order."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    option_id: str = Field(pattern=OPAQUE_ID_PATTERN)
    label: str = Field(min_length=1, max_length=MAX_OPTION_LABEL_CHARACTERS)
    detail: str | None = Field(default=None, min_length=1, max_length=MAX_OPTION_DETAIL_CHARACTERS)

    @field_validator("label", "detail")
    @classmethod
    def text_must_not_be_blank(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("must not be blank")
        return value


class ClarificationRequest(BaseModel):
    """One pause a producer offers, with free text and cancellation always permitted."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    continuation_contract_version: Literal[1] = CONTINUATION_CONTRACT_VERSION
    clarification_id: str = Field(pattern=OPAQUE_ID_PATTERN)
    producer_agent_id: str = Field(pattern=AGENT_ID_PATTERN)
    task_id: str = Field(pattern=OPAQUE_ID_PATTERN)
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARACTERS)
    reason_code: str = Field(pattern=REASON_CODE_PATTERN)
    selection_mode: SelectionMode
    options: tuple[ClarificationOption, ...] = Field(
        min_length=1, max_length=MAX_CLARIFICATION_OPTIONS
    )
    freshness_token: str = Field(pattern=OPAQUE_ID_PATTERN)

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be blank")
        return value

    @model_validator(mode="after")
    def options_must_be_distinct(self) -> Self:
        # Labels as well as IDs: two identically labelled options are the same choice to
        # the person answering, and an audit of offered options and their display order
        # cannot say which one was shown.
        for field, values in (
            ("option_id", [option.option_id for option in self.options]),
            ("label", [option.label for option in self.options]),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"options must have a distinct {field}")
        return self

    @property
    def option_ids(self) -> tuple[str, ...]:
        """Offered option IDs in display order."""

        return tuple(option.option_id for option in self.options)

    def expires_at(self, published_at: datetime) -> datetime:
        """The instant this pause stops being answerable once published."""

        if published_at.tzinfo is None:
            raise ValueError("published_at must be timezone aware")
        return published_at + CLARIFICATION_PAUSE

    def accept(self, response: ClarificationResponse) -> ClarificationResponse:
        """Return the response this request permits, or raise a bounded rejection.

        Cancellation and bounded free text are always permitted. A typed selection must
        name known option IDs, and `single` admits exactly one.
        """

        if not isinstance(response, OptionSelectionResponse):
            return response
        offered = set(self.option_ids)
        unknown = [value for value in response.option_ids if value not in offered]
        if unknown:
            raise ClarificationContractError(
                "unknown_option", f"{len(unknown)} submitted option IDs were not offered"
            )
        if self.selection_mode == "single" and len(response.option_ids) != 1:
            raise ClarificationContractError(
                "selection_count", "single selection admits exactly one option ID"
            )
        return response


class OptionSelectionResponse(BaseModel):
    """A typed selection of one or more offered option IDs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    form: Literal["options"] = "options"
    option_ids: tuple[str, ...] = Field(min_length=1, max_length=MAX_CLARIFICATION_OPTIONS)

    @field_validator("option_ids")
    @classmethod
    def option_ids_must_be_distinct(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("option_ids must be distinct")
        return value


class FreeTextResponse(BaseModel):
    """Bounded free text, always available whatever the selection mode."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    form: Literal["free_text"] = "free_text"
    text: str = Field(min_length=1, max_length=MAX_FREE_TEXT_CHARACTERS)

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("text must not be blank")
        return stripped


class CancelResponse(BaseModel):
    """Decline the question, always available."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    form: Literal["cancel"] = "cancel"


# A discriminated union rather than one model with optional fields: it makes a mixed
# payload -- option IDs together with free text -- unrepresentable instead of something a
# validator has to notice, and `extra="forbid"` on each member rejects the flat mixed form.
ClarificationResponse = Annotated[
    OptionSelectionResponse | FreeTextResponse | CancelResponse,
    Field(discriminator="form"),
]

CLARIFICATION_RESPONSE_ADAPTER: Final = TypeAdapter[ClarificationResponse](ClarificationResponse)


class ContinuationIdentity(BaseModel):
    """Distinct opaque IDs for the session, the child run thread, and the active pause.

    The run ID is one child invocation's LangGraph `thread_id`, never the session ID; the
    clarification ID is the only resume authority.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    session_id: str = Field(pattern=OPAQUE_ID_PATTERN)
    run_id: str = Field(pattern=OPAQUE_ID_PATTERN)
    clarification_id: str = Field(pattern=OPAQUE_ID_PATTERN)

    @model_validator(mode="after")
    def identifiers_must_be_distinct(self) -> Self:
        values = (self.session_id, self.run_id, self.clarification_id)
        if len(set(values)) != len(values):
            raise ValueError("session, run, and clarification IDs must be distinct")
        return self


def order_clarifications(
    requests: Iterable[ClarificationRequest],
    *,
    plan_task_ids: Sequence[str] = (),
) -> tuple[ClarificationRequest, ...]:
    """Order clarifications deterministically, never by the order they arrived.

    Parallel producers finish in whatever order their work happens to take, so ordering by
    completion would show the same user a different question on a retry of the same run.
    The order here comes from the plan: a request for a task the plan lists earlier is
    asked first, tasks outside the plan follow, and task ID then clarification ID settle
    the remainder. The result is stable across replicas and across reruns.
    """

    position = {task_id: index for index, task_id in enumerate(plan_task_ids)}
    unplanned = len(position)
    return tuple(
        sorted(
            requests,
            key=lambda request: (
                position.get(request.task_id, unplanned),
                request.task_id,
                request.clarification_id,
            ),
        )
    )


@dataclass(frozen=True, slots=True)
class ClarificationQueue:
    """The clarifications a paused run still owes, with exactly one of them active.

    Holding the queue rather than one request is what lets a producer report several
    ambiguities at once without the boundary having to show them all: the head is asked,
    the tail stays in checkpointed state, and answering the head promotes the next one
    before any newly dependent work is dispatched.
    """

    pending: tuple[ClarificationRequest, ...] = ()

    def __post_init__(self) -> None:
        identifiers = [request.clarification_id for request in self.pending]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("a queue must not hold the same clarification twice")

    @property
    def active(self) -> ClarificationRequest | None:
        """The one clarification a user is being asked right now."""

        return self.pending[0] if self.pending else None

    @property
    def queued_ids(self) -> tuple[str, ...]:
        """The clarifications waiting behind the active one, in the order they will be asked."""

        return tuple(request.clarification_id for request in self.pending[1:])

    def extend(
        self,
        requests: Iterable[ClarificationRequest],
        *,
        plan_task_ids: Sequence[str] = (),
    ) -> ClarificationQueue:
        """Add newly reported clarifications and reorder the ones not yet asked.

        An already-active clarification keeps its place: it has been shown to the user, so
        a later arrival cannot displace the question they are looking at.
        """

        known = {request.clarification_id for request in self.pending}
        arriving = tuple(request for request in requests if request.clarification_id not in known)
        head = self.pending[:1]
        tail = order_clarifications(self.pending[1:] + arriving, plan_task_ids=plan_task_ids)
        return ClarificationQueue(pending=head + tail)

    def answered(self, clarification_id: str) -> ClarificationQueue:
        """Drop the answered clarification, promoting the next one to active."""

        if self.active is None or self.active.clarification_id != clarification_id:
            raise ClarificationContractError(
                "unknown_clarification", "only the active clarification can be answered"
            )
        return ClarificationQueue(pending=self.pending[1:])


def is_clarification_expired(expires_at: datetime, *, at: datetime) -> bool:
    """Whether a pause published with this expiry is unanswerable at `at`.

    Expiry is inclusive: an answer arriving exactly at the boundary is refused, so the
    same instant can never be both answerable and expired.
    """

    if expires_at.tzinfo is None or at.tzinfo is None:
        raise ValueError("expiry comparison requires timezone aware instants")
    return at >= expires_at
