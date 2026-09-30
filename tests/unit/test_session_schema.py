from datetime import UTC, datetime, timedelta

import pytest
from orchestration_core import (
    CLARIFICATION_PAUSE,
    ClarificationOption,
    ClarificationRequest,
)
from pydantic import ValidationError

from agentic_orchestration.sessions.schema import (
    ACTIVE_PENDING_RUN_STATUSES,
    TURN_SCHEMA_VERSION,
    ClarificationAuditDocument,
    PendingRunDocument,
    SessionDocument,
    StoredContinuation,
    StoredMessage,
    TurnDocument,
    validate_session_document,
    validate_turn_document,
)

CREATED_AT = datetime(2026, 9, 15, 12, 0, tzinfo=UTC)


def assistant_message() -> StoredMessage:
    return StoredMessage(id="assistant", type="ai", content="answer")


def human_message() -> StoredMessage:
    return StoredMessage(id="human", type="human", content="question")


def clarification_request() -> ClarificationRequest:
    return ClarificationRequest(
        clarification_id="clarification-1",
        producer_agent_id="scripted_producer",
        task_id="task-1",
        question="Which market should the summary cover?",
        reason_code="ambiguous_market",
        selection_mode="single",
        options=(
            ClarificationOption(option_id="us", label="United States"),
            ClarificationOption(option_id="eu", label="Europe"),
        ),
        freshness_token="snapshot-1",
    )


CLAIMED_STATUSES = {"claimed", "consumed", "canceled"}


def submission_fields(status: str) -> dict[str, object]:
    """A submitted status records both the submission and the response it claimed."""

    if status not in CLAIMED_STATUSES:
        return {}
    return {
        "submission_id": "submission-1",
        "claimed_response": {"form": "options", "option_ids": ["us"]},
    }


def pending_run(**overrides: object) -> PendingRunDocument:
    values: dict[str, object] = {
        "_id": "run-1",
        "session_id": "session-1",
        "run_id": "run-1",
        "clarification_id": "clarification-1",
        "active_session_key": "session-1",
        "agent_id": "router",
        "selected_agent_id": "scripted_producer",
        "expected_session_revision": None,
        "original_input_message": human_message().model_dump(mode="python"),
        "route_reason_code": "scripted_producer",
        "clarification": clarification_request().model_dump(mode="python"),
        "trace_id": "trace-1",
        "created_at": CREATED_AT,
        "expires_at": CREATED_AT + CLARIFICATION_PAUSE,
    }
    values.update(overrides)
    return PendingRunDocument.model_validate(values)


def audit_entry(**overrides: object) -> ClarificationAuditDocument:
    values: dict[str, object] = {
        "_id": "run-1:1",
        "session_id": "session-1",
        "run_id": "run-1",
        "clarification_id": "clarification-1",
        "sequence": 1,
        "event": "published",
        "selection_mode": "single",
        "question": clarification_request().question,
        "options": [option.model_dump(mode="python") for option in clarification_request().options],
        "attempt_trace_id": "trace-1",
        "recorded_at": CREATED_AT,
    }
    values.update(overrides)
    return ClarificationAuditDocument.model_validate(values)


@pytest.mark.unit
def test_turn_document_retains_delegated_route_affinity() -> None:
    turn = TurnDocument(
        _id="session:1",
        session_id="session",
        turn_number=1,
        input_message=human_message(),
        generated_messages=[assistant_message()],
        final_message_id="assistant",
        routing_outcome="delegate",
        selected_agent_id="marketing_science",
        trace_id="trace-1",
        committed_at=datetime.now(UTC),
    )

    assert turn.routing_outcome == "delegate"
    assert turn.selected_agent_id == "marketing_science"
    assert turn.trace_id == "trace-1"
    assert turn.schema_version == 3


@pytest.mark.unit
def test_turn_document_rejects_inconsistent_route_affinity() -> None:
    with pytest.raises(ValidationError, match="selected_agent_id"):
        TurnDocument(
            _id="session:1",
            session_id="session",
            turn_number=1,
            input_message=human_message(),
            generated_messages=[assistant_message()],
            final_message_id="assistant",
            routing_outcome="answer",
            selected_agent_id="marketing_science",
            committed_at=datetime.now(UTC),
        )


@pytest.mark.unit
def test_turn_document_hydrates_non_delegated_turn_without_optional_route_field() -> None:
    turn = TurnDocument.model_validate(
        {
            "_id": "session:1",
            "session_id": "session",
            "turn_number": 1,
            "input_message": human_message().model_dump(mode="python"),
            "generated_messages": [assistant_message().model_dump(mode="python")],
            "final_message_id": "assistant",
            "routing_outcome": "answer",
            "committed_at": datetime.now(UTC),
        }
    )

    assert turn.selected_agent_id is None


@pytest.mark.unit
def test_pre_router_v1_turn_is_explicitly_incompatible() -> None:
    with pytest.raises(ValidationError, match="schema_version"):
        TurnDocument.model_validate(
            {
                "_id": "legacy:1",
                "schema_version": 1,
                "session_id": "legacy",
                "turn_number": 1,
                "input_message": human_message().model_dump(mode="python"),
                "generated_messages": [assistant_message().model_dump(mode="python")],
                "final_message_id": "assistant",
                "committed_at": datetime.now(UTC),
            }
        )


@pytest.mark.unit
def test_router_era_v1_turn_is_promoted_for_read_compatibility() -> None:
    turn = validate_turn_document(
        {
            "_id": "router-v1:1",
            "schema_version": 1,
            "session_id": "router-v1",
            "turn_number": 1,
            "input_message": human_message().model_dump(mode="python"),
            "generated_messages": [assistant_message().model_dump(mode="python")],
            "final_message_id": "assistant",
            "routing_outcome": "answer",
            "committed_at": datetime.now(UTC),
        }
    )

    assert turn.schema_version == 3
    assert turn.routing_outcome == "answer"


@pytest.mark.unit
def test_router_era_v2_turn_is_promoted_for_read_compatibility() -> None:
    """Turns stored before continuation lineage existed still hydrate unchanged."""

    turn = validate_turn_document(
        {
            "_id": "router-v2:1",
            "schema_version": 2,
            "session_id": "router-v2",
            "turn_number": 1,
            "input_message": human_message().model_dump(mode="python"),
            "generated_messages": [assistant_message().model_dump(mode="python")],
            "final_message_id": "assistant",
            "routing_outcome": "delegate",
            "selected_agent_id": "marketing_science",
            "trace_id": "trace-1",
            "committed_at": CREATED_AT,
        }
    )

    assert turn.schema_version == TURN_SCHEMA_VERSION
    assert turn.continuation is None
    assert turn.selected_agent_id == "marketing_science"


@pytest.mark.unit
def test_stored_session_document_written_before_pending_runs_still_validates() -> None:
    """Publishing a pause adds a collection rather than changing the session shape."""

    session = validate_session_document(
        {
            "_id": "session-1",
            "schema_version": 1,
            "agent_id": "router",
            "revision": 2,
            "status": "active",
            "created_at": CREATED_AT,
            "updated_at": CREATED_AT,
        }
    )

    assert session.revision == 2
    assert session.tenant_id is None


@pytest.mark.unit
def test_session_document_still_refuses_a_revision_below_one() -> None:
    with pytest.raises(ValidationError, match="revision"):
        SessionDocument(
            _id="session-1",
            agent_id="router",
            revision=0,
            created_at=CREATED_AT,
            updated_at=CREATED_AT,
        )


@pytest.mark.unit
def test_first_turn_pause_needs_no_committed_session() -> None:
    """A pause on a session's first request publishes without a session document."""

    record = pending_run()

    assert record.expected_session_revision is None
    assert record.status == "awaiting_response"
    assert record.active_session_key == "session-1"
    assert record.expires_at - record.created_at == CLARIFICATION_PAUSE


@pytest.mark.unit
def test_pause_on_a_later_turn_records_the_revision_it_expects() -> None:
    record = pending_run(expected_session_revision=2)

    assert record.expected_session_revision == 2

    with pytest.raises(ValidationError, match="expected_session_revision"):
        pending_run(expected_session_revision=0)


@pytest.mark.unit
def test_pending_run_embeds_the_offered_options_and_its_contract_version() -> None:
    record = pending_run()

    assert record.continuation_contract_version == 1
    assert record.clarification.option_ids == ("us", "eu")
    assert record.clarification.selection_mode == "single"


@pytest.mark.unit
@pytest.mark.parametrize("status", sorted(ACTIVE_PENDING_RUN_STATUSES))
def test_an_active_pending_run_keys_its_session_for_the_unique_index(status: str) -> None:
    record = pending_run(status=status, **submission_fields(status))

    assert record.active_session_key == record.session_id

    with pytest.raises(ValidationError, match="must key its session"):
        pending_run(status=status, active_session_key=None, **submission_fields(status))


@pytest.mark.unit
@pytest.mark.parametrize("status", ["canceled", "expired"])
def test_a_closed_pending_run_releases_the_session_run_slot(status: str) -> None:
    record = pending_run(status=status, active_session_key=None, **submission_fields(status))

    assert record.active_session_key is None

    with pytest.raises(ValidationError, match="must release its session key"):
        pending_run(status=status, **submission_fields(status))


@pytest.mark.unit
def test_a_cancellation_records_the_submission_that_closed_the_run() -> None:
    """Cancelling is itself a submitted response, so the run keeps what closed it."""

    record = pending_run(
        status="canceled", active_session_key=None, **submission_fields("canceled")
    )

    assert record.submission_id == "submission-1"
    assert record.claimed_response is not None
    assert record.claimed_response.form == "options"


@pytest.mark.unit
def test_an_expired_run_records_no_submission() -> None:
    record = pending_run(status="expired", active_session_key=None)

    assert record.submission_id is None
    assert record.claimed_response is None

    with pytest.raises(ValidationError, match="submission_id"):
        pending_run(status="expired", active_session_key=None, **submission_fields("canceled"))


@pytest.mark.unit
def test_a_submission_and_its_claimed_response_are_recorded_together() -> None:
    with pytest.raises(ValidationError, match="submission_id"):
        pending_run(status="awaiting_response", submission_id="submission-1")

    with pytest.raises(ValidationError, match="claimed_response"):
        pending_run(status="claimed", submission_id="submission-1")

    with pytest.raises(ValidationError, match="claimed_response"):
        pending_run(
            status="awaiting_response",
            claimed_response={"form": "options", "option_ids": ["us"]},
        )


@pytest.mark.unit
def test_pending_run_identity_stays_distinct_and_self_consistent() -> None:
    with pytest.raises(ValidationError, match="must be distinct"):
        pending_run(run_id="session-1", _id="session-1", active_session_key="session-1")

    with pytest.raises(ValidationError, match="_id must be the run ID"):
        pending_run(run_id="run-2")

    with pytest.raises(ValidationError, match="must match the embedded request"):
        pending_run(clarification_id="clarification-2")

    with pytest.raises(ValidationError, match="must be the producing agent"):
        pending_run(selected_agent_id="marketing_science")


@pytest.mark.unit
def test_pending_run_rejects_unknown_fields_and_a_non_user_original_request() -> None:
    with pytest.raises(ValidationError) as raised:
        pending_run(raw_provider_response={"choices": []})

    assert raised.value.errors()[0]["type"] == "extra_forbidden"

    with pytest.raises(ValidationError, match="must be the user request"):
        pending_run(original_input_message=assistant_message().model_dump(mode="python"))


@pytest.mark.unit
def test_pending_run_expiry_must_follow_its_creation() -> None:
    with pytest.raises(ValidationError, match="expires_at must follow"):
        pending_run(expires_at=CREATED_AT - timedelta(seconds=1))


@pytest.mark.unit
def test_published_audit_entry_retains_the_offered_options_in_display_order() -> None:
    entry = audit_entry()

    assert entry.event == "published"
    assert [option.option_id for option in entry.options or []] == ["us", "eu"]
    assert entry.response is None


@pytest.mark.unit
def test_answered_audit_entry_retains_the_response_and_its_submission() -> None:
    entry = audit_entry(
        _id="run-1:2",
        sequence=2,
        event="answered",
        selection_mode=None,
        question=None,
        options=None,
        response={"form": "options", "option_ids": ["us"]},
        response_text="United States",
        submission_id="submission-1",
        attempt_trace_id="trace-2",
    )

    assert entry.response is not None
    assert entry.response.form == "options"
    assert entry.response_text == "United States"
    assert entry.submission_id == "submission-1"


@pytest.mark.unit
@pytest.mark.parametrize("event", ["canceled", "expired"])
def test_closing_audit_entries_carry_no_response(event: str) -> None:
    submission = "submission-1" if event == "canceled" else None
    entry = audit_entry(
        _id="run-1:2",
        sequence=2,
        event=event,
        selection_mode=None,
        question=None,
        options=None,
        submission_id=submission,
    )

    assert entry.response is None

    with pytest.raises(ValidationError, match="response is recorded exactly"):
        audit_entry(
            _id="run-1:2",
            sequence=2,
            event=event,
            selection_mode=None,
            question=None,
            options=None,
            submission_id=submission,
            response={"form": "cancel"},
            response_text="Cancelled this request.",
        )


@pytest.mark.unit
def test_a_cancellation_audit_entry_names_the_submission_that_closed_the_run() -> None:
    entry = audit_entry(
        _id="run-1:2",
        sequence=2,
        event="canceled",
        selection_mode=None,
        question=None,
        options=None,
        submission_id="submission-1",
    )

    assert entry.submission_id == "submission-1"

    with pytest.raises(ValidationError, match="submission ID is recorded exactly"):
        audit_entry(
            _id="run-1:2",
            sequence=2,
            event="canceled",
            selection_mode=None,
            question=None,
            options=None,
        )


@pytest.mark.unit
def test_an_expiry_audit_entry_names_no_submission() -> None:
    with pytest.raises(ValidationError, match="submission ID is recorded exactly"):
        audit_entry(
            _id="run-1:2",
            sequence=2,
            event="expired",
            selection_mode=None,
            question=None,
            options=None,
            submission_id="submission-1",
        )


@pytest.mark.unit
def test_audit_entry_must_match_its_event_shape() -> None:
    with pytest.raises(ValidationError, match="records exactly the offered options"):
        audit_entry(
            event="answered",
            response={"form": "cancel"},
            response_text="Cancelled this request.",
            submission_id="submission-1",
        )

    with pytest.raises(ValidationError, match="_id must be the run ID and sequence"):
        audit_entry(_id="run-1:7")

    with pytest.raises(ValidationError, match="submission ID is recorded exactly"):
        audit_entry(
            _id="run-1:2",
            sequence=2,
            event="answered",
            selection_mode=None,
            question=None,
            options=None,
            response={"form": "cancel"},
            response_text="Cancelled this request.",
            submission_id=None,
        )


@pytest.mark.unit
def test_audit_entry_rejects_a_mixed_response_form() -> None:
    with pytest.raises(ValidationError) as raised:
        audit_entry(
            _id="run-1:2",
            sequence=2,
            event="answered",
            selection_mode=None,
            question=None,
            options=None,
            response={"form": "options", "option_ids": ["us"], "text": "and Canada"},
            response_text="United States",
            submission_id="submission-1",
        )

    assert raised.value.errors()[0]["type"] == "extra_forbidden"


@pytest.mark.unit
def test_continued_turn_retains_its_pause_lineage() -> None:
    turn = TurnDocument(
        _id="session-1:1",
        session_id="session-1",
        turn_number=1,
        input_message=human_message(),
        generated_messages=[assistant_message()],
        final_message_id="assistant",
        routing_outcome="delegate",
        selected_agent_id="scripted_producer",
        trace_id="trace-1",
        continuation=StoredContinuation(
            run_id="run-1",
            clarification_ids=["clarification-1"],
            continuation_contract_version=1,
            attempt_trace_ids=["trace-1", "trace-2"],
            closed_by="answered",
        ),
        committed_at=CREATED_AT,
    )

    assert turn.continuation is not None
    assert turn.continuation.closed_by == "answered"
    assert turn.trace_id == "trace-1"
    assert turn.continuation.attempt_trace_ids == ["trace-1", "trace-2"]


@pytest.mark.unit
def test_pause_lineage_belongs_only_to_a_delegated_turn() -> None:
    with pytest.raises(ValidationError, match="continuation lineage belongs to delegated turns"):
        TurnDocument(
            _id="session-1:1",
            session_id="session-1",
            turn_number=1,
            input_message=human_message(),
            generated_messages=[assistant_message()],
            final_message_id="assistant",
            routing_outcome="clarify",
            continuation=StoredContinuation(
                run_id="run-1",
                clarification_ids=["clarification-1"],
                continuation_contract_version=1,
                closed_by="canceled",
            ),
            committed_at=CREATED_AT,
        )


@pytest.mark.unit
def test_pause_lineage_must_be_distinct() -> None:
    with pytest.raises(ValidationError, match="clarification_ids must be distinct"):
        StoredContinuation(
            run_id="run-1",
            clarification_ids=["clarification-1", "clarification-1"],
            continuation_contract_version=1,
            closed_by="answered",
        )

    with pytest.raises(ValidationError, match="attempt_trace_ids must be distinct"):
        StoredContinuation(
            run_id="run-1",
            clarification_ids=["clarification-1"],
            continuation_contract_version=1,
            attempt_trace_ids=["trace-1", "trace-1"],
            closed_by="answered",
        )
