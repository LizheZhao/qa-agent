"""Strict validation of the clarification pause contract."""

from datetime import UTC, datetime, timedelta

import pytest
from orchestration_core import (
    CLARIFICATION_PAUSE,
    CLARIFICATION_RESPONSE_ADAPTER,
    CONTINUATION_CONTRACT_VERSION,
    MAX_CLARIFICATION_OPTIONS,
    MAX_FREE_TEXT_CHARACTERS,
    ClarificationContractError,
    ClarificationOption,
    ClarificationRequest,
    ContinuationIdentity,
    is_clarification_expired,
)
from pydantic import ValidationError

PUBLISHED_AT = datetime(2026, 9, 15, 12, 0, tzinfo=UTC)


def request(
    selection_mode: str = "single", options: tuple[ClarificationOption, ...] | None = None
) -> ClarificationRequest:
    return ClarificationRequest(
        clarification_id="clarification-1",
        producer_agent_id="scripted_producer",
        task_id="task-1",
        question="Which market should the summary cover?",
        reason_code="ambiguous_market",
        selection_mode=selection_mode,  # type: ignore[arg-type]
        options=(
            (
                ClarificationOption(option_id="us", label="United States"),
                ClarificationOption(option_id="eu", label="Europe"),
            )
            if options is None
            else options
        ),
        freshness_token="snapshot-1",
    )


@pytest.mark.unit
def test_request_retains_offered_options_in_display_order() -> None:
    offered = request()

    assert offered.option_ids == ("us", "eu")
    assert offered.continuation_contract_version == CONTINUATION_CONTRACT_VERSION


@pytest.mark.unit
def test_single_selection_admits_exactly_one_offered_option() -> None:
    accepted = request().accept(
        CLARIFICATION_RESPONSE_ADAPTER.validate_python({"form": "options", "option_ids": ["us"]})
    )

    assert accepted.form == "options"
    assert accepted.option_ids == ("us",)


@pytest.mark.unit
def test_single_selection_rejects_more_than_one_option() -> None:
    with pytest.raises(ClarificationContractError) as raised:
        request().accept(
            CLARIFICATION_RESPONSE_ADAPTER.validate_python(
                {"form": "options", "option_ids": ["us", "eu"]}
            )
        )

    assert raised.value.reason_code == "selection_count"


@pytest.mark.unit
def test_multiple_selection_admits_several_offered_options() -> None:
    accepted = request("multiple").accept(
        CLARIFICATION_RESPONSE_ADAPTER.validate_python(
            {"form": "options", "option_ids": ["eu", "us"]}
        )
    )

    assert accepted.form == "options"
    assert accepted.option_ids == ("eu", "us")


@pytest.mark.unit
@pytest.mark.parametrize("selection_mode", ["single", "multiple"])
def test_unknown_option_ids_are_rejected_in_every_selection_mode(selection_mode: str) -> None:
    with pytest.raises(ClarificationContractError) as raised:
        request(selection_mode).accept(
            CLARIFICATION_RESPONSE_ADAPTER.validate_python(
                {"form": "options", "option_ids": ["mx"]}
            )
        )

    assert raised.value.reason_code == "unknown_option"
    # The rejection reason is a bounded category and never echoes the submitted value.
    assert "mx" not in str(raised.value)


@pytest.mark.unit
@pytest.mark.parametrize("selection_mode", ["single", "multiple"])
def test_free_text_and_cancel_are_always_available(selection_mode: str) -> None:
    offered = request(selection_mode)

    free_text = offered.accept(
        CLARIFICATION_RESPONSE_ADAPTER.validate_python(
            {"form": "free_text", "text": "  Canada, please  "}
        )
    )
    cancel = offered.accept(CLARIFICATION_RESPONSE_ADAPTER.validate_python({"form": "cancel"}))

    assert free_text.form == "free_text"
    assert free_text.text == "Canada, please"
    assert cancel.form == "cancel"


@pytest.mark.unit
def test_repeated_option_ids_in_one_selection_are_rejected() -> None:
    with pytest.raises(ValidationError, match="option_ids must be distinct"):
        CLARIFICATION_RESPONSE_ADAPTER.validate_python(
            {"form": "options", "option_ids": ["us", "us"]}
        )


@pytest.mark.unit
def test_mixed_response_forms_are_unrepresentable() -> None:
    with pytest.raises(ValidationError) as raised:
        CLARIFICATION_RESPONSE_ADAPTER.validate_python(
            {"form": "options", "option_ids": ["us"], "text": "and also Canada"}
        )

    assert raised.value.errors()[0]["type"] == "extra_forbidden"


@pytest.mark.unit
def test_unknown_response_form_is_rejected() -> None:
    with pytest.raises(ValidationError) as raised:
        CLARIFICATION_RESPONSE_ADAPTER.validate_python({"form": "approve"})

    assert raised.value.errors()[0]["type"] == "union_tag_invalid"


@pytest.mark.unit
@pytest.mark.parametrize("text", ["", "   ", "x" * (MAX_FREE_TEXT_CHARACTERS + 1)])
def test_free_text_outside_its_bounds_is_rejected(text: str) -> None:
    with pytest.raises(ValidationError):
        CLARIFICATION_RESPONSE_ADAPTER.validate_python({"form": "free_text", "text": text})


@pytest.mark.unit
def test_duplicate_offered_options_are_rejected() -> None:
    with pytest.raises(ValidationError, match="distinct option_id"):
        request(
            options=(
                ClarificationOption(option_id="us", label="United States"),
                ClarificationOption(option_id="us", label="US"),
            )
        )

    with pytest.raises(ValidationError, match="distinct label"):
        request(
            options=(
                ClarificationOption(option_id="us", label="United States"),
                ClarificationOption(option_id="usa", label="United States"),
            )
        )


@pytest.mark.unit
def test_request_requires_at_least_one_bounded_option_set() -> None:
    with pytest.raises(ValidationError):
        request(options=())

    with pytest.raises(ValidationError):
        request(
            options=tuple(
                ClarificationOption(option_id=f"option-{index}", label=f"Option {index}")
                for index in range(MAX_CLARIFICATION_OPTIONS + 1)
            )
        )


@pytest.mark.unit
def test_request_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError) as raised:
        ClarificationRequest.model_validate(
            {
                **request().model_dump(mode="json"),
                "raw_model_response": {"choices": []},
            }
        )

    assert raised.value.errors()[0]["type"] == "extra_forbidden"


@pytest.mark.unit
def test_pause_expires_ten_minutes_after_publication() -> None:
    expires_at = request().expires_at(PUBLISHED_AT)

    assert timedelta(minutes=10) == CLARIFICATION_PAUSE
    assert expires_at == PUBLISHED_AT + timedelta(minutes=10)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("offset", "expired"),
    [
        (timedelta(0), False),
        (timedelta(minutes=9, seconds=59), False),
        (timedelta(minutes=10), True),
        (timedelta(minutes=10, seconds=1), True),
    ],
)
def test_expiry_is_inclusive_at_its_boundary(offset: timedelta, expired: bool) -> None:
    expires_at = request().expires_at(PUBLISHED_AT)

    assert is_clarification_expired(expires_at, at=PUBLISHED_AT + offset) is expired


@pytest.mark.unit
def test_expiry_refuses_naive_instants() -> None:
    with pytest.raises(ValueError, match="timezone aware"):
        request().expires_at(PUBLISHED_AT.replace(tzinfo=None))

    with pytest.raises(ValueError, match="timezone aware"):
        is_clarification_expired(
            request().expires_at(PUBLISHED_AT), at=PUBLISHED_AT.replace(tzinfo=None)
        )


@pytest.mark.unit
def test_identity_keeps_session_run_and_clarification_distinct() -> None:
    identity = ContinuationIdentity(
        session_id="session-1", run_id="run-1", clarification_id="clarification-1"
    )

    assert identity.run_id != identity.session_id

    with pytest.raises(ValidationError, match="must be distinct"):
        ContinuationIdentity(
            session_id="session-1", run_id="session-1", clarification_id="clarification-1"
        )
