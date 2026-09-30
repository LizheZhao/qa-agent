"""Turning flagged fields into ClarificationRequest objects.

Every decision here came from a constraint the runtime imposes or a client's real data, so each
test names the one it pins.
"""

from __future__ import annotations

import pytest
from ask_genome_agent.nodes.question_understanding.build_clarification_list import (
    freshness_token_for,
    likely_values_for,
)
from ask_genome_agent.nodes.question_understanding.support.clarification_requests import (
    build_clarification_requests,
)
from orchestration_core import MAX_CLARIFICATION_OPTIONS, ClarificationRequest

_TOKEN = "filter-abc123"


def _build(fields, **kwargs):
    return build_clarification_requests(fields, subquery_id=0, freshness_token=_TOKEN, **kwargs)


@pytest.mark.unit
def test_one_request_per_field_because_only_one_interrupt_is_live_at_a_time() -> None:
    """The invocation boundary rejects more than one live interrupt, so the old shape -- several
    fields listed in a single message -- is not expressible."""

    requests = _build({"country": ["us", "uk"], "kpi": ["conversions", "all"]})

    assert len(requests) == 2
    assert {r.clarification_id for r in requests} == {"sq0-country", "sq0-kpi"}
    assert all(r.task_id == "subquery-0" for r in requests)
    assert all(r.freshness_token == _TOKEN for r in requests)


@pytest.mark.unit
def test_option_ids_are_sanitized_while_labels_keep_the_real_value() -> None:
    """option_id is opaque-safe, so a value with spaces cannot be the id -- the same split
    build_field_extraction_model uses for tool parameters."""

    request = _build({"sub brand": ["cottonelle dry blue", "huggies wipes"]})[0]

    assert request.option_ids == ("cottonelle_dry_blue", "huggies_wipes")
    assert [o.label for o in request.options] == ["cottonelle dry blue", "huggies wipes"]


@pytest.mark.unit
def test_intention_is_single_select_and_everything_else_is_multiple() -> None:
    """Matches the source's _format_user_selections: it takes selected[0] for intention and
    leaves every other field as the list it arrived as. A filter can span several countries; a
    question has one intention."""

    requests = {r.clarification_id: r for r in _build({"intention": ["spend"], "country": ["us"]})}

    assert requests["sq0-intention"].selection_mode == "single"
    assert requests["sq0-country"].selection_mode == "multiple"


@pytest.mark.unit
def test_out_of_scope_is_not_offered_as_a_choice() -> None:
    """It is the system's conclusion rather than something a user picks, and dropping it is also
    what brings LINKEDIN's intention list from 11 to exactly 10."""

    request = _build({"intention": ["margin roi", "out-of-scope", "spend"]})[0]

    assert "out_of_scope" not in request.option_ids
    assert request.option_ids == ("margin_roi", "spend")


@pytest.mark.unit
def test_a_field_over_the_cap_offers_its_likeliest_and_says_how_many_more_exist() -> None:
    """KCC's sub brand has 35 values and LINKEDIN's start has 16, so no cap setting covers both.
    Free text is always accepted, so the question offers the plausible values and admits the rest
    exist rather than being skipped."""

    values = [f"brand-{i:02d}" for i in range(35)]
    request = _build({"sub brand": values}, likely_values={"sub brand": {"brand-30"}})[0]

    assert len(request.options) == MAX_CLARIFICATION_OPTIONS
    assert request.options[0].label == "brand-30"  # the likely one survived the truncation
    assert "25 more exist" in request.question
    assert "type another" in request.question


@pytest.mark.unit
def test_ranking_only_reorders_and_never_drops_a_value_that_fits() -> None:
    values = ["alpha", "beta", "gamma"]
    request = _build({"kpi": values}, likely_values={"kpi": {"gamma"}})[0]

    assert [o.label for o in request.options] == ["gamma", "alpha", "beta"]
    assert "more exist" not in request.question


@pytest.mark.unit
def test_ids_that_collide_after_sanitizing_keep_the_first_value() -> None:
    """A client holding both "sub brand" and "sub_brand" would otherwise make the whole request
    invalid, since the contract requires distinct option ids."""

    request = _build({"kpi": ["net sales", "net_sales", "spend"]})[0]

    assert request.option_ids == ("net_sales", "spend")


@pytest.mark.unit
def test_a_field_with_nothing_selectable_produces_no_request() -> None:
    """options is min_length=1, so such a field cannot be a question at all. It stays in
    clarification_fields and aggregate_results discloses it."""

    assert _build({"intention": ["out-of-scope"], "country": []}) == ()


@pytest.mark.unit
def test_every_request_validates_against_the_shared_contract() -> None:
    """The ids, reason code and agent id all have patterns the contract enforces, and a field name
    containing spaces has to survive all of them."""

    requests = _build(
        {"business unit focused marketing": ["brand", "premium"]},
        reason_codes={"business unit focused marketing": "halo_mismatch"},
    )

    assert ClarificationRequest.model_validate(requests[0].model_dump(mode="json"))
    assert requests[0].reason_code == "halo_mismatch"
    assert requests[0].producer_agent_id == "ask_genome"


@pytest.mark.unit
def test_likely_values_come_from_spacy_captures_and_the_question_text() -> None:
    """Both signals already exist in state; the field-to-bucket mapping is the source's own
    spacy_key_map."""

    available = {"business unit focused marketing": ["brand", "premium"], "country": ["us", "uk"]}
    level_type = {"halo_tagging": ["business unit focused marketing"], "general_level": ["country"]}

    likely = likely_values_for(
        available,
        level_type,
        {"halo_tagging": ["premium"], "custom_level": []},
        "what was the margin roi for premium in the uk",
    )

    assert likely["business unit focused marketing"] == {"premium"}
    assert likely["country"] == {"uk"}  # named in the question, though spaCy matched nothing


@pytest.mark.unit
def test_freshness_token_tracks_the_filter_and_is_stable_across_reruns() -> None:
    """The runtime refuses a resume whose token no longer matches the checkpoint, and an
    interrupted node reruns from its start, so the token must be deterministic."""

    first = freshness_token_for({"intention": "spend", "country": "us"})

    assert first == freshness_token_for({"country": "us", "intention": "spend"})
    assert first != freshness_token_for({"intention": "margin roi", "country": "us"})
    assert ClarificationRequest.model_fields["freshness_token"] is not None
