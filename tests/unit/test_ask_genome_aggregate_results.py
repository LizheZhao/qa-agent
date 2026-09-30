"""Unit tests for ask_genome_agent.nodes.aggregate_results."""

from __future__ import annotations

import pytest
from ask_genome_agent.contracts import AskGenomePlan, Subquery, SubqueryKind
from ask_genome_agent.nodes.aggregate_results import aggregate_results

_PLAN = AskGenomePlan(
    original_query="q",
    rephrased_query="q",
    is_multi=False,
    is_sequential=False,
    subqueries=(Subquery(id=0, query="q", kind=SubqueryKind.analytical, coarse_intent="spend"),),
)


_MULTI_PLAN = AskGenomePlan(
    original_query="compare spend and margin roi for paid search last quarter",
    rephrased_query="compare spend and margin roi for paid search last quarter",
    is_multi=True,
    is_sequential=False,
    subqueries=(
        Subquery(id=0, query="spend", kind=SubqueryKind.analytical, coarse_intent="spending"),
        Subquery(id=1, query="roi", kind=SubqueryKind.analytical, coarse_intent="margin roi"),
    ),
)


@pytest.mark.unit
async def test_each_part_of_a_multi_part_question_says_what_it_is_about() -> None:
    """Two subqueries that match the same number of rows printed the same sentence twice, with
    nothing telling them apart, which reads as a bug rather than two real results. Seen live on
    "compare spend and margin roi for paid search last quarter"."""

    state = {
        "plan": _MULTI_PLAN,
        "ner_state": {
            0: {"row_count": 8385, "data_source": {"intention": "spending"}},
            1: {"row_count": 8385, "data_source": {"intention": "margin roi"}},
        },
    }

    content = (await aggregate_results(state))["messages"][0].content

    assert "For spending, found 8385 matching rows." in content
    assert "For margin roi, found 8385 matching rows." in content
    # Reported in plan order, so the parts line up with how the question was asked.
    assert content.index("spending") < content.index("margin roi")


@pytest.mark.unit
async def test_a_refused_subquery_is_reported_not_skipped() -> None:
    """Confirmed live on "what was the spend for the top two best performing tactics": the
    planner splits it correctly and marks part two as depending on part one, but part one
    produces a filter rather than an answer, so nothing names the two tactics and
    resolve_dependency refuses. Refusing is right. Reporting only part one's row count, as if
    the question had been answered, is not."""

    state = {
        "plan": _MULTI_PLAN,
        "ner_state": {0: {"row_count": 254338, "data_source": {"intention": "performance"}}},
        "unanswered": {1: "predecessor context unavailable"},
    }

    content = (await aggregate_results(state))["messages"][0].content

    assert "For performance, found 254338 matching rows." in content
    assert "I couldn't answer 'roi': predecessor context unavailable." in content


@pytest.mark.unit
async def test_a_single_part_question_is_not_labelled() -> None:
    """There is nothing to tell it apart from, so naming the intention is just noise."""

    state = {"plan": _PLAN, "ner_state": {0: {"row_count": 12}}}

    content = (await aggregate_results(state))["messages"][0].content

    assert "Found 12 matching rows." in content
    assert "For " not in content


@pytest.mark.unit
async def test_aggregate_results_stays_plain_when_nothing_is_unresolved() -> None:
    state = {"plan": _PLAN, "ner_state": {0: {"clarification_fields": {}}}}
    result = await aggregate_results(state)
    content = result["messages"][0].content
    assert content == "I extracted 1 subquery from your request."


@pytest.mark.unit
async def test_aggregate_results_surfaces_fields_that_were_never_resolved() -> None:
    """Confirmed live: a run can finalize with a flagged field still missing from ner_results, and
    finalize_filter used to silently drop it. Cancelling a clarification reaches the same place.
    Must be surfaced, not hidden."""

    state = {
        "plan": _PLAN,
        "ner_state": {
            0: {
                "clarification_fields": {
                    "business_unit_for_kpi": ["lms", "lss"],
                    "detailed_kpi": ["business conversions"],
                }
            }
        },
    }
    result = await aggregate_results(state)
    content = result["messages"][0].content
    assert "Still unsure about" in content
    assert "business_unit_for_kpi" in content
    assert "detailed_kpi" in content
