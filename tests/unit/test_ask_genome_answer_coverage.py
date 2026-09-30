"""Coverage: routed to answer_coverage, facts from the data, answer to the user and dependents."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import pytest
from ask_genome_agent.config import FeasibilityConfig, NerData, SpacyConfig
from ask_genome_agent.contracts import AskGenomePlan, Subquery, SubqueryKind
from ask_genome_agent.nodes import answer_coverage as module
from ask_genome_agent.nodes.answer_coverage import answer_coverage
from orchestration_core import AgentDependencies

from tests.fakes import ScriptedChatModel

pytestmark = pytest.mark.unit

_FEASIBILITY = FeasibilityConfig(
    data_sources={
        "bi": {
            "dimensions": {
                "country": {
                    "values": {
                        "uk": {
                            "period_coverage": {
                                "quarter": {"count": 14, "start": 202301, "end": 202606},
                                "year": {"count": 3, "start": 202301, "end": 202512},
                            },
                            "metrics_present": ["roi", "spend"],
                        },
                        "us": {"period_coverage": {}, "metrics_present": ["spend"]},
                    }
                },
                "kpi": {"values": {"delivered bookings": {}}},
            }
        }
    }
)


@pytest.fixture(autouse=True)
def _config(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = pd.DataFrame(
        {"country": ["uk", "uk", "us"], "kpi": ["delivered bookings", "sqo", "delivered bookings"]}
    )
    monkeypatch.setattr(module, "load_feasibility_config", lambda: _FEASIBILITY)
    monkeypatch.setattr(module, "load_spacy_config", lambda: SpacyConfig(core_filters={}))
    monkeypatch.setattr(module, "load_ner_data", lambda: NerData(df_bi=frame, df_br=frame))


def _state(query: str, hints: dict[str, Any] | None = None) -> dict[str, Any]:
    subquery = Subquery(id=0, query=query, kind=SubqueryKind.coverage, spacy_hints=hints or {})
    plan = AskGenomePlan(
        original_query=query,
        rephrased_query=query,
        is_multi=False,
        is_sequential=False,
        subqueries=(subquery,),
    )
    return {"plan": plan, "active_index": 0, "ner_state": {}}


def _deps(model: ScriptedChatModel) -> AgentDependencies:
    return AgentDependencies(model=model, tools=())


def _spec(**spec: Any) -> ScriptedChatModel:
    return ScriptedChatModel().queue_tool_call("CoverageSpec", spec)


async def test_the_facts_come_from_the_config_and_the_model_only_words_them() -> None:
    model = _spec(
        operation="count_periods", dimensions=["country"], values=["UK"], period_type="quarter"
    ).queue_text("There are 14 quarters of UK data, from 2023Q1 to 2026Q2.")
    update = await answer_coverage(_state("how many quarters of uk data do we have"), _deps(model))

    entry = update["ner_state"][0]
    assert entry["answer"] == "There are 14 quarters of UK data, from 2023Q1 to 2026Q2."
    facts = entry["coverage"]["facts"]
    assert facts["found"] is True
    assert facts["period_coverage"] == {"quarter": {"count": 14, "start": 202301, "end": 202606}}
    # The model said "UK"; the data stores "uk".
    assert entry["coverage"]["spec"]["values"] == ["uk"]
    phrasing = model.calls[1]["messages"][0].content
    assert '"count": 14' in phrasing


async def test_a_value_the_data_does_not_hold_is_reported_as_not_found() -> None:
    model = _spec(operation="existence", dimensions=["country"], values=["france"]).queue_text(
        "There is no data for France."
    )
    update = await answer_coverage(_state("do we have data for france"), _deps(model))
    facts = update["ner_state"][0]["coverage"]["facts"]
    assert facts["found"] is False
    assert "dimension_tracked" not in facts, "country is tracked; france is simply absent"


async def test_a_dimension_the_config_does_not_track_is_not_reported_as_no_data() -> None:
    """Live, this said paid social had no data. It does; the config just doesn't list channels."""

    model = _spec(
        operation="count_periods",
        dimensions=["business_driver_detail"],
        values=["paid social"],
        period_type="quarter",
    ).queue_text("Quarter coverage is not recorded for channels.")
    update = await answer_coverage(_state("how many quarters did we run paid social"), _deps(model))

    facts = update["ner_state"][0]["coverage"]["facts"]
    assert facts["dimension_tracked"] is False
    assert "It does not mean there is none." in facts["note"]
    assert "It does not mean there is none." in model.calls[1]["messages"][0].content


async def test_values_that_co_occur_are_counted_live_from_the_data() -> None:
    model = _spec(
        operation="co_occurrence", dimensions=["country", "kpi"], needs_join=True
    ).queue_text("Three pairs.")
    update = await answer_coverage(_state("which countries have which kpis"), _deps(model))
    facts = update["ner_state"][0]["coverage"]["facts"]
    assert facts["pair_count"] == 3
    assert {"country": "us", "kpi": "delivered bookings"} in facts["pairs"]


async def test_a_spacy_match_overrides_the_model_s_column(monkeypatch: pytest.MonkeyPatch) -> None:
    """As upstream: the core filter knows the column, the model only guesses."""

    core = {"britain": [{"country": ["UK"], "source": "bi"}]}
    monkeypatch.setattr(module, "load_spacy_config", lambda: SpacyConfig(core_filters=core))
    model = _spec(operation="metric_availability", dimensions=["market"], values=["britain"])
    model.queue_text("ROI and spend.")
    state = _state("what metrics exist for britain", hints={"custom_level": ["britain"]})
    update = await answer_coverage(state, _deps(model))

    coverage = update["ner_state"][0]["coverage"]
    assert coverage["spec"]["dimensions"] == ["country"]
    assert coverage["facts"]["metrics_present"] == ["roi", "spend"]


async def test_a_dependent_subquery_gets_the_answer_and_the_facts() -> None:
    """As upstream does, so the item list isn't lost in the wording."""

    model = _spec(operation="list_values", dimensions=["country"]).queue_text("UK and US.")
    update = await answer_coverage(_state("which countries do we have"), _deps(model))
    carried = update["last_subquery_result"]
    assert carried.startswith("UK and US.")
    assert json.loads(carried.split("\n\n", 1)[1])["values"] == ["uk", "us"]


async def test_a_failed_phrasing_still_reports_the_facts() -> None:
    model = _spec(operation="list_values", dimensions=["country"]).queue_error(
        RuntimeError("gateway timeout")
    )
    update = await answer_coverage(_state("which countries do we have"), _deps(model))
    entry = update["ner_state"][0]
    assert '"values": ["uk", "us"]' in entry["answer"]
    assert entry["response_error"] == "RuntimeError: gateway timeout"


async def test_the_final_message_carries_the_coverage_answer() -> None:
    from ask_genome_agent.nodes.aggregate_results import aggregate_results

    model = _spec(operation="list_values", dimensions=["country"]).queue_text("UK and US.")
    state = _state("which countries do we have")
    state.update(await answer_coverage(state, _deps(model)))
    state["unanswered"] = {}
    result = await aggregate_results(state)  # type: ignore[arg-type]
    assert "UK and US." in result["messages"][0].content


# ------------------------------------------------------------------ routing


def test_coverage_goes_to_answer_coverage_and_then_writes_context() -> None:
    from ask_genome_agent.graph import create_graph

    graph = create_graph(_deps(ScriptedChatModel())).compile().get_graph()
    edges = {(edge.source, edge.target) for edge in graph.edges}
    assert ("answer_coverage", "write_context") in edges
    assert ("assemble_plan", "answer_coverage") in edges
    assert ("apply_resolution", "answer_coverage") in edges


@pytest.mark.parametrize(
    ("kind", "depends_on", "expected"),
    [
        (SubqueryKind.coverage, None, "answer_coverage"),
        (SubqueryKind.analytical, None, "extract_entities"),
        (SubqueryKind.coverage, 0, "resolve_dependency"),
    ],
)
def test_route_next(kind: SubqueryKind, depends_on: int | None, expected: str) -> None:
    from ask_genome_agent.graph import _route_next

    first = Subquery(id=0, query="a", kind=SubqueryKind.analytical, coarse_intent="roi")
    intent = "roi" if kind == SubqueryKind.analytical else None
    second = Subquery(id=1, query="b", kind=kind, coarse_intent=intent, depends_on=depends_on)
    subqueries = (
        (first, second) if depends_on is not None else (second.model_copy(update={"id": 0}),)
    )
    plan = AskGenomePlan(
        original_query="q",
        rephrased_query="q",
        is_multi=len(subqueries) > 1,
        is_sequential=depends_on is not None,
        subqueries=subqueries,
    )
    index = len(subqueries) - 1
    assert _route_next({"plan": plan, "active_index": index}) == expected  # type: ignore[typeddict-item]


def test_a_rephrased_dependent_coverage_subquery_is_answered_as_coverage() -> None:
    from ask_genome_agent.contracts import DependencyResolution
    from ask_genome_agent.graph import _route_after_resolution

    first = Subquery(id=0, query="a", kind=SubqueryKind.analytical, coarse_intent="roi")
    second = Subquery(id=1, query="b", kind=SubqueryKind.coverage, depends_on=0)
    plan = AskGenomePlan(
        original_query="q",
        rephrased_query="q",
        is_multi=True,
        is_sequential=True,
        subqueries=(first, second),
    )
    state = {
        "plan": plan,
        "active_index": 1,
        "resolution": DependencyResolution(kind="rephrased", raw_text=("b resolved",)),
    }
    assert _route_after_resolution(state) == "answer_coverage"  # type: ignore[arg-type]
