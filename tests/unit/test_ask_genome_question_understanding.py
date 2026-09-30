"""Per-node tests for ask-genome Question Understanding: one test per porting decision.

Each of these pins a single decision made while porting from ask-genome-core -- a divergence from
the source, or a bug found live -- so it can't silently revert. Each node is exercised against
hand-built `ner_state`, monkeypatching one config loader per test.

Scope, stated precisely because an earlier version of this docstring got it wrong: this validates
each **node's** logic in isolation. It does not validate the **chain**, and cannot -- every key a
node-under-test reads is supplied by the test itself, so a producer that stopped writing a key
would leave this file green. The node-to-node handoffs are covered in
test_ask_genome_plan_wiring.py, which runs the chain in sequence and injects no `ner_state` keys
at all. Keep new handoff assertions there; keep decision records here.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest
from ask_genome_agent.config import NerData, NerDataConfig
from ask_genome_agent.contracts import AskGenomePlan, Subquery, SubqueryKind
from ask_genome_agent.nodes.question_understanding import (
    build_clarification_list as build_clarification_list_module,
)
from ask_genome_agent.nodes.question_understanding import classify_fields as classify_fields_module
from ask_genome_agent.nodes.question_understanding import (
    fill_remaining_fields as fill_remaining_fields_module,
)
from ask_genome_agent.nodes.question_understanding.build_clarification_list import (
    build_clarification_list,
)
from ask_genome_agent.nodes.question_understanding.extract_entities import extract_entities
from ask_genome_agent.nodes.question_understanding.fill_remaining_fields import (
    fill_remaining_fields,
)
from ask_genome_agent.nodes.question_understanding.finalize_filter import finalize_filter
from orchestration_core import AgentDependencies

from tests.fakes import ScriptedChatModel

_QUESTIONNAIRE = {
    "classification": {"field": ["intention"], "context": "", "prompt": "classify"},
    "time": {"field": ["start", "end"], "context": "", "prompt": "extract time"},
}
_DF_BI = pd.DataFrame({"time": ["fiscal quarter 1 2024"], "intention": ["spend"]})


def _plan_with_one_subquery(spacy_hints: dict[str, Any] | None = None) -> AskGenomePlan:
    subquery = Subquery(
        id=0,
        query="what was linkedin spend last quarter",
        kind=SubqueryKind.analytical,
        coarse_intent="spend",
        spacy_hints=spacy_hints,
    )
    return AskGenomePlan(
        original_query=subquery.query,
        rephrased_query=subquery.query,
        is_multi=False,
        is_sequential=False,
        subqueries=(subquery,),
    )


def _base_state(messages: list[Any] | None = None, **ner_state_overrides: Any) -> dict[str, Any]:
    return {
        "plan": _plan_with_one_subquery(spacy_hints={"core_dimension": ["linkedin"]}),
        "active_index": 0,
        "ner_state": {0: ner_state_overrides} if ner_state_overrides else {},
        "messages": messages or [],
    }


@pytest.mark.unit
async def test_a_coverage_subquery_reaching_extract_entities_is_a_routing_bug() -> None:
    """The graph sends coverage to answer_coverage, so one here is a bug."""

    plan = AskGenomePlan(
        original_query="q",
        rephrased_query="q",
        is_multi=False,
        is_sequential=False,
        subqueries=(Subquery(id=0, query="q", kind=SubqueryKind.coverage),),
    )
    with pytest.raises(RuntimeError, match="answer_coverage"):
        await extract_entities({"plan": plan, "active_index": 0, "ner_state": {}})


@pytest.mark.unit
async def test_fill_remaining_fields_degrades_to_clarification_instead_of_raising(
    monkeypatch,
) -> None:
    """The bug found in ask-genome-core: exhausting retries used to re-raise and crash the whole
    request despite logging "adding to clarification list". Here it must actually degrade."""

    single_dimension = {"classification": _QUESTIONNAIRE["classification"]}
    ner_data = NerData(df_bi=_DF_BI, questionnaire=single_dimension, level_type={})
    monkeypatch.setattr(fill_remaining_fields_module, "load_ner_data", lambda: ner_data)

    fake = ScriptedChatModel()
    for _ in range(3):
        fake.queue_error(RuntimeError("gateway timeout"))

    state = _base_state(predictions={}, probabilities={}, extended_query="what was spend")
    result = await fill_remaining_fields(state, AgentDependencies(model=fake, tools=()))

    entry = result["ner_state"][0]
    assert "intention" in entry["clarification_list"]
    assert len(fake.calls) == 3  # all 3 retries exhausted, none succeeded


@pytest.mark.unit
async def test_fill_remaining_fields_fills_via_structured_tool_call(monkeypatch) -> None:
    ner_data = NerData(df_bi=_DF_BI, questionnaire=_QUESTIONNAIRE, level_type={})
    monkeypatch.setattr(fill_remaining_fields_module, "load_ner_data", lambda: ner_data)

    fake = ScriptedChatModel().queue_tool_call("FieldExtractionOutput", {"intention": "spend"})
    state = _base_state(
        predictions={}, probabilities={}, extended_query="what was linkedin spend last quarter"
    )
    result = await fill_remaining_fields(state, AgentDependencies(model=fake, tools=()))

    entry = result["ner_state"][0]
    assert entry["ner_results"]["intention"] == "spend"
    assert entry["field_status"]["intention"] == "llm"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('["202504"]', ["202504"]),  # the exact shape seen live in a real tool call
        ('["202504", "202506"]', ["202504", "202506"]),
        ("[202504]", ["202504"]),  # unquoted numbers coerced to strings too
        ("202504", "202504"),  # a plain value is left alone
        ("[not json", "[not json"),  # not decodable -> untouched
        ("[]", []),
        (["202504"], ["202504"]),  # an actual list is already correct
        (None, None),
    ],
)
def test_normalize_extracted_value_decodes_stringified_lists(raw, expected) -> None:
    """The field schema accepts `list[str] | str | None`, and that ambiguity makes the model
    sometimes encode a list as a string -- confirmed live: {"start": "[\\"202504\\"]"}. Left
    as-is, downstream time filtering gets the literal string instead of a usable date."""

    assert fill_remaining_fields_module._normalize_extracted_value(raw) == expected


@pytest.mark.unit
async def test_fill_remaining_fields_normalizes_stringified_list_from_the_model(
    monkeypatch,
) -> None:
    single_dimension = {"time": {"field": ["start"], "context": "", "prompt": "extract time"}}
    ner_data = NerData(df_bi=_DF_BI, questionnaire=single_dimension, level_type={})
    monkeypatch.setattr(fill_remaining_fields_module, "load_ner_data", lambda: ner_data)

    fake = ScriptedChatModel().queue_tool_call("FieldExtractionOutput", {"start": '["202504"]'})
    state = _base_state(predictions={}, probabilities={}, extended_query="last quarter")
    result = await fill_remaining_fields(state, AgentDependencies(model=fake, tools=()))

    assert result["ner_state"][0]["ner_results"]["start"] == ["202504"]


@pytest.mark.unit
async def test_build_clarification_list_flags_halo_mismatch(monkeypatch) -> None:
    df_with_halo_column = _DF_BI.assign(halo_level=["own_brand"])
    ner_data = NerData(
        df_bi=df_with_halo_column,
        questionnaire=_QUESTIONNAIRE,
        level_type={"halo_level": ["halo_level"], "halo_tagging": ["halo_tagging"]},
    )
    monkeypatch.setattr(build_clarification_list_module, "load_ner_data", lambda: ner_data)
    monkeypatch.setattr(
        build_clarification_list_module,
        "load_ner_data_config",
        lambda: NerDataConfig(metric_info={"spend": {}}),
    )

    state = _base_state(
        ner_results={},
        probabilities={"halo": {"yes": 0.9}},
        spacy_filter={"halo_tagging": ["competitor"]},
        clarification_list=[],
    )
    result = await build_clarification_list(state)
    assert result["ner_state"][0]["clarification_fields"]


@pytest.mark.unit
def test_process_bert_predictions_remaps_fiscal_labels_for_non_fiscal_client() -> None:
    non_fiscal_df = pd.DataFrame({"time": ["quarter 1 2024"]})
    probabilities = {
        "period_type": {"fiscal quarter": 0.7, "fiscal half": 0.2, "fiscal year": 0.1},
        "business_driver_class": {"paid": 0.9, "organic": 0.1},
    }

    predictions, remapped = classify_fields_module.process_bert_predictions(
        probabilities, non_fiscal_df
    )

    assert "business_driver" in remapped and "business_driver_class" not in remapped
    assert set(remapped["period_type"]) == {"quarter", "half year", "year"}
    assert predictions["period_type"] == "quarter"
    assert predictions["business_driver"] == "paid"


@pytest.mark.unit
def test_finalize_filter_tags_every_field_auto_confirmed() -> None:
    """finalize_filter tags whatever ner_results already holds, it merges nothing. Answers land on
    the filter in coordinate_clarification, before this node runs."""

    state = _base_state(ner_results={"intention": "spend", "platform": "linkedin"})

    result = finalize_filter(state)
    entry = result["ner_state"][0]

    assert entry["ner_results"] == {"intention": "spend", "platform": "linkedin"}
    assert entry["field_status"] == {"intention": "auto-confirmed", "platform": "auto-confirmed"}
    assert result["last_subquery_result"] == "intention=spend, platform=linkedin"
