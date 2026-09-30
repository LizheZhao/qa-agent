"""Wiring tests: nodes run in sequence, as the graph actually runs them.

These exist because a unit test that builds a `Subquery(spacy_hints=...)` by hand passes happily
while the real pipeline drops the field -- `assemble_plan` was constructing `Subquery(...)` without
`spacy_hints`, so `annotate_feasibility`'s work was silently discarded and `extract_entities`
always took its recompute fallback. Testing each node in isolation could never have caught that.

The same reasoning applies to the whole Question Understanding chain, which is why the second half
of this file exists. Those nodes communicate only by writing and reading keys in one shared
`ner_state[subquery.id]` dict, and `test_ask_genome_question_understanding.py` hand-supplies every
key its node-under-test reads -- so a producer that stopped writing a key, or wrote it under a
different name, would leave that whole suite green. Five of the eight such seams had no coverage
at all. The chain tests below start from *only* what the planning phase produces and never inject
an `ner_state` key, so a dropped handoff surfaces as a KeyError instead of passing silently.

The last section compiles a real graph with a checkpointer, which is the only way to exercise the
interrupt/resume clarification path at all.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest
from ask_genome_agent.config import (
    FeasibilityConfig,
    NarrowingConfig,
    NerData,
    NerDataConfig,
    SpacyConfig,
)
from ask_genome_agent.contracts import (
    AskGenomePlan,
    Subquery,
    SubqueryKind,
    build_field_extraction_model,
    sanitize_field_name,
)
from ask_genome_agent.nodes import annotate_feasibility as annotate_feasibility_module
from ask_genome_agent.nodes.aggregate_results import aggregate_results
from ask_genome_agent.nodes.annotate_feasibility import annotate_feasibility
from ask_genome_agent.nodes.assemble_plan import assemble_plan
from ask_genome_agent.nodes.question_understanding import (
    apply_data_filters as apply_data_filters_module,
)
from ask_genome_agent.nodes.question_understanding import (
    build_clarification_list as build_clarification_list_module,
)
from ask_genome_agent.nodes.question_understanding import classify_fields as classify_fields_module
from ask_genome_agent.nodes.question_understanding import (
    extract_entities as extract_entities_module,
)
from ask_genome_agent.nodes.question_understanding import (
    fill_remaining_fields as fill_remaining_fields_module,
)
from ask_genome_agent.nodes.question_understanding import (
    reconcile_intent as reconcile_intent_module,
)
from ask_genome_agent.nodes.question_understanding import (
    resolve_data_source as resolve_data_source_module,
)
from ask_genome_agent.nodes.question_understanding.apply_data_filters import apply_data_filters
from ask_genome_agent.nodes.question_understanding.build_clarification_list import (
    build_clarification_list,
    get_available_values,
)
from ask_genome_agent.nodes.question_understanding.classify_fields import classify_fields
from ask_genome_agent.nodes.question_understanding.coordinate_clarification import (
    after_clarification,
    coordinate_clarification,
    should_clarify,
)
from ask_genome_agent.nodes.question_understanding.extract_entities import extract_entities
from ask_genome_agent.nodes.question_understanding.fill_remaining_fields import (
    fill_remaining_fields,
)
from ask_genome_agent.nodes.question_understanding.finalize_filter import finalize_filter
from ask_genome_agent.nodes.question_understanding.reconcile_intent import reconcile_intent
from ask_genome_agent.nodes.question_understanding.resolve_data_source import resolve_data_source
from ask_genome_agent.nodes.question_understanding.support import table as table_module
from ask_genome_agent.nodes.question_understanding.support.time_filter import validated_indicators
from ask_genome_agent.nodes.question_understanding.support.time_periods import format_today
from ask_genome_agent.state import AskGenomeState
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from orchestration_core import AgentDependencies, ClarificationRequest

from agentic_orchestration.execution.checkpointing import (
    MemoryGraphCheckpointer,
    graph_checkpointer_scope,
)
from tests.fakes import ScriptedChatModel

_HINTS = {"core_dimension": ["paid search"], "custom_level": ["us"], "halo_tagging": []}

_PLAN_OUTPUT: dict[str, Any] = {
    "rephrased_query": "what was margin roi for paid search last quarter",
    "is_multi": False,
    "subqueries": [
        {
            "id": 0,
            "query": "what was margin roi for paid search last quarter",
            "kind": "analytical",
            "coarse_intent": "margin roi",
            "depends_on": None,
        }
    ],
}


@pytest.fixture
def _stub_configs(monkeypatch) -> None:
    monkeypatch.setattr(
        annotate_feasibility_module, "load_feasibility_config", lambda: FeasibilityConfig()
    )
    monkeypatch.setattr(annotate_feasibility_module, "load_spacy_config", lambda: SpacyConfig())
    monkeypatch.setattr(
        annotate_feasibility_module,
        "load_ner_data_config",
        lambda: NerDataConfig(metric_info={"margin roi": {"data": "bi"}}),
    )
    monkeypatch.setattr(
        annotate_feasibility_module, "compute_spacy_hints", lambda _config, _query: dict(_HINTS)
    )


@pytest.mark.unit
async def test_spacy_hints_survive_annotate_feasibility_into_the_assembled_plan(
    _stub_configs,
) -> None:
    """The B1 regression: hints computed in annotate_feasibility must reach Subquery.spacy_hints."""

    state: dict[str, Any] = {
        "query": _PLAN_OUTPUT["rephrased_query"],
        "plan_output": _PLAN_OUTPUT,
    }
    state.update(await annotate_feasibility(state))

    assert state["plan_output"]["subqueries"][0]["spacy_hints"] == _HINTS

    state.update(assemble_plan(state))

    assert state["plan"].subqueries[0].spacy_hints == _HINTS


@pytest.mark.unit
async def test_extract_entities_reuses_the_hints_the_real_pipeline_produced(
    _stub_configs, monkeypatch
) -> None:
    """End of the same chain: extract_entities must not recompute what the plan already carries."""

    state: dict[str, Any] = {
        "query": _PLAN_OUTPUT["rephrased_query"],
        "plan_output": _PLAN_OUTPUT,
    }
    state.update(await annotate_feasibility(state))
    state.update(assemble_plan(state))
    state["ner_state"] = {}

    def _fail(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("extract_entities recomputed spaCy instead of reusing the hints")

    monkeypatch.setattr(extract_entities_module, "compute_spacy_hints", _fail)
    monkeypatch.setattr(
        extract_entities_module, "load_ner_data_config", lambda: NerDataConfig(acronym_dict={})
    )

    result = await extract_entities(state)

    assert result["ner_state"][0]["spacy_filter"] == _HINTS


@pytest.mark.unit
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("business unit focused marketing", "business_unit_focused_marketing"),
        ("business_unit_for_kpi", "business_unit_for_kpi"),
        ("sub brand", "sub_brand"),
        ("detailed_kpi", "detailed_kpi"),
        ("kpi (net)", "kpi__net"),
        ("  padded  ", "padded"),
    ],
)
def test_sanitize_field_name_produces_identifier_safe_tool_params(raw, expected) -> None:
    """The halo bug: a questionnaire field containing spaces makes an invalid tool-call parameter
    name, and the gateway rejects the whole tool definition with HTTP 500 -- taking every other
    field in that dimension down with it. Verified live: with spaces -> HTTP 500; sanitized ->
    resolves correctly."""

    assert sanitize_field_name(raw) == expected


@pytest.mark.unit
def test_field_extraction_schema_uses_sanitized_names_and_keeps_the_real_name_visible() -> None:
    fields = ["business_unit_for_kpi", "business unit focused marketing"]
    schema = build_field_extraction_model(fields)

    properties = schema.model_json_schema()["properties"]
    assert set(properties) == {"business_unit_for_kpi", "business_unit_focused_marketing"}
    # The model still needs to know which real field it is answering about.
    assert (
        "business unit focused marketing"
        in properties["business_unit_focused_marketing"]["description"]
    )


@pytest.mark.unit
def test_field_extraction_schema_rejects_names_that_collide_when_sanitized() -> None:
    with pytest.raises(ValueError, match="collide after sanitizing"):
        build_field_extraction_model(["sub brand", "sub_brand"])


@pytest.mark.unit
def test_custom_cols_is_general_level_and_tagging_only() -> None:
    """Matches the source's `custom_cols` (filter_generator.py:312). Including halo_* here would
    wrongly route the halo dimension through the classification prompt instead of the default
    answer-format branch."""

    ner_data = NerData(
        df_bi=pd.DataFrame(),
        questionnaire={"classification": {"field": ["intention"]}},
        level_type={
            "general_level": ["country", "kpi"],
            "general_tagging": ["tag_a"],
            "halo_level": ["business_unit_for_kpi"],
            "halo_tagging": ["business unit focused marketing"],
            "diag_tagging": ["diag"],
        },
    )
    assert set(ner_data.custom_cols) == {"tag_a", "country", "kpi"}


# --------------------------------------------------------------------------------------------
# The Question Understanding chain, run in sequence on one coherent fake client.
#
# One config satisfies all six nodes at once -- that is the point. The per-node suite
# monkeypatches a different loader per test, so no two nodes there ever agree on what the client
# looks like, which is another reason a mismatched key between them is invisible.
# --------------------------------------------------------------------------------------------

_CHAIN_QUERY = "what was margin roi for paid search last quarter in the us"

# 'business unit focused marketing' is deliberately kept as a real LINKEDIN/12 field name with
# spaces in it, so the chain exercises build_field_extraction_model's sanitize round-trip through
# the real _fill_dimension rather than only through the unit tests above.
_HALO_TAGGING_FIELD = "business unit focused marketing"

_CHAIN_QUESTIONNAIRE: dict[str, Any] = {
    "classification": {"field": ["intention"], "context": "", "prompt": "classify the intention"},
    # The real client CSV stores a template here, with placeholders and doubled JSON braces.
    "time": {
        "field": ["start", "end"],
        "context": "",
        "prompt": 'Today is {date}. Question: {question}. Reply like {{"start": "YYYYMM"}}',
    },
    "geography": {"field": ["country"], "context": "geo", "prompt": "which country"},
    "halo": {
        "field": ["business_unit_for_kpi", _HALO_TAGGING_FIELD],
        "context": "halo",
        "prompt": "halo fields",
    },
}
_CHAIN_LEVEL_TYPE: dict[str, list[str]] = {
    "general_level": ["country"],
    "general_tagging": [],
    "halo_level": ["business_unit_for_kpi"],
    "halo_tagging": [_HALO_TAGGING_FIELD],
}
_CHAIN_DF_BI = pd.DataFrame(
    {
        # non-fiscal, so process_bert_predictions must strip the 'fiscal ' prefix off period_type
        "time": ["quarter 1 2024", "quarter 2 2024"],
        "country": ["us", "uk"],
        "business_unit_for_kpi": ["lms", "lss"],
        # The real client has a column for the halo tagging field too. Without one,
        # get_available_values finds no values and silently drops it, so a halo mismatch would
        # produce one question instead of two.
        _HALO_TAGGING_FIELD: ["premium", "brand"],
        # Data filtering reads these; Question Understanding never looks at them. The three
        # hierarchy columns are the triple get_specific_values names outright when resolving
        # business_driver, so a frame without them is a misconfigured client, not a valid one.
        "activity_group": ["paid search", "digital display"],
        "business_driver_detail": ["paid search", "digital display"],
        "business_driver": ["marketing", "marketing"],
        "measure_group": ["sem", "display"],
        "measure": ["lms bing search", "lms dv360"],
        "value": [1.1, 2.2],
        "metric": ["gross roi", "gross roi"],
        "start": [202401, 202404],
        "end": [202403, 202406],
        "period_type": ["quarter", "quarter"],
    }
)
_CHAIN_METRIC_INFO = {
    "margin roi": {"data": "bi", "metric": ["roi"], "mainMetric": ["roi"]},
    "spend": {"data": "br", "metric": ["spend"], "mainMetric": ["spend"]},
}
_CHAIN_METRIC_CONFIGS = pd.DataFrame(
    {"code": ["roi", "spend"], "kpiName": ["Gross ROI", "Gross Spend"]}
)
_CHAIN_DATA_LEVELS = {"biLevels": ["country"], "brLevels": ["country"], "brLevelsToIgnore": []}


def _bert_response(*, halo_yes: float) -> dict[str, dict[str, float]]:
    """The classifier's real response shape, already reduced to {field: {label: prob}}."""

    return {
        "intention": {"margin roi": 1.0, "spend": 0.0},
        "business_driver_class": {"specific": 1.0, "all": 0.0},
        "trend": {"no": 1.0, "yes": 0.0},
        "period_type": {"fiscal quarter": 0.99, "fiscal half": 0.01},
        "rank": {"na": 1.0},
        "halo": {"yes": halo_yes, "no": 1.0 - halo_yes},
    }


def _chain_plan_with_query(query: str, spacy_hints: dict[str, Any]) -> AskGenomePlan:
    subquery = Subquery(
        id=0,
        query=query,
        kind=SubqueryKind.analytical,
        coarse_intent="margin roi",
        spacy_hints=spacy_hints,
    )
    return AskGenomePlan(
        original_query=query,
        rephrased_query=query,
        is_multi=False,
        is_sequential=False,
        subqueries=(subquery,),
    )


def _chain_plan(spacy_hints: dict[str, Any]) -> AskGenomePlan:
    return _chain_plan_with_query(_CHAIN_QUERY, spacy_hints)


def _scripted_field_answers() -> ScriptedChatModel:
    """One tool call per questionnaire dimension that still needs the LLM, in questionnaire order:
    time, geography, halo. BERT covers 'intention' at full confidence, so 'classification' is
    skipped -- which is itself part of the classify_fields -> fill_remaining_fields contract.

    The halo answers are keyed by *sanitized* names, because that is what the real tool schema
    exposes; _fill_dimension maps them back to the questionnaire's real names.
    """

    return (
        ScriptedChatModel()
        .queue_tool_call("FieldExtractionOutput", {"start": "202401", "end": "202403"})
        .queue_tool_call("FieldExtractionOutput", {"country": "specific"})
        .queue_tool_call(
            "FieldExtractionOutput",
            {
                "business_unit_for_kpi": "irrelevant",
                sanitize_field_name(_HALO_TAGGING_FIELD): "irrelevant",
            },
        )
    )


def _chain_ner_data() -> NerData:
    return NerData(
        df_bi=_CHAIN_DF_BI,
        metric_configs=_CHAIN_METRIC_CONFIGS,
        questionnaire=_CHAIN_QUESTIONNAIRE,
        level_type=_CHAIN_LEVEL_TYPE,
    )


def _chain_narrowing(
    core_filters: dict[str, Any] | None = None, level_type: dict[str, Any] | None = None
) -> NarrowingConfig:
    """One narrowing config, shared so the chain and postprocess never diverge on it."""

    return NarrowingConfig(
        client_code="TESTCO",
        metric_info={},
        bi_levels=("activity_group", "measure_group", "measure"),
        br_levels=("business_driver", "business_driver_detail", "activity_group"),
        core_filters=core_filters
        or {"paid search": [{"source": "bi", "activity_group": ["paid search"]}]},
        level_type=level_type or {"general_level": [], "halo_level": [], "halo_tagging": []},
        sub_cols=(),
        fingerprint="chain",
    )


@pytest.fixture
def _chain_client(monkeypatch) -> None:
    """Patch every loader the chain nodes read, to one consistent client."""

    ner_data = _chain_ner_data()
    ner_data_config = NerDataConfig(
        metric_info=_CHAIN_METRIC_INFO, data_levels=_CHAIN_DATA_LEVELS, acronym_dict={}
    )
    # A real filter, not just a source tag: the narrowing steps resolve captured terms
    # against these, so a term configured with no columns would select nothing.
    spacy_config = SpacyConfig(
        core_filters={"paid search": [{"source": "bi", "activity_group": ["paid search"]}]}
    )

    reads_ner_data = (
        classify_fields_module,
        fill_remaining_fields_module,
        build_clarification_list_module,
        resolve_data_source_module,
        apply_data_filters_module,
        # rehydrate_result reads it too, so a replay uses the same client as the live pass.
        table_module,
    )
    reads_ner_data_config = (
        extract_entities_module,
        reconcile_intent_module,
        build_clarification_list_module,
        resolve_data_source_module,
    )
    for module in reads_ner_data:
        monkeypatch.setattr(module, "load_ner_data", lambda: ner_data)
    for module in reads_ner_data_config:
        monkeypatch.setattr(module, "load_ner_data_config", lambda: ner_data_config)
    for module in (extract_entities_module, reconcile_intent_module):
        monkeypatch.setattr(module, "load_spacy_config", lambda: spacy_config)
    monkeypatch.setattr(
        apply_data_filters_module,
        "load_time_indicators",
        lambda: validated_indicators({}, (_CHAIN_DF_BI,)),
    )
    # Built from the same spacy config the rest of the chain uses, so the narrowing steps run on
    # what extract_entities actually captured rather than on a second, divergent fixture.
    narrowing = _chain_narrowing(spacy_config.core_filters, ner_data.level_type)
    for module in (apply_data_filters_module, table_module):
        monkeypatch.setattr(module, "load_narrowing_config", lambda: narrowing)
    monkeypatch.setenv("ASK_GENOME_CLIENT_CODE", "TESTCO")
    monkeypatch.setenv("ASK_GENOME_MODEL_GROUP_ID", "1")
    monkeypatch.setattr(table_module, "source_stamp", lambda source: (1, 2))


async def _run_chain(
    state: dict[str, Any], model: ScriptedChatModel, monkeypatch, *, halo_yes: float
) -> list[set[str]]:
    """Run the six state-producing QU nodes in graph order, returning the ner_state entry's key
    set after each one. Only the external classifier boundary is faked."""

    async def _fake_classify(query: str, client: Any = None) -> dict[str, dict[str, float]]:
        del query, client
        return _bert_response(halo_yes=halo_yes)

    monkeypatch.setattr(classify_fields_module, "classify_query", _fake_classify)

    dependencies = AgentDependencies(model=model, tools=())
    keys_after: list[set[str]] = []
    for node in (extract_entities, classify_fields, reconcile_intent):
        state.update(await node(state))
        keys_after.append(set(state["ner_state"][0]))
    state.update(await fill_remaining_fields(state, dependencies))
    keys_after.append(set(state["ner_state"][0]))
    state.update(await build_clarification_list(state))
    keys_after.append(set(state["ner_state"][0]))
    return keys_after


@pytest.mark.unit
async def test_question_understanding_chain_resolves_every_field_without_injected_state(
    _chain_client, monkeypatch
) -> None:
    """The whole chain, starting from only what the planning phase produces.

    `ner_state` is absent at the start and no test code ever writes a key into it, so every value
    each node reads had to have been produced by an earlier node. This is what the per-node suite
    cannot do: there, a producer could stop writing a key entirely and nothing would fail.
    """

    model = _scripted_field_answers()
    state: dict[str, Any] = {
        "plan": _chain_plan({"core_dimension": ["paid search"], "halo_tagging": []}),
        "active_index": 0,
        "messages": [HumanMessage(content=_CHAIN_QUERY)],
    }

    await _run_chain(state, model, monkeypatch, halo_yes=0.02)
    entry = state["ner_state"][0]

    # BERT's own fields survived reconcile_intent's overwrite and fill_remaining_fields' merge.
    assert entry["ner_results"]["intention"] == "margin roi"
    assert entry["field_status"]["intention"] == "bert"
    # ...including the non-fiscal period_type remap, which only classify_fields performs.
    assert entry["ner_results"]["period_type"] == "quarter"

    # The LLM-filled fields, including the space-containing name mapped back from its sanitized
    # tool parameter.
    assert entry["ner_results"]["start"] == "202401"
    assert entry["ner_results"]["country"] == "specific"
    assert entry["ner_results"][_HALO_TAGGING_FIELD] == "irrelevant"
    assert entry["field_status"]["country"] == "llm"

    # Nothing ambiguous left, so the graph routes past clarification and finalizes.
    assert entry["clarification_fields"] == {}
    assert should_clarify(state) == "finalize_filter"

    state.update(finalize_filter(state))
    # finalize_filter must not flatten the provenance it inherited (the B4 fix).
    assert state["ner_state"][0]["field_status"]["intention"] == "bert"
    assert "intention=margin roi" in state["last_subquery_result"]


@pytest.mark.unit
async def test_the_filter_the_chain_resolved_is_what_reaches_the_client_data(
    _chain_client, monkeypatch
) -> None:
    """The new seam: finalize_filter -> resolve_data_source -> apply_data_filters.

    Nothing here injects `intention`, `data_source` or `ner_results` -- each is read from what the
    node before it wrote. A producer that renamed any of them would surface as a KeyError instead
    of a quietly empty result, which is the failure mode this file exists for.
    """

    state: dict[str, Any] = {
        "plan": _chain_plan({"core_dimension": ["paid search"], "halo_tagging": []}),
        "active_index": 0,
        "messages": [HumanMessage(content=_CHAIN_QUERY)],
    }

    await _run_chain(state, _scripted_field_answers(), monkeypatch, halo_yes=0.02)
    state.update(finalize_filter(state))
    state.update(await resolve_data_source(state))

    source = state["ner_state"][0]["data_source"]
    # BERT's intention survived all the way into the dataset choice, and the metric group it names
    # resolved to a real kpiName rather than an empty list.
    assert source["intention"] == "margin roi"
    assert source["source"] == "bi"
    assert source["main_metric"] == ["gross roi"]

    state.update(await apply_data_filters(state))
    entry = state["ner_state"][0]

    # The time range the LLM extracted reached the data, resolved against the periods that exist
    # rather than used raw: one requested quarter is padded with the same quarter a year earlier,
    # so a comparison is possible. Only 202401 is in this fixture, so one row survives.
    assert entry["applied_filter"]["start"] == [202301, 202401]
    assert entry["applied_filter"]["period_type"] == ["quarter"]
    assert entry["row_count"] == 1
    assert entry["records"][0]["time"] == "quarter 1 2024"
    assert "1 rows matched" in state["last_subquery_result"]

    # country came back as 'specific', which is an answer *format* meaning "spaCy will narrow
    # this", so for now it widens to every real value rather than narrowing. That is the
    # _get_specific_values / spaCy-merge step apply_data_filters documents as not yet ported;
    # this pins the current behaviour so porting it shows up here as a failure.
    assert sorted(entry["applied_filter"]["country"]) == ["uk", "us"]

    message = (await aggregate_results(state))["messages"][0]
    assert "Found 1 matching rows" in message.content


@pytest.mark.unit
async def test_extract_entities_extended_query_is_what_reaches_the_field_extraction_model(
    _chain_client, monkeypatch
) -> None:
    """The extract_entities -> fill_remaining_fields seam, asserted on the payload rather than on
    the absence of a KeyError: the text the LLM is asked about must be the `extended_query` that
    extract_entities produced, not the raw subquery or something reconstructed downstream."""

    model = _scripted_field_answers()
    state: dict[str, Any] = {
        "plan": _chain_plan({"core_dimension": ["paid search"], "halo_tagging": []}),
        "active_index": 0,
        "messages": [HumanMessage(content=_CHAIN_QUERY)],
    }

    await _run_chain(state, model, monkeypatch, halo_yes=0.02)

    extended_query = state["ner_state"][0]["extended_query"]
    assert model.calls, "the field-extraction model was never called"
    for call in model.calls:
        rendered = " ".join(str(message.content) for message in call["messages"])
        assert extended_query in rendered


@pytest.mark.unit
async def test_chain_accumulates_exactly_the_documented_ner_state_keys(
    _chain_client, monkeypatch
) -> None:
    """Pins the producer/consumer contract in state.py's `ner_state` docstring. Each node adds its
    own keys and removes none -- a rename shows up here as a diff instead of as a silent
    downstream KeyError in production."""

    state: dict[str, Any] = {
        "plan": _chain_plan({"core_dimension": ["paid search"], "halo_tagging": []}),
        "active_index": 0,
        "messages": [HumanMessage(content=_CHAIN_QUERY)],
    }

    keys_after = await _run_chain(state, _scripted_field_answers(), monkeypatch, halo_yes=0.02)

    assert keys_after[0] == {"extended_query", "detected_pair", "spacy_filter"}
    assert keys_after[1] == keys_after[0] | {"predictions", "probabilities"}
    assert keys_after[2] == keys_after[1] | {"clarification_list", "source_conflict"}
    assert keys_after[3] == keys_after[2] | {"ner_results", "field_status"}
    assert keys_after[4] == keys_after[3] | {"clarification_fields"}


@pytest.mark.unit
def test_intention_offers_only_real_intentions_plus_out_of_scope() -> None:
    """The source appends "all" and "irrelevant" to any field it has no explicit options for,
    which reached `intention` and offered 13 choices for a client with 10 intentions. Neither
    extra is answerable: there is no "every intention at once", and an analytical question always
    has one. Raised by Lizhe after seeing the clarification message.
    """

    ner_data = _chain_ner_data()
    config = NerDataConfig(metric_info={"margin roi": {}, "spend": {}, "source of change": {}})

    options = get_available_values(["intention"], ner_data, config)["intention"]

    # "out-of-scope" sorts alphabetically with the rest rather than trailing them.
    assert options == ["margin roi", "out-of-scope", "source of change", "spend"]
    assert "all" not in options
    assert "irrelevant" not in options

    # A tagging column still gets them, because there they are real answers.
    tagging = get_available_values(["business_unit_for_kpi"], ner_data, config)
    assert "all" in tagging["business_unit_for_kpi"]


@pytest.mark.unit
async def test_a_dataset_conflict_is_reported_and_never_silently_substituted(
    _chain_client, monkeypatch
) -> None:
    """The behaviour agreed in review, replacing the source's.

    spaCy matched a br-sourced dimension while BERT predicted a bi-sourced intention. The source
    replaces the user's intention with that source's default and asks them to "clarify intention".
    Confirmed live that this is wrong twice over: the substitution is unconditional, so the filter
    committed to 'source of change' even where the user never agreed to it; and a
    dataset incompatibility is not evidence the user was ambiguous.

    The user's intention must survive, and the conflict must reach the final message instead.
    """

    monkeypatch.setattr(
        reconcile_intent_module,
        "load_spacy_config",
        lambda: SpacyConfig(core_filters={"brand refresh": [{"source": "br"}]}),
    )

    state: dict[str, Any] = {
        "plan": _chain_plan({"core_dimension": ["brand refresh"], "halo_tagging": []}),
        "active_index": 0,
        "messages": [HumanMessage(content=_CHAIN_QUERY)],
    }

    await _run_chain(state, _scripted_field_answers(), monkeypatch, halo_yes=0.02)
    entry = state["ner_state"][0]

    # What the user asked for is what the filter says.
    assert entry["predictions"]["intention"] == "margin roi"
    assert entry["ner_results"]["intention"] == "margin roi"

    # The conflict is recorded rather than acted on.
    conflict = entry["source_conflict"]
    assert conflict["intention"] == "margin roi"
    assert conflict["dimensions"] == ["brand refresh"]
    assert conflict["dimension_source"] == "br"
    assert conflict["alternative_intention"] == "source of change"

    # It is not treated as an ambiguous intention, so no clarification is raised for it.
    assert "intention" not in entry["clarification_list"]
    assert "intention" not in entry["clarification_fields"]
    assert should_clarify(state) == "finalize_filter"

    # ...and the user is told, in their own terms.
    message = (await aggregate_results(state))["messages"][0]
    assert "margin roi" in message.content
    assert "brand refresh" in message.content
    assert "source of change" in message.content


@pytest.mark.unit
async def test_unresolved_fields_reach_the_final_message_from_the_real_chain(
    _chain_client, monkeypatch
) -> None:
    """The last uncovered seam: build_clarification_list writes `clarification_fields` and
    aggregate_results reads it to build the "still unsure about" disclosure. The cancel test below
    walks the same key through a compiled graph, but it hands the nodes a prepared ner_state. This
    one starts from only what planning produces, so it also proves the key is the one the producer
    actually writes.

    Whenever a field ends the run unresolved the user has to be told. Silently dropping one is the
    live bug the disclosure exists for.
    """

    state: dict[str, Any] = {
        "plan": _chain_plan({"core_dimension": ["paid search"], "halo_tagging": []}),
        "active_index": 0,
        "messages": [HumanMessage(content=_CHAIN_QUERY)],
    }

    await _run_chain(state, _scripted_field_answers(), monkeypatch, halo_yes=0.9)

    # Ambiguous, so the graph pauses here rather than finalizing.
    assert state["ner_state"][0]["clarification_fields"]
    assert should_clarify(state) == "coordinate_clarification"

    # If those questions go unanswered, finalizing must not quietly bury them.
    state.update(finalize_filter(state))

    message = (await aggregate_results(state))["messages"][0]
    assert "Still unsure about" in message.content
    assert "business_unit_for_kpi" in message.content


@pytest.mark.unit
async def test_no_prompt_placeholder_ever_reaches_the_model(_chain_client, monkeypatch) -> None:
    """The live defect: questionnaire.csv's prompt column is a *template*, and this port passed it
    through raw, so the model received the literal characters "Today is {date}." for the time
    dimension -- which is why two queries differing only by "in the us" resolved the same "last
    quarter" to 202504-202506 and 202601-202603.

    Asserted across every prompt the chain sends, not just time's, because every dimension's
    template has unfilled placeholders.
    """

    model = _scripted_field_answers()
    state: dict[str, Any] = {
        "plan": _chain_plan({"core_dimension": ["paid search"], "halo_tagging": []}),
        "active_index": 0,
        "messages": [HumanMessage(content=_CHAIN_QUERY)],
    }

    await _run_chain(state, model, monkeypatch, halo_yes=0.02)

    assert model.calls, "no prompt was sent"
    for call in model.calls:
        rendered = " ".join(str(message.content) for message in call["messages"])
        for placeholder in (
            "{date}",
            "{question}",
            "{example}",
            "{predefined_list}",
            "{cols}",
            "{answer_format}",
            "{context}",
            "{dimension}",
        ):
            assert placeholder not in rendered, f"unfilled {placeholder} sent to the model"

    # The time dimension must actually carry a real date, not merely lack a placeholder.
    time_calls = [
        c for c in model.calls if "Today is" in " ".join(str(m.content) for m in c["messages"])
    ]
    assert time_calls, "the time dimension prompt was never sent"
    rendered = " ".join(str(m.content) for m in time_calls[0]["messages"])
    assert format_today() in rendered
    # ...and the doubled braces in the client template must unescape, not survive as '{{'.
    assert "{{" not in rendered


# --------------------------------------------------------------------------------------------
# Pause and resume, through a graph compiled with a checkpointer.
#
# The node-level tests above call functions directly, which cannot exercise interrupt() at all:
# LangGraph needs a compiled graph and a saver to suspend and continue one. That gap is exactly
# how the old clarification path stayed green while the graph had stopped calling it.
# --------------------------------------------------------------------------------------------


def _clarification_graph() -> StateGraph[Any]:
    """Just the clarification stretch of the real graph, wired the same way."""

    builder: StateGraph[Any] = StateGraph(AskGenomeState)
    builder.add_node("build_clarification_list", build_clarification_list)
    builder.add_node("coordinate_clarification", coordinate_clarification)
    builder.add_node("finalize_filter", finalize_filter)
    builder.add_edge(START, "build_clarification_list")
    builder.add_conditional_edges(
        "build_clarification_list",
        should_clarify,
        path_map={
            "coordinate_clarification": "coordinate_clarification",
            "finalize_filter": "finalize_filter",
        },
    )
    builder.add_conditional_edges(
        "coordinate_clarification",
        after_clarification,
        path_map={
            "coordinate_clarification": "coordinate_clarification",
            "finalize_filter": "finalize_filter",
        },
    )
    builder.add_edge("finalize_filter", END)
    return builder


def _halo_mismatch_state() -> dict[str, Any]:
    """BERT says there is a halo, the LLM answered 'irrelevant' for both halo fields."""

    return {
        "plan": _chain_plan({"core_dimension": ["paid search"], "halo_tagging": ["premium"]}),
        "active_index": 0,
        "messages": [HumanMessage(content=_CHAIN_QUERY)],
        "ner_state": {
            0: {
                "extended_query": _CHAIN_QUERY,
                "spacy_filter": {"halo_tagging": ["premium"]},
                "probabilities": {"halo": {"yes": 0.9, "no": 0.1}},
                "clarification_list": [],
                "user_confirmed": {},
                "ner_results": {
                    "intention": "margin roi",
                    "business_unit_for_kpi": "irrelevant",
                    _HALO_TAGGING_FIELD: "irrelevant",
                },
                "field_status": {"intention": "bert"},
            }
        },
    }


@pytest.mark.unit
async def test_the_graph_pauses_with_a_typed_request_and_resumes_in_place(_chain_client) -> None:
    """The whole point of the rework: the run suspends rather than ending the turn, and continues
    with ner_state intact. Under the old design the reply came back as a new turn, the planner
    rephrased it, and the captured dimension and time range were lost."""

    async with graph_checkpointer_scope(MemoryGraphCheckpointer()) as checkpointer:
        graph = _clarification_graph().compile(checkpointer=checkpointer.saver)
        config = {"configurable": {"thread_id": "conversation-under-test"}}

        result = await graph.ainvoke(_halo_mismatch_state(), config)

        # Paused, not finished, and carrying a request the shared contract accepts.
        assert "__interrupt__" in result
        assert len(result["__interrupt__"]) == 1, "one live interrupt per invocation"
        first = ClarificationRequest.model_validate(result["__interrupt__"][0].value)
        assert first.producer_agent_id == "ask_genome"
        assert first.reason_code == "halo_mismatch"

        answered = [first.clarification_id]
        while "__interrupt__" in result:
            request = ClarificationRequest.model_validate(result["__interrupt__"][0].value)
            answered.append(request.clarification_id)
            result = await graph.ainvoke(
                Command(resume={"form": "options", "option_ids": [request.option_ids[0]]}), config
            )

    entry = result["ner_state"][0]

    # Both halo fields were asked, one at a time, and both answers landed on the filter.
    assert len(set(answered)) == 2
    assert entry["field_status"]["business_unit_for_kpi"] == "user-confirmed"
    assert entry["field_status"][_HALO_TAGGING_FIELD] == "user-confirmed"
    assert entry["clarification_fields"] == {}

    # What the pause preserved: extraction done before it is still there, untouched.
    assert entry["spacy_filter"] == {"halo_tagging": ["premium"]}
    assert entry["ner_results"]["intention"] == "margin roi"
    assert entry["field_status"]["intention"] == "bert"


@pytest.mark.unit
async def test_cancelling_leaves_the_field_unresolved_and_disclosed(_chain_client) -> None:
    """Cancel is always an available response. It must not be recorded as an answer the user never
    gave, so the field stays in clarification_fields for aggregate_results to disclose."""

    async with graph_checkpointer_scope(MemoryGraphCheckpointer()) as checkpointer:
        graph = _clarification_graph().compile(checkpointer=checkpointer.saver)
        config = {"configurable": {"thread_id": "cancelled-conversation"}}
        result = await graph.ainvoke(_halo_mismatch_state(), config)
        while "__interrupt__" in result:
            result = await graph.ainvoke(Command(resume={"form": "cancel"}), config)

    entry = result["ner_state"][0]
    assert entry["field_status"].get("business_unit_for_kpi") != "user-confirmed"
    assert set(entry["clarification_fields"]) == {"business_unit_for_kpi", _HALO_TAGGING_FIELD}

    message = (await aggregate_results(result))["messages"][0].content
    assert "Still unsure about" in message


@pytest.mark.unit
async def test_free_text_is_taken_as_the_value(_chain_client) -> None:
    """free_text_allowed is fixed True on every question, so a user can type a value the option
    list did not offer -- which is what makes truncating a 35-value field acceptable."""

    async with graph_checkpointer_scope(MemoryGraphCheckpointer()) as checkpointer:
        graph = _clarification_graph().compile(checkpointer=checkpointer.saver)
        config = {"configurable": {"thread_id": "typed-conversation"}}
        result = await graph.ainvoke(_halo_mismatch_state(), config)
        while "__interrupt__" in result:
            result = await graph.ainvoke(
                Command(resume={"form": "free_text", "text": "lts"}), config
            )

    entry = result["ner_state"][0]
    assert entry["ner_results"]["business_unit_for_kpi"] == "lts"
    assert entry["field_status"]["business_unit_for_kpi"] == "user-confirmed"


@pytest.mark.unit
async def test_postprocess_receives_what_the_chain_actually_produced(
    _chain_client, monkeypatch
) -> None:
    """The seam the live run broke on, from the planning phase with nothing injected.

    Three bugs got past the per-node suites because every fixture there hand-supplied the keys
    that were missing in reality:

    - the node read the recipe's own filter, which predates the spaCy merge and so carries no
      `core_dimension`;
    - `time`, `period_type` and `level` are derived after filtering and were never carried;
    - `map_dict` names metric groups and the data names kpis, so nothing ranked. process_data
      now expands them itself, so what is left to pin is the filter.

    Each one left postprocessing producing nothing while every test stayed green.
    """

    from ask_genome_agent.nodes.postprocess import run_postprocess as postprocess_module
    from ask_genome_agent.nodes.postprocess.run_postprocess import _readout_filter
    from ask_genome_agent.nodes.question_understanding.support.table import (
        TableRecipe,
        rehydrate_result,
    )

    monkeypatch.setattr(postprocess_module, "load_narrowing_config", lambda: _chain_narrowing())

    state: dict[str, Any] = {
        "plan": _chain_plan({"core_dimension": ["paid search"], "halo_tagging": []}),
        "active_index": 0,
        "messages": [HumanMessage(content=_CHAIN_QUERY)],
    }
    await _run_chain(state, _scripted_field_answers(), monkeypatch, halo_yes=0.02)
    state.update(finalize_filter(state))
    state.update(await resolve_data_source(state))
    state.update(await apply_data_filters(state))

    entry = state["ner_state"][0]
    recipe = TableRecipe.model_validate(entry["table_recipe"])
    rebuilt = rehydrate_result(recipe)

    # The recipe is the input to filtering; these are its output, and the node derives them.
    assert "core_dimension" not in recipe.ner_filter, "the recipe stores the pre-merge filter"
    assert "core_dimension" in rebuilt.ner_filter, "the merge is what adds it"

    resolved = _readout_filter(rebuilt, entry)
    assert resolved["core_dimension"], "the branch fields must reach postprocessing"
    assert resolved["time"], "derived from the filtered rows, not from the question"
    assert resolved["period_type"]
    assert "halo_tagging" in resolved["level"], "the client's tagging columns"
    assert "latest_time" in resolved
