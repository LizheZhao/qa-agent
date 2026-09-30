"""Per-node tests for the data-filtering stage: one test per porting decision.

Same split as the Question Understanding suites. Decisions and divergences from ask-genome-core
are pinned here against hand-built state; the handoffs between these nodes and the ones before
them belong in test_ask_genome_plan_wiring.py, which runs the chain for real.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest
from ask_genome_agent.config import NarrowingConfig, NerData, NerDataConfig
from ask_genome_agent.contracts import AskGenomePlan, Subquery, SubqueryKind
from ask_genome_agent.nodes.aggregate_results import aggregate_results
from ask_genome_agent.nodes.question_understanding import (
    apply_data_filters as apply_data_filters_module,
)
from ask_genome_agent.nodes.question_understanding import (
    resolve_data_source as resolve_data_source_module,
)
from ask_genome_agent.nodes.question_understanding.apply_data_filters import apply_data_filters
from ask_genome_agent.nodes.question_understanding.resolve_data_source import (
    after_data_source,
    resolve_data_source,
)
from ask_genome_agent.nodes.question_understanding.support import table as table_module
from ask_genome_agent.nodes.question_understanding.support.filtering import (
    answer_dict_for,
    apply_answer_dict,
    rejection_message_for,
)
from ask_genome_agent.nodes.question_understanding.support.time_filter import validated_indicators

from tests.source_parity import SOURCE_SRC, source_available, source_functions

# Two rows per source so a wrong-frame bug shows up as a row count, not just a column error.
_DF_BI = pd.DataFrame(
    {
        "metric": ["gross roi", "activity", "gross roi"],
        "country": ["us", "us", "uk"],
        "business_driver": ["media", "media", "non-media"],
        # The hierarchy the level step reads. It names all five outright, so a frame without
        # them is a misconfigured client rather than a case to tolerate.
        "business_driver_detail": ["paid search"] * 3,
        "activity_group": ["paid search"] * 3,
        "measure_group": ["sem"] * 3,
        "measure": ["lms bing search"] * 3,
        "time": ["quarter 1 2024"] * 3,
        "value": [1.1, 2.2, 3.3],
        "start": [202401, 202401, 202401],
        "end": [202403, 202403, 202403],
        "period_type": ["quarter", "quarter", "quarter"],
    }
)
_DF_BR = pd.DataFrame(
    {
        "metric": ["sovc", "contribution"],
        "country": ["us", "us"],
        "business_driver": ["media", "base"],
        "business_driver_detail": ["paid search", "non marketing"],
        "activity_group": ["paid search", "other"],
        "measure_group": ["sem", "other"],
        "measure": ["lms bing search", "other"],
        "time": ["quarter 1 2024"] * 2,
        "value": [4.4, 5.5],
        "start": [202401, 202401],
        "end": [202403, 202403],
        "period_type": ["quarter", "quarter"],
    }
)
_METRIC_CONFIGS = pd.DataFrame(
    {
        "code": ["roi", "activity", "sovc", "cost_per_activity"],
        "kpiName": ["Gross ROI", "Activity", "SOVC", "CPM"],
    }
)
_METRIC_INFO = {
    "margin roi": {"data": "bi", "metric": ["roi"], "mainMetric": ["roi"]},
    "source of change": {"data": "br", "metric": ["sovc"], "mainMetric": ["sovc"]},
    "cost per": {
        "data": "bi",
        "metric": ["cost_per_activity"],
        "mainMetric": ["cost_per_activity"],
    },
}
_DATA_LEVELS = {
    "biLevels": ["country", "business_driver"],
    "brLevels": ["country", "business_driver"],
    "brLevelsToIgnore": ["business_driver"],
}


def _ner_data() -> NerData:
    return NerData(
        df_bi=_DF_BI,
        df_br=_DF_BR,
        metric_configs=_METRIC_CONFIGS,
        questionnaire={"classification": {"field": ["intention"]}},
        level_type={"general_level": ["country"]},
    )


_INDICATORS = validated_indicators(
    {"period_priority": ["year", "half", "quarter", "month"], "non_trend_report_length": [1] * 4},
    (_DF_BI, _DF_BR),
)


@pytest.fixture
def _stub_data(monkeypatch) -> None:
    config = NerDataConfig(metric_info=_METRIC_INFO, data_levels=_DATA_LEVELS)
    for module in (resolve_data_source_module, apply_data_filters_module):
        monkeypatch.setattr(module, "load_ner_data", _ner_data, raising=False)
        monkeypatch.setattr(module, "load_ner_data_config", lambda: config, raising=False)
    monkeypatch.setattr(
        apply_data_filters_module, "load_time_indicators", lambda: _INDICATORS, raising=False
    )
    # No spaCy config in these fixtures: the narrowing chain runs but captures nothing, which is
    # what makes these per-node assertions about the filter alone.
    narrowing = NarrowingConfig(
        client_code="TESTCO",
        metric_info={},
        bi_levels=("activity_group", "measure_group", "measure"),
        br_levels=("business_driver", "business_driver_detail", "activity_group"),
        core_filters={},
        level_type={"general_level": ["country"]},
        sub_cols=(),
        fingerprint="cfg0",
    )
    for module in (apply_data_filters_module, table_module):
        monkeypatch.setattr(module, "load_narrowing_config", lambda: narrowing, raising=False)
    # The recipe records which client and which file the table came from. There is no data
    # directory here, so the identity is stubbed the same way the loaders are.
    monkeypatch.setenv("ASK_GENOME_CLIENT_CODE", "TESTCO")
    monkeypatch.setenv("ASK_GENOME_MODEL_GROUP_ID", "1")
    monkeypatch.setattr(table_module, "source_stamp", lambda source: (1, 2))


def _state(**entry: Any) -> dict[str, Any]:
    subquery = Subquery(id=0, query="q", kind=SubqueryKind.analytical, coarse_intent="margin roi")
    plan = AskGenomePlan(
        original_query="q",
        rephrased_query="q",
        is_multi=False,
        is_sequential=False,
        subqueries=(subquery,),
    )
    # Both come from earlier nodes; the filtering node reads them without producing them.
    entry.setdefault("extended_query", "q")
    entry.setdefault("spacy_filter", {})
    return {"plan": plan, "active_index": 0, "ner_state": {0: entry}}


@pytest.mark.unit
async def test_intention_decides_which_dataset_is_read(_stub_data) -> None:
    """bi vs br is per intention, not per client, which is the whole reason br.csv is loaded."""

    bi = await resolve_data_source(_state(ner_results={"intention": "margin roi"}))
    br = await resolve_data_source(_state(ner_results={"intention": "source of change"}))

    assert bi["ner_state"][0]["data_source"]["source"] == "bi"
    assert bi["ner_state"][0]["data_source"]["row_count_before_filtering"] == len(_DF_BI)
    assert br["ner_state"][0]["data_source"]["source"] == "br"
    assert br["ner_state"][0]["data_source"]["row_count_before_filtering"] == len(_DF_BR)
    # br carries fields the answer must not filter on; bi does not.
    assert br["ner_state"][0]["data_source"]["ignore_fields"] == ["business_driver"]
    assert bi["ner_state"][0]["data_source"]["ignore_fields"] == []


@pytest.mark.unit
async def test_metric_groups_resolve_to_real_kpi_names(_stub_data) -> None:
    """The join-key fix, pinned. map_dict.csv names groups by standardized_metrics' `code`
    ('cost_per_activity'), while the source looks them up by a lowercased `metricName`
    ('cost per activity'). Every multi-word group silently resolved to nothing, which for LINKEDIN
    left the whole 'cost per' intention with no main metric and therefore no possible answer."""

    result = await resolve_data_source(_state(ner_results={"intention": "cost per"}))

    assert result["ner_state"][0]["data_source"]["main_metric"] == ["cpm"]


@pytest.mark.unit
async def test_out_of_scope_ends_the_turn_instead_of_raising(_stub_data) -> None:
    """The source raises ValueError and takes the request down. Out-of-scope is an expected
    answer here -- it is even offered in the intention clarification -- so it degrades."""

    result = await resolve_data_source(_state(ner_results={"intention": "out-of-scope"}))
    entry = result["ner_state"][0]

    assert entry["out_of_scope"] is True
    assert entry["data_source"] is None
    assert after_data_source({**_state(**entry), "ner_state": {0: entry}}) == "write_context"

    message = (await aggregate_results({**_state(**entry), "ner_state": {0: entry}}))["messages"][0]
    assert "isn't something I can answer" in message.content


@pytest.mark.unit
async def test_filtering_produces_real_records_and_a_row_count(_stub_data) -> None:
    state = _state(
        ner_results={"intention": "margin roi", "country": "us", "start": 202401, "end": 202403},
        data_source={
            "intention": "margin roi",
            "source": "bi",
            "metric": ["gross roi"],
            "main_metric": ["gross roi"],
            "data_levels": ["country", "business_driver"],
            "ignore_fields": [],
            "row_count_before_filtering": len(_DF_BI),
        },
    )
    result = await apply_data_filters(state)
    entry = result["ner_state"][0]

    # us only, and only the metric this intention reports on: the uk row and the activity row
    # are both dropped. Records are plain dicts, never a DataFrame.
    assert entry["row_count"] == 1
    assert {r["country"] for r in entry["records"]} == {"us"}
    assert {r["metric"] for r in entry["records"]} == {"gross roi"}
    assert all(isinstance(record, dict) for record in entry["records"])
    assert entry["rejection_message"] == ""
    assert "1 rows matched" in result["last_subquery_result"]


@pytest.mark.unit
async def test_a_large_result_is_sampled_rather_than_checkpointed_whole(
    _stub_data, monkeypatch
) -> None:
    """State is checkpointed into one document capped at 16MB. Confirmed live: a LINKEDIN spend
    question matches ~96k rows, and inlining them raised DocumentTooLarge and failed the whole
    request. row_count stays the real total."""

    # All one metric and every row a distinct value, so this stays a test about sampling rather
    # than about filtering: repeated rows would be collapsed by the two dedupes.
    wide = pd.concat([_DF_BI[_DF_BI["metric"] == "gross roi"]] * 60, ignore_index=True)
    wide = wide.assign(value=range(len(wide)))
    state = _state(
        ner_results={"intention": "margin roi"},
        data_source={
            "intention": "margin roi",
            "source": "bi",
            "metric": ["gross roi"],
            "main_metric": ["gross roi"],
            "data_levels": ["country"],
            "ignore_fields": [],
            "row_count_before_filtering": len(wide),
        },
    )
    monkeypatched = NerData(
        df_bi=wide,
        df_br=_DF_BR,
        metric_configs=_METRIC_CONFIGS,
        questionnaire={"classification": {"field": ["intention"]}},
        level_type={"general_level": ["country"]},
    )
    monkeypatch.setattr(apply_data_filters_module, "load_ner_data", lambda: monkeypatched)

    result = await apply_data_filters(state)
    entry = result["ner_state"][0]

    assert entry["row_count"] == len(wide), "the count must stay the real total"
    assert len(entry["records"]) == 50
    assert entry["records_truncated"] is True


def _bi_source(**overrides: Any) -> dict[str, Any]:
    source: dict[str, Any] = {
        "intention": "margin roi",
        "source": "bi",
        "metric": ["gross roi"],
        "main_metric": ["gross roi"],
        "data_levels": ["country", "business_driver"],
        "ignore_fields": [],
        "row_count_before_filtering": len(_DF_BI),
    }
    source.update(overrides)
    return source


@pytest.mark.unit
async def test_the_intention_s_metrics_now_narrow_the_rows(_stub_data) -> None:
    """Before the narrowing chain was wired in, ner_filter never reached the apply step, so the
    metric the intention reports on was resolved and then never used. A margin ROI question
    counted activity rows as if they were ROI rows."""

    state = _state(ner_results={"intention": "margin roi"}, data_source=_bi_source())

    entry = (await apply_data_filters(state))["ner_state"][0]

    assert {r["metric"] for r in entry["records"]} == {"gross roi"}
    assert entry["applied_filter"]["metric"] == ["gross roi"]


@pytest.mark.unit
async def test_exact_duplicate_rows_are_removed_once_the_spacy_step_runs(
    _stub_data, monkeypatch
) -> None:
    """combine_filters ends with drop_duplicates, so wiring it in removes rows that repeat in
    full. Not the duplicate-hierarchy issue in this module's docstring: those rows differ in
    `value`, so they survive this and are still counted twice."""

    narrowing = NarrowingConfig(
        client_code="TESTCO",
        metric_info={},
        bi_levels=("activity_group", "measure_group", "measure"),
        br_levels=("business_driver", "business_driver_detail", "activity_group"),
        core_filters={"paid search": [{"source": "bibr", "country": ["us"]}]},
        level_type={"general_level": ["country"]},
        sub_cols=(),
        fingerprint="cfg1",
    )
    monkeypatch.setattr(apply_data_filters_module, "load_narrowing_config", lambda: narrowing)

    doubled = pd.concat([_DF_BI, _DF_BI], ignore_index=True)
    differing = doubled.assign(value=range(len(doubled)))
    for frame, expected in ((doubled, 1), (differing, 2)):
        monkeypatch.setattr(
            apply_data_filters_module,
            "load_ner_data",
            lambda frame=frame: NerData(
                df_bi=frame,
                df_br=_DF_BR,
                metric_configs=_METRIC_CONFIGS,
                questionnaire={"classification": {"field": ["intention"]}},
                level_type={"general_level": ["country"]},
            ),
        )
        state = _state(
            ner_results={"intention": "margin roi", "country": "us"},
            data_source=_bi_source(row_count_before_filtering=len(frame)),
        )

        entry = (await apply_data_filters(state))["ner_state"][0]

        assert entry["row_count"] == expected


@pytest.mark.unit
def test_a_filter_that_would_empty_the_result_is_skipped_and_named() -> None:
    """The source applies filters as a running conjunction and refuses to let one empty it,
    recording the smallest conflicting combination instead. That is what makes the message
    actionable: an empty table tells the user nothing about which pair disagreed."""

    answer_dict = {"country": ["uk"], "business_driver": ["media"]}
    filtered, log = apply_answer_dict(_DF_BI, answer_dict)

    # uk applies; uk + media is empty, so business_driver is skipped rather than zeroing the run.
    assert len(filtered) == 1
    assert log["conflicts"][0]["filter"] == "business_driver"
    assert log["conflicts"][0]["conflicts_with"] == ["country"]
    assert "no data in combination with country" in rejection_message_for(log)


@pytest.mark.unit
def test_a_value_absent_from_the_data_is_reported_not_applied() -> None:
    filtered, log = apply_answer_dict(_DF_BI, {"country": ["fr"]})

    assert len(filtered) == len(_DF_BI)  # skipped, so nothing was narrowed
    assert log["skipped_no_value"] == {"country": ["fr"]}
    assert "No data matches country" in rejection_message_for(log)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("all", ["us", "uk"]),  # every real value
        ("specific", ["us", "uk"]),  # same until spaCy narrows it
        ("us", ["us"]),
        ("fr", None),  # nothing real, so the field is left out entirely
    ],
)
def test_model_answers_map_to_values_that_exist_in_the_column(answer, expected) -> None:
    """`all`/`specific`/`irrelevant` are answer *formats*, not column values. Passing them
    through to the filter would match nothing."""

    resolved = answer_dict_for(_DF_BI, {"country": answer}, ["country"])

    if expected is None:
        assert "country" not in resolved
    else:
        assert sorted(resolved["country"]) == sorted(expected)


@pytest.mark.unit
def test_time_is_an_overlap_not_an_equality() -> None:
    """A row covers the request if its span crosses it at all. Requiring equality would drop
    every row whose period merely contains the quarter asked about."""

    answer_dict = answer_dict_for(_DF_BI, {"start": 202402, "end": 202402}, [])
    filtered, _ = apply_answer_dict(_DF_BI, answer_dict)

    assert len(filtered) == len(_DF_BI)


@pytest.mark.unit
async def test_rows_without_the_intention_s_main_metric_are_disclosed(_stub_data) -> None:
    """Rows can survive every filter and still be useless: if none of them carry a metric the
    intention reports on, the answer would be built from unrelated measures."""

    state = _state(
        ner_results={"intention": "margin roi", "country": "us"},
        data_source={
            "intention": "margin roi",
            "source": "bi",
            "metric": ["gross roi"],
            "main_metric": ["a metric no row carries"],
            "data_levels": ["country"],
            "ignore_fields": [],
            "row_count_before_filtering": len(_DF_BI),
        },
    )
    result = await apply_data_filters(state)
    entry = result["ner_state"][0]

    assert entry["row_count"] > 0
    assert "No rows carry a metric" in entry["rejection_message"]

    message = (await aggregate_results({**state, "ner_state": result["ner_state"]}))["messages"][0]
    assert "No rows carry a metric" in message.content


@pytest.mark.unit
def test_metric_configs_degrade_when_a_client_has_no_formatting_file(tmp_path, monkeypatch) -> None:
    """The source selects its display columns out of an empty frame and raises KeyError, which
    is five of the six onboarded clients. Unformatted numbers beat no answer."""

    from ask_genome_agent.local import get_metric_configs

    client = tmp_path / "NOFMT" / "1"
    client.mkdir(parents=True)
    pd.DataFrame(
        {
            "kpiName": ["gross roi", "contribution"],
            "domain": ["bi", "calculated"],
            "standardizedMetricId": [1, 2],
            "alias": ["ROI", ""],
        }
    ).to_csv(client / "metric_mapping.csv", index=False)
    pd.DataFrame(
        {
            "id": [1, 2],
            "metricName": ["Margin ROI", "Contribution"],
            "code": ["roi", "contribution"],
        }
    ).to_csv(tmp_path / "standardized_metrics.csv", index=False)
    monkeypatch.setenv("ASK_GENOME_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ASK_GENOME_CLIENT_CODE", "NOFMT")
    monkeypatch.setenv("ASK_GENOME_MODEL_GROUP_ID", "1")

    configs = get_metric_configs("NOFMT", 1)

    # Two names per group: the code map_dict uses, and the standardized name the GPT readout
    # matches on.
    assert list(configs["code"]) == ["roi", "contribution"]
    assert list(configs["metricGroup"]) == ["Margin ROI", "Contribution"]
    # Neither name is ever blank. The chain is MetricName -> alias -> kpiName, so a metric the
    # mapping gave no alias falls back to its raw kpiName, not to the standardized metricName.
    assert list(configs["alias"]) == ["ROI", "contribution"]
    assert list(configs["MetricName"]) == ["ROI", "contribution"]
    # Defaults stand in for the missing formatting; a calculated metric reads as a percentage.
    assert configs["NumberDisplayDecimals"].tolist() == [2, 2]
    assert configs.loc[configs["domain"] == "calculated", "Postfix"].iloc[0] == "%"
    assert configs.loc[configs["domain"] == "bi", "Postfix"].iloc[0] == ""


@pytest.mark.unit
@pytest.mark.parametrize(
    ("ner_results", "expected"),
    [
        (
            {"trend": "yes", "rank": "top", "how_many": "5"},
            {"trend": "yes", "rank": "top", "how_many": 5},
        ),
        (
            {"trend": "no", "rank": "bottom", "how_many": 3},
            {"trend": "no", "rank": "bottom", "how_many": 3},
        ),
        # Anything the readout would not recognise reads as 'not asked'.
        (
            {"trend": "maybe", "rank": "sideways", "how_many": "lots"},
            {"trend": "na", "rank": "na", "how_many": "na"},
        ),
        (
            {"trend": "na", "rank": "na", "how_many": "na"},
            {"trend": "na", "rank": "na", "how_many": "na"},
        ),
    ],
)
def test_answers_about_the_question_reach_the_filter(ner_results, expected) -> None:
    """trend, rank and how_many name no column, so the column branches skip them -- but the
    readout reads how_many to decide how many drivers to name. Dropping it silently capped every
    answer at the default three however many the user asked for."""

    resolved = answer_dict_for(_DF_BI, dict(ner_results), ["country"])

    assert {k: resolved[k] for k in expected} == expected


@pytest.mark.unit
def test_a_relevant_answer_resolves_to_yes() -> None:
    """A boolean-ish column is answered 'relevant' rather than with a value."""

    frame = pd.DataFrame({"portfolio": ["yes", "no"], "value": [1.0, 2.0]})

    assert answer_dict_for(frame, {"portfolio": "relevant"}, ["portfolio"]) == {
        "portfolio": ["yes"]
    }


_COLGUS_FRAME = pd.DataFrame(
    {
        "product": ["masterbrand", "optic white", "overall"],
        "country": ["us", "us", "us"],
        "metric": ["gross roi"] * 3,
        "value": [1.0, 2.0, 3.0],
    }
)

_PRODUCT_CASES: tuple[tuple[str, dict[str, Any], list[str]], ...] = (
    ("a real product value", {"product": "optic white"}, ["product"]),
    # 'ahw' is an offered answer for COLGUS but no column carries it.
    ("an answer with no row", {"product": "ahw"}, ["product"]),
    ("equity, likewise", {"product": "equity"}, ["product"]),
    # The prompt warns the classifier against splitting a product name off a tactic.
    ("a misparse", {"product": "ahw media"}, ["product"]),
    ("the other misparse", {"product": "pillars"}, ["product"]),
    # Nothing matched at all, so product falls back to the roll-up where other fields drop out.
    ("nothing matched", {"product": "something else"}, ["product"]),
    ("a field that is not product", {"country": "somewhere else"}, ["country"]),
)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("name", "ner_results", "fields"), _PRODUCT_CASES, ids=[c[0] for c in _PRODUCT_CASES]
)
@pytest.mark.skipif(not source_available(), reason="ask-genome-core is not checked out")
def test_product_and_misparsed_answers_match_the_source(name, ner_results, fields) -> None:
    """`product` is classified in 7 of the 30 onboarded configurations, so these branches are
    shared behaviour rather than one client's legacy."""

    del name
    source = source_functions(SOURCE_SRC / "model" / "data_filtering.py", ["_get_answer_dict"])[
        "_get_answer_dict"
    ]

    want: dict[str, Any] = {}
    for key, value in ner_results.items():
        want.update(source(_COLGUS_FRAME, key, [key], {key: value}))
    got = answer_dict_for(_COLGUS_FRAME, dict(ner_results), fields)

    assert got == want


@pytest.mark.unit
def test_the_product_cases_are_not_all_the_same_answer() -> None:
    """Guards the parity above: the fallback, the acceptance and the rejection must differ."""

    def resolve(answer: str) -> dict[str, Any]:
        return answer_dict_for(_COLGUS_FRAME, {"product": answer}, ["product"])

    assert resolve("optic white") == {"product": ["optic white"]}
    assert resolve("ahw") == {"product": ["ahw"]}
    assert resolve("ahw media") == {"product": ["overall"]}
    # A non-product field with no match drops out entirely instead.
    assert answer_dict_for(_COLGUS_FRAME, {"country": "nowhere"}, ["country"]) == {}


@pytest.mark.unit
def test_a_multi_word_group_resolves_by_code_and_by_name(tmp_path, monkeypatch) -> None:
    """Filtering and upstream's ranking look groups up by code, the GPT readout by the lower-cased
    name. With only one of them, 'cost per activity' resolved to nothing on one side or the other.
    Ranking is checked through process_data's own normaliser, on map_dict.csv's raw strings."""

    from ask_genome_agent.local import get_metric_configs
    from ask_genome_agent.vendor.ask_genome_core.model.readout_utils import normalize_map_dict

    client = tmp_path / "MULTI" / "1"
    client.mkdir(parents=True)
    pd.DataFrame(
        {"kpiName": ["CPM"], "domain": ["bi"], "standardizedMetricId": [2], "alias": ["CPM"]}
    ).to_csv(client / "metric_mapping.csv", index=False)
    pd.DataFrame(
        {"id": [2], "metricName": ["Cost per Activity"], "code": ["cost_per_activity"]}
    ).to_csv(tmp_path / "standardized_metrics.csv", index=False)
    monkeypatch.setenv("ASK_GENOME_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ASK_GENOME_CLIENT_CODE", "MULTI")
    monkeypatch.setenv("ASK_GENOME_MODEL_GROUP_ID", "1")

    configs = get_metric_configs("MULTI", 1)

    by_name = configs[configs["metricGroup"].str.lower().isin(["cost per activity"])]
    assert list(by_name["kpiName"]) == ["CPM"]
    row = {
        "mainMetric": "cost_per_activity",
        "rankMetric": "cost_per_activity",
        "sortMetric": "cost_per_activity",
        "relatedKPIs": '["cost_per_activity"]',
    }
    ranked = normalize_map_dict({"cost per": row}, configs)
    assert ranked["cost per"]["mainMetric"] == ["cpm"]
