"""The boundary between our postprocess contract and ask-genome-core's vendored process_data.

process_data itself is upstream's and is checked against upstream by the sync script, not
re-tested here. What is pinned is the translation both ways, and that the vendored copy imports
the way the agent imports it: from any working directory, with the agent's settings only.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from ask_genome_agent.nodes.postprocess.support import process_data_adapter as adapter
from ask_genome_agent.nodes.postprocess.support.contract import PostprocessRequest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / "packages/agents/ask-genome/src/ask_genome_agent/vendor/ask_genome_core"
UPSTREAM = Path(os.environ.get("ASK_GENOME_CORE_CHECKOUT", "/home/yvyas/new-ask-genome-core"))


def _request(query: str = "", **filter_overrides: Any) -> PostprocessRequest:
    table = pd.DataFrame(
        {
            "activity_group": ["paid search", None, None],
            "measure_group": [None, "brand", None],
            "measure": [None, None, "exact"],
            "value": [1.0, 2.0, 3.0],
            "record_level": ["activity_group", "measure_group", "measure"],
        }
    )
    return PostprocessRequest(
        table=table,
        ner_filter={"intention": ["margin roi"], **filter_overrides},
        ner_results={"intention": ["margin roi"], "time_tag": "snapshot"},
        data_levels=("activity_group", "measure_group", "measure"),
        source="bi",
        query=query,
    )


def _returned(**overrides: Any) -> tuple[Any, ...]:
    """process_data's fourteen values, in its order."""

    values: dict[str, Any] = {
        "context_str": "paid search gross roi is 2.4 in 2025.",
        "data": pd.DataFrame({"Tactics": ["Paid Search"], "ROI 2025": ["2.40"]}),
        "benchmark_data": pd.DataFrame(),
        "benchmark_str": "benchmark text",
        "planner_data": pd.DataFrame(),
        "spend_share_principle": pd.DataFrame(),
        "pretext": "Overall ROI is 3.1.",
        "pivot_biz_table": pd.DataFrame({"Tactics": ["Search"], "ROI 2025": ["2.40"]}),
        "pretext_table": pd.DataFrame(),
        "pretext_table_trend": pd.DataFrame(),
        "match": {
            "search_key": "N~Y~N~Y~N~Y",
            "detail_view": "detail_m",
            "agg_view": "detail_ag",
            "readout": "readout_ag, readout_m",
        },
        "principle_pretext": "",
        "overall_view": pd.DataFrame(),
        "readout_adj": True,
    }
    values.update(overrides)
    return tuple(values.values())


@pytest.fixture
def _upstream(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Stand-ins for process_data and its config loader, recording what they were handed."""

    seen: dict[str, Any] = {"returns": _returned()}

    def fake_process_data(client_code, model_group_id, readout_data, indicator, configs):  # type: ignore[no-untyped-def]
        seen.update(client=(client_code, model_group_id), readout_data=readout_data)
        return seen["returns"]

    def fake_build_readout(client_code, model_group_id, query, detail, aggregate, *rest):  # type: ignore[no-untyped-def]
        seen.update(readout_query=query, readout_detail=detail)
        if isinstance(seen.get("readout"), Exception):
            raise seen["readout"]
        return seen.get("readout", "GPT readout: roi 2.4.")

    monkeypatch.setattr(adapter, "process_data", fake_process_data)
    monkeypatch.setattr(adapter.insight_utils, "build_readout", fake_build_readout)
    monkeypatch.setattr(
        adapter.readout_utils, "get_metric_configs", lambda code, group: pd.DataFrame()
    )
    return seen


def test_the_rebuilt_table_reaches_upstream_as_its_three_level_frames(_upstream: dict) -> None:
    adapter.postprocess(_request(), "LINKEDIN", 12)
    readout_data = _upstream["readout_data"]
    assert readout_data.df_activity_group["value"].tolist() == [1.0]
    assert readout_data.df_measure_group["value"].tolist() == [2.0]
    assert readout_data.df_measure["value"].tolist() == [3.0]
    assert _upstream["client"] == ("LINKEDIN", 12)


def test_the_filter_keeps_its_own_values_over_the_defaults(_upstream: dict) -> None:
    """The defaults only fill keys generate_readoutdata sets and our filter does not carry."""

    adapter.postprocess(_request(rejection_message="one dimension dropped"), "LINKEDIN", 12)
    filters = _upstream["readout_data"].ner_filters
    assert filters["rejection_message"] == "one dimension dropped"
    assert filters["metadata_mode"] == "bypass"
    assert _upstream["readout_data"].ner_results["time_tag"] == "snapshot"


def test_period_bounds_reach_upstream_as_lists(_upstream: dict) -> None:
    """Upstream's time prompt answers with lists; ours answers with one string."""

    request = _request()
    request.ner_results.update(start="202501", end="202512", period_type="year")
    adapter.postprocess(request, "LINKEDIN", 12)
    results = _upstream["readout_data"].ner_results
    assert (results["start"], results["end"]) == (["202501"], ["202512"])
    assert results["period_type"] == "year", "only the period bounds change shape"


@pytest.mark.parametrize("value", ["irrelevant", "all", ["202501"]])
def test_bounds_already_in_upstream_s_shape_are_left_alone(value: Any) -> None:
    assert adapter.upstream_results({"start": value}, {})["start"] == value


@pytest.mark.parametrize(("answer", "coerced"), [("irrelevant", "na"), ("5", 5), ("na", "na")])
def test_how_many_reaches_upstream_as_filtering_coerced_it(answer: str, coerced: Any) -> None:
    """Live, a trend question on br answered nothing: every level raised int('irrelevant'),
    because the br path reads how_many from the results, where ours said 'irrelevant'."""

    results = adapter.upstream_results({"how_many": answer}, {"how_many": coerced})
    assert results["how_many"] == coerced


def test_intention_reaches_upstream_as_a_list() -> None:
    """Upstream indexes it: `ner_res_dict.get("intention")[0]`. A string would give "p"."""

    assert adapter.upstream_results({"intention": "planner"}, {})["intention"] == ["planner"]
    assert adapter.upstream_results({"intention": ["planner"]}, {})["intention"] == ["planner"]


def test_upstream_s_time_check_no_longer_reports_year_0() -> None:
    """Live, every answer opened with "You requested data for year 0": process_data takes the min
    over `start`, and over a string that is its smallest digit. Runs upstream's own check."""

    from ask_genome_agent.vendor.ask_genome_core.model import readout_utils

    ner_filter = {
        "time": ["year 2024", "year 2025"],
        "period_type": ["year"],
        "latest_time": 202606,
        "latest_period_type": "quarter",
    }
    data = pd.DataFrame({"end": [202412, 202512]})
    raw = {"start": "202501", "end": "202512", "period_type": "year"}

    assert "year 0" in readout_utils._validate_time_range(raw, ner_filter, data)
    assert "year 0" not in readout_utils._validate_time_range(
        adapter.upstream_results(raw, ner_filter), ner_filter, data
    )


def test_upstream_s_values_land_in_our_contract(_upstream: dict) -> None:
    result = adapter.postprocess(_request(), "LINKEDIN", 12)

    assert result.response_context == "paid search gross roi is 2.4 in 2025."
    assert result.fixtext == "Overall ROI is 3.1."
    assert result.benchmark_text == "benchmark text"
    assert result.principle_pretext == ""
    assert "Paid Search" in result.detail_table_text
    assert "Search" in result.aggregate_table_text
    assert not result.is_planner_answer
    assert result.warnings == ()


def test_the_model_reads_the_gpt_readout_built_from_process_data_s_tables(
    _upstream: dict,
) -> None:
    result = adapter.postprocess(_request("How did search do?"), "LINKEDIN", 12)
    assert result.insight_context == "GPT readout: roi 2.4."
    assert _upstream["readout_query"] == "How did search do?"
    assert _upstream["readout_detail"]["Tactics"].tolist() == ["Paid Search"]
    assert result.answerable


def test_a_table_free_readout_is_asked_the_page_s_fixed_query(_upstream: dict) -> None:
    result = adapter.postprocess(_request("How did search do?"), "LINKEDIN", 12)
    assert result.insight_query == "Generate insights with given context."

    _upstream["returns"] = _returned(readout_adj=False)
    result = adapter.postprocess(_request("How did search do?"), "LINKEDIN", 12)
    assert result.insight_query == "How did search do?"


def test_a_failed_gpt_readout_falls_back_to_the_readout_and_says_so(_upstream: dict) -> None:
    _upstream["readout"] = KeyError("Metric")
    result = adapter.postprocess(_request(), "LINKEDIN", 12)
    assert result.insight_context == "paid search gross roi is 2.4 in 2025."
    (warning,) = result.warnings
    assert (warning.stage, warning.code) == ("insight", "insight_fallback")
    assert "KeyError" in warning.message


def test_a_planner_question_reads_its_scenarios_as_json(
    _upstream: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the rules for its kind of question, from insight_instruction.csv, in file order."""

    rules = pd.DataFrame(
        {
            "configName": ["system", "specific", "general", "benchmark"],
            "instruction": ["System rules.", "Specific rules.", "General rules.", "Bench rules."],
        }
    )
    monkeypatch.setattr(adapter.readout_utils, "load_insight_instructions", lambda *a, **k: rules)
    monkeypatch.setattr(
        adapter.readout_utils, "build_planner_specific_context", lambda *a: '{"scenario": 1}'
    )
    monkeypatch.setattr(adapter.readout_utils, "build_planner_general_context", lambda *a: "")
    _upstream["returns"] = _returned(planner_data=pd.DataFrame({"Driver": ["Audio"]}))
    request = _request("Should I move spend?", intention=["planner"], core_dimension=["audio"])

    result = adapter.postprocess(request, "LINKEDIN", 12)
    assert result.insight_context == '{"scenario": 1}'
    assert result.insight_query == (
        "System rules.\n\nSpecific rules.\n\nUser question: Should I move spend?"
    )
    assert "readout_query" not in _upstream, "planner questions do not use build_readout"


def test_without_tables_there_is_nothing_for_the_model(_upstream: dict) -> None:
    """The page's own condition: no detail, aggregate or planner table, no model call."""

    _upstream["returns"] = _returned(data=pd.DataFrame(), pivot_biz_table=pd.DataFrame())
    assert not adapter.postprocess(_request(), "LINKEDIN", 12).answerable


def test_the_lookup_row_explains_the_selection(_upstream: dict) -> None:
    selection = adapter.postprocess(_request(), "LINKEDIN", 12).selection_metadata
    assert selection.search_key == "N~Y~N~Y~N~Y"
    assert (selection.detail_level, selection.aggregate_level) == ("m", "ag")
    assert selection.readout_levels == ("ag", "m")
    assert selection.degraded_levels == ()


def test_a_planner_table_makes_it_a_planner_answer(_upstream: dict) -> None:
    """The Streamlit app's own rule for putting the reference-only notice after the answer."""

    _upstream["returns"] = _returned(planner_data=pd.DataFrame({"Driver": ["Audio"]}))
    assert adapter.postprocess(_request(), "LINKEDIN", 12).is_planner_answer


def test_nothing_produced_is_flagged(_upstream: dict) -> None:
    _upstream["returns"] = _returned(context_str="", pretext="", match={})
    result = adapter.postprocess(_request(), "LINKEDIN", 12)
    assert result.is_empty
    assert {w.code for w in result.warnings} == {"empty_result", "no_lookup_match"}


def test_a_pretext_alone_is_not_empty(_upstream: dict) -> None:
    """The planner's "cannot answer" message arrives as a pretext with no readout."""

    _upstream["returns"] = _returned(context_str="", pretext="Please use the GPSE Planner.")
    assert not adapter.postprocess(_request(), "LINKEDIN", 12).is_empty


# ------------------------------------------------------------- the vendored copy itself


def test_the_vendored_copy_imports_from_anywhere_with_the_agent_s_settings_only(
    tmp_path: Path,
) -> None:
    """readout.py reads a YAML file and builds its dispatch tables at import. Upstream reads it
    relative to the working directory, which would break the agent anywhere but upstream's repo."""

    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("DATA_DIR", "CLIENT_CODE", "MODEL_GROUP_ID")
    }
    completed = subprocess.run(
        [sys.executable, "-c", "import ask_genome_agent.vendor.ask_genome_core.model.readout"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]


def test_no_vendored_file_reads_the_gateway_s_client_code() -> None:
    """Bare CLIENT_CODE belongs to the enterprise LLM gateway. The sync renames every read."""

    bare = re.compile(r"""os\.getenv\(\s*["'](DATA_DIR|CLIENT_CODE|MODEL_GROUP_ID)["']""")
    offenders = [
        f"{path.relative_to(VENDOR)}:{number}"
        for path in VENDOR.rglob("*.py")
        for number, line in enumerate(path.read_text().splitlines(), 1)
        if bare.search(line)
    ]
    assert not offenders


@pytest.mark.skipif(
    not (UPSTREAM / ".git").exists(), reason="ask-genome-core is not checked out beside this repo"
)
def test_the_vendored_copy_matches_its_recorded_commit() -> None:
    """A hand edit to a vendored file would be lost on the next sync, so it is caught here."""

    completed = subprocess.run(
        [sys.executable, "scripts/sync_ask_genome_core.py", "--source", str(UPSTREAM), "--check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
