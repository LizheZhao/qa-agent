"""Runs ask-genome-core's own process_data and returns our postprocess contract.

The postprocessing logic still changes with client requests, so it is not re-ported. The
vendored copy under vendor/ask_genome_core, written by scripts/sync_ask_genome_core.py, is the
implementation, and this module only translates at the boundary:

  in   the rebuilt table, split into the three level frames upstream expects
  out  text and selection metadata, never frames, since graph state is checkpointed into one
       capped document

It also builds what the GPT readout gives the model, while the tables still exist: the text
pages/insights_only.py assembles from process_data's output, which generate_response then sends.

`readout` is the whole DeepSeek context for now. Upstream returns no separate readout without the
scaffolding it adds, so a dependent subquery sees that text rather than the GPT one.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from ask_genome_agent.nodes.postprocess.support.contract import (
    PostprocessRequest,
    PostprocessResult,
    PostprocessWarning,
    SelectionMetadata,
)
from ask_genome_agent.vendor.ask_genome_core.data.data_interface import ReadoutData
from ask_genome_agent.vendor.ask_genome_core.insight_generation import utils as insight_utils
from ask_genome_agent.vendor.ask_genome_core.model import readout_utils
from ask_genome_agent.vendor.ask_genome_core.model.readout import process_data

logger = logging.getLogger(__name__)

_LEVELS = ("ag", "mg", "m")

# What the page asks in place of the question when the readout carries no table.
_FIXED_QUERY = "Generate insights with given context."

# Keys generate_readoutdata sets that process_data does not read. Present so the filter has the
# shape upstream builds; a filter that already carries them wins.
_FILTER_DEFAULTS = {"rejection_message": "", "metadata_mode": "bypass"}

_PERIOD_BOUNDS = ("start", "end")
_UNSPECIFIED = ("irrelevant", "all")


def postprocess(
    request: PostprocessRequest, client_code: str, model_group_id: int
) -> PostprocessResult:
    frames = split_levels(request)
    readout_data = ReadoutData(
        ner_filters={**_FILTER_DEFAULTS, **request.ner_filter},
        ner_results=upstream_results(request.ner_results, request.ner_filter),
        df_activity_group=frames["ag"],
        df_measure_group=frames["mg"],
        df_measure=frames["m"],
    )
    metric_configs = readout_utils.get_metric_configs(client_code, model_group_id)
    (
        context,
        detail,
        _benchmark_data,
        benchmark_text,
        planner_data,
        _spend_share,
        pretext,
        aggregate,
        _pretext_table,
        _pretext_trend,
        match,
        principle_pretext,
        _overall,
        no_table_readout,
    ) = process_data(client_code, model_group_id, readout_data, None, metric_configs)

    aggregate_text, detail_text, _, _ = readout_utils.table_to_text(
        aggregate, detail, take_head=False
    )
    context, pretext = str(context or ""), str(pretext or "")
    warnings = list(_warnings(context, pretext, match))
    insight_query, insight_context = insight_input(
        request.query,
        readout_data,
        context,
        detail,
        aggregate,
        planner_data,
        bool(no_table_readout),
        client_code,
        model_group_id,
        metric_configs,
        warnings,
    )
    return PostprocessResult(
        response_context=context,
        readout=context,
        aggregate_table_text=str(aggregate_text or ""),
        detail_table_text=str(detail_text or ""),
        selection_metadata=selection(match),
        principle_pretext=str(principle_pretext or ""),
        benchmark_text=str(benchmark_text or ""),
        fixtext=pretext,
        # The Streamlit app's own test for which way round the answer and notice go.
        is_planner_answer=isinstance(planner_data, pd.DataFrame) and not planner_data.empty,
        insight_query=insight_query,
        insight_context=insight_context,
        answerable=not (_empty(detail) and _empty(aggregate) and _empty(planner_data)),
        warnings=tuple(warnings),
    )


def insight_input(
    query: str,
    readout_data: Any,
    context: str,
    detail: pd.DataFrame,
    aggregate: pd.DataFrame,
    planner_data: pd.DataFrame,
    no_table_readout: bool,
    client_code: str,
    model_group_id: int,
    metric_configs: pd.DataFrame,
    warnings: list[PostprocessWarning],
) -> tuple[str, str]:
    """What the model is asked, and what it reads. Ported from pages/insights_only.py, which
    holds this as inline page code rather than a function, so it is the one part not vendored.

    A planner question reads its scenarios as JSON, or the readout and scenario tables where no
    JSON context can be built. Anything else reads build_readout, falling back to the readout
    when that fails, as the page does.
    """

    insight_query = _FIXED_QUERY if no_table_readout else query
    ner_filter = readout_data.ner_filters
    source = ""
    if "planner" in ner_filter["intention"]:
        instructions = readout_utils.load_insight_instructions(
            client_code, model_group_id, intention="planner"
        )
        specific = general = ""
        if ner_filter.get("core_dimension"):
            specific = readout_utils.build_planner_specific_context(
                client_code, int(model_group_id), readout_data
            )
        if not specific:
            general = readout_utils.build_planner_general_context(
                client_code, int(model_group_id), readout_data
            )
        # The shared rules, the row for this kind of question, and the benchmark row when the
        # context compares against one, in file order.
        names = ["system", "specific" if specific else "general"]
        if '"benchmark_comparison"' in (specific or general):
            names.append("benchmark")
        if not instructions.empty and "configName" in instructions.columns:
            chosen = instructions[instructions["configName"].isin(names)]
            instructions = chosen if not chosen.empty else instructions
        rules = (
            "\n\n".join(instructions["instruction"].dropna().tolist())
            if not instructions.empty
            else ""
        )
        json_context = specific or general
        if json_context:
            insight_query = (f"{rules}\n\n" if rules else "") + "User question: " + query
            source = json_context
        else:
            if rules:
                insight_query = f"{rules}\n\n{insight_query}"
            source = _planner_tables(context, planner_data, ner_filter, client_code, model_group_id)
    else:
        try:
            source = insight_utils.build_readout(
                client_code, model_group_id, query, detail, aggregate, readout_data, metric_configs
            )
        except Exception as error:  # the page's own fallback, recorded rather than shown inline
            logger.warning("build_readout failed, falling back to the readout: %s", error)
            warnings.append(
                PostprocessWarning(
                    stage="insight",
                    code="insight_fallback",
                    message=f"{type(error).__name__}: {error}",
                )
            )
    return insight_query, str(source or context)


def _planner_tables(
    context: str,
    planner_data: pd.DataFrame,
    ner_filter: dict[str, Any],
    client_code: str,
    model_group_id: int,
) -> str:
    """The readout plus the scenario summary and driver detail, for a planner question no JSON
    context could be built for. The summary is narrowed to the business units named."""

    _, summary = readout_utils.load_diminishing_returns_data(client_code, model_group_id)
    units = ner_filter.get("business_unit_for_kpi", [])
    units = [units] if isinstance(units, str) else list(units)
    named = [v.lower() for v in units if v not in ("irrelevant", "all", "overall")]
    parts = [context] if context else []
    if not summary.empty:
        if named and "business_unit_for_kpi" in summary.columns:
            summary = summary[summary["business_unit_for_kpi"].str.lower().isin(named)]
        if not summary.empty:
            parts.append("=== SCENARIO SUMMARY ===\n" + summary.to_string(index=False))
    if not planner_data.empty:
        parts.append("=== DRIVER DETAIL ===\n" + planner_data.to_string(index=False))
    return "\n\n".join(parts)


def _empty(frame: Any) -> bool:
    return not isinstance(frame, pd.DataFrame) or frame.empty


def upstream_results(ner_results: dict[str, Any], ner_filter: dict[str, Any]) -> dict[str, Any]:
    """The results in the shape ask-genome-core's own question understanding gives them.

    Its time prompt answers `{"start": ["YYYYMM"], "end": ["YYYYMM"]}`, and process_data takes the
    min and max over those lists. Ours come back from a tool schema as one "YYYYMM" string, which
    process_data would walk character by character and report as "data for year 0".

    Its trend prompt answers `how_many` as a number or 'na', and the br path int()s anything else.
    Ours can answer 'irrelevant', so the filter's copy is used: filtering already coerced the same
    answer to that vocabulary.

    `intention` is read as a list and indexed, so a bare string would silently become its first
    letter.
    """

    results = dict(ner_results)
    for key in _PERIOD_BOUNDS:
        value = results.get(key)
        if isinstance(value, str) and value not in _UNSPECIFIED:
            results[key] = [value]
    if "how_many" in results and "how_many" in ner_filter:
        results["how_many"] = ner_filter["how_many"]
    if isinstance(results.get("intention"), str):
        results["intention"] = [results["intention"]]
    return results


def split_levels(request: PostprocessRequest) -> dict[str, pd.DataFrame]:
    """The rebuilt table as upstream's three level frames, coarsest first.

    The table arrives as one frame tagged by `record_level`, which is ask-genome-core's three
    dataframes held as one.
    """

    table = request.table
    frames: dict[str, pd.DataFrame] = {}
    for key, level in zip(_LEVELS, request.data_levels, strict=False):
        if "record_level" in table.columns:
            frames[key] = table[table["record_level"] == level].reset_index(drop=True)
        else:
            frames[key] = table
    for key in _LEVELS:
        frames.setdefault(key, table.iloc[0:0])
    return frames


def selection(match: Any) -> SelectionMetadata:
    """Which level answered, read off the result_lookup row upstream matched."""

    row = dict(match) if isinstance(match, dict) else {}
    readout = str(row.get("readout") or "")
    return SelectionMetadata(
        detail_level=_level(row.get("detail_view")),
        aggregate_level=_level(row.get("agg_view")),
        overall_level=_level(row.get("overall_view")),
        readout_levels=tuple(_level(part) for part in readout.split(",") if part.strip()),
        search_key=str(row.get("search_key") or ""),
        matched=row,
    )


def _level(view: Any) -> str:
    """'detail_m' -> 'm'. The lookup names a view by its kind and level."""

    text = str(view or "").strip()
    return text.rsplit("_", 1)[-1] if "_" in text else text


def _warnings(context: str, pretext: str, match: Any) -> tuple[PostprocessWarning, ...]:
    warnings: list[PostprocessWarning] = []
    if not match:
        warnings.append(
            PostprocessWarning(
                stage="selection",
                code="no_lookup_match",
                message="no result_lookup row matched the levels that produced rows",
            )
        )
    if not context.strip() and not pretext.strip():
        warnings.append(
            PostprocessWarning(
                stage="level_pipeline", code="empty_result", message="no level produced a readout"
            )
        )
    return tuple(warnings)
