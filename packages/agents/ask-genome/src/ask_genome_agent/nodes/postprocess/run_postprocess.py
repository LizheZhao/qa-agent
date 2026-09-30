"""Deterministic: turns one subquery's filtered table into the text response generation reads.

Thin on purpose: rebuild the table from its recipe, hand it to ask-genome-core's own
process_data through support/process_data_adapter.py, and write back a bounded summary.
"""

from __future__ import annotations

import logging
from typing import Any

from ask_genome_agent.config import load_narrowing_config
from ask_genome_agent.nodes.postprocess.support.contract import (
    PostprocessRequest,
    PostprocessResult,
)
from ask_genome_agent.nodes.postprocess.support.process_data_adapter import postprocess
from ask_genome_agent.nodes.question_understanding.support.table import (
    StaleTableRecipeError,
    TableRecipe,
    rehydrate_result,
)
from ask_genome_agent.state import AskGenomeState

logger = logging.getLogger(__name__)

# The context is prompt-sized, but state is checkpointed into one capped document, so it is
# truncated here rather than left to fail the write.
_MAX_CONTEXT = 24_000
# The GPT readout is the model's whole input and runs to ~250k characters for a spend question, so
# it gets far more room. Well inside the 16MB checkpoint even for several subqueries.
_MAX_INSIGHT_CONTEXT = 600_000
_MAX_TABLE_TEXT = 8_000
# What a dependent subquery is handed to resolve a pronoun with. Long enough to name the driver
# and its figure, short enough not to crowd the resolution prompt.
_MAX_CARRIED_READOUT = 1_500


async def run_postprocess(state: AskGenomeState) -> dict[str, Any]:
    subquery = state["plan"].subqueries[state["active_index"]]
    ner_state = dict(state["ner_state"])
    entry = dict(ner_state[subquery.id])

    recipe_data = entry.get("table_recipe")
    if not recipe_data:
        # Nothing was filtered: an out-of-scope or rejected subquery. Not an error.
        return {}

    recipe = TableRecipe.model_validate(recipe_data)

    try:
        # The result, not just the frame: the recipe's own filter predates the spaCy merge and
        # lacks core_dimension and the tagging fields the per-level pipeline branches on.
        rebuilt = rehydrate_result(recipe)
    except StaleTableRecipeError as error:
        # Loud: the recipe no longer describes the table its id promises.
        logger.warning("postprocess could not rebuild table for %s: %s", subquery.id, error)
        entry["postprocess_error"] = f"stale table recipe: {error}"
        ner_state[subquery.id] = entry
        return {"ner_state": ner_state}

    request = PostprocessRequest(
        table=rebuilt.frame,
        ner_filter=_readout_filter(rebuilt, entry),
        # generate_readoutdata copies it onto the results too; br growth reads it from there.
        ner_results={**rebuilt.ner_results, "time_tag": _time_tag(entry)},
        data_levels=tuple(entry.get("data_levels", ())),
        source=recipe.source,
        query=subquery.query,
    )

    result = postprocess(request, recipe.client_code, recipe.model_group_id)
    if result.is_empty:
        # Postprocessing ran and produced nothing. Without this the filtering step's row count
        # stands as the subquery's outcome and the turn reports a number as though it answered.
        entry["postprocess_error"] = _empty_reason(result)
    entry.update(_summarise(result))
    ner_state[subquery.id] = entry
    return {
        "ner_state": ner_state,
        "last_subquery_result": _summary_line(result, entry.get("row_count", 0)),
    }


def _readout_filter(rebuilt: Any, entry: dict[str, Any]) -> dict[str, Any]:
    """The filter as postprocessing expects it.

    Ported from the tail of filter_generator.py:generate_readoutdata, which enriches the filter
    once the rows are known. `time` and `period_type` come off the filtered frame, so they say
    what the answer covers rather than what was asked. `level` carries the tagging columns the
    halo filter and overall-figure branch read.

    None of it belongs in the recipe: the recipe is filtering's input, these are its output.
    """

    frame = rebuilt.frame
    return {
        **rebuilt.ner_filter,
        "level": load_narrowing_config().level_type,
        "time": frame["time"].unique().tolist() if "time" in frame.columns else [],
        "period_type": (
            frame["period_type"].unique().tolist() if "period_type" in frame.columns else []
        ),
        "latest_time": int(entry.get("latest_time") or 0),
        "latest_period_type": entry.get("latest_period_type", "irrelevant"),
        "time_tag": _time_tag(entry),
    }


def _time_tag(entry: dict[str, Any]) -> str:
    """Snapshot, change, multiple periods or trend. Filtering stores it beside the filter, but the
    source keeps it in the filter, and the GPT readout reads it from there to pick its tables."""

    return str(entry.get("time_tag") or "snapshot")


def _summarise(result: PostprocessResult) -> dict[str, Any]:
    """What goes into state: text and metadata, bounded, no frames."""

    metadata = result.selection_metadata
    return {
        "response_context": result.response_context[:_MAX_CONTEXT],
        "readout": result.readout[:_MAX_CONTEXT],
        "aggregate_table_text": result.aggregate_table_text[:_MAX_TABLE_TEXT],
        "detail_table_text": result.detail_table_text[:_MAX_TABLE_TEXT],
        "benchmark_text": result.benchmark_text,
        "fixtext": result.fixtext,
        # None where nothing generated it; the boundary that needs a string converts it there.
        "principle_pretext": result.principle_pretext,
        "is_planner_answer": result.is_planner_answer,
        "insight_query": result.insight_query[:_MAX_INSIGHT_CONTEXT],
        "insight_context": result.insight_context[:_MAX_INSIGHT_CONTEXT],
        "insight_context_truncated": len(result.insight_context) > _MAX_INSIGHT_CONTEXT,
        "answerable": result.answerable,
        "selection": {
            "detail_level": metadata.detail_level,
            "aggregate_level": metadata.aggregate_level,
            "overall_level": metadata.overall_level,
            "readout_levels": list(metadata.readout_levels),
            "search_key": metadata.search_key,
            "degraded_levels": list(metadata.degraded_levels),
        },
        "postprocess_warnings": [
            {"stage": w.stage, "code": w.code, "message": w.message, "level": w.level}
            for w in result.warnings
        ],
    }


def _empty_reason(result: PostprocessResult) -> str:
    """Why nothing came back. process_data re-raises any level failure other than "no valid
    data", so by here every level was simply empty, or the lookup had no row for the result."""

    if not result.selection_metadata.matched:
        return "no result_lookup row matched the levels that produced rows"
    return "no level produced a readout"


def _summary_line(result: PostprocessResult, row_count: int) -> str:
    """What a dependent subquery gets if generate_response writes no answer.

    The readout leads: resolve_dependency hands this to an LLM to work out what "its" refers to,
    and "answered at level mg from 123 rows" names no channel.
    """

    if result.is_empty:
        return f"{row_count} rows matched but no level produced a readout"

    level = result.selection_metadata.detail_level or "?"
    provenance = f"(answered at level {level} from {row_count} rows"
    if result.selection_metadata.degraded_levels:
        provenance += (
            f"; selection degraded: {', '.join(result.selection_metadata.degraded_levels)}"
        )
    provenance += ")"

    carried = result.readout.strip()[:_MAX_CARRIED_READOUT]
    return f"{carried}\n\n{provenance}" if carried else provenance
