"""Deterministic: applies the resolved filter to the client's real data.

The node itself is thin. It asks support/filtering.py to turn the model's answers into real
column values, support/time_filter.py to resolve the time range against the periods that exist,
and then support/narrowing.py for the chain ask-genome-core runs before the filter is applied.

What it keeps is a sample of the rows, the true count, and a recipe that rebuilds the whole
result on demand. The recipe is taken where the chain splits, after the two steps that only
produce dicts and before the ones that narrow the frame. See support/table.py.

Two dedupes run in here and neither is the open one. combine_filters drops rows that repeat in
full; the hierarchy step drops one figure recorded at several levels, keyed on `value`.

The open one: the raw files carry several model outputs, so one hierarchy can appear twice with
*different* values, and because both dedupes key on value those rows survive and are both counted.
Michael found ~28 in a slice of LINKEDIN/12 on 18 Sept, and Lizhe's stated handling is to dedupe
on the filter fields keeping the first row. Which columns count as filter fields was never pinned
down, and reading it as every column but `value` drops 6,284 rows from bi -- two orders of
magnitude more than he saw, so that reading is probably wrong. Lizhe also offered to trace it
through preprocessing, in which case it does not belong in an agent at all. Row counts stay
slightly high until that is settled.

Still deferred, and neither changes the shape of what this returns: the readout metadata bundle
beyond latest_time, and the `data_filtering_other_category` variants for clients running with
legacy preprocessing disabled.
"""

from __future__ import annotations

import logging
from typing import Any

from ask_genome_agent.config import load_narrowing_config, load_ner_data, load_time_indicators
from ask_genome_agent.nodes.question_understanding.support.filtering import (
    answer_dict_for,
    rejection_message_for,
)
from ask_genome_agent.nodes.question_understanding.support.narrowing import (
    build_ner_filter,
    map_keyword_in_query,
    narrow_and_apply,
    postprocess_filter,
)
from ask_genome_agent.nodes.question_understanding.support.table import (
    recipe_from_result,
    table_id,
)
from ask_genome_agent.nodes.question_understanding.support.time_filter import (
    latest_period,
    resolve_time_period,
)
from ask_genome_agent.state import AskGenomeState

logger = logging.getLogger(__name__)

# Rows kept for inspection in the diagnostics UI. Small because a result is unbounded and state
# is checkpointed into one capped document.
_SAMPLE_ROWS = 50


async def apply_data_filters(state: AskGenomeState) -> dict[str, Any]:
    subquery = state["plan"].subqueries[state["active_index"]]
    ner_data = load_ner_data()
    narrowing = load_narrowing_config()
    indicators = load_time_indicators()
    ner_state = dict(state["ner_state"])
    entry = dict(ner_state[subquery.id])
    source = entry["data_source"]

    frame = ner_data.df_bi if source["source"] == "bi" else ner_data.df_br
    fields = list(source["data_levels"]) + [
        f for fields_ in ner_data.level_type.values() for f in fields_
    ]

    answer_dict = answer_dict_for(frame, entry["ner_results"], fields)
    # Time is resolved against the data rather than taken at face value: an unstated range means
    # the most recent periods, and an explicit one snaps to real period boundaries.
    resolved_time = resolve_time_period(
        frame, entry["ner_results"], indicators, [source["intention"]]
    )
    entry["time_tag"] = resolved_time.pop("time_tag", "")
    answer_dict.update(resolved_time)

    ner_filter = build_ner_filter(source, answer_dict, entry["ner_results"])
    ner_filter, ner_results = postprocess_filter(
        ner_filter, entry["ner_results"], frame, indicators, narrowing.level_type
    )
    # Reported, never filtered on: how current the data is at one rung coarser than the
    # question. generate_readoutdata computes it here, before any narrowing.
    entry["latest_time"], entry["latest_period_type"] = latest_period(frame, ner_filter, indicators)

    frame, ner_filter, ignore_fields = map_keyword_in_query(
        frame, entry["extended_query"], ner_filter, list(source["ignore_fields"])
    )

    # The recipe boundary: nothing after this reads the question, and nothing before it depends
    # on which rows survive, so these four values are what a stored table replays from.
    narrowed = narrow_and_apply(
        frame, ner_filter, ner_results, ignore_fields, entry["spacy_filter"], narrowing
    )
    filtered = narrowed.frame
    recipe = recipe_from_result(
        source=source["source"],
        ner_filter=ner_filter,
        ner_results=ner_results,
        ignore_fields=ignore_fields,
        spacy_filter=entry["spacy_filter"],
        row_count=len(filtered),
    )

    rejection = rejection_message_for(narrowed.log)
    metric_column: list[Any] = (
        list(filtered["metric"].unique()) if "metric" in filtered.columns else []
    )
    has_main_metric = any(m in metric_column for m in source["main_metric"])
    if len(filtered) and not has_main_metric:
        rejection = (
            rejection + "\n\n" if rejection else ""
        ) + "No rows carry a metric this intention reports on."

    # What the filter came to, as the apply step saw it: the fields it did not skip. The recipe
    # holds the rest, which is machinery rather than something to show.
    entry["applied_filter"] = {
        field: values
        for field, values in narrowed.ner_filter.items()
        if isinstance(values, list)
        and field in filtered.columns
        and field not in narrowed.ignore_fields
    }
    entry["row_count"] = len(filtered)
    # The three levels this result reports at. The rows carry `record_level`, so the three
    # dataframes ask-genome-core returns are levels.split_by_level away. See support/levels.py.
    entry["data_levels"] = narrowed.data_levels
    entry["rejection_message"] = rejection
    # A sample, not the result: a LINKEDIN spend question matches ~96k rows, and inlining those
    # blew the checkpoint document's 16MB limit and failed the request. Nothing reads them yet.
    # When response generation needs the real rows they belong in the artifact store, with only a
    # reference held here.
    entry["records"] = filtered.head(_SAMPLE_ROWS).to_dict(orient="records")
    entry["records_truncated"] = len(filtered) > _SAMPLE_ROWS

    # A few hundred bytes that rebuild those rows on demand. See support/table.py.
    entry["table_recipe"] = recipe.model_dump(mode="json")
    entry["table_id"] = table_id(recipe)
    ner_state[subquery.id] = entry

    summary = f"{len(filtered)} rows matched for intention={source['intention']}"
    if rejection:
        summary += f"; {rejection.splitlines()[0]}"
    return {"ner_state": ner_state, "last_subquery_result": summary}
