"""Deterministic: turns the resolved intention into which data to read and which metrics count.

Ports filter_generator.py's get_metric_by_intention / get_data_level_by_intention and the
intention half of filter_generator_refactor.py:map_filter_to_value.

The dataset is chosen per intention, not per client: 'margin roi' reads bi while 'source of
change' reads br, for the same question shape. That is why both frames are loaded.

One divergence. The source raises ValueError("Intention is out-of-scope, question rejected") when
the intention is unusable, which takes the whole request down with it. Out-of-scope is a real,
expected answer here, and the user picked it from a clarification list in some cases, so it ends
the turn with a plain sentence instead. Same reasoning as the retry-exhaustion and coverage
guards: an outcome the design anticipates is not a crash.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import pandas as pd

from ask_genome_agent.config import NerData, NerDataConfig, load_ner_data, load_ner_data_config
from ask_genome_agent.state import AskGenomeState

logger = logging.getLogger(__name__)

# Ported from map_filter_to_value's _check_intention_rejection call.
_REJECTION_LABELS = frozenset({"none", "out-of-scope"})


def is_rejected(intentions: list[str], available: list[str]) -> bool:
    """Ported verbatim from filter_generator.py:_check_intention_rejection."""

    if any(x for x in intentions if x in _REJECTION_LABELS):
        return True
    return not any(x in available for x in intentions)


def metrics_for_intention(
    intention: str, ner_data_config: NerDataConfig, metric_configs: pd.DataFrame
) -> dict[str, Any]:
    """Ported from filter_generator.py:get_metric_by_intention.

    A metric group names a set of real kpiName values in the client's data. `metric` is everything
    the intention may read, `main_metric` is what has to be present for the answer to mean
    anything. The source's temporary in-place rewrite of metric_info is dropped: it mutated the
    cached config, which here is shared across requests.
    """

    metric_info = ner_data_config.metric_info[intention]
    lookup: defaultdict[str, list[str]] = defaultdict(list)
    for group, kpi in zip(metric_configs["code"], metric_configs["kpiName"], strict=False):
        lookup[str(group).lower()].append(str(kpi).lower())

    unknown = [m for m in metric_info["metric"] if not lookup.get(m)]
    if unknown:
        logger.warning("intention %r names metric groups with no rows: %s", intention, unknown)

    return {
        "data": metric_info["data"],
        "metric": [k for m in metric_info["metric"] for k in lookup[m]],
        "main_metric": [k for m in metric_info["mainMetric"] for k in lookup[m]],
    }


def data_for_intention(
    ner_data_config: NerDataConfig, ner_data: NerData, source: str
) -> tuple[list[str], list[str], pd.DataFrame]:
    """Ported verbatim from filter_generator.py:get_data_level_by_intention."""

    levels = ner_data_config.data_levels
    if source == "bi":
        return list(levels["biLevels"]), [], ner_data.df_bi
    return list(levels["brLevels"]), list(levels["brLevelsToIgnore"]), ner_data.df_br


async def resolve_data_source(state: AskGenomeState) -> dict[str, Any]:
    subquery = state["plan"].subqueries[state["active_index"]]
    ner_data = load_ner_data()
    ner_data_config = load_ner_data_config()
    ner_state = dict(state["ner_state"])
    entry = dict(ner_state[subquery.id])

    raw = entry["ner_results"].get("intention", "none")
    intentions = [raw] if isinstance(raw, str) else list(raw)

    if is_rejected(intentions, list(ner_data_config.metric_info)):
        logger.info("intention %s is out of scope for this client", intentions)
        entry["data_source"] = None
        entry["out_of_scope"] = True
        ner_state[subquery.id] = entry
        return {"ner_state": ner_state}

    # The source takes the first intention and ignores the rest, so a multi-intention answer
    # silently resolves to one. Kept, because the clarification for intention is single-select.
    intention = next(x for x in intentions if x in ner_data_config.metric_info)
    metrics = metrics_for_intention(intention, ner_data_config, ner_data.metric_configs)
    data_levels, ignore_fields, frame = data_for_intention(
        ner_data_config, ner_data, metrics["data"]
    )

    entry["out_of_scope"] = False
    entry["data_source"] = {
        "intention": intention,
        "source": metrics["data"],
        "metric": metrics["metric"],
        "main_metric": metrics["main_metric"],
        "data_levels": data_levels,
        "ignore_fields": ignore_fields,
        # The frame itself never enters state: observability's capture raises on a DataFrame and
        # then marks this node's whole state delta unsupported, so the row count stands in for it.
        "row_count_before_filtering": len(frame),
    }
    ner_state[subquery.id] = entry
    return {"ner_state": ner_state}


def after_data_source(state: AskGenomeState) -> str:
    """Out-of-scope questions skip filtering; there is nothing to filter against."""

    subquery = state["plan"].subqueries[state["active_index"]]
    entry = state["ner_state"][subquery.id]
    return "write_context" if entry.get("out_of_scope") else "apply_data_filters"
