"""Which granularity each filtered row reports at, and which three to report on.

The client's data stacks several granularities in one table: a row can describe a business
driver, a channel, or a single measure, and the hierarchy columns it leaves null are what say
which. `record_levels` reads that off each row, `report_levels` picks the three worth reporting,
and `dedup_levels` drops the same figure where it appears at more than one of them.

ask-genome-core ends by splitting the result into three dataframes, one per level. That split is
a pure partition on `record_level` -- `split_by_level` below is the whole of it -- so this package
keeps one frame carrying the column and the chosen triple beside it. The three frames are one
line away for whoever wants them, and the table recipe stays a single table.

Not to be confused with the duplicate rows Michael raised. `dedup_levels` keys on `value`, so two
rows with the same dimensions and different values both survive it; it resolves one figure
appearing at several levels, which is a different thing.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

# The full hierarchy, coarsest first. Ported verbatim from _get_level / get_report_levels.
POSSIBLE_LEVELS = [
    "business_driver",
    "business_driver_detail",
    "activity_group",
    "measure_group",
    "measure",
]

_REPORT_TARGET = 3


def record_levels(df: pd.DataFrame, levels: list[str] | None = None) -> pd.DataFrame:
    """Ported from data_filtering.py:_get_level. Tags each row with the deepest hierarchy column it
    carries; later columns win, so the order of `levels` defines depth.
    """

    levels = levels or POSSIBLE_LEVELS
    tagged = df.copy()
    tagged["record_level"] = "invalid"
    for level in levels:
        tagged.loc[~pd.isna(tagged[level]) & (tagged[level] != "invalid"), "record_level"] = level
    return tagged


def levels_to_target(possible_levels: list[str], valid: list[str], target: int = 3) -> list[str]:
    """Ported verbatim from data_filtering.py:get_report_levels_to_target. Pads a short run out to
    three rungs, reaching towards the coarser end first, so the report is always three deep.
    """

    if len(valid) >= target:
        return valid[-target:]

    first_idx = possible_levels.index(valid[0])
    last_idx = possible_levels.index(valid[-1])
    needed = target - len(valid)

    front_available = possible_levels[:first_idx]
    take_front = min(len(front_available), needed)
    front = front_available[-take_front:] if take_front > 0 else []
    needed -= take_front

    back = possible_levels[last_idx + 1 : last_idx + 1 + needed]
    return front + valid + back


def most_granular_level(possible_levels: list[str], valid: list[str]) -> list[str]:
    """Ported verbatim from data_filtering.py:most_granular_level."""

    rank = {level: index for index, level in enumerate(possible_levels)}
    candidates = [c for c in valid if c in rank]
    return [max(candidates, key=lambda c: rank[c])]


def report_levels(
    df: pd.DataFrame, level: list[str], ner_filter: dict[str, Any], is_halo_tagging: bool
) -> list[str]:
    """Ported from data_filtering.py:get_report_levels.

    bi keeps the levels the intention came with. br takes the deepest captured term's column as
    the detail rung and builds back from there.
    """

    data_source = ner_filter.get("data", "bi")
    core_dimensions = ner_filter.get("core_dimension", {})
    tagged = record_levels(df)
    levels_in_data = [x for x in POSSIBLE_LEVELS if x in tagged["record_level"].unique()]

    if data_source == "bi":
        return level
    if is_halo_tagging:
        return levels_to_target(POSSIBLE_LEVELS, levels_in_data, target=_REPORT_TARGET)
    if not core_dimensions:
        return level

    has_bdd = False
    all_filter_levels: set[str] = set()
    for filters in core_dimensions.values():
        filter_levels = [
            key
            for term_filter in filters
            for key in term_filter
            if key in levels_in_data and "br" in term_filter.get("source", "na")
        ]
        all_filter_levels.update(filter_levels)
        deepest = next((x for x in levels_in_data[::-1] if x in filter_levels), level[-1])
        if deepest == "business_driver_detail":
            has_bdd = True

    if not all_filter_levels:
        all_filter_levels = set(levels_in_data)
    detail_level = most_granular_level(POSSIBLE_LEVELS, list(all_filter_levels))
    result = levels_to_target(POSSIBLE_LEVELS, detail_level, target=_REPORT_TARGET)

    if "business_driver" not in result and has_bdd and "business_driver_detail" not in result:
        result = ["business_driver_detail", *result[1:]]
    return result


def dedup_levels(
    df: pd.DataFrame, level: list[str], level_type: dict[str, list[str]]
) -> pd.DataFrame:
    """Ported from data_filtering.py:_dedup_df. Drops rows outside the three levels, then one figure
    recorded at several of them, keeping the most granular.

    `value` is in the key, so the unresolved same-dimensions-different-value duplicates survive.
    """

    hierarchy_levels = list(level_type.get("general_level", [])) + list(
        level_type.get("halo_level", [])
    )
    tagged = record_levels(df)
    tagged["record_level"] = tagged["record_level"].map({k: k for k in level}).fillna("invalid")
    tagged = tagged[tagged["record_level"].isin(level)]

    priority = {name: index for index, name in enumerate(level)}
    return (
        tagged.assign(_rank=tagged["record_level"].map(priority))
        .sort_values("_rank")
        .drop_duplicates(subset=[*hierarchy_levels, "time", "metric", "value"], keep="last")
        .drop(columns="_rank")
    )


def split_by_level(
    df: pd.DataFrame, level: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Ported from data_filtering.py:_get_level_answer. The whole of the source's three-dataframe
    result: a partition, so holding one frame and calling this where the three are wanted loses
    nothing.
    """

    return tuple(  # type: ignore[return-value]
        df[df["record_level"] == name].reset_index(drop=True).copy() for name in level
    )
