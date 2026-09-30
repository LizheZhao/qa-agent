"""The filter primitives: turning answers into column values, and applying them.

Ports data_filtering.py's _get_answer_dict, _minimal_conflicting_subset and
_apply_answer_dict_with_log, plus filter_generator_refactor.py:log_to_message.

Two layers, and the split matters. The first translates what the model answered into values that
exist in this client's columns: 'all' means every real value, 'irrelevant' means the 'overall'
roll-up row, 'specific' means the same as 'all' until spaCy narrows it. Only then does the second
intersect those values with the data, one field at a time.

Filters apply as a running conjunction, and a field that would empty the result is skipped rather
than applied, with the smallest conflicting combination recorded.

Pure functions of a frame and a dict, which is what lets table.py replay them. The live path and
the replay path call the same code and cannot drift.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

import pandas as pd

_TIME_FIELDS = ("start", "end")
# Roll-up values that are never a real answer for a hierarchy column, from _get_answer_dict.
_ROLLUP = ("overall", "portfolio")
# Answers that describe the question rather than a column, normalised the same way the source
# does so the readout can rely on the shape.
_QUESTION_FIELDS = ("trend", "rank", "how_many")
_TREND_ANSWERS = ("yes", "no", "na")
_RANK_ANSWERS = ("top", "bottom", "na")
# Answers the classifier is told it may give that no column holds: COLGUS offers 'ahw' for
# `product` and the data never carries it.
_ANSWERS_WITHOUT_A_ROW = ("equity", "ahw")
# Misparses the classifier is warned against. 'ahw media' splits a product name off a tactic;
# 'pillars' appears in no onboarded configuration and is carried only because the source pairs
# the two.
_MISPARSED_ANSWERS = ("ahw media", "pillars")


def answer_dict_for(
    frame: pd.DataFrame, ner_results: dict[str, Any], fields: list[str]
) -> dict[str, list[Any]]:
    """Ported from data_filtering.py:_get_answer_dict.

    `trend`, `rank` and `how_many` name no column but still belong in the filter: the readout
    reads `how_many` to decide how many drivers to name. A field whose answer matches nothing is
    omitted, which is how the apply step tells 'not asked' from 'asked and impossible'.

    Only the `measure_group` substring branch is left out -- no onboarded configuration
    classifies that field, checked across all 30 client/model-group questionnaires.
    """

    resolved: dict[str, Any] = {}
    for field, raw in ner_results.items():
        values = raw if isinstance(raw, list) else [raw]
        if field in _TIME_FIELDS:
            resolved[field] = [int(v) for v in values if str(v).lstrip("-").isdigit()]
            continue
        if field in _QUESTION_FIELDS:
            resolved[field] = _question_answer(field, values)
            continue
        if field not in frame.columns or field not in fields:
            continue

        column = frame[field].dropna().unique()
        if "irrelevant" in values:
            if "overall" in column:
                resolved[field] = ["overall"]
        elif "all" in values or "specific" in values:
            real = [x for x in column if x not in _ROLLUP]
            if real:
                resolved[field] = real
        elif "overall" in values:
            resolved[field] = ["overall"]
        elif "relevant" in values:
            resolved[field] = ["yes"]
        elif field == "business_driver":
            resolved_driver = _business_driver_values(values, column)
            if resolved_driver:
                resolved[field] = resolved_driver
        else:
            real = [
                v
                for v in values
                if (v in column or v in _ANSWERS_WITHOUT_A_ROW) and v not in _MISPARSED_ANSWERS
            ]
            if real:
                resolved[field] = real
            elif field == "product":
                # The roll-up, so a product question the data cannot place still reports.
                resolved[field] = ["overall"]
    return resolved


def _question_answer(field: str, values: list[Any]) -> Any:
    """Ported from _get_answer_dict's trend/rank/how_many branches. Coerced rather than passed
    through, so an answer phrased differently reads as 'not asked' instead of unrecognised.
    """

    if field == "how_many":
        try:
            return "na" if "na" in values else int(values[0])
        except (TypeError, ValueError):
            return "na"
    allowed = _TREND_ANSWERS if field == "trend" else _RANK_ANSWERS
    matching = [v for v in values if v in allowed]
    return matching[0] if matching else "na"


def _business_driver_values(values: list[Any], column: Any) -> list[Any]:
    """Ported from _get_answer_dict's business_driver branch: the coarse answers are umbrellas
    over several real values, so they expand rather than match directly."""

    expanded = list(values)
    if "marketing" in values:
        expanded += ["marketing", "media", "non-media", "promotions"]
    elif "media" in values:
        expanded += ["media"]
    elif "base" in values:
        expanded += ["base", "other"]
    return [x for x in set(expanded) if x in column]


def _minimal_conflict(
    frame: pd.DataFrame, applied: list[tuple[str, pd.Series]], offender: pd.Series
) -> list[str]:
    """Ported verbatim from data_filtering.py:_minimal_conflicting_subset. The smallest set of
    applied filters this one empties -- which pair conflicts is what the user needs told.
    """

    for size in range(1, len(applied) + 1):
        for combo in combinations(range(len(applied)), size):
            mask = offender.copy()
            for index in combo:
                mask &= applied[index][1]
            if not mask.any():
                return [applied[index][0] for index in combo]
    return []


def _ordered_fields(
    answer_dict: dict[str, Any], prioritized: list[str] | None, ignore_fields: list[str]
) -> list[str]:
    """Prioritised fields first, then the rest in filter order, each field only once."""

    seen: set[str] = set()
    ordered = []
    for field in list(prioritized or []) + list(answer_dict):
        if field in seen or field in ignore_fields:
            continue
        seen.add(field)
        ordered.append(field)
    return ordered


def apply_answer_dict(
    frame: pd.DataFrame,
    answer_dict: dict[str, list[Any]],
    *,
    ner_results: dict[str, Any] | None = None,
    ignore_fields: list[str] | None = None,
    prioritized_indicator: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Ported from data_filtering.py:_apply_answer_dict_with_log.

    `prioritized_indicator` moves fields to the front, which decides conflicts: a filter that
    would empty the result is skipped, so whichever comes first wins and the later one is dropped.
    """

    ignore_fields = ignore_fields or []
    ner_results = ner_results or {}
    log: dict[str, Any] = {"skipped_no_value": {}, "conflicts": []}

    remaining = frame.copy()
    applied: list[tuple[str, pd.Series]] = []

    for field in _ordered_fields(answer_dict, prioritized_indicator, ignore_fields):
        values = answer_dict.get(field)
        if field not in frame.columns or not isinstance(values, list):
            continue

        if field in _TIME_FIELDS:
            starts = answer_dict.get("start") or []
            ends = answer_dict.get("end") or []
            if not starts or not ends:
                continue
            # Overlap, not equality: a row covers the request if its span crosses it at all.
            mask = pd.Series(False, index=frame.index)
            for start, end in zip(starts, ends, strict=False):
                mask |= (frame["start"].astype(int) <= int(end)) & (
                    frame["end"].astype(int) >= int(start)
                )
        elif all(value not in frame[field].unique() for value in values):
            log["skipped_no_value"][field] = values
            continue
        elif ner_results.get(field) == "all":
            # 'all' includes rows that never carried the dimension at all.
            mask = frame[field].isin(values) | frame[field].isna()
        else:
            mask = frame[field].isin(values)

        narrowed = remaining.index.intersection(frame.index[mask])
        if len(narrowed) == 0:
            log["conflicts"].append(
                {
                    "filter": field,
                    "values": values,
                    "conflicts_with": _minimal_conflict(frame, applied, mask),
                    # Nothing reads this; kept so the log matches the source's exactly.
                    "all_applied_before": [key for key, _ in applied],
                }
            )
            # Skipped, not fatal: later filters still apply and their conflicts still surface.
            continue

        remaining = remaining.loc[narrowed]
        applied.append((field, mask))

    return remaining.reset_index(drop=True), log


def apply_answer_dict_unlogged(
    frame: pd.DataFrame,
    answer_dict: dict[str, Any],
    *,
    ner_results: dict[str, Any] | None = None,
    ignore_fields: list[str] | None = None,
    prioritized_indicator: list[str] | None = None,
) -> pd.DataFrame:
    """Ported from data_filtering.py:_apply_answer_dict, the variant without the log. Not the logged
    one with the log thrown away: this narrows in place, so an emptying filter empties for good.
    """

    ner_results = ner_results or {}
    narrowed = frame.copy()

    for field in _ordered_fields(answer_dict, prioritized_indicator, []):
        values = answer_dict.get(field)
        if not isinstance(values, list) or field not in frame.columns:
            continue
        if field in _TIME_FIELDS:
            mask = pd.Series(False, index=narrowed.index)
            for start, end in zip(
                answer_dict.get("start") or [], answer_dict.get("end") or [], strict=False
            ):
                mask |= (narrowed["start"].astype(int) <= int(end)) & (
                    narrowed["end"].astype(int) >= int(start)
                )
            narrowed = narrowed[mask]
        elif all(value not in narrowed[field].unique() for value in values):
            continue
        elif field not in (ignore_fields or []):
            if ner_results.get(field, "") == "all":
                narrowed = narrowed[narrowed[field].isin(values) | narrowed[field].isna()]
            else:
                narrowed = narrowed[narrowed[field].isin(values)]
    return narrowed.reset_index(drop=True)


def rejection_message_for(log: dict[str, Any]) -> str:
    """Ported from filter_generator_refactor.py:log_to_message."""

    parts: list[str] = []
    for field, values in log["skipped_no_value"].items():
        parts.append(
            f"No data matches {field} = {values} after applying keyword filtering;"
            " this filter was skipped."
        )
    for conflict in log["conflicts"]:
        if conflict["conflicts_with"]:
            combined = " + ".join(conflict["conflicts_with"])
            parts.append(
                f"{conflict['filter']} = {conflict['values']} has no data in combination"
                f" with {combined}."
            )
        else:
            parts.append(f"{conflict['filter']} = {conflict['values']} produced no data.")
    return "\n\n".join(parts)
