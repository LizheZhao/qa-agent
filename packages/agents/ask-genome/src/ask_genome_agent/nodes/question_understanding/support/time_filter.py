"""Resolves a question's time range against the periods the client's data actually has.

Ports data_filtering.py's _get_time_period_answer_dict plus its period arithmetic helpers
(get_period_delta, subtract_periods, expand_forward_periods, find_closer_time_index) and
filter_generator.py's _validate_postprocess_indicators.

Two jobs, and the second is the one that was missing. It picks a period_type the data has, then
turns the model's start/end into real period boundaries. 'irrelevant' does not mean "no time
filter": it means the user did not say, so the answer defaults to the most recent periods. Without
that, a question with no time reference matched every period in the client's history. Confirmed
live at 254,338 rows.

It also snaps explicit dates to period boundaries and expands a span into the individual periods
it covers, so "last quarter" becomes that quarter rather than a raw month range.

`time_tag` (snapshot / period_over_period_change / multiple_periods / trend) is carried through
even though nothing reads it yet. insight_generation picks its template from it, and one branch
below rewrites start/end when it is a period-over-period comparison, so dropping it would change
the filter rather than just omit a label.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd

from ask_genome_agent.nodes.question_understanding.support.time_periods import _shift_months

# Months per period, replacing the source's dateutil relativedelta. See time_periods._shift_months
# for why python-dateutil is kept out of this package.
_PERIOD_MONTHS = (("month", 1), ("quarter", 3), ("half", 6), ("year", 12))

# Ported from _get_time_period_answer_dict: these two intentions report at a coarser granularity
# than the rest, capped by performance_period_granularity.
_COARSE_INTENTIONS = ("performance", "margin roi")

_FISCAL_PERIODS = ("fiscal year", "fiscal half", "fiscal quarter", "fiscal month")
_CUSTOM_PERIODS = ("year", "half", "quarter", "month")


def period_months(period: str) -> int:
    """Ported from data_filtering.py:get_period_delta."""

    for name, months in _PERIOD_MONTHS:
        if name in period:
            return months
    raise ValueError(f"Unsupported period type: {period}")


def subtract_periods(date_value: Any, count: int, period: str) -> list[int]:
    """Ported verbatim from data_filtering.py:subtract_periods.

    `count` periods back from `date_value`, inclusive of both ends.
    """

    date = datetime.strptime(str(date_value), "%Y%m")
    months = period_months(period)
    return [int(_shift_months(date, -months * i).strftime("%Y%m")) for i in range(count + 1)]


def expand_forward_periods(start: Any, end: Any, period: str) -> tuple[list[int], list[int]]:
    """Ported verbatim from data_filtering.py:expand_forward_periods.

    A span becomes the individual periods inside it, so a filter can match each one.
    """

    first = datetime.strptime(str(start), "%Y%m")
    last = datetime.strptime(str(end), "%Y%m")
    months = period_months(period)

    starts: list[int] = []
    current = first
    while current <= last:
        starts.append(int(current.strftime("%Y%m")))
        current = _shift_months(current, months)

    ends: list[int] = []
    current = last
    while current >= first:
        ends.append(int(current.strftime("%Y%m")))
        current = _shift_months(current, -months)
    return sorted(starts), sorted(ends)


def closer_time_index(month: str, period: str, position: str) -> str:
    """Ported verbatim from data_filtering.py:find_closer_time_index.

    Snaps a month to the boundary of the period containing it, so a mid-period date becomes the
    period the user meant.
    """

    value = int(month)
    if "year" in period and "half" not in period:
        return "01" if position == "start" else "12"
    if "half" in period:
        if position == "start":
            return "01" if value <= 6 else "07"
        return "06" if value <= 6 else "12"
    if "quarter" in period:
        boundaries = (("01", "03"), ("04", "06"), ("07", "09"), ("10", "12"))
        first, final = boundaries[(value - 1) // 3]
        return first if position == "start" else final
    return month


def validated_indicators(
    indicators: dict[str, Any], frames: tuple[pd.DataFrame, ...]
) -> dict[str, Any]:
    """Ported from filter_generator.py:_validate_postprocess_indicators.

    The client's CSV values are names and lists; this turns them into the period vocabulary the
    data actually uses and the per-period report lengths keyed by that vocabulary. The source's
    filtering of period_priority against the real period types sits commented out there, so the
    full ordered vocabulary is what wins, and that is kept.
    """

    available = set()
    for frame in frames:
        if "period_type" in frame.columns:
            available |= set(frame["period_type"].dropna().unique())

    fiscal = any("fiscal" in str(period) for period in available)
    periods = list(_FISCAL_PERIODS if fiscal else _CUSTOM_PERIODS)

    lengths = indicators.get("non_trend_report_length") or [2, 2, 2, 12]
    trend_lengths = indicators.get("trend_report_length") or [2, 2, 4, 12]

    granularity = indicators.get("performance_period_granularity")
    if granularity not in available:
        # The CSV can name a period this client's data does not have; fall back to the finest
        # granularity it does.
        finer_first = [p for p in reversed(periods) if p in available]
        granularity = finer_first[0] if finer_first else periods[-1]

    return {
        **indicators,
        "period_priority": periods,
        "calendar_type": "fiscal" if fiscal else "custom",
        "performance_period_granularity": granularity,
        "report_yoy": bool(indicators.get("report_yoy", 0)),
        "non_trend_report_length": {k: int(v) for k, v in zip(periods, lengths, strict=False)},
        "trend_report_length": {k: int(v) for k, v in zip(periods, trend_lengths, strict=False)},
    }


def _resolve_period_type(
    frame: pd.DataFrame,
    ner_results: dict[str, Any],
    indicators: dict[str, Any],
    intentions: list[str],
) -> tuple[list[str], bool]:
    """Which period_type to report at, given what the user asked and what the data has."""

    trend = ner_results.get("trend", "no")
    requested = ner_results.get("period_type", "irrelevant")
    if isinstance(requested, list):
        requested = requested[0] if requested else "irrelevant"
    unstated = requested == "irrelevant"

    priority: list[str] = list(indicators["period_priority"])
    in_data = [p for p in priority if p in set(frame["period_type"].dropna().unique())]
    if not in_data:
        return [], unstated

    if trend == "yes" and unstated:
        # A trend with no requested granularity reports at the coarsest the data has.
        return [in_data[0]], unstated

    if any(item in _COARSE_INTENTIONS for item in intentions):
        granularity = indicators["performance_period_granularity"]
        cap = priority.index(granularity) if granularity in priority else len(priority) - 1
        ordered = priority[: cap + 1][::-1]
    else:
        ordered = priority[::-1]

    if indicators.get("calendar_type") == "custom" and isinstance(requested, str):
        requested = requested.replace("fiscal", "").strip()
    named = (
        [x.strip() for x in requested.split(",") if x.strip() in ordered]
        if isinstance(requested, str)
        else []
    )

    valid = next((x for x in named if x in ordered), in_data[0])
    chosen = next(
        (x for x in ordered[: ordered.index(valid) + 1][::-1] if x in in_data), in_data[-1]
    )
    return [chosen], unstated


def _latest_periods(
    frame: pd.DataFrame, period_types: list[str], period: str, count: int
) -> tuple[list[int], list[int]]:
    """The most recent `count` periods present in the data, as parallel start/end lists."""

    scoped = frame[frame["period_type"].isin(period_types)]
    if scoped.empty:
        return [], []
    latest_start = max(scoped["start"].astype(int).unique())
    latest_end = max(scoped["end"].astype(int).unique())
    return (
        subtract_periods(latest_start, count - 1, period),
        subtract_periods(latest_end, count - 1, period),
    )


def resolve_time_period(
    frame: pd.DataFrame,
    ner_results: dict[str, Any],
    indicators: dict[str, Any],
    intentions: list[str],
) -> dict[str, Any]:
    """Ported from data_filtering.py:_get_time_period_answer_dict.

    Returns the `start`, `end` and `period_type` the filter should use, plus the `time_tag`
    describing the shape of the request.
    """

    period_types, unstated_period = _resolve_period_type(frame, ner_results, indicators, intentions)
    if not period_types:
        return {}

    period = period_types[0]
    trend = ner_results.get("trend", "no")
    report_lengths = indicators["non_trend_report_length"]
    default_length = int(indicators.get("trend_time_length") or 2)
    count = int(report_lengths.get(period, default_length))

    start = ner_results.get("start", "irrelevant")
    end = ner_results.get("end", "irrelevant")
    starts = [str(x) for x in (start if isinstance(start, list) else [start])]
    ends = [str(x) for x in (end if isinstance(end, list) else [end])]
    unstated_range = "irrelevant" in starts or "irrelevant" in ends

    scoped = frame[frame["period_type"].isin(period_types)]

    if unstated_range:
        if trend == "yes":
            # A trend with no range reports the whole history at this granularity.
            start_res = sorted(scoped["start"].astype(int).unique().tolist())
            end_res = sorted(scoped["end"].astype(int).unique().tolist())
            time_tag = _trend_tag(len(start_res))
        else:
            start_res, end_res = _latest_periods(frame, period_types, period, count)
            time_tag = "snapshot" if unstated_period else "multiple_periods"
    elif _out_of_range(scoped, starts, ends):
        # Rows that overlap the request at any granularity, not just the chosen one. The source
        # searches the whole frame here, so a range beyond the quarters can still land on a year.
        overlapping = frame[_overlap_mask(frame, starts, ends)]
        if overlapping.empty:
            # Nothing overlaps at all, so fall back to the most recent periods. report_yoy steps
            # that fallback by year rather than by the reporting period: the same period a year
            # back, not the period before it. Only this branch uses it; the unstated-range branch
            # above has the equivalent line commented out in the source.
            fallback = "year" if indicators.get("report_yoy") else period
            span = int(report_lengths.get(fallback, default_length))
            start_res, end_res = _latest_periods(frame, period_types, fallback, span)
        else:
            period_types = _period_type_from(overlapping, period_types, indicators)
            period = period_types[0]
            span = int(report_lengths.get(period, default_length))
            start_res, end_res = _latest_periods(overlapping, period_types, period, span)
        time_tag = (
            _trend_tag(len(start_res))
            if trend == "yes"
            else ("snapshot" if len(start_res) <= 1 else "multiple_periods")
        )
    else:
        start_res, end_res = [], []
        for one_start, one_end in zip(starts, ends, strict=False):
            snapped_start = one_start[:-2] + closer_time_index(one_start[-2:], period, "start")
            snapped_end = one_end[:-2] + closer_time_index(one_end[-2:], period, "end")
            expanded_starts, expanded_ends = expand_forward_periods(
                snapped_start, snapped_end, period
            )
            start_res.extend(expanded_starts)
            end_res.extend(expanded_ends)

        requested = len(start_res)
        if requested == 1:
            # A single period is padded with the same period a year earlier, so there is
            # something to compare it against.
            start_res.append(_same_period_last_year(min(start_res)))
            end_res.append(_same_period_last_year(min(end_res)))
        if trend == "yes":
            time_tag = "period_over_period_change" if requested <= 3 else "trend"
        else:
            time_tag = "snapshot" if requested == 1 else "multiple_periods"

    if time_tag == "period_over_period_change":
        start_res, end_res = _adjacent_pair(scoped, start_res, end_res)

    return {
        "start": sorted({int(x) for x in start_res}),
        "end": sorted({int(x) for x in end_res}),
        "period_type": period_types,
        "time_tag": time_tag,
    }


def _same_period_last_year(period: int) -> int:
    """YYYYMM with the year decremented, keeping the month. From the source's padding branch."""

    text = str(period)
    return int(f"{int(text[:4]) - 1}{text[-2:]}")


def _trend_tag(period_count: int) -> str:
    """A trend over one period is a snapshot, and over two or three is a comparison."""

    if period_count <= 1:
        return "snapshot"
    if period_count <= 3:
        return "period_over_period_change"
    return "trend"


def _overlap_mask(frame: pd.DataFrame, starts: list[str], ends: list[str]) -> pd.Series:
    """Rows whose span crosses any of the requested spans."""

    mask = pd.Series(False, index=frame.index)
    for one_start, one_end in zip(starts, ends, strict=False):
        mask |= (frame["start"].astype(int) <= int(one_end)) & (
            frame["end"].astype(int) >= int(one_start)
        )
    return mask


def _period_type_from(
    overlapping: pd.DataFrame, period_types: list[str], indicators: dict[str, Any]
) -> list[str]:
    """Re-pick the reporting period from what the overlapping rows actually offer.

    Ported from the non-empty branch of the source's out-of-range path: the requested granularity
    may not exist among the rows that matched, so it walks the priority order to the nearest one
    that does.
    """

    priority: list[str] = list(indicators["period_priority"])
    available = [p for p in priority if p in set(overlapping["period_type"].dropna().unique())]
    if not available:
        return period_types

    chosen = period_types[0]
    if chosen in priority:
        index = priority.index(chosen)
        coarser_than_all = index > max(priority.index(p) for p in available)
        if coarser_than_all:
            return [available[-1]]
        priority = priority[index:]

    valid = next((x for x in period_types if x in priority), available[0])
    return [
        next(
            (x for x in priority[: priority.index(valid) + 1][::-1] if x in available), available[0]
        )
    ]


def _out_of_range(scoped: pd.DataFrame, starts: list[str], ends: list[str]) -> bool:
    """Whether the requested span falls entirely outside what the data covers."""

    if scoped.empty:
        return True
    requested_starts = [int(x) for x in starts]
    requested_ends = [int(x) for x in ends]
    return min(requested_starts) > max(scoped["end"].astype(int)) or max(requested_ends) < min(
        scoped["start"].astype(int)
    )


def _adjacent_pair(
    scoped: pd.DataFrame, start_res: list[int], end_res: list[int]
) -> tuple[list[int], list[int]]:
    """Ported from the tail of _get_time_period_answer_dict.

    A period-over-period comparison that resolved to one real period is widened to that period
    and the one before it, so there is something to compare against.
    """

    if scoped.empty:
        return start_res, end_res
    periods = scoped[["start", "end"]].drop_duplicates().sort_values("start")
    matched = periods[periods["start"].isin([int(x) for x in start_res])]
    if len(matched) != 1:
        return start_res, end_res
    selected = periods[periods["start"] <= matched["start"].iloc[0]].tail(2)
    if len(selected) != 2:
        return start_res, end_res
    return selected["start"].tolist(), selected["end"].tolist()


def latest_period(
    data: pd.DataFrame, ner_filter: dict[str, Any], indicators: dict[str, Any]
) -> tuple[int, str]:
    """Ported from filter_generator.py:find_latest_time.

    The newest period the data holds at one rung coarser than the question asked about, which is
    what the response phase uses to say how current an answer is. It reports; it never filters.
    Custom aggregations are excluded because they are not a period the client reports on.
    """

    periods_asked = ner_filter.get("period_type") or []
    frame = data[data["custom_aggregated"] == "no"] if "custom_aggregated" in data.columns else data

    priority = indicators.get("period_priority") or []
    valid_periods = [x for x in priority if x in frame["period_type"].unique()]
    index = min(valid_periods.index(periods_asked[0]) + 1, len(valid_periods) - 1)
    coarser = valid_periods[index]
    return int(frame[frame["period_type"] == coarser]["end"].max()), coarser
