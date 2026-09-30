"""Deterministic relative-period arithmetic, and the time dimension's few-shot examples.

Ported from ask-genome-core/src/model/filter_generator.py:generate_time_periods and
get_few_shot_learning_examples_for_time.

The source never asks the model to do calendar arithmetic unaided: it states today's date in the
prompt and renders 18 worked examples through this function at prompt-construction time, so every
example is anchored to the real current date. Without that, "last quarter" resolved to a different
period on each call.

"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

_SUB_PERIOD_MONTHS = {
    "Q1": (1, 3),
    "Q2": (4, 6),
    "Q3": (7, 9),
    "Q4": (10, 12),
    "H1": (1, 6),
    "H2": (7, 12),
}

_IRRELEVANT: dict[str, Any] = {"start": "irrelevant", "end": "irrelevant"}

# The source's floor for an open-ended "since ..." range (only an end date given).
_EARLIEST_PERIOD = "201701"


def _to_yyyymm(value: datetime) -> str:
    return value.strftime("%Y%m")


def _shift_months(value: datetime, months: int) -> datetime:
    """The source uses dateutil's relativedelta here. Every call site works on a day-1 date and
    only reads back YYYYMM, so plain month arithmetic is equivalent, and it keeps
    python-dateutil out of this package's dependencies, where it isn't declared (it arrives
    transitively via pandas today, which a separately versioned distribution shouldn't rely on).
    """

    total = value.year * 12 + (value.month - 1) + months
    return datetime(total // 12, total % 12 + 1, 1)


def _sub_period_range(year: int, tag: str) -> tuple[str, str]:
    if tag not in _SUB_PERIOD_MONTHS:
        raise ValueError(f"Invalid sub-period tag: {tag}")
    month_start, month_end = _SUB_PERIOD_MONTHS[tag]
    return f"{year}{month_start:02d}", f"{year}{month_end:02d}"


def generate_time_periods(
    period_type: str | None = None,
    num_periods: int | None = None,
    specific_sub_period: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    today: datetime | None = None,
    include_current: bool = False,
) -> dict[str, Any]:
    """Ported from filter_generator.py:generate_time_periods.

    - period_type + num_periods: that many periods back, aligned to the latest *completed* one.
    - plus specific_sub_period: that sub-period (Q1/H2/...) across that many past years.
    - explicit start and/or end: returned directly; a lone start ends today, a lone end starts at
      the source's 201701 floor.
    - neither: 'irrelevant', which is how "no time reference in the question" is expressed.
    """

    if today is None:
        today = datetime.today()

    offset = 0
    if include_current:
        offset = 1
    elif specific_sub_period:
        _, sub_end = _sub_period_range(today.year, specific_sub_period)
        if _to_yyyymm(today) > sub_end:
            # This year's instance of the sub-period has already finished, so it counts.
            offset = 1

    if start_date and end_date:
        return {"start": [start_date], "end": [end_date]}
    if start_date:
        return {"start": [start_date], "end": [_to_yyyymm(today)]}
    if end_date:
        return {"start": [_EARLIEST_PERIOD], "end": [end_date]}

    if period_type and num_periods and specific_sub_period:
        results: dict[str, Any] = {"start": [], "end": []}
        for i in range(num_periods):
            year = today.year - (num_periods - offset - i)
            sub_start, sub_end = _sub_period_range(year, specific_sub_period)
            results["start"].append(sub_start)
            results["end"].append(sub_end)
        return results

    if not (period_type and num_periods):
        return dict(_IRRELEVANT)

    results = {"start": [], "end": []}
    current = today.replace(day=1)

    # Align to the start of the period in progress; the loop then steps back, so the newest period
    # returned is always the latest *completed* one. The source spells the quarter/half cases out
    # as month-range if/elifs; this is the same mapping.
    #
    # The one deliberate divergence in this function: the source's month branch subtracts a month
    # here *and* subtracts `step` in the loop, so "last month" comes back as the month before last.
    # Every other branch subtracts once. An off-by-one that will drive real data filtering isn't
    # worth reproducing for fidelity, so month is aligned like the rest.
    if period_type == "month":
        pass  # `current` is already the first of the month in progress.
    elif period_type == "quarter":
        current = datetime(today.year, ((today.month - 1) // 3) * 3 + 1, 1)
    elif period_type == "half year":
        current = datetime(today.year, 1 if today.month <= 6 else 7, 1)
    elif period_type == "year":
        current = datetime(today.year, 1, 1)
    else:
        raise ValueError(f"Invalid period type: {period_type}")

    for i in range(num_periods):
        step = num_periods - offset - i
        if period_type == "month":
            start = _shift_months(current, -step)
            end = start
        elif period_type == "quarter":
            start = _shift_months(current, -step * 3)
            end = _shift_months(start, 2)
        elif period_type == "half year":
            start = _shift_months(current, -step * 6)
            end = _shift_months(start, 5)
        else:  # year
            start = datetime(current.year - step, 1, 1)
            end = datetime(start.year, 12, 1)
        results["start"].append(_to_yyyymm(start))
        results["end"].append(_to_yyyymm(end))

    return results


# Ported from get_few_shot_learning_examples_for_time: the questions, and the parameters each
# answer is generated from. Storing parameters rather than dates is what keeps the examples
# anchored to the current date; they're rendered fresh on every prompt build.
_TIME_EXAMPLES: tuple[tuple[str, dict[str, Any]], ...] = (
    (
        "What are the results from the last 4 quarters?",
        {"period_type": "quarter", "num_periods": 4},
    ),
    ("Show me the data for last year.", {"period_type": "year", "num_periods": 1}),
    (
        "Give me the performance data from January to March 2025.",
        {"start_date": "202501", "end_date": "202503"},
    ),
    (
        "Compare the first half of the last two years.",
        {"period_type": "half year", "num_periods": 2, "specific_sub_period": "H1"},
    ),
    ("How did we perform in April 2025?", {"start_date": "202504", "end_date": "202504"}),
    (
        "Compare the performance of search in the last three Q1s.",
        {"period_type": "quarter", "num_periods": 3, "specific_sub_period": "Q1"},
    ),
    (
        "How did PMAX campaigns perform compared to last year?",
        {"period_type": "year", "num_periods": 2, "include_current": True},
    ),
    ("How did PMAX campaigns perform since 2019?", {"start_date": "201901"}),
    (
        "how has the pd handraiser campaign performed over the last 3 years?",
        {"period_type": "year", "num_periods": 3},
    ),
    (
        "what is our most efficient marketing tactic over this year?",
        {"period_type": "year", "num_periods": 1, "include_current": True},
    ),
    (
        "what have category impacts been to the pd business over the last year?",
        {"period_type": "year", "num_periods": 1},
    ),
    ("How have search trends changed year over year?", {}),
    (
        "what was the month-over-month trend for roi in 2024?",
        {"start_date": "202401", "end_date": "202412"},
    ),
    (
        "what was the household penetration efficiencies in 2024 h2 and q2?",
        {"period_type": "quarter", "start_date": "202404", "end_date": "202412"},
    ),
    ("Show campaign impact.", {}),
    ("what's the contribution before 2019?", {"start_date": "201701", "end_date": "201812"}),
    (
        "what was the year - over - year trend for total shopper in 2024",
        {"start_date": "202401", "end_date": "202412"},
    ),
    (
        "How much did the paid search ROI change from Q3 FY25 to Q4 FY25?",
        {
            "start_date": "202507",
            "end_date": "202512",
            "period_type": "quarter",
            "num_periods": 2,
        },
    ),
)


def get_few_shot_learning_examples_for_time(today: datetime | None = None) -> str:
    """Ported from filter_generator.py:get_few_shot_learning_examples_for_time.

    Renders each stored example's parameters through generate_time_periods against `today`, so the
    worked answers the model sees are always in the current date's frame.
    """

    blocks = []
    for index, (question, params) in enumerate(_TIME_EXAMPLES):
        output = dict(_IRRELEVANT) if not params else generate_time_periods(**params, today=today)
        blocks.append(
            f"Input {index}:  \n    {question}  \n    Output {index}:  \n    {json.dumps(output)}"
        )
    return "\n\n".join(blocks)


def format_today(today: datetime | None = None) -> str:
    """The source's `date` string: 'September, 2026 (202609)' (construct_prompt)."""

    now = today or datetime.now()
    return f"{now.strftime('%B')}, {now.year} ({now.year}{now.month:02d})"
