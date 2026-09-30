"""Time resolution against the periods a client's data actually has.

One test per decision in the port of `_get_time_period_answer_dict`. The headline is that
'irrelevant' is not "no time filter": it means the user did not say, and the answer defaults to
the most recent periods. Live, before this, a question with no time reference matched 254,338
rows because start/end resolved to an empty list and the filter was skipped entirely.
"""

from __future__ import annotations

import pandas as pd
import pytest
from ask_genome_agent.nodes.question_understanding.support.time_filter import (
    closer_time_index,
    expand_forward_periods,
    resolve_time_period,
    subtract_periods,
    validated_indicators,
)

# Eight quarters and two years, so "latest N periods" has something to reach back through.
_QUARTERS = [
    (202401, 202403),
    (202404, 202406),
    (202407, 202409),
    (202410, 202412),
    (202501, 202503),
    (202504, 202506),
]
_FRAME = pd.DataFrame(
    {
        "start": [s for s, _ in _QUARTERS] + [202401, 202501],
        "end": [e for _, e in _QUARTERS] + [202412, 202512],
        "period_type": ["quarter"] * len(_QUARTERS) + ["year", "year"],
    }
)
_INDICATORS = validated_indicators({"non_trend_report_length": [2, 2, 2, 12]}, (_FRAME,))


@pytest.mark.unit
def test_indicators_take_their_vocabulary_from_the_data() -> None:
    """The CSV names periods; which vocabulary applies depends on whether the client's own
    period_type values are fiscal. LINKEDIN's are plain, so 'fiscal quarter' in the CSV must not
    leak through as the reporting granularity."""

    indicators = validated_indicators(
        {"performance_period_granularity": "fiscal quarter"}, (_FRAME,)
    )

    assert indicators["calendar_type"] == "custom"
    assert indicators["period_priority"] == ["year", "half", "quarter", "month"]
    # Not in the data, so it falls back to the finest granularity that is.
    assert indicators["performance_period_granularity"] == "quarter"
    assert indicators["non_trend_report_length"]["quarter"] == 2


@pytest.mark.unit
def test_an_unstated_range_defaults_to_the_latest_periods() -> None:
    """The gap this module closes. 'irrelevant' start/end previously produced an empty list, the
    filter was skipped, and every period in the client's history matched."""

    resolved = resolve_time_period(
        _FRAME,
        {"period_type": "quarter", "start": "irrelevant", "end": "irrelevant", "trend": "no"},
        _INDICATORS,
        ["spending"],
    )

    # The two most recent quarters, not the whole history.
    assert resolved["period_type"] == ["quarter"]
    assert resolved["start"] == [202501, 202504]
    assert resolved["end"] == [202503, 202506]


@pytest.mark.unit
def test_an_unstated_period_type_resolves_to_one_the_data_has() -> None:
    resolved = resolve_time_period(
        _FRAME,
        {"period_type": "irrelevant", "start": "irrelevant", "end": "irrelevant", "trend": "no"},
        _INDICATORS,
        ["spending"],
    )

    assert resolved["period_type"][0] in {"quarter", "year"}
    assert resolved["start"], "an unstated period type must still produce a range"


@pytest.mark.unit
def test_an_explicit_range_snaps_to_period_boundaries() -> None:
    """A mid-period month means the period containing it. 202405 is inside 202404-202406."""

    resolved = resolve_time_period(
        _FRAME,
        {"period_type": "quarter", "start": 202405, "end": 202405, "trend": "no"},
        _INDICATORS,
        ["spending"],
    )

    assert 202404 in resolved["start"]
    assert 202406 in resolved["end"]


@pytest.mark.unit
def test_one_requested_period_is_padded_with_the_year_before() -> None:
    """Ported from the source: a single period gets the same period a year earlier alongside it,
    so there is something to compare against."""

    resolved = resolve_time_period(
        _FRAME,
        {"period_type": "quarter", "start": 202501, "end": 202503, "trend": "no"},
        _INDICATORS,
        ["spending"],
    )

    assert resolved["start"] == [202401, 202501]
    assert resolved["end"] == [202403, 202503]


@pytest.mark.unit
def test_a_range_outside_the_data_falls_back_to_the_latest_periods() -> None:
    """Asking about 2019 when the data starts in 2024 should answer about what exists, not
    return nothing at all."""

    resolved = resolve_time_period(
        _FRAME,
        {"period_type": "quarter", "start": 201901, "end": 201903, "trend": "no"},
        _INDICATORS,
        ["spending"],
    )

    assert resolved["start"] == [202501, 202504]


@pytest.mark.unit
def test_a_coarse_intention_reports_no_finer_than_its_granularity() -> None:
    """`performance` and `margin roi` are capped by performance_period_granularity; every other
    intention may report at whatever granularity was asked for.

    The cap works by dropping finer periods from the candidate order, so a capped intention that
    asks for one of them finds nothing to match and falls back to the coarsest the data has.
    """

    monthly = pd.concat(
        [_FRAME, pd.DataFrame({"start": [202501], "end": [202501], "period_type": ["month"]})],
        ignore_index=True,
    )
    indicators = validated_indicators(
        {"performance_period_granularity": "quarter", "non_trend_report_length": [2, 2, 2, 12]},
        (monthly,),
    )
    asked = {"period_type": "month", "start": "irrelevant", "end": "irrelevant", "trend": "no"}

    uncapped = resolve_time_period(monthly, asked, indicators, ["spending"])
    capped = resolve_time_period(monthly, asked, indicators, ["margin roi"])

    assert uncapped["period_type"] == ["month"]
    assert capped["period_type"] != ["month"]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("month", "period", "position", "expected"),
    [
        ("05", "quarter", "start", "04"),
        ("05", "quarter", "end", "06"),
        ("11", "quarter", "start", "10"),
        ("07", "half", "start", "07"),
        ("07", "half", "end", "12"),
        ("07", "year", "start", "01"),
        ("07", "year", "end", "12"),
        ("07", "month", "start", "07"),
    ],
)
def test_a_month_snaps_to_its_period_boundary(month, period, position, expected) -> None:
    assert closer_time_index(month, period, position) == expected


@pytest.mark.unit
def test_period_arithmetic_walks_whole_periods() -> None:
    # Three quarters back from 202504, inclusive of both ends.
    assert subtract_periods(202504, 3, "quarter") == [202504, 202501, 202410, 202407]
    # A year spans four quarters.
    starts, ends = expand_forward_periods(202401, 202410, "quarter")
    assert starts == [202401, 202404, 202407, 202410]
    assert ends == [202401, 202404, 202407, 202410]
