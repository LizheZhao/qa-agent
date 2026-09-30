"""Unit tests for the ported relative-period arithmetic and the time prompt's construction.

`today` is pinned to 2026-09-08 -- the date the live inconsistency was measured on -- so every
expected value below is checkable by hand.
"""

from __future__ import annotations

from datetime import datetime

import pytest
from ask_genome_agent.nodes.question_understanding.support.time_periods import (
    format_today,
    generate_time_periods,
    get_few_shot_learning_examples_for_time,
)
from ask_genome_agent.prompts import DYNAMIC_TIME_PROMPT

_TODAY = datetime(2026, 9, 8)


@pytest.mark.unit
def test_format_today_matches_the_sources_date_string() -> None:
    """construct_prompt builds 'September, 2026 (202609)'."""

    assert format_today(_TODAY) == "September, 2026 (202609)"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("params", "expected"),
    [
        # "last quarter" on 2026-09-08: Q3 is in progress, so the latest *completed* quarter is
        # Q2 2026. This is the case that was resolving to 202504-202506 and 202601-202603 live.
        ({"period_type": "quarter", "num_periods": 1}, {"start": ["202604"], "end": ["202606"]}),
        (
            {"period_type": "quarter", "num_periods": 4},
            {
                "start": ["202507", "202510", "202601", "202604"],
                "end": ["202509", "202512", "202603", "202606"],
            },
        ),
        # "last month" on 2026-09-08 is August. The source returns July here -- see the divergence
        # note in generate_time_periods: its month branch subtracts a month twice.
        ({"period_type": "month", "num_periods": 1}, {"start": ["202608"], "end": ["202608"]}),
        (
            {"period_type": "month", "num_periods": 3},
            {"start": ["202606", "202607", "202608"], "end": ["202606", "202607", "202608"]},
        ),
        ({"period_type": "year", "num_periods": 1}, {"start": ["202501"], "end": ["202512"]}),
        # include_current shifts the window forward by one, so this year counts.
        (
            {"period_type": "year", "num_periods": 1, "include_current": True},
            {"start": ["202601"], "end": ["202612"]},
        ),
        # September > June, so H1 2026 has finished and counts as one of the two.
        (
            {"period_type": "half year", "num_periods": 2, "specific_sub_period": "H1"},
            {"start": ["202501", "202601"], "end": ["202506", "202606"]},
        ),
        # Explicit ranges pass straight through.
        (
            {"start_date": "202501", "end_date": "202503"},
            {"start": ["202501"], "end": ["202503"]},
        ),
        # A lone start runs to today; a lone end starts at the source's 201701 floor.
        ({"start_date": "201901"}, {"start": ["201901"], "end": ["202609"]}),
        ({"end_date": "201812"}, {"start": ["201701"], "end": ["201812"]}),
        # No period information at all is how "the question has no time reference" is expressed.
        ({}, {"start": "irrelevant", "end": "irrelevant"}),
    ],
)
def test_generate_time_periods(params, expected) -> None:
    assert generate_time_periods(**params, today=_TODAY) == expected


@pytest.mark.unit
def test_quarter_alignment_never_returns_an_in_progress_period() -> None:
    """The source aligns to the latest completed period. Checked across a whole year, since the
    arithmetic here replaces the source's explicit month-range if/elifs."""

    for month, expected_quarter_end in [
        (1, "202512"),  # Jan -> latest completed quarter is Q4 of last year
        (3, "202512"),
        (4, "202603"),  # Apr -> Q1 2026
        (6, "202603"),
        (7, "202606"),  # Jul -> Q2 2026
        (9, "202606"),
        (10, "202609"),  # Oct -> Q3 2026
        (12, "202609"),
    ]:
        result = generate_time_periods(
            period_type="quarter", num_periods=1, today=datetime(2026, month, 15)
        )
        assert result["end"] == [expected_quarter_end], f"month {month}"


@pytest.mark.unit
def test_year_end_is_december_of_the_start_year() -> None:
    result = generate_time_periods(period_type="year", num_periods=3, today=_TODAY)
    assert result == {
        "start": ["202301", "202401", "202501"],
        "end": ["202312", "202412", "202512"],
    }


@pytest.mark.unit
def test_invalid_period_type_is_rejected() -> None:
    with pytest.raises(ValueError, match="Invalid period type"):
        generate_time_periods(period_type="fortnight", num_periods=1, today=_TODAY)


@pytest.mark.unit
def test_invalid_sub_period_is_rejected() -> None:
    with pytest.raises(ValueError, match="Invalid sub-period tag"):
        generate_time_periods(
            period_type="quarter", num_periods=1, specific_sub_period="Q9", today=_TODAY
        )


@pytest.mark.unit
def test_few_shot_examples_are_anchored_to_the_given_date() -> None:
    """The examples store *parameters*, not literal dates, and are rendered per prompt build --
    that is what keeps every worked answer in the current date's frame."""

    rendered = get_few_shot_learning_examples_for_time(_TODAY)

    assert "Input 0:" in rendered and "Output 0:" in rendered
    # "last 4 quarters" example, computed against 2026-09-08.
    assert '"start": ["202507", "202510", "202601", "202604"]' in rendered
    # The two no-time-reference examples must teach the 'irrelevant' answer.
    assert '{"start": "irrelevant", "end": "irrelevant"}' in rendered
    # A year later, the same examples must move with it.
    assert rendered != get_few_shot_learning_examples_for_time(datetime(2027, 9, 8))


@pytest.mark.unit
def test_time_prompt_renders_with_no_placeholder_left_behind() -> None:
    """The actual live defect: the model received the literal characters 'Today is {date}.'"""

    rendered = DYNAMIC_TIME_PROMPT.format(
        date=format_today(_TODAY), example=get_few_shot_learning_examples_for_time(_TODAY)
    )

    assert "{date}" not in rendered
    assert "{example}" not in rendered
    assert "Today is September, 2026 (202609)." in rendered
    assert "Quarter 3: July to September" in rendered
