"""Parity between ask-genome-core's time resolution and ours.

Every other test in this repo checks our code against our own expectations, which proves internal
consistency and nothing about fidelity to the source. This one runs the real
`_get_time_period_answer_dict` beside `resolve_time_period` on identical inputs and compares all
four outputs.

It earned its place immediately: the first run disagreed on both out-of-range cases, because the
source steps that fallback by year when `report_yoy` is set and we stepped it by the reporting
period. LINKEDIN/12 has `report_yoy` on, so that was a live divergence no unit test would have
caught.

Skipped unless ask-genome-core is checked out beside this repo, so ordinary collection never
depends on a second working tree.
"""

from __future__ import annotations

import sys
from typing import Any

import pandas as pd
import pytest
from ask_genome_agent.nodes.question_understanding.support.time_filter import (
    resolve_time_period,
    validated_indicators,
)

from tests.source_parity import SOURCE_SRC

_SOURCE_SRC = SOURCE_SRC


def _source_function() -> Any:
    if str(_SOURCE_SRC) not in sys.path:
        sys.path.insert(0, str(_SOURCE_SRC))
    from model.data_filtering import _get_time_period_answer_dict  # type: ignore[import-not-found]

    return _get_time_period_answer_dict


pytestmark = pytest.mark.skipif(
    not _SOURCE_SRC.is_dir(), reason="ask-genome-core is not checked out beside this repo"
)

# Six quarters and two years, so the two granularities disagree about what a range covers and the
# out-of-range branches have somewhere to land.
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

_CASES: tuple[tuple[str, dict[str, Any], list[str]], ...] = (
    (
        "no time reference",
        {"period_type": "irrelevant", "start": "irrelevant", "end": "irrelevant", "trend": "no"},
        ["spending"],
    ),
    (
        "one explicit quarter",
        {"period_type": "quarter", "start": ["202501"], "end": ["202503"], "trend": "no"},
        ["spending"],
    ),
    (
        "mid-period month",
        {"period_type": "quarter", "start": ["202405"], "end": ["202405"], "trend": "no"},
        ["spending"],
    ),
    (
        "multi-quarter range",
        {"period_type": "quarter", "start": ["202401"], "end": ["202410"], "trend": "no"},
        ["spending"],
    ),
    (
        "explicit year",
        {"period_type": "year", "start": ["202401"], "end": ["202412"], "trend": "no"},
        ["spending"],
    ),
    (
        "out of range, past",
        {"period_type": "quarter", "start": ["201901"], "end": ["201903"], "trend": "no"},
        ["spending"],
    ),
    (
        "out of range, future",
        {"period_type": "quarter", "start": ["202901"], "end": ["202903"], "trend": "no"},
        ["spending"],
    ),
    (
        "trend, no dates",
        {"period_type": "irrelevant", "start": "irrelevant", "end": "irrelevant", "trend": "yes"},
        ["spending"],
    ),
    (
        "trend, with dates",
        {"period_type": "quarter", "start": ["202401"], "end": ["202506"], "trend": "yes"},
        ["spending"],
    ),
    (
        "coarse intention",
        {"period_type": "irrelevant", "start": "irrelevant", "end": "irrelevant", "trend": "no"},
        ["margin roi"],
    ),
    (
        "period type absent",
        {"period_type": "month", "start": "irrelevant", "end": "irrelevant", "trend": "no"},
        ["spending"],
    ),
    # Past the quarters but inside a year row, so the overlap search finds rows and has to re-pick
    # the reporting granularity from them.
    (
        "beyond quarters, year covers",
        {"period_type": "quarter", "start": ["202507"], "end": ["202512"], "trend": "no"},
        ["spending"],
    ),
    (
        "beyond quarters, coarse",
        {"period_type": "quarter", "start": ["202510"], "end": ["202512"], "trend": "no"},
        ["margin roi"],
    ),
    (
        "half not in data",
        {"period_type": "half", "start": ["202401"], "end": ["202406"], "trend": "no"},
        ["spending"],
    ),
    (
        "two disjoint ranges",
        {
            "period_type": "quarter",
            "start": ["202401", "202501"],
            "end": ["202403", "202503"],
            "trend": "no",
        },
        ["spending"],
    ),
)


def _indicators(report_yoy: bool) -> dict[str, Any]:
    return validated_indicators(
        {
            "non_trend_report_length": [2, 2, 2, 12],
            "trend_report_length": [2, 2, 4, 12],
            "report_yoy": report_yoy,
            "trend_time_length": "2",
            "performance_period_granularity": "quarter",
        },
        (_FRAME,),
    )


@pytest.mark.unit
@pytest.mark.parametrize("report_yoy", [True, False], ids=["yoy_on", "yoy_off"])
@pytest.mark.parametrize(("name", "ner_results", "intentions"), _CASES, ids=[c[0] for c in _CASES])
def test_time_resolution_matches_the_source(name, ner_results, intentions, report_yoy) -> None:
    del name
    source = _source_function()
    indicators = _indicators(report_yoy)

    expected = source(_FRAME.copy(), dict(ner_results), dict(indicators), intentions)
    actual = resolve_time_period(_FRAME.copy(), dict(ner_results), dict(indicators), intentions)

    for key in ("start", "end", "period_type", "time_tag"):
        assert actual.get(key) == expected.get(key), key
