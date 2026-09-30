"""Parity between ask-genome-core's hierarchy-level steps and ours.

These decide which granularities a result reports at and drop the same figure where it appears
at more than one of them, so they change row counts as well as shape.

Skipped unless ask-genome-core is checked out beside this repo, so ordinary collection never
depends on a second working tree.
"""

from __future__ import annotations

import sys
from typing import Any

import pandas as pd
import pytest
from ask_genome_agent.nodes.question_understanding.support.levels import (
    dedup_levels,
    levels_to_target,
    most_granular_level,
    record_levels,
    report_levels,
    split_by_level,
)

from tests.source_parity import SOURCE_SRC

_SOURCE_SRC = SOURCE_SRC

pytestmark = pytest.mark.skipif(
    not _SOURCE_SRC.is_dir(), reason="ask-genome-core is not checked out beside this repo"
)


def _source(name: str) -> Any:
    if str(_SOURCE_SRC) not in sys.path:
        sys.path.insert(0, str(_SOURCE_SRC))
    import model.data_filtering as source  # type: ignore[import-not-found]

    return getattr(source, name)


_POSSIBLE = [
    "business_driver",
    "business_driver_detail",
    "activity_group",
    "measure_group",
    "measure",
]

# One row per depth, so record_level takes a different value on each and the level steps have
# something to tell apart. None means the row does not reach that rung.
_FRAME = pd.DataFrame(
    {
        "business_driver": ["marketing", "marketing", "marketing", "marketing", "base"],
        "business_driver_detail": [None, "paid search", "paid search", "digital display", None],
        "activity_group": [None, None, "paid search", "digital display", None],
        "measure_group": [None, None, "sem", "display", None],
        "measure": [None, None, "lms bing search", None, None],
        "country": ["us", "us", "us", "uk", "us"],
        "kpi": ["delivered bookings"] * 5,
        "time": ["quarter 1 2024"] * 5,
        "metric": ["spend"] * 5,
        "value": [1.0, 2.0, 3.0, 4.0, 5.0],
    }
)
# The same figure recorded at two depths, which is exactly what the dedupe resolves.
_REPEATED = pd.concat(
    [
        _FRAME,
        _FRAME.iloc[[2]].assign(measure=None, measure_group=None),
        _FRAME.iloc[[2]].assign(measure=None),
    ],
    ignore_index=True,
)

_LEVEL_TYPE = {"general_level": ["country", "kpi"], "halo_level": []}
_BI_LEVELS = ["activity_group", "measure_group", "measure"]
_BR_LEVELS = ["business_driver", "business_driver_detail", "activity_group"]


@pytest.mark.unit
@pytest.mark.parametrize("frame", [_FRAME, _REPEATED], ids=["distinct", "repeated"])
def test_record_level_matches_the_source(frame) -> None:
    want = _source("_get_level")(frame.copy())
    got = record_levels(frame.copy())

    pd.testing.assert_frame_equal(got, want)


@pytest.mark.unit
def test_the_frame_is_not_mutated_in_place() -> None:
    """A deliberate divergence. The source writes record_level onto the frame it was handed, so
    a caller that reuses that frame silently gets the column. Ours copies; the returned frame is
    identical either way, which is what parity above checks."""

    frame = _FRAME.copy()
    _source("_get_level")(frame)
    assert "record_level" in frame.columns

    frame = _FRAME.copy()
    record_levels(frame)
    assert "record_level" not in frame.columns


_TARGET_CASES: tuple[tuple[str, list[str], int], ...] = (
    ("already three", ["business_driver_detail", "activity_group", "measure_group"], 3),
    ("more than three", _POSSIBLE, 3),
    ("one in the middle", ["activity_group"], 3),
    ("one at the coarse end", ["business_driver"], 3),
    ("one at the fine end", ["measure"], 3),
    ("two adjacent", ["measure_group", "measure"], 3),
    ("two at the coarse end", ["business_driver", "business_driver_detail"], 3),
    ("target of one", ["activity_group"], 1),
)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("name", "valid", "target"), _TARGET_CASES, ids=[c[0] for c in _TARGET_CASES]
)
def test_padding_to_three_levels_matches_the_source(name, valid, target) -> None:
    del name
    want = _source("get_report_levels_to_target")(list(_POSSIBLE), list(valid), target)
    got = levels_to_target(list(_POSSIBLE), list(valid), target)

    assert got == want


@pytest.mark.unit
@pytest.mark.parametrize(
    "valid",
    [["activity_group"], ["business_driver", "measure"], ["measure", "business_driver"], _POSSIBLE],
)
def test_most_granular_matches_the_source(valid) -> None:
    assert most_granular_level(list(_POSSIBLE), list(valid)) == _source("most_granular_level")(
        list(_POSSIBLE), list(valid)
    )


_PAID_SEARCH_BR = {"source": "br", "activity_group": ["paid search"]}
_DRIVER_BR = {"source": "br", "business_driver_detail": ["paid search"]}
_BI_ONLY = {"source": "bi", "activity_group": ["paid search"]}
_MEASURE_BR = {"source": "br", "measure": ["lms bing search"]}
# One term resolving to business_driver_detail and another to measure: the deepest of the two
# decides the rungs, which leaves business_driver_detail out of them and fires the override.
_BDD_AND_MEASURE = {"driver": [_DRIVER_BR], "measure": [_MEASURE_BR]}

_REPORT_CASES: tuple[tuple[str, dict[str, Any], bool, list[str]], ...] = (
    # bi keeps whatever the intention came with, whatever was captured.
    (
        "bi passes through",
        {"data": "bi", "core_dimension": {"x": [_PAID_SEARCH_BR]}},
        False,
        _BI_LEVELS,
    ),
    ("br with no captures", {"data": "br", "core_dimension": {}}, False, _BR_LEVELS),
    # Halo tagging ignores the captures and reads the depths present in the data.
    ("br halo tagging", {"data": "br", "core_dimension": {}}, True, _BR_LEVELS),
    (
        "br capture at activity_group",
        {"data": "br", "core_dimension": {"paid search": [_PAID_SEARCH_BR]}},
        False,
        _BR_LEVELS,
    ),
    # The has_bdd branch: the deepest captured rung is business_driver_detail.
    (
        "br capture at business_driver_detail",
        {"data": "br", "core_dimension": {"paid search": [_DRIVER_BR]}},
        False,
        _BR_LEVELS,
    ),
    (
        "br capture on both rungs",
        {"data": "br", "core_dimension": {"paid search": [_DRIVER_BR, _PAID_SEARCH_BR]}},
        False,
        _BR_LEVELS,
    ),
    # Configured for bi only, so the br run finds no filter levels and falls back to the data.
    (
        "br capture that fires the business_driver_detail override",
        {"data": "br", "core_dimension": _BDD_AND_MEASURE},
        False,
        _BR_LEVELS,
    ),
    # The same captures against bi, which returns its own levels and computes nothing.
    (
        "bi ignores the captures entirely",
        {"data": "bi", "core_dimension": _BDD_AND_MEASURE},
        False,
        _BI_LEVELS,
    ),
    (
        "br capture configured for bi",
        {"data": "br", "core_dimension": {"paid search": [_BI_ONLY]}},
        False,
        _BR_LEVELS,
    ),
)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("name", "ner_filter", "is_halo", "level"), _REPORT_CASES, ids=[c[0] for c in _REPORT_CASES]
)
def test_report_levels_match_the_source(name, ner_filter, is_halo, level) -> None:
    del name
    want = _source("get_report_levels")(
        df=_FRAME.copy(), level=list(level), ner_filter=dict(ner_filter), is_halo_tagging=is_halo
    )
    got = report_levels(_FRAME.copy(), list(level), dict(ner_filter), is_halo)

    assert got == want


@pytest.mark.unit
def test_the_report_level_cases_are_not_all_the_same_answer() -> None:
    """Guards the parity above. get_report_levels returns `level` untouched for bi and for br
    with no captures, so a fixture where every branch agrees would look green for the wrong
    reason."""

    answers = {
        tuple(report_levels(_FRAME.copy(), list(level), dict(ner_filter), is_halo))
        for _, ner_filter, is_halo, level in _REPORT_CASES
    }
    assert len(answers) > 1


@pytest.mark.unit
@pytest.mark.parametrize("frame", [_FRAME, _REPEATED], ids=["distinct", "repeated"])
@pytest.mark.parametrize("level", [_BI_LEVELS, _BR_LEVELS], ids=["bi", "br"])
def test_hierarchy_dedupe_matches_the_source(frame, level) -> None:
    want = _source("_dedup_df")(df=frame.copy(), level=list(level), level_type=dict(_LEVEL_TYPE))
    got = dedup_levels(frame.copy(), list(level), dict(_LEVEL_TYPE))

    pd.testing.assert_frame_equal(got, want)


@pytest.mark.unit
def test_the_dedupe_actually_removes_something() -> None:
    """Guards the parity above from a fixture where nothing is duplicated."""

    kept = dedup_levels(_REPEATED.copy(), list(_BI_LEVELS), dict(_LEVEL_TYPE))
    assert len(kept) < len(_REPEATED[_REPEATED["measure"].notna()]) + 2


@pytest.mark.unit
def test_rows_with_the_same_dimensions_but_different_values_both_survive() -> None:
    """The open duplicate issue, pinned as out of scope for this step. `value` is part of the
    dedupe key, so this does not address what Michael and Lizhe were discussing."""

    same_dims = pd.concat([_FRAME.iloc[[2]], _FRAME.iloc[[2]].assign(value=99.0)])

    kept = dedup_levels(same_dims, list(_BI_LEVELS), dict(_LEVEL_TYPE))

    assert len(kept) == 2


@pytest.mark.unit
@pytest.mark.parametrize("level", [_BI_LEVELS, _BR_LEVELS], ids=["bi", "br"])
def test_splitting_by_level_matches_the_source(level) -> None:
    """The three dataframes ask-genome-core returns. Ours holds one frame plus record_level, and
    this is the whole of the difference."""

    deduped = dedup_levels(_REPEATED.copy(), list(level), dict(_LEVEL_TYPE))

    want = _source("_get_level_answer")(deduped.copy(), level=list(level))
    got = split_by_level(deduped.copy(), list(level))

    for got_part, want_part in zip(got, want, strict=True):
        pd.testing.assert_frame_equal(got_part, want_part)


@pytest.mark.unit
@pytest.mark.parametrize("level", [_BI_LEVELS, _BR_LEVELS], ids=["bi", "br"])
def test_the_split_loses_nothing(level) -> None:
    """Why one frame is enough: the dedupe has already dropped every row outside the triple, so
    the partition is total and the three frames add up to what we keep."""

    deduped = dedup_levels(_REPEATED.copy(), list(level), dict(_LEVEL_TYPE))

    assert sum(len(part) for part in split_by_level(deduped, list(level))) == len(deduped)
