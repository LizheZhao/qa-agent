"""Parity between ask-genome-core's narrowing steps and ours.

These four functions each return a mutated ner_filter, ner_res, ignore_fields and frame into the
next, so a per-function test cannot prove the chain is right: every one could match while the
composition differs. Both levels live here, and the pipeline test is the one that matters.

Skipped unless ask-genome-core is checked out beside this repo, so ordinary collection never
depends on a second working tree.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest
from ask_genome_agent.config import NarrowingConfig
from ask_genome_agent.nodes.question_understanding.support.filtering import apply_answer_dict
from ask_genome_agent.nodes.question_understanding.support.narrowing import (
    apply_filtering_logic,
    combine_filters,
    get_specific_values,
    map_keyword_in_query,
    postprocess_filter,
    reconcile_ner,
)
from ask_genome_agent.nodes.question_understanding.support.time_filter import validated_indicators

from tests.source_parity import SOURCE_SRC, source_available, source_functions, source_module

pytestmark = pytest.mark.skipif(
    not source_available(), reason="ask-genome-core is not checked out beside this repo"
)


def _source(name: str) -> Any:
    return getattr(source_module(), name)


_SOURCE_SPACY = (
    source_functions(
        SOURCE_SRC / "model" / "filter_generator.py",
        [
            "_get_mask",
            "_get_target_columns",
            "_label_data_with_term",
            "categorize_filters",
            "apply_categorized_filters",
            "combine_filters",
            "_ner_reconciliation",
        ],
        extra_globals={"legacy_filtering": source_module()},
    )
    if source_available()
    else {}
)


# Six quarters and two years, plus one general_tagging column so rule 5 has something to act on.
_FRAME = pd.DataFrame(
    {
        "start": [202401, 202404, 202407, 202410, 202501, 202504, 202401, 202501],
        "end": [202403, 202406, 202409, 202412, 202503, 202506, 202412, 202512],
        "period_type": ["quarter"] * 6 + ["year", "year"],
        "brand": ["a", "b", "a", "b", "a", "b", "a", "b"],
    }
)
_INDICATORS = validated_indicators(
    {
        "trend_report_length": [2, 2, 4, 12],
        "non_trend_report_length": [2, 2, 2, 12],
        "trend_time_length": "2",
    },
    (_FRAME,),
)

_QUARTER = {"period_type": ["quarter"], "start": [202504], "end": [202506]}

_CASES: tuple[tuple[str, dict[str, Any], dict[str, Any], dict[str, Any]], ...] = (
    ("plain", {"intention": ["margin roi"], **_QUARTER}, {}, {}),
    # Rule 1: this intention forces trend off, which then feeds rule 2.
    ("source of change", {"intention": ["source of change"], **_QUARTER}, {}, {}),
    (
        "trend at quarter",
        {"intention": ["spending"], "trend": "yes", **_QUARTER},
        {"trend": "yes"},
        {},
    ),
    (
        "trend at year",
        {
            "intention": ["spending"],
            "trend": "yes",
            "period_type": ["year"],
            "start": [202501],
            "end": [202512],
        },
        {"trend": "yes"},
        {},
    ),
    # Already more than one period, so rule 2 leaves the range alone.
    (
        "trend, range already wide",
        {
            "intention": ["spending"],
            "trend": "yes",
            "period_type": ["quarter"],
            "start": [202501, 202504],
            "end": [202503, 202506],
        },
        {"trend": "yes"},
        {},
    ),
    # Rule 5: 'all' becomes the column's real values, so the apply step stops widening for it.
    (
        "tagging answered all",
        {"intention": ["spending"], **_QUARTER},
        {"brand": "all"},
        {"general_tagging": ["brand"]},
    ),
    (
        "tagging answered otherwise",
        {"intention": ["spending"], **_QUARTER},
        {"brand": "irrelevant"},
        {"general_tagging": ["brand"]},
    ),
)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("name", "ner_filter", "ner_res", "level_type"), _CASES, ids=[c[0] for c in _CASES]
)
def test_postprocess_matches_the_source(name, ner_filter, ner_res, level_type) -> None:
    del name
    source = _source("_postprocess_filter_given_indicator")

    expected = source(
        dict(ner_filter), dict(ner_res), _FRAME.copy(), dict(_INDICATORS), dict(level_type)
    )
    actual = postprocess_filter(
        dict(ner_filter), dict(ner_res), _FRAME.copy(), dict(_INDICATORS), dict(level_type)
    )

    assert actual == expected


_KEYWORD_FRAME = pd.DataFrame(
    {
        "activity_group": ["google paid search", "amazon dsp", "brand tv", None],
        "measure_group": ["paid media", "retail media", "brand", None],
        "measure": ["google sem", "amazon display", "tv spot", None],
    }
)
_RETAILER_FRAME = _KEYWORD_FRAME.assign(retailer=["yes", "yes", "no", "no"])

_KEYWORD_CASES: tuple[tuple[str, pd.DataFrame, str, dict[str, Any], list[str]], ...] = (
    ("no keyword", _KEYWORD_FRAME, "what was margin roi for paid search last quarter", {}, []),
    ("one platform", _KEYWORD_FRAME, "how did google perform", {}, []),
    ("two platforms", _KEYWORD_FRAME, "compare google and amazon", {}, []),
    # 'masterbrand' is in the map but excluded from the platform list.
    ("non-platform keyword", _KEYWORD_FRAME, "what about masterbrand", {}, []),
    # Named, but nothing in the frame carries it, so no platform is recorded.
    ("keyword with no match", _KEYWORD_FRAME, "how did tiktok perform", {}, []),
    ("retail without the column", _KEYWORD_FRAME, "retail media performance", {}, []),
    ("retail with the column", _RETAILER_FRAME, "retail media performance", {}, []),
    (
        "existing filter and ignores",
        _KEYWORD_FRAME,
        "how did google perform",
        {"country": ["us"]},
        ["kpi"],
    ),
)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("name", "frame", "query", "ner_filter", "ignore_fields"),
    _KEYWORD_CASES,
    ids=[c[0] for c in _KEYWORD_CASES],
)
def test_keyword_mapping_matches_the_source(name, frame, query, ner_filter, ignore_fields) -> None:
    del name
    source = _source("map_keyword_in_query")

    want_frame, want_filter, want_ignore = source(
        frame.copy(), query, dict(ner_filter), list(ignore_fields)
    )
    got_frame, got_filter, got_ignore = map_keyword_in_query(
        frame.copy(), query, dict(ner_filter), list(ignore_fields)
    )

    assert got_filter == want_filter
    assert got_ignore == want_ignore
    pd.testing.assert_frame_equal(got_frame, want_frame)


@pytest.mark.unit
def test_a_frame_without_the_keyword_columns_does_not_crash() -> None:
    """A deliberate divergence. The source reads `res_df.activity_group` unguarded and raises
    AttributeError for a client whose data lacks it. Since the whole function is inert for row
    selection, crashing on a column it only reads to match dead keywords has no upside."""

    frame = pd.DataFrame({"country": ["us", "uk"]})

    with pytest.raises(AttributeError):
        _source("map_keyword_in_query")(frame.copy(), "how did google perform", {}, [])

    _, filters, ignored = map_keyword_in_query(frame.copy(), "how did google perform", {}, [])

    assert "platform" not in filters
    assert ignored == ["platform", "retailer"]


# LINKEDIN-shaped: the columns combine_filters can match on, plus the time columns the chain test
# needs. `custom_aggregated` really is only yes/no for this client -- no row is 'diagnostic' -- so
# the diagnostic split below is exercised by _SPLIT_FRAME instead.
_DATA_FRAME = pd.DataFrame(
    {
        "activity_group": [
            "paid search",
            "digital display",
            "print",
            "paid search",
            "digital video",
            "digital display",
            "paid search",
            "paid social",
        ],
        "measure_group": ["sem", "display", "print", "sem", "video", "display", "sem", "social"],
        "measure": [
            "lms bing search",
            "lms dv360 display",
            "print insert",
            "lms google search",
            "lms youtube",
            "lms gdn",
            "lms bing search",
            "lms facebook",
        ],
        "business_driver": ["marketing"] * 6 + ["base", "marketing"],
        "business_driver_detail": [
            "paid search",
            "digital display",
            "print",
            "paid search",
            "digital video",
            "digital display",
            "non marketing",
            "paid social",
        ],
        "country": ["us", "us", "uk", "uk", "us", "ca", "us", "us"],
        "kpi": ["delivered bookings"] * 4 + ["signups"] * 4,
        "detailed_kpi": ["business delivered bookings"] * 4 + ["member signups"] * 4,
        "custom_aggregated": ["no", "no", "no", "no", "yes", "no", "no", "no"],
        # Read only by reconciliation: the portfolio flag it pins and the halo level it checks.
        "portfolio": ["no", "no", "yes", "no", "no", "yes", "no", "no"],
        "business_unit_for_kpi": ["lms", "lss", "overall", "lms", "lss", "overall", "lms", "lss"],
        "metric": ["spend"] * 7 + ["margin roi"],
        "period_type": ["quarter"] * 8,
        "start": [202504] * 8,
        "end": [202506] * 8,
    }
)

# Same rows plus the diagnostic columns, so the custom_aggregated == 'diagnostic' branch and the
# diag_tagging bucket get covered even though no client in hand produces them.
_SPLIT_FRAME = _DATA_FRAME.assign(
    custom_aggregated=["no", "no", "diagnostic", "no", "yes", "diagnostic", "no", "no"],
    split=["none", "none", "incremental", "none", "none", "incremental", "none", "none"],
    split_type=["none", "none", "lift", "none", "none", "lift", "none", "none"],
)

_LEVEL_TYPE = {
    "general_level": ["country", "kpi", "detailed_kpi"],
    "halo_level": ["business_unit_for_kpi"],
    "halo_tagging": ["business unit focused marketing"],
}
_SUB_COLS = [
    "activity_group",
    "measure_group",
    "measure",
    "business_driver",
    "business_driver_detail",
]

# The first four entries are copied from LINKEDIN/12's core_dimension_filter.json. The rest are
# shapes that file has no example of -- AKA aliasing, a diagnostic split, a halo-only filter, the
# custom-aggregation exclusion -- kept so every bucket categorize_filters can produce is covered.
_CORE_FILTERS: dict[str, list[dict[str, Any]]] = {
    "paid search": [
        {"business_driver_detail": ["paid search"], "source": "bibr", "detail": ""},
        {"activity_group": ["paid search"], "source": "bibr", "detail": ""},
    ],
    "digital display": [
        {"business_driver_detail": ["digital display"], "source": "bibr", "detail": ""},
        {"activity_group": ["digital display"], "source": "bibr", "detail": ""},
    ],
    "print": [
        {"business_driver_detail": ["print"], "source": "br", "detail": ""},
        {"activity_group": ["print"], "source": "br", "detail": ""},
    ],
    "non marketing": [
        {"source": "br", "business_driver": ["base"], "composite": "0", "halo_to_ignore": "0"},
    ],
    # Deliberately uneven metadata between these two. They target the same column and must group
    # together, which only holds if grouping ignores the keys that are not columns.
    "united states": [
        {"source": "bibr", "country": ["us"], "AKA": "0", "org_term": "united states"}
    ],
    "canada": [{"source": "bibr", "country": ["ca"]}],
    "bookings": [{"source": "bibr", "kpi": ["delivered bookings"]}],
    "signups": [{"source": "bibr", "kpi": ["signups"]}],
    # The term is itself a detailed_kpi value while its filter only names kpi. A filter that
    # says nothing about a column falls back to the captured term, which is what pins the other
    # column here.
    "member signups": [{"source": "bibr", "kpi": ["signups"]}],
    "incrementality": [{"source": "bibr", "split": ["incremental"], "split_type": ["lift"]}],
    "sem": [
        {"source": "bibr", "activity_group": ["paid search"], "AKA": "1", "org_term": "paid search"}
    ],
    "owned": [
        {"source": "bibr", "activity_group": ["digital video"], "detail": "ignore when cust_agg"}
    ],
    "focused bu": [
        {"source": "bi", "business_driver_detail": ["digital display"], "halo_to_ignore": "1"}
    ],
}

_BASE_FILTER = {"intention": ["spending"], "data": "bi", "metric": ["spend"], "main_metric": []}


def _combine_case(**overrides: Any) -> dict[str, Any]:
    case: dict[str, Any] = {
        "frame": _DATA_FRAME,
        "ner_filter": dict(_BASE_FILTER),
        "ner_res": {},
        "spacy_filter": {},
        "ignore_fields": [],
        "level_type": _LEVEL_TYPE,
    }
    case.update(overrides)
    return case


_COMBINE_CASES: tuple[tuple[str, dict[str, Any]], ...] = (
    ("nothing captured", _combine_case()),
    ("one core dimension", _combine_case(spacy_filter={"core_dimension": ["paid search"]})),
    # Two terms of the same type widen rather than narrow: this should keep both channels.
    (
        "two core dimensions",
        _combine_case(spacy_filter={"core_dimension": ["paid search", "digital display"]}),
    ),
    # 'print' is configured source='br' only, so against bi its filters are skipped and the term
    # narrows nothing at all.
    (
        "term not configured for this source",
        _combine_case(spacy_filter={"core_dimension": ["print"]}),
    ),
    (
        "same term against its own source",
        _combine_case(
            ner_filter={**_BASE_FILTER, "data": "br"}, spacy_filter={"core_dimension": ["print"]}
        ),
    ),
    # Different column groups AND together: united states and bookings must both hold.
    (
        "two custom levels on different columns",
        _combine_case(spacy_filter={"core_dimension": ["united states", "bookings"]}),
    ),
    # Same column group ORs: either country.
    (
        "two custom levels on one column",
        _combine_case(spacy_filter={"core_dimension": ["united states", "canada"]}),
    ),
    (
        "custom level and core dimension",
        _combine_case(spacy_filter={"core_dimension": ["paid search", "united states"]}),
    ),
    (
        "business driver",
        _combine_case(
            ner_filter={**_BASE_FILTER, "data": "br"},
            spacy_filter={"business_driver": ["non marketing"]},
        ),
    ),
    # No configured filter, so it lands in terms_from_insights and selects nothing.
    ("unconfigured term", _combine_case(spacy_filter={"core_dimension": ["response curve"]})),
    (
        "configured and unconfigured together",
        _combine_case(spacy_filter={"core_dimension": ["paid search", "response curve"]}),
    ),
    # A captured metric is merged into the filter's metric list, not used to select rows.
    (
        "metric capture",
        _combine_case(spacy_filter={"core_dimension": ["paid search"], "metric": ["margin roi"]}),
    ),
    ("metric already present", _combine_case(spacy_filter={"metric": ["spend"]})),
    # Composite captures pull the halo columns into sub_cols on top of spacyOverrideColumns.
    (
        "composite capture",
        _combine_case(spacy_filter={"core_dimension_composite": ["paid search"]}),
    ),
    ("halo_tagging captures are skipped", _combine_case(spacy_filter={"halo_tagging": ["print"]})),
    # halo_to_ignore drops the filter unless a halo field was answered with real values.
    (
        "halo filter without a halo answer",
        _combine_case(spacy_filter={"core_dimension": ["focused bu"]}),
    ),
    (
        "halo filter with a halo answer",
        _combine_case(
            ner_filter={**_BASE_FILTER, "business unit focused marketing": ["talent"]},
            spacy_filter={"core_dimension": ["focused bu"]},
        ),
    ),
    ("aka labelling", _combine_case(spacy_filter={"core_dimension": ["sem"]})),
    (
        "custom aggregation exclusion",
        _combine_case(spacy_filter={"core_dimension": ["owned"]}),
    ),
    (
        "diagnostic split",
        _combine_case(frame=_SPLIT_FRAME, spacy_filter={"core_dimension": ["incrementality"]}),
    ),
    (
        "diagnostic split alongside a core dimension",
        _combine_case(
            frame=_SPLIT_FRAME,
            spacy_filter={"core_dimension": ["incrementality", "digital display"]},
        ),
    ),
    (
        "diagnostic frame, no diagnostic term",
        _combine_case(frame=_SPLIT_FRAME, spacy_filter={"core_dimension": ["paid search"]}),
    ),
    (
        "existing ignore fields are kept",
        _combine_case(
            spacy_filter={"core_dimension": ["paid search"]}, ignore_fields=["platform", "retailer"]
        ),
    ),
)


@pytest.mark.unit
@pytest.mark.parametrize(("name", "case"), _COMBINE_CASES, ids=[c[0] for c in _COMBINE_CASES])
def test_combine_filters_matches_the_source(name, case) -> None:
    del name
    source = _SOURCE_SPACY["combine_filters"]

    def call(fn: Any) -> tuple[pd.DataFrame, dict[str, Any], list[str]]:
        return fn(
            case["frame"].copy(),
            {k: list(v) if isinstance(v, list) else v for k, v in case["ner_filter"].items()},
            dict(case["ner_res"]),
            {k: list(v) for k, v in case["spacy_filter"].items()},
            _CORE_FILTERS,
            list(case["ignore_fields"]),
            list(_SUB_COLS),
            _LEVEL_TYPE,
        )

    want_frame, want_filter, want_ignore = call(source)
    got_frame, got_filter, got_ignore = call(combine_filters)

    assert got_filter == want_filter
    # Both build the tail of this list from a set, so only its contents are meaningful.
    assert sorted(got_ignore) == sorted(want_ignore)
    pd.testing.assert_frame_equal(
        got_frame.reset_index(drop=True), want_frame.reset_index(drop=True)
    )


# A frame with a row every field is recorded 'overall' on, which is the fallback step four takes
# for a field nothing pinned.
_OVERALL_FRAME = pd.concat(
    [
        _DATA_FRAME,
        _DATA_FRAME.head(1).assign(country="overall", kpi="overall", detailed_kpi="overall"),
    ],
    ignore_index=True,
)
_TAGGING_LEVEL_TYPE = {**_LEVEL_TYPE, "general_tagging": ["business_driver_detail"]}

_SPECIFIC_CASES: tuple[
    tuple[str, pd.DataFrame, dict[str, Any], dict[str, Any], dict[str, Any]], ...
] = (
    ("nothing answered", _DATA_FRAME, {"core_dimension": ["paid search"]}, {}, _LEVEL_TYPE),
    # Answered 'specific' without saying which value: the captured term settles it.
    (
        "level answered specific",
        _DATA_FRAME,
        {"core_dimension": ["united states"]},
        {"country": "specific"},
        _LEVEL_TYPE,
    ),
    # Not answered specific, but a captured term landed in the custom_level bucket, which is
    # enough on its own to run.
    (
        "level not answered but captured",
        _DATA_FRAME,
        {"core_dimension": ["united states"]},
        {"country": "irrelevant"},
        _LEVEL_TYPE,
    ),
    (
        "two levels captured",
        _DATA_FRAME,
        {"core_dimension": ["united states", "bookings"]},
        {"country": "specific", "kpi": "specific"},
        _LEVEL_TYPE,
    ),
    # Nothing captured for it, so it falls back to 'overall' where the data has such a row.
    (
        "specific with nothing to pin it, overall available",
        _OVERALL_FRAME,
        {},
        {"country": "specific"},
        _LEVEL_TYPE,
    ),
    (
        "specific with nothing to pin it, no overall row",
        _DATA_FRAME,
        {},
        {"country": "specific"},
        _LEVEL_TYPE,
    ),
    (
        "business driver from a core dimension capture",
        _DATA_FRAME,
        {"core_dimension": ["paid search"]},
        {"business_driver": "specific"},
        _LEVEL_TYPE,
    ),
    (
        "business driver from its own capture",
        _DATA_FRAME,
        {"business_driver": ["non marketing"]},
        {"business_driver": "specific"},
        _LEVEL_TYPE,
    ),
    (
        "custom tagging",
        _DATA_FRAME,
        {"core_dimension": ["digital display"]},
        {"business_driver_detail": "specific"},
        _TAGGING_LEVEL_TYPE,
    ),
    # A field that is neither a level nor a tagging is left alone entirely.
    ("field outside the level types", _DATA_FRAME, {}, {"trend": "yes"}, _LEVEL_TYPE),
    (
        "a capture that pins two columns at once",
        _DATA_FRAME,
        {"core_dimension": ["member signups"]},
        {"kpi": "specific"},
        _LEVEL_TYPE,
    ),
    (
        "a captured term that is not a real value",
        _DATA_FRAME,
        {"core_dimension": ["response curve"]},
        {"country": "specific"},
        _LEVEL_TYPE,
    ),
)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("name", "frame", "spacy_filter", "ner_res", "level_type"),
    _SPECIFIC_CASES,
    ids=[c[0] for c in _SPECIFIC_CASES],
)
def test_get_specific_values_matches_the_source(name, frame, spacy_filter, ner_res, level_type):
    """Driven from what the real combine_filters hands it, not a hand-built approximation of it:
    this step reads the capture buckets and the narrowed frame that step three produces."""

    del name
    source = _source("_get_specific_values")
    narrowed, ner_filter, ignore = _SOURCE_SPACY["combine_filters"](
        frame.copy(),
        dict(_BASE_FILTER),
        dict(ner_res),
        dict(spacy_filter),
        _CORE_FILTERS,
        [],
        list(_SUB_COLS),
        level_type,
    )

    def call(fn: Any) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
        return fn(
            narrowed.copy(),
            copy.deepcopy(ner_filter),
            dict(ner_res),
            level_type,
            list(ignore),
        )

    want_filter, want_res, want_ignore = call(source)
    got_filter, got_res, got_ignore = call(get_specific_values)

    assert got_filter == want_filter
    assert got_res == want_res
    assert sorted(got_ignore) == sorted(want_ignore)


_CHAIN_CASES: tuple[tuple[str, str, dict[str, Any], dict[str, Any], pd.DataFrame], ...] = (
    (
        "paid search last quarter",
        "what was spend for paid search last quarter",
        {"core_dimension": ["paid search"]},
        {},
        _DATA_FRAME,
    ),
    (
        "two channels",
        "compare spend for paid search and digital display",
        {"core_dimension": ["paid search", "digital display"]},
        {},
        _DATA_FRAME,
    ),
    (
        "channel and level",
        "what was spend for digital display in the united states",
        {"core_dimension": ["digital display", "united states"]},
        {},
        _DATA_FRAME,
    ),
    # A trend widens the time range in step one, and the widened range must reach step three.
    (
        "trend",
        "show me the spend trend for paid search",
        {"core_dimension": ["paid search"]},
        {"trend": "yes"},
        _DATA_FRAME,
    ),
    (
        "source of change",
        "what drove the change in spend",
        {"core_dimension": ["digital display"]},
        {},
        _DATA_FRAME,
    ),
    (
        "a keyword and a capture",
        "how did google paid search perform",
        {"core_dimension": ["paid search"]},
        {},
        _DATA_FRAME,
    ),
    ("nothing captured", "what was spend last quarter", {}, {}, _DATA_FRAME),
    # The cases that reach step four: a field the captured terms can settle, and one they cannot.
    (
        "a level to pin",
        "what was spend for digital display in the united states",
        {"core_dimension": ["digital display", "united states"]},
        {"country": "specific", "kpi": "irrelevant"},
        _DATA_FRAME,
    ),
    (
        "a level nothing pins",
        "what was spend last quarter",
        {},
        {"country": "specific"},
        _OVERALL_FRAME,
    ),
    (
        "business driver and a trend",
        "show me the spend trend for paid search",
        {"core_dimension": ["paid search"]},
        {"trend": "yes", "business_driver": "specific"},
        _DATA_FRAME,
    ),
)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("name", "query", "spacy_filter", "ner_res", "frame"),
    _CHAIN_CASES,
    ids=[c[0] for c in _CHAIN_CASES],
)
def test_the_whole_chain_matches_the_source(name, query, spacy_filter, ner_res, frame) -> None:
    """The one that matters. Each step feeds the next, so all four can match alone while the
    composition does not: a filter mutated in place, an ignore field added at the wrong point, a
    frame passed on with the wrong index."""

    del name
    intention = "source of change" if "drove the change" in query else "spending"
    ner_filter = {
        **_BASE_FILTER,
        "intention": [intention],
        "period_type": ["quarter"],
        "start": [202504],
        "end": [202506],
    }
    if "trend" in ner_res:
        ner_filter["trend"] = ner_res["trend"]

    def run(ours: bool) -> tuple[Any, ...]:
        if ours:
            postprocess, keywords, combine = (
                postprocess_filter,
                map_keyword_in_query,
                combine_filters,
            )
            specific, reconcile, owned = (
                get_specific_values,
                reconcile_ner,
                apply_filtering_logic,
            )
        else:
            postprocess = _source("_postprocess_filter_given_indicator")
            keywords = _source("map_keyword_in_query")
            combine = _SOURCE_SPACY["combine_filters"]
            specific = _source("_get_specific_values")
            reconcile = _SOURCE_SPACY["_ner_reconciliation"]
            owned = _source("_apply_filtering_logic")

        filt, res = postprocess(
            dict(ner_filter), dict(ner_res), frame.copy(), dict(_INDICATORS), dict(_LEVEL_TYPE)
        )
        narrowed, filt, ignore = keywords(frame.copy(), query, filt, [])
        # generate_readoutdata resets the index here before the spacy step.
        narrowed = narrowed.reset_index(drop=True).copy()
        narrowed, filt, ignore = combine(
            narrowed,
            filt,
            res,
            dict(spacy_filter),
            _CORE_FILTERS,
            ignore,
            list(_SUB_COLS),
            _LEVEL_TYPE,
        )
        filt, res, ignore = specific(narrowed, filt, res, _LEVEL_TYPE, ignore)
        if ours:
            filt, res, ignore, priority = reconcile(
                frame.copy(), filt, res, ignore, _narrowing_config(level_type=_LEVEL_TYPE)
            )
        else:
            filt, res, ignore, priority = reconcile(
                "LINKEDIN",
                12,
                filt,
                res,
                SimpleNamespace(level_type=_LEVEL_TYPE, df_bi=frame, df_br=frame),
                SimpleNamespace(metric_info=_METRIC_INFO),
                ignore,
                None,
            )
        filtered, _ = _apply_with_log(ours, narrowed, filt, res, ignore, priority)
        filtered = owned("LINKEDIN", filt, filtered)
        return filtered, filt, res, ignore

    want_frame, want_filter, want_res, want_ignore = run(ours=False)
    got_frame, got_filter, got_res, got_ignore = run(ours=True)

    assert got_filter == want_filter
    assert got_res == want_res
    assert sorted(got_ignore) == sorted(want_ignore)
    pd.testing.assert_frame_equal(got_frame, want_frame)


# Carries every column reconciliation reads: the portfolio flag, the halo level, and the two
# level columns _check_valid_level falls back on.
_RECONCILE_FRAME = pd.DataFrame(
    {
        "portfolio": ["yes", "no", "no", "yes"],
        "business_unit_for_kpi": ["overall", "lms", "lss", "overall"],
        "business unit focused marketing": ["talent", "talent", "premium", "premium"],
        "country": ["us", "us", "uk", "ca"],
        "kpi": ["delivered bookings", "signups", "overall", "signups"],
        "detailed_kpi": ["business delivered bookings", "member signups", "overall", "overall"],
        "business_driver": ["marketing", "other", "other", "marketing"],
        "activity_group": ["paid search", "email", "events", "digital display"],
        "metric": ["spend"] * 4,
        "start": [202504] * 4,
        "end": [202506] * 4,
    }
)
_METRIC_INFO = {
    "contribution": {"metric": ["contribution"], "mainMetric": ["contribution"]},
    "source of change": {"metric": ["sovc"], "mainMetric": ["sovc"]},
}
_HALO_LEVEL = "business_unit_for_kpi"


def _apply_with_log(
    ours: bool,
    frame: pd.DataFrame,
    ner_filter: dict[str, Any],
    ner_res: dict[str, Any],
    ignore_fields: list[str],
    priority: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """The filter application, ours or the source's, with the prioritised fields."""

    if ours:
        return apply_answer_dict(
            frame,
            ner_filter,
            ner_results=ner_res,
            ignore_fields=ignore_fields,
            prioritized_indicator=priority,
        )
    return _source("_apply_answer_dict_with_log")(
        frame,
        ner_filter,
        res_dict=ner_res,
        ignore_fields=ignore_fields,
        prioritized_indicator=priority,
    )


def _narrowing_config(client_code: str = "LINKEDIN", **overrides: Any) -> NarrowingConfig:
    values: dict[str, Any] = {
        "client_code": client_code,
        "core_filters": _CORE_FILTERS,
        "level_type": _LEVEL_TYPE_HALO,
        "sub_cols": tuple(_SUB_COLS),
        "bi_levels": ("activity_group", "measure_group", "measure"),
        "br_levels": ("business_driver", "business_driver_detail", "activity_group"),
        "metric_info": _METRIC_INFO,
        "fingerprint": "cfg0",
    }
    values.update(overrides)
    return NarrowingConfig(**values)


_LEVEL_TYPE_HALO = {
    **_LEVEL_TYPE,
    "halo_level": [_HALO_LEVEL],
    "halo_tagging": ["business unit focused marketing"],
}

_RECONCILE_CASES: tuple[tuple[str, str, dict[str, Any], dict[str, Any], list[str]], ...] = (
    # A country but no business unit means the portfolio roll-up, and the halo level stops being
    # filtered on at all.
    (
        "country specific, no business unit",
        "LINKEDIN",
        {"intention": ["spending"], "metric": ["spend"], "country": ["us"]},
        {"country": "specific"},
        [],
    ),
    (
        "business unit specific",
        "LINKEDIN",
        {
            "intention": ["spending"],
            "metric": ["spend"],
            "country": ["us"],
            _HALO_LEVEL: ["lms"],
        },
        {"country": "specific", _HALO_LEVEL: "specific"},
        [],
    ),
    # Asked for the portfolio by name.
    (
        "portfolio named in the halo level",
        "LINKEDIN",
        {"intention": ["spending"], "metric": ["spend"], _HALO_LEVEL: ["portfolio"]},
        {_HALO_LEVEL: "specific"},
        [],
    ),
    # Only 'overall' left, so the level falls back to it rather than being dropped.
    (
        "halo level answered overall only",
        "LINKEDIN",
        {"intention": ["spending"], "metric": ["spend"], _HALO_LEVEL: ["overall"]},
        {_HALO_LEVEL: "specific", "country": "irrelevant"},
        [],
    ),
    (
        "neither country nor business unit",
        "LINKEDIN",
        {"intention": ["spending"], "metric": ["spend"]},
        {},
        [],
    ),
    # Ungated by client: a captured term turns a sales question into a contribution one.
    (
        "sales with a capture",
        "LINKEDIN",
        {
            "intention": ["sales"],
            "metric": ["spend"],
            "core_dimension": {"paid search": [{"activity_group": ["paid search"]}]},
            "core_dimension_composite": {},
        },
        {},
        [],
    ),
    (
        "sales without a capture",
        "LINKEDIN",
        {
            "intention": ["sales"],
            "metric": ["spend"],
            "core_dimension": {},
            "core_dimension_composite": {},
        },
        {},
        [],
    ),
    (
        "executive summary pulls in recommendation",
        "LINKEDIN",
        {
            "intention": ["spending"],
            "metric": ["spend"],
            "terms_from_insights": ["executive summary"],
        },
        {},
        [],
    ),
    # Another client: the halo conversion runs and the portfolio block does not.
    (
        "another client with a halo answer",
        "SCOTTS",
        {
            "intention": ["spending"],
            "metric": ["spend"],
            "data": "bi",
            _HALO_LEVEL: ["lms"],
        },
        {_HALO_LEVEL: ["lms"]},
        [],
    ),
    (
        "existing ignore fields are kept",
        "LINKEDIN",
        {"intention": ["spending"], "metric": ["spend"], "country": ["us"]},
        {"country": "specific"},
        ["platform", "retailer"],
    ),
)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("name", "client", "ner_filter", "ner_res", "ignore_fields"),
    _RECONCILE_CASES,
    ids=[c[0] for c in _RECONCILE_CASES],
)
def test_reconciliation_matches_the_source(name, client, ner_filter, ner_res, ignore_fields):
    del name
    source = _SOURCE_SPACY["_ner_reconciliation"]
    ner_data = SimpleNamespace(
        level_type=_LEVEL_TYPE_HALO, df_bi=_RECONCILE_FRAME, df_br=_RECONCILE_FRAME
    )
    ner_data_config = SimpleNamespace(metric_info=_METRIC_INFO)
    base = {"data": "bi", "core_dimension": {}, "core_dimension_composite": {}, **ner_filter}

    want_filter, want_res, want_ignore, want_priority = source(
        client,
        12,
        copy.deepcopy(base),
        dict(ner_res),
        ner_data,
        ner_data_config,
        list(ignore_fields),
        None,
    )
    got_filter, got_res, got_ignore, got_priority = reconcile_ner(
        _RECONCILE_FRAME.copy(),
        copy.deepcopy(base),
        dict(ner_res),
        list(ignore_fields),
        _narrowing_config(client),
    )

    assert got_filter == want_filter
    assert got_res == want_res
    assert got_ignore == want_ignore
    assert got_priority == want_priority


_OWNED_CASES: tuple[tuple[str, str, dict[str, Any]], ...] = (
    ("linkedin roi, nothing captured", "LINKEDIN", {"intention": ["margin roi"]}),
    ("linkedin spend, nothing captured", "LINKEDIN", {"intention": ["spending"]}),
    ("linkedin performance", "LINKEDIN", {"intention": ["performance"]}),
    # An intention the rule does not name.
    ("linkedin contribution", "LINKEDIN", {"intention": ["contribution"]}),
    # Asked for owned media by name, so it stays.
    (
        "owned media captured",
        "LINKEDIN",
        {
            "intention": ["margin roi"],
            "core_dimension": {"email": [{"org_term": "email", "activity_group": ["email"]}]},
        },
    ),
    (
        "a capture that is not owned media",
        "LINKEDIN",
        {
            "intention": ["margin roi"],
            "core_dimension": {
                "paid search": [{"org_term": "paid search", "activity_group": ["paid search"]}]
            },
        },
    ),
    ("another client", "SCOTTS", {"intention": ["margin roi"]}),
)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("name", "client", "ner_filter"), _OWNED_CASES, ids=[c[0] for c in _OWNED_CASES]
)
def test_owned_media_exclusion_matches_the_source(name, client, ner_filter) -> None:
    del name
    source = _source("_apply_filtering_logic")
    base = {"core_dimension": {}, **ner_filter}

    want = source(client, copy.deepcopy(base), _RECONCILE_FRAME.copy())
    got = apply_filtering_logic(client, copy.deepcopy(base), _RECONCILE_FRAME.copy())

    pd.testing.assert_frame_equal(got.reset_index(drop=True), want.reset_index(drop=True))


# No row is both portfolio=yes and country=us, so these two filters cannot both apply and
# whichever runs first is the one that survives.
_CONFLICT_FRAME = pd.DataFrame(
    {
        "portfolio": ["yes", "yes", "no", "no"],
        "country": ["ca", "uk", "us", "us"],
        "metric": ["spend"] * 4,
    }
)

_PRIORITY_CASES: tuple[tuple[str, dict[str, Any], list[str]], ...] = (
    ("no priority", {"country": ["us"], "portfolio": ["yes"]}, []),
    ("portfolio first", {"country": ["us"], "portfolio": ["yes"]}, ["portfolio"]),
    # Already first in the dict, so prioritising it changes nothing.
    ("priority already first", {"portfolio": ["yes"], "country": ["us"]}, ["portfolio"]),
    # Named twice: the source dedupes rather than applying it again.
    ("priority duplicates a filter", {"portfolio": ["yes"], "country": ["us"]}, ["portfolio"]),
    ("priority on a field with no filter", {"country": ["us"]}, ["portfolio"]),
)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("name", "answer_dict", "priority"), _PRIORITY_CASES, ids=[c[0] for c in _PRIORITY_CASES]
)
def test_prioritised_fields_match_the_source(name, answer_dict, priority) -> None:
    """Reconciliation's whole effect on the portfolio field is to put it first. That only shows
    up when something conflicts with it, because a filter that would empty the result is skipped
    rather than applied -- so order decides which of the two is dropped."""

    del name
    want_frame, want_log = _apply_with_log(
        False, _CONFLICT_FRAME.copy(), dict(answer_dict), {}, [], list(priority)
    )
    got_frame, got_log = _apply_with_log(
        True, _CONFLICT_FRAME.copy(), dict(answer_dict), {}, [], list(priority)
    )

    pd.testing.assert_frame_equal(got_frame, want_frame)
    assert got_log == want_log


@pytest.mark.unit
def test_the_priority_cases_are_not_all_the_same_answer() -> None:
    """Guards the parity above from passing on a fixture where order makes no difference."""

    without, _ = _apply_with_log(
        True, _CONFLICT_FRAME.copy(), {"country": ["us"], "portfolio": ["yes"]}, {}, [], []
    )
    with_priority, _ = _apply_with_log(
        True,
        _CONFLICT_FRAME.copy(),
        {"country": ["us"], "portfolio": ["yes"]},
        {},
        [],
        ["portfolio"],
    )

    assert list(without["country"]) == ["us", "us"]
    assert list(with_priority["portfolio"]) == ["yes", "yes"]
