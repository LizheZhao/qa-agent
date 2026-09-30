"""The steps that narrow a filter before it is applied.

ask-genome-core runs four of these between map_filter_to_value and the filter application, each
returning a mutated ner_filter, ner_res, ignore_fields and frame into the next:

    _postprocess_filter_given_indicator -> map_keyword_in_query -> combine_filters
    -> _get_specific_values -> (_ner_reconciliation) -> _apply_answer_dict_with_log

Order is load-bearing. _get_specific_values reads the frame combine_filters already narrowed, so
running them the other way round gives it the wrong data even if both are ported correctly. They
are ported here in source order, and tests/unit/test_ask_genome_narrowing_parity.py compares the
whole chain against the real functions rather than each one alone.

One shape decision. The source carries intention, metrics, resolved field values and time in one
`ner_filter` dict, while the rest of this package keeps them apart in `applied_filter` and
`data_source`. These functions read that combined shape, so it is rebuilt here rather than each
function being bent to our layout: a verbatim port can be compared to the original, an adapted one
cannot.

_ner_reconciliation stays deferred, as agreed. It sits between _get_specific_values and the apply,
and it produces the `prioritized_indicator` that _apply_answer_dict_with_log takes, so parity runs
the source with that argument left out too.

The chain splits in two at combine_filters. The first two steps read the frame but only return
dicts; from combine_filters on, the frame itself is narrowed. `narrow_and_apply` below is that
second half, and it is the single entry point both the live node and table.py's replay call --
which is what lets a stored recipe record the dicts at the split and rebuild the rows from them.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, NamedTuple

import numpy as np
import pandas as pd

from ask_genome_agent.config import NarrowingConfig
from ask_genome_agent.nodes.question_understanding.support.filtering import (
    apply_answer_dict,
    apply_answer_dict_unlogged,
)
from ask_genome_agent.nodes.question_understanding.support.levels import (
    dedup_levels,
    report_levels,
)
from ask_genome_agent.nodes.question_understanding.support.time_filter import subtract_periods

logger = logging.getLogger(__name__)

_SOURCE_OF_CHANGE = "source of change"

# Keys on a core-dimension filter that describe the filter itself rather than a column to match.
_FILTER_META_KEYS = frozenset({"AKA", "org_term", "source", "halo_to_ignore", "detail"})

# Capture types whose terms are ORed together, and those ORed within a column group and ANDed
# across groups. 'brand or tv' should widen; 'us' and 'video' should both have to hold.
_UNION_TYPES = frozenset(
    {"core_dimension", "core_dimension_composite", "diag_tagging", "business_driver_captured"}
)
_GROUPED_TYPES = frozenset({"custom_level", "custom_tagging"})

# Verbatim from map_keyword_in_query. None of it is LINKEDIN vocabulary.
_KEYWORD_MAP: dict[str, list[str]] = {
    "amazon": ["amazon", "amz"],
    "meta": ["meta", "facebook", "fb"],
    "tiktok": ["tiktok"],
    "ttd(the trade desk)": ["ttd", "trade desk"],
    "youtube": ["youtube", "trueview"],
    "google": ["google"],
    "pinterest": ["pinterest"],
    "twitter": ["twitter"],
    "walmart": ["walmart"],
    "bing": ["bing"],
    "target": ["target"],
    "iheart": ["iheart"],
    "kroger": ["kroger"],
    "sams club": ["sams club", "sam's club"],
    "podcast": ["podcast"],
    "instacart": ["instacart"],
    "ahw": ["ahw"],
    "equity": ["masterbrand"],
    "masterbrand": ["masterbrand"],
    "sub-brand": ["optic white", "total tp"],
}
_NON_PLATFORM_KEYWORDS = frozenset({"ahw", "equity", "masterbrand", "sub-brand"})

# Verbatim from _apply_filtering_logic. Four of the seven are real LINKEDIN activity_groups.
_OWNED_MEDIA = [
    "lol non lan",
    "events",
    "seo",
    "email",
    "owned",
    "owned marketing",
    "owned media",
]


class NarrowedResult(NamedTuple):
    """What the data-dependent half of the chain produces.

    `frame` carries a `record_level` column and `data_levels` names the three levels reported on,
    which is ask-genome-core's three dataframes held as one. levels.split_by_level produces them.
    """

    frame: pd.DataFrame
    ner_filter: dict[str, Any]
    ner_results: dict[str, Any]
    ignore_fields: list[str]
    log: dict[str, Any]
    data_levels: list[str]


def narrow_and_apply(
    frame: pd.DataFrame,
    ner_filter: dict[str, Any],
    ner_results: dict[str, Any],
    ignore_fields: list[str],
    spacy_filter: dict[str, list[str]],
    config: NarrowingConfig,
) -> NarrowedResult:
    """Everything from combine_filters onwards: the steps whose result depends on the rows.

    The live path and table.py's replay both come through here, from the same four values, so a
    rebuilt table cannot drift from the one that was answered with. Everything earlier in the
    chain produces only those four values, which is why a recipe does not have to store the
    question.
    """

    # The steps below mutate what they are handed, and the caller keeps these as the recipe.
    ner_filter, ner_results, ignore_fields = (
        dict(ner_filter),
        dict(ner_results),
        list(ignore_fields),
    )
    # generate_readoutdata resets the index before the spacy step, and _get_mask builds masks
    # positionally, so replay has to start from the same shape the live path did.
    original = frame
    frame = frame.reset_index(drop=True).copy()
    # generate_readoutdata runs the spacy step only for a client that has the config for it, and
    # skipping it matters beyond the captures: combine_filters also drops exact duplicate rows.
    if config.core_filters:
        frame, ner_filter, ignore_fields = combine_filters(
            frame,
            ner_filter,
            ner_results,
            spacy_filter,
            config.core_filters,
            ignore_fields,
            list(config.sub_cols),
            config.level_type,
        )
    ner_filter, ner_results, ignore_fields = get_specific_values(
        frame, ner_filter, ner_results, config.level_type, ignore_fields
    )
    # Reconciliation reads the unnarrowed frame, as the source does: it re-reads NerData rather
    # than using whatever the spacy step left.
    ner_filter, ner_results, ignore_fields, prioritized = reconcile_ner(
        original, ner_filter, ner_results, ignore_fields, config
    )
    filtered, log = apply_answer_dict(
        frame,
        ner_filter,
        ner_results=ner_results,
        ignore_fields=ignore_fields,
        prioritized_indicator=prioritized,
    )
    filtered = apply_filtering_logic(config.client_code, ner_filter, filtered)

    # The hierarchy step, gated on the intention's metric being present exactly as the source
    # gates it: with nothing to report there is no level to report it at. Where the source then
    # returns three empty frames, the rows are kept here and apply_data_filters says so in the
    # rejection message instead -- an unusable answer is worth being able to look at.
    data_levels = config.levels_for(str(ner_filter.get("data", "bi")))
    if _has_main_metric(filtered, ner_filter):
        is_halo_tagging = any(ner_filter.get(x) for x in config.level_type.get("halo_tagging", []))
        data_levels = report_levels(filtered, data_levels, ner_filter, is_halo_tagging)
        filtered = dedup_levels(filtered, data_levels, config.level_type).reset_index(drop=True)

    return NarrowedResult(filtered, ner_filter, ner_results, ignore_fields, log, data_levels)


def _has_main_metric(frame: pd.DataFrame, ner_filter: dict[str, Any]) -> bool:
    if "metric" not in frame.columns:
        return False
    present = frame["metric"].unique()
    return any(x for x in ner_filter.get("main_metric", []) if x in present)


def build_ner_filter(
    data_source: dict[str, Any],
    answer_dict: dict[str, list[Any]],
    ner_results: dict[str, Any],
) -> dict[str, Any]:
    """The combined dict the source's narrowing steps read, assembled from our split state.

    `data_source` is resolve_data_source's entry; `data` is the source's name for which of the
    two frames this intention reads.
    """

    combined: dict[str, Any] = {
        "intention": [data_source["intention"]],
        "data": data_source["source"],
        "metric": list(data_source["metric"]),
        "main_metric": list(data_source["main_metric"]),
        **answer_dict,
    }
    if "trend" in ner_results:
        combined["trend"] = ner_results["trend"]
    return combined


def postprocess_filter(
    ner_filter: dict[str, Any],
    ner_dict: dict[str, Any],
    data: pd.DataFrame,
    indicators: dict[str, Any],
    level_type: dict[str, list[str]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Ported from data_filtering.py:_postprocess_filter_given_indicator.

    Three live rules. Rules 3 and 4 are commented out in the source and stay that way: rule 3
    moved into sequential NER, rule 4 into reconciliation.

    Rule 5 is the one with reach beyond time. It replaces a general_tagging column answered 'all'
    with the column's real values in both dicts, which means the apply step no longer sees 'all'
    for it and stops widening to rows that never carried the field. LINKEDIN has no
    general_tagging, so it is inert there.
    """

    ner_filter = dict(ner_filter)
    ner_dict = dict(ner_dict)
    general_tagging = level_type.get("general_tagging", [])
    ner_dict["intention"] = ner_filter["intention"]

    intentions = ner_filter.get("intention", [])
    if any(item == _SOURCE_OF_CHANGE for item in intentions):
        ner_dict["trend"] = "no"
        ner_filter["trend"] = "no"

    if _SOURCE_OF_CHANGE in intentions or ner_filter.get("trend") == "yes":
        _widen_time_for_trend(ner_filter, data, indicators)

    for column in general_tagging:
        if ner_dict.get(column, "irrelevant") == "all":
            real = list(data[column].dropna().unique())
            ner_dict[column] = real
            ner_filter[column] = real

    return ner_filter, ner_dict


def map_keyword_in_query(
    frame: pd.DataFrame, query: str, ner_filter: dict[str, Any], ignore_fields: list[str]
) -> tuple[pd.DataFrame, dict[str, Any], list[str]]:
    """Ported from data_filtering.py:map_keyword_in_query.

    Almost inert, and ported anyway so the chain's ner_filter and ignore_fields match the source
    exactly. The line that would narrow the frame by these keywords is commented out there, and
    both keys it sets go straight into ignore_fields, so the apply step skips them. Nothing
    between here and the filter reads either one.

    The keyword map is other clients' vocabulary: retailers, social platforms, Colgate sub-brands.
    LINKEDIN matches none of it and has no `retailer` column, so for this client the whole
    function reduces to two more entries in ignore_fields.
    """

    result_filter = dict(ner_filter)
    matched_any = False
    platforms: list[str] = []

    for keyword, variants in _KEYWORD_MAP.items():
        if keyword not in query.lower():
            continue
        if keyword not in _NON_PLATFORM_KEYWORDS:
            platforms.append(keyword)
        for variant in variants:
            for column in ("activity_group", "measure_group", "measure"):
                if column in frame.columns:
                    matched_any |= any(
                        variant in value for value in frame[column].dropna().unique()
                    )

    if matched_any:
        result_filter["platform"] = platforms
    if ("retailer" in query.lower() or "retail" in query.lower()) and "retailer" in frame.columns:
        result_filter["retailer"] = "yes"

    return frame.copy(), result_filter, [*ignore_fields, "platform", "retailer"]


def categorize_filters(
    spacy_filter: dict[str, list[str]],
    core_dimension_filters: dict[str, list[dict[str, Any]]],
    level_type: dict[str, list[str]],
    product_halo_indicator: bool,
    ner_filter: dict[str, Any],
) -> dict[str, Any]:
    """Ported from filter_generator.py:categorize_filters.

    Turns the terms spaCy captured into buckets keyed by how they should combine. A term's
    bucket comes from the columns its configured filters touch, not from the capture type: a
    term that resolves to a `split` column is diagnostic tagging whatever spaCy called it.

    Terms with no configured filter at all fall out into `terms_from_insights`, which nothing
    here applies. That is how 'response curve' survives capture without narrowing anything.
    """

    core_dim_res: dict[str, list[dict[str, Any]]] = defaultdict(list)
    core_dim_composite_res: dict[str, list[dict[str, Any]]] = defaultdict(list)
    custom_level_res: dict[str, list[dict[str, Any]]] = defaultdict(list)
    custom_tagging_res: dict[str, list[dict[str, Any]]] = defaultdict(list)
    diag_tagging_res: dict[str, list[dict[str, Any]]] = defaultdict(list)
    bd_res: dict[str, list[dict[str, Any]]] = defaultdict(list)
    terms_from_insights: list[str] = []
    data_source = ner_filter.get("data", "br")

    specific_levels = level_type.get("general_level", [])
    specific_taggings = level_type.get("general_tagging", [])

    for captured_type, captured_terms in spacy_filter.items():
        if captured_type == "halo_tagging":
            continue
        for captured_term in captured_terms:
            core_dim_filter = list(core_dimension_filters.get(captured_term, []))
            # A halo filter only makes sense once a halo field was actually asked for.
            if not product_halo_indicator and data_source == "bi":
                core_dim_filter = [
                    f for f in core_dim_filter if f.get("halo_to_ignore", "0") == "0"
                ]

            for term_filter in core_dim_filter:
                if data_source not in term_filter.get("source", ""):
                    continue
                if captured_type == "core_dimension_composite":
                    core_dim_composite_res[captured_term].append(term_filter)
                elif captured_type == "business_driver" and "business_driver" in term_filter:
                    bd_res[captured_term].append(term_filter)
                elif captured_type == "metric":
                    pass
                elif {"split", "split_type"} & term_filter.keys():
                    diag_tagging_res[captured_term].append(term_filter)
                elif set(specific_taggings) & term_filter.keys():
                    custom_tagging_res[captured_term].append(term_filter)
                elif set(specific_levels) & term_filter.keys():
                    custom_level_res[captured_term].append(term_filter)
                else:
                    core_dim_res[captured_term].append(term_filter)
            if not core_dim_filter:
                terms_from_insights.append(captured_term)

    return {
        "core_dimension": core_dim_res,
        "core_dimension_composite": core_dim_composite_res,
        "custom_level": custom_level_res,
        "custom_tagging": custom_tagging_res,
        "diag_tagging": diag_tagging_res,
        "terms_from_insights": terms_from_insights,
        "business_driver_captured": bd_res,
    }


def apply_categorized_filters(
    df: pd.DataFrame,
    categorized_filter: dict[str, Any],
    ignore_columns: list[str],
    combine_all: bool = False,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Ported from filter_generator.py:apply_categorized_filters.

    Builds one mask per capture type and reduces them. `combine_all` ORs those masks instead of
    ANDing them, which is how combine_filters takes a first wide pass before narrowing.

    An empty categorized filter means no spaCy term constrained anything, so everything matches.
    """

    filtered_data = df.copy()
    skip_types = {"terms_from_insights", *ignore_columns}

    def apply_aka_labeling(
        data: pd.DataFrame, captured_term: str, term_filters: list[dict[str, Any]]
    ) -> pd.DataFrame:
        """Records the term the user actually said on the rows an alias filter matched."""

        aka_filters = [f for f in term_filters if f.get("AKA", "0") == "1"]
        if aka_filters:
            org_term = next((f.get("org_term", captured_term) for f in aka_filters), captured_term)
            aka_mask = _get_mask(data, aka_filters, ignore_columns)
            data = _label_data_with_term(data, aka_mask, org_term)
        return data

    and_masks: list[np.ndarray] = []
    for captured_type, captured_term_dict in categorized_filter.items():
        if captured_type in skip_types:
            continue

        if captured_type in _UNION_TYPES:
            or_masks = []
            for captured_term, term_filters in captured_term_dict.items():
                filtered_data = apply_aka_labeling(filtered_data, captured_term, term_filters)
                or_masks.append(_get_mask(filtered_data, term_filters, ignore_columns))
            if or_masks:
                and_masks.append(np.logical_or.reduce(or_masks))

        elif captured_type in _GROUPED_TYPES:
            col_group_masks: dict[frozenset[str], list[np.ndarray]] = defaultdict(list)
            for captured_term, term_filters in captured_term_dict.items():
                filtered_data = apply_aka_labeling(filtered_data, captured_term, term_filters)
                term_mask = _get_mask(filtered_data, term_filters, ignore_columns)
                col_group_masks[_target_columns(term_filters)].append(term_mask)
            if col_group_masks:
                group_masks = [np.logical_or.reduce(m) for m in col_group_masks.values()]
                and_masks.append(np.logical_and.reduce(group_masks))

        else:
            logger.info("Captured term type: %s not supported.", captured_type)

    if not and_masks:
        spacy_mask = np.ones(len(filtered_data), dtype=bool)
    elif combine_all:
        spacy_mask = np.logical_or.reduce(and_masks)
    else:
        spacy_mask = np.logical_and.reduce(and_masks)

    return spacy_mask, filtered_data


def combine_filters(
    data: pd.DataFrame,
    ner_filter: dict[str, Any],
    ner_res: dict[str, Any],
    spacy_filter: dict[str, list[str]],
    core_dimension_filters: dict[str, list[dict[str, Any]]],
    ignore_fields: list[str],
    sub_cols: list[str],
    level_type: dict[str, list[str]],
) -> tuple[pd.DataFrame, dict[str, Any], list[str]]:
    """Ported from filter_generator.py:combine_filters.

    The step that makes 'print' select print. The classifier answers one value per configured
    field, which cannot express a term that spans several columns; this resolves the captured
    terms against the spaCy config and narrows the frame by them directly.

    Two passes. The first ORs every capture type to drop rows no captured term mentions at all,
    the second ANDs them for real. In between the frame splits on `custom_aggregated`, because
    diagnostic rows carry the split columns that normal rows do not and would otherwise be
    filtered away by a diagnostic term they cannot satisfy.

    Every column a captured term resolved goes into ignore_fields, so the later apply step does
    not filter the same column again from the classifier's coarser answer.
    """

    def covered_columns(categorized: dict[str, Any]) -> set[str]:
        covered: set[str] = set()
        for bucket in categorized.values():
            if not isinstance(bucket, dict):
                continue
            for term_filters in bucket.values():
                for term_filter in term_filters:
                    covered.update(col for col in term_filter if col in data.columns)
        return covered

    # Copied, not appended to: the source extends the caller's list, which for us is cached
    # config that every later subquery would then see grown.
    sub_cols = list(sub_cols)
    if spacy_filter.get("core_dimension_composite", []):
        sub_cols += level_type.get("halo_level", []) + level_type.get("halo_tagging", [])

    combined_filter: dict[str, Any] = {}
    core_dimensions = spacy_filter.get("core_dimension", []) + spacy_filter.get(
        "core_dimension_composite", []
    )
    if (
        core_dimensions
        and any(x for x in core_dimensions if x in core_dimension_filters)
        and len(core_dimension_filters) > 0
    ):
        ignore_fields += sub_cols
    combined_filter.update(dict(ner_filter))
    combined_filter["metric"] = list(combined_filter["metric"]) + [
        x for x in spacy_filter.get("metric", []) if x not in ner_filter["metric"]
    ]

    # A list here means the halo field was answered with real values rather than left open.
    halo_tagging = next(iter(level_type.get("halo_tagging") or []), None)
    product_halo_indicator = halo_tagging is not None and isinstance(
        ner_filter.get(halo_tagging), list
    )

    categorized_filter = categorize_filters(
        spacy_filter=spacy_filter,
        core_dimension_filters=core_dimension_filters,
        level_type=level_type,
        product_halo_indicator=product_halo_indicator,
        ner_filter=ner_filter,
    )

    spacy_mask_combined, filtered_data_combined = apply_categorized_filters(
        df=data, categorized_filter=categorized_filter, ignore_columns=[], combine_all=True
    )
    filtered_data_combined = filtered_data_combined[spacy_mask_combined]

    diag_data_filtered = pd.DataFrame()
    if "custom_aggregated" in filtered_data_combined.columns:
        norm_data = filtered_data_combined[
            filtered_data_combined["custom_aggregated"] != "diagnostic"
        ]
        diag_data = filtered_data_combined[
            filtered_data_combined["custom_aggregated"] == "diagnostic"
        ]

        norm_data_mask, norm_data_filtered = apply_categorized_filters(
            df=norm_data,
            categorized_filter=categorized_filter,
            ignore_columns=["diag_tagging"],
            combine_all=False,
        )
        norm_data_filtered = norm_data_filtered[norm_data_mask].copy()

        if len(diag_data) > 0 and len(categorized_filter.get("diag_tagging", {})):
            diag_data_mask, diag_data_filtered = apply_categorized_filters(
                df=diag_data,
                categorized_filter=categorized_filter,
                ignore_columns=[],
                combine_all=False,
            )
            diag_data_filtered = diag_data_filtered[diag_data_mask].copy()
    else:
        norm_data_mask, norm_data_filtered = apply_categorized_filters(
            df=filtered_data_combined,
            categorized_filter=categorized_filter,
            ignore_columns=[],
            combine_all=False,
        )
        norm_data_filtered = norm_data_filtered[norm_data_mask].copy()

    if len(diag_data_filtered):
        combined_filter.update(categorized_filter)
        covered_cols = covered_columns(categorized_filter)
    else:
        res_filter = {k: v for k, v in categorized_filter.items() if k != "diag_tagging"}
        covered_cols = covered_columns(res_filter)
        combined_filter.update(res_filter)

    frames = [f for f in (norm_data_filtered, diag_data_filtered) if len(f)]
    filtered_data = (
        pd.concat(frames, ignore_index=True).drop_duplicates()
        if frames
        else norm_data_filtered.reset_index(drop=True)
    )
    ignore_fields.extend(covered_cols)
    return filtered_data, dict(combined_filter), ignore_fields


def get_specific_values(
    df: pd.DataFrame,
    ner_filter: dict[str, Any],
    ner_res: dict[str, Any],
    level_type: dict[str, list[str]],
    ignore_fields: list[str],
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    """Ported from data_filtering.py:_get_specific_values.

    The classifier can answer a field 'specific' without saying which value, and combine_filters
    leaves its captured terms in the filter without saying which field they belong to. This
    settles both against the data: a captured term that is a real value of one of the field's
    columns pins that field to it, and a field nothing pins falls back to 'overall' where the
    data has such a row.

    It reads the frame combine_filters already narrowed, which is why the order of these two is
    load-bearing rather than incidental.
    """

    corrections: dict[str, str] = {}
    custom_levels = level_type.get("general_level", [])
    custom_taggings = level_type.get("general_tagging", [])

    for key, val in ner_res.items():
        if key == "business_driver":
            em_cols = ["activity_group", "business_driver_detail", "business_driver"]
            spacy_types = ["core_dimension", "business_driver_captured"]
        elif key in custom_levels:
            em_cols, spacy_types = custom_levels, ["custom_level"]
        elif key in custom_taggings:
            em_cols, spacy_types = custom_taggings, ["custom_tagging"]
        else:
            continue

        answered_specific = isinstance(val, str) and val == "specific"
        if not answered_specific and not any(ner_filter.get(st) for st in spacy_types):
            continue

        # The values these columns really hold, after combine_filters narrowed the frame.
        present = {c: set(df[c].dropna()) for c in em_cols}
        if key == "business_driver":
            present = {key: set(df[em_cols].stack().dropna())}

        captured: set[Any] = set()
        for spacy_type in spacy_types:
            bucket = ner_filter.get(spacy_type)
            if isinstance(bucket, dict):
                if spacy_type == "business_driver_captured":
                    for term, filters in bucket.items():
                        captured.update(
                            t for f in filters for t in f.get("business_driver", [term])
                        )
                # A filter that says nothing about a column contributes the captured term itself,
                # which then survives only if the column really holds a value by that name.
                for term, filters in bucket.items():
                    captured.update(t for f in filters for c in em_cols for t in f.get(c, [term]))
            elif isinstance(bucket, list):
                captured.update(bucket)

        matched = {col: [v for v in captured if v in values] for col, values in present.items()}
        pinned = {col: values for col, values in matched.items() if values}
        if key in pinned:
            ner_filter.update(pinned)
            ignore_fields += [key]
            for col in pinned:
                corrections[col] = "specific"
        else:
            corrections[key] = "irrelevant"
            if "overall" in df[key].unique():
                ner_filter.update({key: ["overall"]})

    ner_res.update(corrections)
    return dict(ner_filter), dict(ner_res), ignore_fields


def reconcile_ner(
    frame: pd.DataFrame,
    ner_filter: dict[str, Any],
    ner_results: dict[str, Any],
    ignore_fields: list[str],
    config: NarrowingConfig,
) -> tuple[dict[str, Any], dict[str, Any], list[str], list[str]]:
    """Ported from filter_generator.py:_ner_reconciliation.

    Hardcoded per-client corrections, kept as the source has them rather than generalised -- its
    own comment is `TODO: general reconciliation logic`. Two of the three matter for LINKEDIN:

    - portfolio. A question that names a country but no business unit is asking about the
      portfolio roll-up, so `portfolio` is pinned and made the first filter applied. Otherwise it
      is pinned to the detail rows and the level fields are checked for something real to match.
      That single field splits bi 38k/460k, so it decides the answer, not just narrows it.
    - sales -> contribution. Ungated by client: once a captured term is in play, a 'sales'
      question is really a contribution question, so the intention and its metrics are rewritten.

    `frame` is the unnarrowed source frame, which is what the source re-reads here.
    """

    prioritized_indicator: list[str] = []
    ner_filter, ner_results = dict(ner_filter), dict(ner_results)
    halo_level = next(iter(config.level_type.get("halo_level") or []), None)
    halo_tagging = next(iter(config.level_type.get("halo_tagging") or []), None)

    if halo_level and halo_tagging and config.client_code not in ("LINKEDIN",):
        ner_filter, ner_results = halo_level_to_tagging(
            halo_level, halo_tagging, frame, ner_filter, ner_results
        )

    if config.client_code == "FTR":
        # The only branch that needs the filter actually applied; elsewhere the source reads
        # nothing off it but the column names, which filtering never changes.
        applied = apply_answer_dict_unlogged(
            frame, ner_filter, ner_results=ner_results, ignore_fields=ignore_fields
        )
        drivers = applied["business_driver"].dropna().unique()
        if all(drivers == "other") and ner_filter["intention"] == ["contribution"]:
            _rewrite_intention(ner_filter, ner_results, "source of change", config)

    captured = ner_filter.get("core_dimension") or {}
    captured_composite = ner_filter.get("core_dimension_composite") or {}
    if len(captured) + len(captured_composite) and ner_filter["intention"] == ["sales"]:
        _rewrite_intention(ner_filter, ner_results, "contribution", config)

    if config.client_code == "LINKEDIN" and "portfolio" in frame.columns and halo_level:
        is_specific_bu = isinstance(ner_filter.get(halo_level), list) and ner_results.get(
            halo_level
        ) not in ("irrelevant", "all")
        is_specific_country = ner_results.get("country") == "specific"
        if (not is_specific_bu and is_specific_country) or (
            "portfolio" in ner_filter.get(halo_level, [])
        ):
            ner_filter["portfolio"] = ["yes"]
            ignore_fields.append(halo_level)
        else:
            ner_filter["portfolio"] = ["no"]
            for column in (halo_level, "kpi", "detailed_kpi"):
                _keep_valid_level(ner_filter, column, ignore_fields, frame)
        prioritized_indicator.append("portfolio")

    if config.client_code == "LINKEDIN":
        na_terms = ner_filter.get("terms_from_insights", [])
        if "executive summary" in na_terms:
            na_terms += ["recommendation"]

    return ner_filter, ner_results, ignore_fields, prioritized_indicator


def halo_level_to_tagging(
    halo_level: str,
    halo_tagging: str,
    data: pd.DataFrame,
    ner_filter: dict[str, Any],
    ner_dict: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Ported from data_filtering.py:_halo_level_to_tagging.

    Moves a halo answer from the level column onto the tagging column and rolls the level up to
    'overall'. Reconciliation gates this off for LINKEDIN, so it is dead for this deployment and
    ported for the clients where it is not.
    """

    if not (
        halo_level
        and halo_tagging
        and ner_filter.get("data", "br") == "bi"
        and ner_filter.get(halo_tagging) is None
    ):
        return dict(ner_filter), dict(ner_dict)

    available = list(data[halo_tagging].dropna().unique())
    if ner_dict.get(halo_level, "irrelevant") == "all":
        ner_filter[halo_tagging] = [x for x in available if x not in ("equity", "ahw")]
        ner_filter[halo_level] = ["overall"]
        ner_dict[halo_tagging] = ner_filter[halo_tagging]
    elif (
        isinstance(ner_dict.get(halo_level), list)
        and "overall" not in ner_filter.get(halo_level, [])
        and "overall" in data[halo_level].unique()
    ):
        mapping = {"optic white tp": "optic white"}
        ner_filter[halo_tagging] = [
            mapping.get(x, x)
            for x in ner_dict.get(halo_level, [])
            if x in set(available).union(mapping)
        ]
        ner_filter[halo_level] = ["overall"]
        ner_dict[halo_tagging] = ner_filter[halo_tagging]
        if len(ner_filter[halo_tagging]) < 1:
            del ner_filter[halo_tagging]

    return dict(ner_filter), dict(ner_dict)


def apply_filtering_logic(
    client_code: str, ner_filter: dict[str, Any], data: pd.DataFrame
) -> pd.DataFrame:
    """Ported from data_filtering.py:_apply_filtering_logic.

    Owned media is not something a ROI, performance or spend question is asking about unless it
    says so, so those rows come out. 11% of LINKEDIN's bi and 15% of its br, which is why this
    runs after the filter rather than being folded into it: the exclusion is about the question,
    not about any field the user answered.
    """

    if client_code != "LINKEDIN" or not any(
        x in ner_filter["intention"] for x in ("margin roi", "performance", "spending")
    ):
        return data

    org_terms = {
        f.get("org_term")
        for filters in ner_filter.get("core_dimension", {}).values()
        for f in filters
    }
    if any(x for x in _OWNED_MEDIA if x in org_terms):
        return data
    return data[~data["activity_group"].isin(_OWNED_MEDIA)].reset_index(drop=True)


def _rewrite_intention(
    ner_filter: dict[str, Any],
    ner_results: dict[str, Any],
    intention: str,
    config: NarrowingConfig,
) -> None:
    """Swap the intention and the metrics that go with it, in both dicts."""

    info = config.metric_info.get(intention, {})
    for target in (ner_filter, ner_results):
        target["intention"] = [intention]
        target["main_metric"] = info.get("mainMetric", [])
        target["metric"] = info.get("metric", [])


def _keep_valid_level(
    ner_filter: dict[str, Any], column: str, ignore_fields: list[str], data: pd.DataFrame
) -> None:
    """Ported from _ner_reconciliation's _check_valid_level. Drop the 'overall' roll-up from a
    level that has real values, fall back to it when it has none, and stop filtering on the
    column entirely when the data has neither."""

    real = [x for x in ner_filter.get(column, []) if x != "overall"]
    if real:
        ner_filter[column] = real
    elif column in data and "overall" in data[column].unique():
        ner_filter[column] = ["overall"]
    else:
        ignore_fields.append(column)


def _get_mask(
    df: pd.DataFrame, dict_list: list[dict[str, Any]], ignored_columns: list[str]
) -> np.ndarray:
    """Ported from filter_generator.py:_get_mask. ANDs the columns within one filter, ORs the
    filters, and ignores any column the caller has already handled."""

    empty = np.zeros(len(df), dtype=bool)
    if not dict_list:
        return empty

    if "custom_aggregated" in df.columns:
        normal_mask = (df["custom_aggregated"] != "yes").to_numpy()
    else:
        normal_mask = empty

    or_masks: list[np.ndarray] = [empty]
    for dic in dict_list:
        valid_cols = [x for x in dic if x in df.columns and x not in ignored_columns]
        if not valid_cols:
            continue
        matched = np.logical_and.reduce([df[col].isin(dic[col]).to_numpy() for col in valid_cols])
        # A filter marked this way only applies to rows that are not custom aggregations.
        if dic.get("detail") == "ignore when cust_agg":
            matched = matched & normal_mask
        or_masks.append(matched)

    reduced: np.ndarray = np.logical_or.reduce(or_masks)
    return reduced


def _target_columns(term_filters: list[dict[str, Any]]) -> frozenset[str]:
    """The data columns a term's filters actually match on, which is what groups terms."""

    return frozenset(k for f in term_filters for k in f if k not in _FILTER_META_KEYS)


def _label_data_with_term(df: pd.DataFrame, mask: np.ndarray, term: str) -> pd.DataFrame:
    """Ported from filter_generator.py:_label_data_with_term."""

    if "core_dimension_term" in df.columns:
        df.loc[mask, "core_dimension_term"] += term + ","
    else:
        df["core_dimension_term"] = ""
        df.loc[mask, "core_dimension_term"] = term + ","
    return df


def _widen_time_for_trend(
    ner_filter: dict[str, Any], data: pd.DataFrame, indicators: dict[str, Any]
) -> None:
    """A trend needs more than one period, so the range reaches back to trend_report_length."""

    periods = ner_filter.get("period_type", [])
    start = ner_filter.get("start", [])
    end = ner_filter.get("end", [])
    if len(start) >= 2:
        return

    if "year" in periods or "fiscal year" in periods:
        scoped = data[data["period_type"].isin(periods)]
        valid_start = list(scoped["start"].unique())
        valid_end = list(scoped["end"].unique())
    else:
        lengths = indicators.get("trend_report_length") or {}
        longest = next((v for k, v in lengths.items() if k in periods), 99)
        valid_start = [
            d for x in start for d in subtract_periods(x, longest - len(start), periods[0])
        ]
        valid_end = [d for x in end for d in subtract_periods(x, longest - len(end), periods[0])]

    ner_filter["start"] = [int(x) for x in sorted(set(valid_start + list(start)))]
    ner_filter["end"] = [int(x) for x in sorted(set(valid_end + list(end)))]
