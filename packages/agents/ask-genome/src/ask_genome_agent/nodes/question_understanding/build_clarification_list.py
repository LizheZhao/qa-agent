"""Deterministic: decides which fields still need user clarification, and what values to offer.

Ports only what's live in ask-genome-core's update_clarification_list: the incomplete-answer
trigger (raised upstream in fill_remaining_fields) and the halo-mismatch check. The confidence,
specificity and time checks sit commented out in the source; restoring them would be a behaviour
change, so discussion required.

Keeps the raw probabilities, spaCy captures and field provenance in ner_state rather than just
the final flags. After discussions with Lizhe, the plan is an LLM-based flagger that reasons over
all filters at once. It can replace this rule as a drop-in node, without re-plumbing extraction.
"""

import hashlib
from typing import Any

import pandas as pd
from orchestration_core import QUEUED_CLARIFICATIONS_CHANNEL

from ask_genome_agent.config import NerData, NerDataConfig, load_ner_data, load_ner_data_config
from ask_genome_agent.nodes.question_understanding.support.clarification_requests import (
    build_clarification_requests,
)
from ask_genome_agent.state import AskGenomeState

# Which spaCy bucket holds candidates for a field, keyed by the field's level type. Taken from
# the source's own spacy_key_map in update_clarification_list, extended with the halo levels.
_SPACY_BUCKET_FOR_LEVEL = {
    "general_level": "custom_level",
    "general_tagging": "custom_tagging",
    "halo_level": "halo_tagging",
    "halo_tagging": "halo_tagging",
    "diag_tagging": "diag_tagging",
    "business_driver": "business_driver",
}


def likely_values_for(
    available: dict[str, list[str]],
    level_type: dict[str, list[str]],
    spacy_filter: dict[str, Any],
    query: str,
) -> dict[str, set[str]]:
    """The values worth putting first when a field has more options than the cap allows.

    Two signals, both already computed: what spaCy matched for that field's bucket, and what the
    question names outright. Ranking only changes which values survive truncation, so a wrong
    guess costs nothing: free text still accepts anything.
    """

    level_of = {field: level for level, fields in level_type.items() for field in fields}
    lowered = query.lower()
    likely: dict[str, set[str]] = {}
    for field, values in available.items():
        bucket = _SPACY_BUCKET_FOR_LEVEL.get(level_of.get(field, ""), "")
        captured = {str(v) for v in spacy_filter.get(bucket, [])} if bucket else set()
        named = {str(v) for v in values if str(v).lower() in lowered}
        likely[field] = captured | named
    return likely


def freshness_token_for(ner_results: dict[str, Any]) -> str:
    """Identifies the extraction state the questions were asked about.

    The runtime refuses a resume whose token no longer matches the checkpoint, so it has to change
    when the filter does and stay identical across a node rerun. A digest of the resolved fields
    gives both.
    """

    material = repr(sorted((str(k), repr(v)) for k, v in ner_results.items()))
    return f"filter-{hashlib.sha256(material.encode()).hexdigest()[:32]}"


def should_clarify_halo(
    ner_results: dict[str, Any],
    probabilities: dict[str, Any],
    spacy_filter: dict[str, Any],
    halo_level: str | None,
    halo_tagging: str | None,
) -> bool:
    """Ported verbatim from filter_generator_refactor.py:_should_clarify_halo."""

    halo_ner = spacy_filter.get("halo_tagging", [])
    is_halo_llm = False
    if halo_level and halo_tagging:
        res1 = ner_results.get(halo_level, [])
        res2 = ner_results.get(halo_tagging, [])
        res1 = [res1] if isinstance(res1, str) else res1
        res2 = [res2] if isinstance(res2, str) else res2
        combined = res1 + res2
        if halo_ner and not set(halo_ner).intersection(combined):
            return True
        is_halo_llm = not (("irrelevant" in res1) or ("irrelevant" in res2))

    is_halo_bert: bool = probabilities.get("halo", {}).get("yes", 0) >= 0.5
    return is_halo_bert != is_halo_llm


def order_clarification_list(fields: list[str], level_type: dict[str, list[str]]) -> list[str]:
    """Ported verbatim from filter_generator_refactor.py:order_clarification_list."""

    order = ["intention", "period_type", "start", "end"]
    level_order = ("general_level", "general_tagging", "halo_level", "halo_tagging", "diag_tagging")
    for level_name in level_order:
        order += level_type.get(level_name, [])
    ordered = [f for f in order if f in fields]
    ordered += [f for f in fields if f not in order]
    return ordered


def get_available_values(
    fields: list[str],
    ner_data: NerData,
    ner_data_config: NerDataConfig,
    data: pd.DataFrame | None = None,
) -> dict[str, list[str]]:
    """Ported from filter_generator_refactor.py:get_available_values, with one fix.

    The source appends "all" and "irrelevant" to any field not listed in additional_options.
    That default is right for tagging columns, where both are real answers, but it reaches
    `intention` too and offers 13 choices where the client only has 10 intentions plus
    "out-of-scope". Neither extra is answerable: you cannot ask for every intention at once, and
    an analytical question always has one. Raised by Lizhe after seeing the clarification message.
    """

    data = ner_data.df_bi if data is None else data
    data_cols = data.columns
    hierarchy_levels = ner_data.level_type.get("halo_level", []) + ner_data.level_type.get(
        "general_level", []
    )
    default_vals = {"trend": ["yes", "no"]}
    additional_options: dict[str, list[str]] = {"period_type": [], "intention": []}
    additional_options.update({lvl: ["all"] for lvl in hierarchy_levels})

    result: dict[str, list[str]] = {}
    for field in fields:
        if field == "intention":
            vals = [*ner_data_config.metric_info.keys(), "out-of-scope"]
        elif field in data_cols:
            vals = list(data[field].dropna().unique())
        else:
            vals = default_vals.get(field, [])
        if vals:
            result[field] = sorted(vals) + additional_options.get(field, ["all", "irrelevant"])
    return result


async def build_clarification_list(state: AskGenomeState) -> dict[str, Any]:
    subquery = state["plan"].subqueries[state["active_index"]]
    ner_data = load_ner_data()
    ner_data_config = load_ner_data_config()
    ner_state = dict(state.get("ner_state", {}))
    entry = dict(ner_state[subquery.id])

    clarification_list = list(entry.get("clarification_list", []))
    halo_level = ner_data.level_type.get("halo_level", [None])[0]
    halo_tagging = ner_data.level_type.get("halo_tagging", [None])[0]
    is_halo_clarify = should_clarify_halo(
        entry["ner_results"],
        entry["probabilities"],
        entry["spacy_filter"],
        halo_level,
        halo_tagging,
    )
    if is_halo_clarify:
        clarification_list.extend(f for f in (halo_level, halo_tagging) if f)

    # Never re-ask a field the user answered. A deliberate divergence: the source's clarification
    # was a single interrupt-once-then-merge pass that couldn't loop, so re-asking was impossible.
    # This one loops, and without the guard it never converges: the halo check compares spaCy's
    # capture against the LLM's answer, and lemmatisation turns 'lms' into 'lm', which matches no
    # real column value.
    user_confirmed = entry.get("user_confirmed", {})
    clarification_list = [f for f in clarification_list if f not in user_confirmed]

    ordered = order_clarification_list(list(set(clarification_list)), ner_data.level_type)
    available = get_available_values(ordered, ner_data, ner_data_config)

    requests = build_clarification_requests(
        available,
        subquery_id=subquery.id,
        freshness_token=freshness_token_for(entry.get("ner_results", {})),
        reason_codes=dict.fromkeys(
            (f for f in (halo_level, halo_tagging) if f and is_halo_clarify), "halo_mismatch"
        ),
        likely_values=likely_values_for(
            available, ner_data.level_type, entry["spacy_filter"], entry.get("extended_query", "")
        ),
    )
    # Every flagged field stays here. coordinate_clarification removes one only when the user
    # actually answers it, so a cancelled field is still disclosed by aggregate_results rather
    # than silently dropped. It is also how a resumed clarification is mapped back to its field.
    entry["clarification_fields"] = available
    ner_state[subquery.id] = entry
    return {"ner_state": ner_state, QUEUED_CLARIFICATIONS_CHANNEL: requests}
