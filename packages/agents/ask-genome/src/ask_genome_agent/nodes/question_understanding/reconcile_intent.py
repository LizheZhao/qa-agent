"""Deterministic: compares BERT's intent prediction against the data source implied by the
client-specific dimensions spaCy matched, and checks for response-modifier keywords.

spaCy and BERT produce independent signals; this node compares them. It does not treat either as
authoritative over the other.

Ported from ask-genome-core/src/model/filter_generator.py: validate_intention and
_check_for_response_modifier, with one deliberate divergence described in validate_intention.
"""

from collections import Counter
from typing import Any

from ask_genome_agent.config import NerDataConfig, load_ner_data_config, load_spacy_config
from ask_genome_agent.state import AskGenomeState

_RESPONSE_MODIFIERS = ("diminishing return", "roi curve", "platform", "response curve", "marginal")


def validate_intention(
    predictions: dict[str, Any],
    spacy_filter: dict[str, Any],
    core_dimension_filters: dict[str, Any],
    ner_data_config: NerDataConfig,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Detects an intention whose dataset disagrees with the dimensions spaCy matched.

    Returns the predictions unchanged, plus a description of the conflict when there is one.

    Deliberate divergence from filter_generator.py:validate_intention, agreed in review. The
    source replaces the user's intention with that data source's default and flags "intention"
    for clarification. Two problems with that, both confirmed live on "what was the margin roi
    for print last quarter": the substitution happens unconditionally, so the filter says
    "source of change" even when the user never agreed to it; and a
    dataset incompatibility is not evidence that the user was ambiguous. They clearly wanted
    margin roi. The real finding is that margin roi may not exist for print, which no answer
    from the user can change.

    So the intention is left alone and the conflict is reported instead. What to ask the user,
    if anything, is open design work; reporting it honestly is strictly better than quietly
    answering a different question.
    """

    predictions = dict(predictions)
    pred_intention = predictions.get("intention", "none")
    pred_data_source = ner_data_config.metric_info.get(pred_intention, {}).get("data")
    captured_core_dims = spacy_filter.get("core_dimension", []) + spacy_filter.get(
        "core_dimension_composite", []
    )
    core_dimension_sources: list[str] = []
    for core_dim in captured_core_dims:
        sources = [x.get("source", "bibr") for x in core_dimension_filters.get(core_dim, [])]
        core_dimension_sources.extend(sources)

    sources_blob = "".join(core_dimension_sources)
    if not (core_dimension_sources and pred_data_source and pred_data_source not in sources_blob):
        return predictions, None

    most_common = Counter(core_dimension_sources).most_common(1)[0][0]
    indicators = ner_data_config.ner_postprocess_indicators or {}
    alternatives = {
        "bi": indicators.get("default_bi_intention", "performance"),
        "br": indicators.get("default_br_intention", "source of change"),
        "bibr": pred_intention,
    }
    conflict = {
        "intention": pred_intention,
        "intention_source": pred_data_source,
        "dimensions": list(captured_core_dims),
        "dimension_source": most_common,
        # What the source would have silently substituted. Kept so a later design can offer it as
        # a choice rather than impose it.
        "alternative_intention": alternatives.get(most_common, "none"),
    }
    return predictions, conflict


def check_for_response_modifier(
    query: str, predictions: dict[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    """Ported verbatim from filter_generator.py:_check_for_response_modifier."""

    predictions = dict(predictions)
    mods = []
    clarification_list: list[str] = []
    for modifier in _RESPONSE_MODIFIERS:
        if modifier in query.lower():
            if modifier in ("diminishing return", "response curve"):
                predictions["intention"] = "planner"
                clarification_list.append("intention")
            if modifier == "marginal" and predictions.get("intention") != "planner":
                continue
            mods.append(modifier)
    predictions["response_modifier"] = mods
    return predictions, clarification_list


async def reconcile_intent(state: AskGenomeState) -> dict[str, Any]:
    subquery = state["plan"].subqueries[state["active_index"]]
    ner_data_config = load_ner_data_config()
    spacy_config = load_spacy_config()
    ner_state = dict(state.get("ner_state", {}))
    entry = dict(ner_state[subquery.id])

    predictions, source_conflict = validate_intention(
        entry["predictions"], entry["spacy_filter"], spacy_config.core_filters, ner_data_config
    )
    predictions, modifier_clarify = check_for_response_modifier(subquery.query, predictions)

    entry["predictions"] = predictions
    entry["clarification_list"] = list(set(modifier_clarify))
    # None when the intention and the matched dimensions agree, which is the usual case.
    entry["source_conflict"] = source_conflict
    ner_state[subquery.id] = entry
    return {"ner_state": ner_state}
