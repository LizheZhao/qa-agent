"""Deterministic feasibility annotation for analytical subqueries only.

spaCy hints are computed per subquery, from that subquery's own (self-contained) query text --
not once for the whole original query -- so an entity belonging to one subquery cannot leak into
another subquery's feasibility check.

The computed hints are persisted onto Subquery.spacy_hints (not just used locally and discarded),
so Question Understanding can reuse the identical extraction instead of recomputing it.
"""

from collections import defaultdict
from typing import Any

from ask_genome_agent.config import (
    FeasibilityConfig,
    load_feasibility_config,
    load_ner_data_config,
    load_spacy_config,
)
from ask_genome_agent.contracts import SubqueryKind
from ask_genome_agent.nodes.support.spacy_hints import compute_spacy_hints
from ask_genome_agent.state import AskGenomeState

_FILTER_META_KEYS = {"AKA", "org_term", "source", "halo_to_ignore", "detail"}
_SPACY_VALUE_KEYS = (
    "custom_level",
    "custom_tagging",
    "halo_tagging",
    "core_dimension",
    "core_dimension_composite",
    "business_driver",
)


def resolve_referenced_terms(
    hints: dict[str, Any], core_filters: dict[str, Any], data_source: str
) -> dict[str, set[str]]:
    resolved: dict[str, set[str]] = defaultdict(set)
    for key in _SPACY_VALUE_KEYS:
        for term in hints.get(key) or []:
            for filt in core_filters.get(term, []):
                if data_source and data_source not in filt.get("source", ""):
                    continue
                for col, vals in filt.items():
                    if col in _FILTER_META_KEYS:
                        continue
                    vals = vals if isinstance(vals, list) else [vals]
                    resolved[col].update(str(v).lower() for v in vals)
    return resolved


async def annotate_feasibility(state: AskGenomeState) -> dict[str, Any]:
    feasibility_config = load_feasibility_config()
    spacy_config = load_spacy_config()
    metric_info = load_ner_data_config().metric_info

    plan_output = dict(state["plan_output"])
    subqueries = [dict(item) for item in plan_output["subqueries"]]
    for subquery in subqueries:
        if subquery["kind"] != SubqueryKind.analytical.value:
            continue
        hints = compute_spacy_hints(spacy_config, subquery["query"])
        subquery["spacy_hints"] = hints
        subquery["feasibility"] = _check_feasibility(
            subquery, hints, spacy_config.core_filters, metric_info, feasibility_config
        )
    plan_output["subqueries"] = subqueries
    return {"plan_output": plan_output}


def _check_feasibility(
    subquery: dict[str, Any],
    hints: dict[str, Any],
    core_filters: dict[str, Any],
    metric_info: dict[str, Any],
    feasibility_config: FeasibilityConfig,
) -> dict[str, Any] | None:
    if not feasibility_config.data_sources:
        return None
    coarse_intent = subquery.get("coarse_intent")
    info = metric_info.get(coarse_intent) if coarse_intent is not None else None
    if info is None:
        return {"status": "infeasible", "reasons": [f"unknown intent '{coarse_intent}'"]}

    data_source = info.get("data")
    ds = feasibility_config.data_sources.get(data_source, {})
    reasons = []

    available_metrics = {str(m).lower() for m in ds.get("metrics", [])}
    intent_metrics = [str(m).lower() for m in (info.get("metric", []) or [])]
    if available_metrics and intent_metrics and not (set(intent_metrics) & available_metrics):
        reasons.append(f"no metric for intent '{coarse_intent}' in {data_source} data")

    dimensions = ds.get("dimensions") or {}
    for col, vals in resolve_referenced_terms(hints, core_filters, data_source).items():
        dim = dimensions.get(col)
        if dim is None:
            continue
        known = {str(k).lower() for k in dim.get("values") or {}}
        if vals and known and vals.isdisjoint(known):
            reasons.append(f"{col} value(s) {sorted(vals)} not found in {data_source} data")

    return {"status": "infeasible" if reasons else "feasible", "reasons": reasons}
