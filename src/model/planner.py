"""Planner layer (v1).

Runs before NER: rephrases, splits a query into sub-queries, classifies each as
coverage vs analytical, assigns a coarse intent, and annotates coarse feasibility.
This module is Streamlit-free: it does planning + stateless helpers only. The
stateful execution loop (clarification-graph interrupts, readout, response,
per-sub-query rendering, sequential threading) lives in pages/orchestrator_page.py, because
clarification pauses for user input and must be driven by the page.

Sequential dependencies use a single mechanism: a dependent sub-query (depends_on set) is
resolved from its predecessor's context via resolve_from_context, which may reject it, rephrase
it to one self-contained sub-query, or split it into several (one per item in the context).
"""
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from src.utils import load_yaml_file
from src.data.data_interface import NerDataConfig, SpacyConfig, FeasibilityConfig, ReadoutData
from src.model.filter_generator import normalize_query, _clean_variant_mapping, get_spacy_filter
from src.integrations.sdk_utils import planner_llm_call, resolve_from_context_call


logger = logging.getLogger(__name__)

# metadata keys inside a core_dimension_filter entry; the remaining keys are real data columns
_FILTER_META_KEYS = {"AKA", "org_term", "source", "halo_to_ignore", "detail"}
# spacy_filter keys whose values are matched query terms (resolved to data cols/values via core_filters)
_SPACY_VALUE_KEYS = ("custom_level", "custom_tagging", "halo_tagging",
                     "core_dimension", "core_dimension_composite", "business_driver")


def resolve_referenced_terms(hints: dict, core_filters: dict, data_source: str) -> dict:
    """Map spacy-matched (stemmed/lemmatized) terms to real {col: set(values)} via core_dimension_filter.

    This mirrors the downstream categorize_filters step, so a referenced entity is checked against
    the actual data columns/values it resolves to (not the raw stemmed text)."""
    resolved = defaultdict(set)
    for key in _SPACY_VALUE_KEYS:
        for term in (hints.get(key) or []):
            for filt in core_filters.get(term, []):
                if data_source and data_source not in filt.get("source", ""):
                    continue
                for col, vals in filt.items():
                    if col in _FILTER_META_KEYS:
                        continue
                    vals = vals if isinstance(vals, list) else [vals]
                    resolved[col].update(str(v).lower() for v in vals)
    return resolved


@dataclass
class SubQuerySpec:
    id: int
    query: str
    kind: str                                  # "coverage" | "analytical"
    coarse_intent: Optional[str] = None        # analytical only; None for coverage
    depends_on: Optional[int] = None           # predecessor id; None if independent


@dataclass
class SubQueryAnnotation:
    spacy_hints: dict = field(default_factory=dict)
    feasibility: Optional[dict] = None         # {"status": "feasible"|"infeasible", "reasons": [...]}


@dataclass
class SubQueryRuntime:
    filled_query: Optional[str] = None
    resolved: Optional[bool] = None
    bert_intent: Optional[str] = None
    intent_conflict: bool = False
    readout_data: Optional[ReadoutData] = None
    response: Optional[str] = None
    rejected: bool = False
    rejection_message: Optional[str] = None


@dataclass
class EnrichedSubQuery:
    spec: SubQuerySpec
    annotation: SubQueryAnnotation
    runtime: SubQueryRuntime = field(default_factory=SubQueryRuntime)


@dataclass
class PlannerPlan:
    original_query: str
    rephrased_query: str
    is_multi: bool
    subqueries: list = field(default_factory=list)        # list[SubQuerySpec]


@dataclass
class PlannerResult:
    plan: PlannerPlan
    subqueries: list = field(default_factory=list)        # list[EnrichedSubQuery]
    is_sequential: bool = False


def spacy_hints(spacy_config: SpacyConfig, query: str) -> dict:
    """Light spacy keyword pass, reused from the NER path, used as a split/feasibility signal."""
    cleaned = _clean_variant_mapping(spacy_config.variant_mapping)
    normalized = normalize_query(query.lower(), cleaned)
    return get_spacy_filter(normalized.replace("'", ""), spacy_config)


def annotate_feasibility(spec: SubQuerySpec, hints: dict, core_filters: dict,
                         metric_info: dict, feasibility: FeasibilityConfig) -> Optional[dict]:
    """Coarse, annotate-only feasibility against the precomputed config (NER decides finally).

    Checks (1) the intent maps to a metric present in its data_source, and (2) referenced
    entities, resolved to real (col, values) via core_dimension_filter, have a footprint in
    the data. All comparisons are case-insensitive (data values are stored lower-cased)."""
    if not feasibility.data_sources:
        return None
    info = metric_info.get(spec.coarse_intent)
    if info is None:
        return {"status": "infeasible", "reasons": [f"unknown intent '{spec.coarse_intent}'"]}

    data_source = info.get("data")
    ds = feasibility.data_sources.get(data_source, {})
    reasons = []

    available_metrics = {str(m).lower() for m in ds.get("metrics", [])}
    intent_metrics = [str(m).lower() for m in (info.get("metric", []) or [])]
    if available_metrics and intent_metrics and not (set(intent_metrics) & available_metrics):
        reasons.append(f"no metric for intent '{spec.coarse_intent}' in {data_source} data")

    dimensions = ds.get("dimensions") or {}
    for col, vals in resolve_referenced_terms(hints, core_filters, data_source).items():
        dim = dimensions.get(col)
        if dim is None:
            continue  # column not enumerated in the coverage config; leave it to NER
        known = {str(k).lower() for k in (dim.get("values") or {}).keys()}
        if vals and known and vals.isdisjoint(known):
            reasons.append(f"{col} value(s) {sorted(vals)} not found in {data_source} data")

    return {"status": "infeasible" if reasons else "feasible", "reasons": reasons}


def generate_plan(client_code: str, model_group_id: int, query: str) -> PlannerResult:
    """Plan + annotate (no execution). The page drives execution of the returned sub-queries."""
    ner_data_config = NerDataConfig.from_local(client_code, model_group_id)
    feasibility = FeasibilityConfig.from_local(client_code, model_group_id)
    spacy_config = SpacyConfig.from_local(client_code, model_group_id)
    intent_catalog = list(ner_data_config.metric_info.keys())

    hints = spacy_hints(spacy_config, query)
    core_filters = spacy_config.core_filters

    prompts = load_yaml_file("./config/prompt.yaml")
    system_prompt = prompts["PLANNER_PROMPT"].replace("INTENT_CATALOG_PLACEHOLDER", ", ".join(intent_catalog))
    plan_model, elapsed = planner_llm_call(system_prompt, query, intent_catalog)
    logger.info(f"planner produced {len(plan_model.subqueries)} sub-queries in {elapsed:.2f}s")

    specs = [SubQuerySpec(id=s.id, query=s.query, kind=s.kind,
                          coarse_intent=s.coarse_intent, depends_on=s.depends_on)
             for s in plan_model.subqueries]
    plan = PlannerPlan(original_query=query, rephrased_query=plan_model.rephrased_query,
                       is_multi=plan_model.is_multi, subqueries=specs)

    enriched = []
    for spec in specs:
        feas = annotate_feasibility(spec, hints, core_filters, ner_data_config.metric_info, feasibility) \
            if spec.kind == "analytical" else None
        enriched.append(EnrichedSubQuery(spec=spec,
                                         annotation=SubQueryAnnotation(spacy_hints=hints, feasibility=feas)))

    is_sequential = any(s.depends_on is not None for s in specs)
    return PlannerResult(plan=plan, subqueries=enriched, is_sequential=is_sequential)


def resolve_from_context(prev_context_str: str, query: str) -> tuple[list, bool, str]:
    """Resolve a dependent sub-query from its predecessor's context. The LLM decides among:
    reject (empty list + resolved=False), rephrase (one self-contained sub-query), or
    split (one self-contained sub-query per relevant item in the context).

    Returns (subqueries: list[str], resolved: bool, reason: str)."""
    prompts = load_yaml_file("./config/prompt.yaml")
    prompt = (prompts["RESOLVE_FROM_CONTEXT_PROMPT"]
              .replace("PREV_RESPONSE_PLACEHOLDER", str(prev_context_str))
              .replace("QUERY_PLACEHOLDER", query))
    out, _ = resolve_from_context_call(prompt, "Return the JSON object.")
    return out.subqueries, out.resolved, out.reason
