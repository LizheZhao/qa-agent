"""Coverage branch (v1).

Answers data-coverage / metadata questions ("how many quarters did we run paid social",
"which countries have which brands") without entering the NER / readout pipeline.

Single-dimension coverage is answered from the precomputed FeasibilityConfig; joint
co-occurrence is answered live via a groupby on df_bi/df_br. Facts are looked up
deterministically (never hallucinated), then an LLM phrases the sentence.
"""
import json
import logging

import pandas as pd

from src.utils import load_yaml_file
from src.integrations.sdk_utils import coverage_spec_call, text_llm_call


logger = logging.getLogger(__name__)

_REFERENCED_VALUE_KEYS = ("custom_level", "halo_level", "custom_tagging", "halo_tagging")


def _hint_summary(hints: dict) -> dict:
    return {k: hints.get(k) for k in _REFERENCED_VALUE_KEYS if hints.get(k)}


def _find_data_source(feasibility, dimensions: list) -> tuple:
    """Pick the data_source (bi/br) whose dimensions cover the referenced columns; default bi."""
    sources = feasibility.data_sources or {}
    for ds_name, ds in sources.items():
        dims = ds.get("dimensions") or {}
        if dimensions and all(d in dims for d in dimensions):
            return ds_name, ds
    for ds_name, ds in sources.items():
        dims = ds.get("dimensions") or {}
        if any(d in dims for d in dimensions):
            return ds_name, ds
    return "bi", sources.get("bi", {})


def _values_table(dim, values):
    return pd.DataFrame({dim: values}) if dim and values else None


def _config_facts(spec, ds: dict) -> tuple:
    """Look up facts for a single-dimension coverage operation. Returns (facts, table)."""
    dimensions = ds.get("dimensions") or {}
    facts = {"operation": spec.operation}

    if spec.operation == "list_values":
        dim = spec.dimensions[0] if spec.dimensions else None
        values = sorted((dimensions.get(dim, {}).get("values") or {}).keys()) if dim else []
        facts.update({"dimension": dim, "values": values})
        return facts, _values_table(dim, values)

    dim = spec.dimensions[0] if spec.dimensions else None
    val = spec.values[0] if spec.values else None
    value_entry = (dimensions.get(dim, {}).get("values") or {}).get(val) if dim and val else None
    facts.update({"dimension": dim, "value": val})

    if value_entry is None:
        facts["found"] = False
        return facts, None
    facts["found"] = True

    if spec.operation in ("count_periods", "time_range"):
        coverage = value_entry.get("period_coverage") or {}
        facts["period_coverage"] = {spec.period_type: coverage.get(spec.period_type)} \
            if spec.period_type else coverage
    elif spec.operation == "metric_availability":
        facts["metrics_present"] = value_entry.get("metrics_present", [])
    # existence: found=True already answers it
    return facts, None


def _co_occurrence_facts(spec, df: pd.DataFrame) -> tuple:
    """Live groupby for joint coverage across 2+ dimensions. Returns (facts, table)."""
    dims = [d for d in spec.dimensions if d in df.columns]
    if len(dims) < 2:
        return {"operation": "co_occurrence", "error": "need two known dimensions", "dimensions": dims}, None

    sub = df
    for dim, val in zip(spec.dimensions, spec.values):
        if dim in sub.columns and val:
            sub = sub[sub[dim].astype(str) == val]

    table = sub[dims].dropna().drop_duplicates().reset_index(drop=True)
    facts = {"operation": "co_occurrence", "dimensions": dims,
             "pair_count": int(len(table)), "pairs": table.head(200).to_dict("records")}
    return facts, table


def answer_coverage(query: str, hints: dict, core_filters: dict, feasibility,
                    df_bi: pd.DataFrame, df_br: pd.DataFrame) -> dict:
    """Answer a coverage sub-query. Returns {response, table, spec, facts}."""
    # local import to keep this module light-importable (planner pulls the heavy NER chain)
    from src.model.planner import resolve_referenced_terms

    prompts = load_yaml_file("./config/prompt.yaml")
    spec_prompt = (prompts["COVERAGE_SPEC_PROMPT"]
                   .replace("HINTS_PLACEHOLDER", json.dumps(_hint_summary(hints)))
                   .replace("QUERY_PLACEHOLDER", query))
    spec, _ = coverage_spec_call(spec_prompt, "Return the JSON object.")
    # data values are stored lower-cased; lower-case the LLM-extracted values before any lookup/filter
    spec.values = [str(v).lower() for v in spec.values]

    # resolve spacy-matched entities to real (col, values) via core_dimension_filter; prefer them
    # over the LLM's guess for single-dimension value lookups (data_source unknown here -> any source)
    resolved = resolve_referenced_terms(hints, core_filters, None)
    if resolved and not (spec.needs_join or spec.operation == "co_occurrence"):
        col = next(iter(resolved))
        spec.dimensions = [col]
        spec.values = sorted(resolved[col])
    logger.info(f"coverage spec: op={spec.operation} dims={spec.dimensions} join={spec.needs_join}")

    if spec.needs_join or spec.operation == "co_occurrence":
        ds_name, _ = _find_data_source(feasibility, spec.dimensions)
        df = df_bi if ds_name == "bi" else df_br
        facts, table = _co_occurrence_facts(spec, df)
    else:
        _, ds = _find_data_source(feasibility, spec.dimensions)
        facts, table = _config_facts(spec, ds)

    phrase_prompt = (prompts["COVERAGE_PHRASING_PROMPT"]
                     .replace("QUERY_PLACEHOLDER", query)
                     .replace("FACTS_PLACEHOLDER", json.dumps(facts, default=str)))
    response, _ = text_llm_call(phrase_prompt, "Write the answer.")

    return {"response": response, "table": table, "spec": spec, "facts": facts}
