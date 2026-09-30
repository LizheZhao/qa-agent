"""Answering coverage questions: what data exists, not what it says.

Ported from answer_coverage in ask-genome-core's src/model/coverage.py. A model turns the question
into a lookup spec, upstream's vendored helpers find the facts, and a model words them. So the
numbers come from the data, never the model.

Both model calls go through the orchestrator's model. One addition: a dimension the feasibility
config doesn't list is reported as untracked, not as no data.
"""

import json
import logging
from typing import Any

from orchestration_core import AgentDependencies

from ask_genome_agent.config import load_feasibility_config, load_ner_data, load_spacy_config
from ask_genome_agent.contracts import CoverageSpec
from ask_genome_agent.nodes.annotate_feasibility import resolve_referenced_terms
from ask_genome_agent.nodes.support.spacy_hints import compute_spacy_hints
from ask_genome_agent.prompts import coverage_phrasing_prompt, coverage_spec_prompt
from ask_genome_agent.state import AskGenomeState
from ask_genome_agent.vendor.ask_genome_core.model import coverage as upstream

logger = logging.getLogger(__name__)


async def answer_coverage(state: AskGenomeState, dependencies: AgentDependencies) -> dict[str, Any]:
    subquery = state["plan"].subqueries[state["active_index"]]
    spacy_config = load_spacy_config()
    hints = subquery.spacy_hints
    if hints is None:
        hints = compute_spacy_hints(spacy_config, subquery.query)

    spec = await _spec_call(subquery.query, hints, dependencies)
    # The data is stored lower-case.
    spec = spec.model_copy(update={"values": [str(v).lower() for v in spec.values]})

    # As upstream does: a spaCy match knows the column better than the model, unless it's a join.
    resolved = resolve_referenced_terms(hints, spacy_config.core_filters, "")
    joined = spec.needs_join or spec.operation == "co_occurrence"
    if resolved and not joined:
        column = next(iter(resolved))
        spec = spec.model_copy(update={"dimensions": [column], "values": sorted(resolved[column])})

    feasibility = load_feasibility_config()
    if joined:
        source, _ = upstream._find_data_source(feasibility, spec.dimensions)
        data = load_ner_data()
        frame = data.df_bi if source == "bi" else data.df_br
        facts, _table = upstream._co_occurrence_facts(spec, frame)
    else:
        _, source_config = upstream._find_data_source(feasibility, spec.dimensions)
        facts, _table = upstream._config_facts(spec, source_config)
        facts.update(_untracked(spec, source_config))

    facts_text = json.dumps(facts, default=str)
    try:
        answer = await _phrasing_call(subquery.query, facts_text, dependencies)
        phrasing_error = None
    except Exception as error:
        # The facts are the answer anyway.
        logger.warning("coverage phrasing failed for %s: %s", subquery.id, error)
        answer = facts_text
        phrasing_error = f"{type(error).__name__}: {error}"

    ner_state = dict(state.get("ner_state", {}))
    entry: dict[str, Any] = {
        "answer": answer,
        "coverage": {"spec": spec.model_dump(), "facts": facts},
    }
    if phrasing_error:
        entry["response_error"] = phrasing_error
    ner_state[subquery.id] = entry
    # Upstream passes dependents the facts too, not just the sentence.
    return {"ner_state": ner_state, "last_subquery_result": f"{answer}\n\n{facts_text}"}


def _untracked(spec: CoverageSpec, source_config: dict[str, Any]) -> dict[str, Any]:
    """Upstream reports an unlisted dimension as not found, and the answer then says "no data".
    Live, that told us paid social never ran; the config just has no channel dimension."""

    dimension = spec.dimensions[0] if spec.dimensions else None
    if not dimension or dimension in (source_config.get("dimensions") or {}):
        return {}
    return {
        "dimension_tracked": False,
        "note": (
            f"Coverage is not recorded for {dimension}, so this lookup cannot say whether the "
            "value has data. It does not mean there is none."
        ),
    }


async def _spec_call(
    query: str, hints: dict[str, Any], dependencies: AgentDependencies
) -> CoverageSpec:
    structured_model = dependencies.model.with_structured_output(
        CoverageSpec, method="function_calling"
    )
    system = coverage_spec_prompt(query, json.dumps(upstream._hint_summary(hints)))
    result = await structured_model.ainvoke(
        [("system", system), ("user", "Return the JSON object.")]
    )
    if not isinstance(result, CoverageSpec):
        raise TypeError("the coverage spec call expected a CoverageSpec result")
    return result


async def _phrasing_call(query: str, facts: str, dependencies: AgentDependencies) -> str:
    system = coverage_phrasing_prompt(query, facts)
    result = await dependencies.model.ainvoke([("system", system), ("user", "Write the answer.")])
    text = result.content if isinstance(result.content, str) else str(result.content)
    if not text.strip():
        raise ValueError("the model returned an empty response")
    return text.strip()
