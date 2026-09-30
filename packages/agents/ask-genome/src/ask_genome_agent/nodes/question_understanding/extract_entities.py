"""Deterministic: first step of Question Understanding for one analytical subquery.

Reuses the spaCy hints annotate_feasibility.py already computed for this subquery, via
Subquery.spacy_hints, instead of running the identical extraction twice. Also expands the client's
known acronyms in the query text.

Coverage never gets here; the graph sends it to answer_coverage. If one does, it's a routing bug.
"""

import re
from typing import Any

from ask_genome_agent.config import load_ner_data_config, load_spacy_config
from ask_genome_agent.contracts import SubqueryKind
from ask_genome_agent.nodes.support.spacy_hints import compute_spacy_hints
from ask_genome_agent.state import AskGenomeState


def extend_acronyms(input_string: str, mapping_dict: dict[str, str]) -> tuple[str, dict[str, str]]:
    """Ported verbatim from ask-genome-core/src/model/filter_generator.py:extend_acronyms."""

    if not mapping_dict:
        return input_string, {}
    cleaned_string = re.sub(r"[^a-zA-Z0-9\s]", "", input_string)
    pattern = r"(?i)\b(" + "|".join(re.escape(key) for key in mapping_dict) + r")(?=\d|\b)"
    result = re.sub(
        pattern, lambda m: mapping_dict[m.group().lower()], cleaned_string, flags=re.IGNORECASE
    )
    detected_pair = {
        k: mapping_dict[k] for k in re.findall(pattern, input_string, flags=re.IGNORECASE)
    }
    return result, detected_pair


async def extract_entities(state: AskGenomeState) -> dict[str, Any]:
    subquery = state["plan"].subqueries[state["active_index"]]
    if subquery.kind != SubqueryKind.analytical:
        raise RuntimeError(
            f"subquery {subquery.id} is {subquery.kind.value}; the graph routes those to "
            "answer_coverage, not question understanding"
        )

    ner_data_config = load_ner_data_config()
    query = subquery.query.lower()
    extended_query, detected_pair = extend_acronyms(query, ner_data_config.acronym_dict or {})

    spacy_filter = subquery.spacy_hints
    if spacy_filter is None:
        # Fallback only. annotate_feasibility.py should already have computed and persisted
        # this for every analytical subquery.
        spacy_filter = compute_spacy_hints(load_spacy_config(), subquery.query)

    ner_state = dict(state.get("ner_state", {}))
    ner_state[subquery.id] = {
        "extended_query": extended_query,
        "detected_pair": detected_pair,
        "spacy_filter": spacy_filter,
    }
    return {"ner_state": ner_state}
