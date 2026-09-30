"""Deterministic Assemble Plan Step
1. Combines the classified subqueries and their feasibility results into the final validated
   AskGenomePlan.
2. Initializes the state required for the graph's processing loop:
    cursor = 0 — starts processing from the first subquery.
    Empty context store — used to save results/context as subqueries are processed.
"""

from typing import Any

from ask_genome_agent.contracts import AskGenomePlan, FeasibilityVerdict, Subquery
from ask_genome_agent.state import AskGenomeState


def assemble_plan(state: AskGenomeState) -> dict[str, Any]:
    plan_output = state["plan_output"]
    subqueries = tuple(
        Subquery(
            id=item["id"],
            query=item["query"],
            kind=item["kind"],
            coarse_intent=item.get("coarse_intent"),
            depends_on=item.get("depends_on"),
            feasibility=(
                FeasibilityVerdict.model_validate(item["feasibility"])
                if item.get("feasibility") is not None
                else None
            ),
            # annotate_feasibility persists these onto the subquery dict; carrying them through
            # here is what lets Question Understanding reuse the extraction instead of
            # recomputing it. Dropping them silently made extract_entities always take its
            # recompute fallback -- the reuse never actually happened.
            spacy_hints=item.get("spacy_hints"),
        )
        for item in plan_output["subqueries"]
    )
    plan = AskGenomePlan(
        original_query=state["query"],
        rephrased_query=plan_output["rephrased_query"],
        is_multi=plan_output["is_multi"],
        is_sequential=any(s.depends_on is not None for s in subqueries),
        subqueries=subqueries,
    )
    return {"plan": plan, "active_index": 0, "context": {}}
