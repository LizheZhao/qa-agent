"""Deterministic plan surgery: apply_resolution.py node updates the current plan after a
dependent subquery has been resolved.

It takes the dependency-resolution result and safely splice it back into the existing plan
while preserving the original ordering behavior.
"""

from typing import Any

from ask_genome_agent.contracts import Subquery, SubqueryKind
from ask_genome_agent.state import AskGenomeState


def apply_resolution(state: AskGenomeState) -> dict[str, Any]:
    plan = state["plan"]
    idx = state["active_index"]
    resolution = state["resolution"]
    assert resolution is not None
    subqueries = list(plan.subqueries)

    if resolution.kind in ("unchanged", "rejected"):
        return {"plan": plan}

    if resolution.kind == "rephrased":
        subqueries[idx] = subqueries[idx].model_copy(
            update={"query": resolution.raw_text[0], "depends_on": None}
        )
        return {"plan": plan.model_copy(update={"subqueries": tuple(subqueries)})}

    # split
    existing_ids = [s.id for s in subqueries]
    next_id = max(existing_ids) + 1 if existing_ids else 0
    spliced = tuple(
        Subquery(
            id=next_id + i,
            query=text,
            kind=SubqueryKind.analytical,
            coarse_intent=None,
            depends_on=None,
            feasibility=None,
        )
        for i, text in enumerate(resolution.raw_text)
    )
    subqueries[idx : idx + 1] = spliced
    updated_plan = plan.model_copy(
        update={
            "subqueries": tuple(subqueries),
            "is_sequential": any(s.depends_on is not None for s in subqueries),
        }
    )
    return {"plan": updated_plan}
