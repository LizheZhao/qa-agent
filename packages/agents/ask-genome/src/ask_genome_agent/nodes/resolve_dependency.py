"""
Deterministic: resolve a subquery's dependency using the predecessor's context.

1. Handles dependent subqueries using the already-completed context from their predecessor.
2. Makes an LLM decision only to: reject the subquery, rephrase it, or split it into multiple
   subqueries.
3. It does not assign IDs or modify the plan directly; apply_resolution.py handles those updates.
4. Mirrors resolve_from_context behavior from ask-genome-core.
5. The predecessor's context and the current dependent query are sent as the user message; the
   system prompt only describes the decision logic.
6. This node runs only when the current subquery has depends_on set.
"""

from typing import Any

from orchestration_core import AgentDependencies

from ask_genome_agent.contracts import DependencyResolution, ResolveOutcome
from ask_genome_agent.prompts import RESOLVE_FROM_CONTEXT_PROMPT
from ask_genome_agent.state import AskGenomeState


async def resolve_dependency(
    state: AskGenomeState, dependencies: AgentDependencies
) -> dict[str, Any]:
    plan = state["plan"]
    subquery = plan.subqueries[state["active_index"]]
    assert subquery.depends_on is not None

    # Keyed by Subquery.id, not list position -- fixes the source's dangling-reference bug.
    predecessor_context = state["context"].get(subquery.depends_on)
    if not predecessor_context:
        return {
            "resolution": DependencyResolution(
                kind="rejected", reason="predecessor context unavailable"
            )
        }

    outcome = await _resolve_from_context_call(predecessor_context, subquery.query, dependencies)
    if not outcome.resolved or not outcome.subqueries:
        return {"resolution": DependencyResolution(kind="rejected", reason=outcome.reason)}

    if len(outcome.subqueries) == 1:
        return {"resolution": DependencyResolution(kind="rephrased", raw_text=outcome.subqueries)}
    return {"resolution": DependencyResolution(kind="split", raw_text=outcome.subqueries)}


async def _resolve_from_context_call(
    context: str, query: str, dependencies: AgentDependencies
) -> ResolveOutcome:
    structured_model = dependencies.model.with_structured_output(
        ResolveOutcome, method="function_calling"
    )
    user_message = f"Previous data context:\n{context}\n\nCurrent follow-up query: {query}"
    result = await structured_model.ainvoke(
        [("system", RESOLVE_FROM_CONTEXT_PROMPT), ("user", user_message)]
    )
    if not isinstance(result, ResolveOutcome):
        raise TypeError("resolve_from_context_call expected a ResolveOutcome result")
    return result
