"""Context/dependency sub-agent: resolves a sub-query that depends on a predecessor.

Port of _resolve_dependency from pages/orchestrator_page.py into a graph node.
resolve_from_context lets the LLM decide among reject, rephrase (one self-contained
sub-query) or split (several). Reject and split are handed to the replan node via
replan_trigger; a rephrase fills filled_query and execution continues.
"""
import logging

from langchain_core.runnables import RunnableConfig

from src.model.planner import resolve_from_context
from src.agents.state import AgentState, current_spec

logger = logging.getLogger(__name__)


def resolve_dependency_node(state: AgentState, config: RunnableConfig) -> dict:
    spec = current_spec(state)
    dep = spec.get("depends_on")
    if dep is None:
        return {}

    context = state.get("contexts", {}).get(dep)
    if not context:
        return {"replan_trigger": {
            "type": "dep_missing",
            "reason": f"This sub-query depends on sub-query {dep + 1}, which returned no data."}}

    subqueries, resolved, reason = resolve_from_context(context, spec["query"])
    if not resolved or not subqueries:
        return {"replan_trigger": {
            "type": "dep_reject",
            "reason": f"Could not derive this sub-query from sub-query {dep + 1}: {reason}"}}
    if len(subqueries) == 1:
        return {"filled_query": subqueries[0]}
    # split: one analytical sub-query per item found in the predecessor's context
    logger.info(f"dependency resolution split sub-query {state['cursor']} into {len(subqueries)}")
    return {"replan_trigger": {"type": "split", "split_queries": subqueries}}
