"""Shared state for the agent orchestrator graph.

Sub-query specs are stored as plain dicts (not dataclasses) so the checkpointer can
serialize them. Each executing node leaves its result in pending_payload /
pending_context; the record node commits them and advances the cursor.
"""
from typing import Optional, TypedDict


class AgentState(TypedDict, total=False):
    # plan
    query: str
    rephrased_query: str
    is_multi: bool
    is_sequential: bool
    subqueries: list                        # list[dict], see spec_to_dict
    cursor: int
    # committed results
    # payload shape: {"index","kind","query","response","tables","rejected","message"}
    payloads: list
    contexts: dict                          # sub-query index -> context string for dependents
    replan_notes: list                      # human-readable notes about plan revisions
    # scratch for the sub-query being executed (cleared by the record node)
    filled_query: Optional[str]
    ner_results: dict
    spacy_filter: dict
    bert_probabilities: dict
    clarification_fields: dict
    user_selections: dict
    field_status: dict
    pending_payload: Optional[dict]
    pending_context: Optional[str]
    # re-plan bookkeeping
    replan_trigger: Optional[dict]          # {"type", "reason", "split_queries"}
    replan_count: int
    attempts: dict                          # sub-query id -> how many times it entered replan


# reset between sub-queries by the record node
CLEARED_SCRATCH = {
    "filled_query": None,
    "ner_results": {},
    "spacy_filter": {},
    "bert_probabilities": {},
    "clarification_fields": {},
    "user_selections": {},
    "field_status": {},
    "pending_payload": None,
    "pending_context": None,
    "replan_trigger": None,
}


def spec_to_dict(enriched) -> dict:
    """Serialize an EnrichedSubQuery from the planner into a checkpointable dict."""
    return {
        "id": enriched.spec.id,
        "query": enriched.spec.query,
        "kind": enriched.spec.kind,
        "coarse_intent": enriched.spec.coarse_intent,
        "depends_on": enriched.spec.depends_on,
        "spacy_hints": enriched.annotation.spacy_hints,
        "feasibility": enriched.annotation.feasibility,
    }


def current_spec(state: AgentState) -> dict:
    return state["subqueries"][state["cursor"]]


def rejection_payload(index: int, kind: str, query: str, message: str) -> dict:
    return {"index": index, "kind": kind, "query": query, "rejected": True, "message": message}
