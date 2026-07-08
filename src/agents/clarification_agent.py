"""Clarification sub-agent: NER extraction plus human-in-the-loop confirmation.

Adapts the nodes of src/model/filter_clarification_graph.py for the orchestrator graph.
The one real difference: coarse_intent comes from the current sub-query spec in graph
state, not from config, because it changes per sub-query within one thread.
merge_and_tag is reused from the original graph unchanged.
"""
import logging

from langgraph.types import interrupt
from langchain_core.runnables import RunnableConfig

import src.model.filter_generator_refactor as filter_generator
from src.model.filter_clarification_graph import merge_and_tag_node, _to_native
from src.agents.state import AgentState, current_spec

logger = logging.getLogger(__name__)


def extract_ner_node(state: AgentState, config: RunnableConfig) -> dict:
    cfg = config["configurable"]
    spec = current_spec(state)
    query = state.get("filled_query") or spec["query"]

    probabilities, spacy_filter, ner_results, clarification_fields = filter_generator.extract_ner_filter(
        cfg["client_code"], cfg["model_group_id"], query,
        llm_priority=cfg.get("llm_priority", False),
        enterprise_ai_ner=cfg.get("enterprise_ai_ner", False),
        coarse_intent=spec.get("coarse_intent"))

    field_status = {}
    for field_name in list(ner_results.keys()) + list(probabilities.keys()):
        if field_name in clarification_fields:
            field_status[field_name] = "waiting_for_verification"
        else:
            field_status[field_name] = "auto-confirmed"

    return _to_native({"bert_probabilities": probabilities,
                       "spacy_filter": spacy_filter,
                       "ner_results": ner_results,
                       "clarification_fields": clarification_fields,
                       "field_status": field_status})


def should_clarify(state: AgentState) -> str:
    return "clarify" if state.get("clarification_fields") else "merge_and_tag"


def clarify_node(state: AgentState) -> dict:
    """Pause the graph until the user confirms the ambiguous fields on the page."""
    spec = current_spec(state)
    user_selections = interrupt({
        "clarification_fields": state["clarification_fields"],
        "ner_results": state.get("ner_results", {}),
        "sub_query_index": state["cursor"],
        "query": state.get("filled_query") or spec["query"],
        "message": "Please confirm or adjust the following fields.",
    })
    return {"user_selections": user_selections}


def merge_and_tag(state: AgentState) -> dict:
    # the original node reads/writes only keys AgentState also has, so it works as-is
    return merge_and_tag_node(state)
