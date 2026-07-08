import os
import sys
import numpy as np
import pandas as pd

from pydantic import BaseModel
from typing import Union, Any, TypedDict
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig

import src.model.filter_generator_refactor as filter_generator


class GraphState(TypedDict):
    query: str
    ner_results: dict[str, Any]
    spacy_filter: dict[str, Any]
    bert_probabilities: dict[str, Any]
    clarification_fields: dict[str, Any]                   # field_name → candidate options
    user_selections: dict[str, list[str]]       # field_name → user-chosen values
    field_status: dict[str, str]                # field_name → "verified" | "auto-confirmed"


def extract_ner_node(graph_state: GraphState, config: RunnableConfig) -> dict:
    """
    Adopting methods from src.model.filter_generator_refactor
    """
    cfg = config["configurable"]
    client_code, model_group_id = cfg["client_code"], cfg["model_group_id"]
    llm_priority = cfg.get("llm_priority", False)
    enterprise_ai_ner = cfg.get("enterprise_ai_ner", False)
    coarse_intent = cfg.get("coarse_intent")
    query = graph_state["query"]
    probabilities, spacy_filter, ner_results, clarification_fields = filter_generator.extract_ner_filter(
        client_code, model_group_id, query, llm_priority=llm_priority, enterprise_ai_ner=enterprise_ai_ner,
        coarse_intent=coarse_intent
    )

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


def should_clarify(graph_state: GraphState) -> str:
    """Router: check if any fields need clarification."""
    clarification_list = graph_state.get("clarification_fields", {})
    if len(clarification_list):
        return "clarify"
    return "merge_and_tag"


def clarify_node(graph_state: GraphState) -> dict:
    """
    Pause the graph and wait for user input.
    Expected resume payload:
        {
            "brand": ["Brand X"],
            "country": ["US", "UK"],
        }
    Values are lists (multi-select).
    """
    clarification_request = {
        "clarification_fields": graph_state["clarification_fields"],
        "message": "Please confirm or adjust the following fields.",
    }
    user_selections = interrupt(clarification_request)

    return {"user_selections": user_selections}


def _format_user_selections(selected: list, field_name: str) -> Union[list, str]:
    if field_name == "intention":
        selected = selected[0]
    elif any(x in selected for x in ["all", "irrelevant"]):
        selected = "all" if "all" in selected else "irrelevant"
    elif field_name in ["start", "end"]:
        selected = [str(x) for x in selected]
    return selected


def merge_and_tag_node(graph_state: GraphState) -> dict:
    """
    Merge user selections (if any) into ner_filters.
    Tag every field as 'verified' (user confirmed) or 'auto-confirmed'.
    """
    ner_filters = dict(graph_state.get("ner_results", {}))
    clarification_fields = dict(graph_state.get("clarification_fields", {}))
    user_selections = dict(graph_state.get("user_selections", {}))
    field_status = dict(graph_state.get("field_status", {}))

    # merge user_selections
    for field_name, vals in user_selections.items():
        if field_name in ner_filters or field_name in clarification_fields:
            # User explicitly confirmed/changed this field
            selected = _format_user_selections(vals, field_name)
            ner_filters[field_name] = selected
            field_status[field_name] = "verified"
        else:
            # NER extracted, no clarification needed
            field_status[field_name] = "auto-confirmed"

    return _to_native({"ner_results": ner_filters, "field_status": field_status})


def build_clarification_graph() -> StateGraph:
    graph = StateGraph(GraphState)

    graph.add_node("extract_ner", extract_ner_node)
    graph.add_node("clarify", clarify_node)
    graph.add_node("merge_and_tag", merge_and_tag_node)

    graph.add_edge(START, "extract_ner")
    graph.add_conditional_edges(
        "extract_ner",
        should_clarify,
        {"clarify": "clarify", "merge_and_tag": "merge_and_tag"},
    )
    graph.add_edge("clarify", "merge_and_tag")
    graph.add_edge("merge_and_tag", END)

    return graph


def create_app():
    """Compile the graph with a MemorySaver checkpointer."""
    graph = build_clarification_graph()
    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


def _to_native(obj):
    """Recursively coerce numpy scalars/arrays to native Python types so the
    MemorySaver checkpointer can msgpack-serialize graph state."""
    if isinstance(obj, dict):
        return {_to_native(k): _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return _to_native(obj.tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    return obj