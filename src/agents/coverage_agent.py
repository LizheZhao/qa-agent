"""Coverage sub-agent: answers data-availability sub-queries.

Port of _process_coverage from pages/orchestrator_page.py into a graph node.
The context stored for dependents is the response plus the structured facts, so a
dependent resolution sees the reliable item list.
"""
import json
import logging

from langchain_core.runnables import RunnableConfig

from src.data.data_interface import NerData, FeasibilityConfig, SpacyConfig
from src.model.coverage import answer_coverage
from src.agents.state import AgentState, current_spec

logger = logging.getLogger(__name__)


def coverage_node(state: AgentState, config: RunnableConfig) -> dict:
    cfg = config["configurable"]
    client_code, model_group_id = cfg["client_code"], cfg["model_group_id"]
    spec = current_spec(state)

    ner_data = NerData.from_local(client_code, model_group_id)
    feasibility = FeasibilityConfig.from_local(client_code, model_group_id)
    spacy_config = SpacyConfig.from_local(client_code, model_group_id)
    result = answer_coverage(spec["query"], spec.get("spacy_hints", {}), spacy_config.core_filters,
                             feasibility, ner_data.df_bi, ner_data.df_br)

    context = result["response"] + "\n\n" + json.dumps(result.get("facts", {}), default=str)
    payload = {"index": state["cursor"], "kind": "coverage", "query": spec["query"],
               "response": result["response"], "tables": [result.get("table")]}
    return {"pending_payload": payload, "pending_context": context}
