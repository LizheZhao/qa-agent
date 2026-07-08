"""Re-plan node: decides what happens to the remaining plan when a sub-query fails.

Triggers and behavior:
- split (from dependency resolution): deterministic, splice the expanded sub-queries
  into the plan at the cursor, exactly like _splice_subqueries on orchestrator_page.
- dep_missing / dep_reject: deterministic drop with the same rejection messages the
  orchestrator page shows.
- empty_data / error: the LLM picks a tool - drop_current or revise_remaining - via
  agent_tool_call (verbose, so tool selection is logged).

Completed work is never touched: only subqueries[cursor:] may be rewritten.
"""
import json
import logging
from typing import Literal, Optional

from pydantic import BaseModel
from langchain_core.runnables import RunnableConfig

from src.utils import load_yaml_file
from src.integrations.vendor import agent_tool_call
from src.agents.state import AgentState, current_spec, rejection_payload

logger = logging.getLogger(__name__)

MAX_REPLANS = 3
MAX_ATTEMPTS_PER_SUBQUERY = 2
MAX_PLAN_SIZE = 10


class RevisedSubQuery(BaseModel):
    query: str
    kind: Literal["coverage", "analytical"] = "analytical"
    coarse_intent: Optional[str] = None
    depends_on: Optional[int] = None


REPLAN_TOOLS = [
    {"name": "drop_current",
     "description": "Give up on the failed sub-query and continue with the rest of the plan. "
                    "Use this when rewording cannot fix the failure.",
     "input_schema": {"type": "object", "properties": {
         "reason": {"type": "string", "description": "short explanation"}}}},
    {"name": "revise_remaining",
     "description": "Replace the remaining sub-queries (the failed one and everything after it) "
                    "with a revised list that can still answer the original question.",
     "input_schema": {"type": "object", "properties": {
         "revised_subqueries": {"type": "array", "items": {"type": "object", "properties": {
             "query": {"type": "string"},
             "kind": {"type": "string", "enum": ["coverage", "analytical"]},
             "coarse_intent": {"type": "string"},
             "depends_on": {"type": "integer"}},
             "required": ["query"]}},
         "reason": {"type": "string", "description": "short explanation"}},
         "required": ["revised_subqueries"]}},
]


def replan_node(state: AgentState, config: RunnableConfig) -> dict:
    trigger = state["replan_trigger"]
    spec = current_spec(state)
    attempts = dict(state.get("attempts", {}))
    attempts[spec["id"]] = attempts.get(spec["id"], 0) + 1

    if trigger["type"] == "split":
        subqueries = _splice(state["subqueries"], state["cursor"], trigger["split_queries"])
        return {"subqueries": subqueries, "replan_trigger": None, "attempts": attempts,
                "filled_query": None}

    over_limit = state.get("replan_count", 0) >= MAX_REPLANS \
        or attempts[spec["id"]] > MAX_ATTEMPTS_PER_SUBQUERY
    if trigger["type"] in ("dep_missing", "dep_reject") or over_limit:
        return _drop_current(state, spec, trigger["reason"], attempts)

    # empty_data / error: let the LLM decide between dropping and revising
    decision = agent_tool_call(*_replan_prompts(state, trigger), tools=REPLAN_TOOLS,
                               sidebar_vendor=config["configurable"].get("llm_vendor"), verbose=True)
    revised = _validate_revision(decision, config["configurable"])
    if revised is None:
        return _drop_current(state, spec, trigger["reason"], attempts)

    note = f"Plan revised: {decision['args'].get('reason', trigger['reason'])}"
    subqueries = state["subqueries"][:state["cursor"]] + _with_fresh_ids(state["subqueries"], revised)
    subqueries = subqueries[:MAX_PLAN_SIZE]
    logger.info(f"{note} ({len(revised)} remaining sub-queries)")
    return {"subqueries": subqueries, "replan_trigger": None, "attempts": attempts,
            "filled_query": None,
            "replan_count": state.get("replan_count", 0) + 1,
            "replan_notes": state.get("replan_notes", []) + [note]}


def _drop_current(state: AgentState, spec: dict, reason: str, attempts: dict) -> dict:
    payload = rejection_payload(state["cursor"], spec["kind"],
                                state.get("filled_query") or spec["query"], reason)
    return {"pending_payload": payload, "pending_context": "",
            "replan_trigger": None, "attempts": attempts}


def _replan_prompts(state: AgentState, trigger: dict) -> tuple:
    completed = [{"query": p["query"], "rejected": p.get("rejected", False),
                  "context": (state.get("contexts", {}).get(p["index"]) or "")[:500]}
                 for p in state.get("payloads", [])]
    remaining = [{"query": s["query"], "kind": s["kind"], "coarse_intent": s["coarse_intent"]}
                 for s in state["subqueries"][state["cursor"]:]]
    prompts = load_yaml_file("./config/prompt.yaml")
    system_prompt = (prompts["REPLAN_PROMPT"]
                     .replace("QUERY_PLACEHOLDER", state["query"])
                     .replace("COMPLETED_PLACEHOLDER", json.dumps(completed, default=str))
                     .replace("REMAINING_PLACEHOLDER", json.dumps(remaining, default=str))
                     .replace("FAILURE_PLACEHOLDER", trigger["reason"]))
    return system_prompt, "Call exactly one tool."


def _validate_revision(decision: dict, cfg: dict) -> Optional[list]:
    """Returns the validated revised sub-queries, or None if the decision is a drop."""
    if decision["tool"] != "revise_remaining":
        return None
    raw = decision["args"].get("revised_subqueries") or []
    revised = [RevisedSubQuery.model_validate(item) for item in raw]
    if not revised:
        return None
    # local import: only needed on the rare revise path
    from src.data.data_interface import NerDataConfig
    ner_data_config = NerDataConfig.from_local(cfg["client_code"], cfg["model_group_id"])
    intent_catalog = list(ner_data_config.metric_info.keys())
    for item in revised:
        if item.coarse_intent not in intent_catalog:
            item.coarse_intent = None       # unknown intent: leave the decision to NER
    return revised


def _with_fresh_ids(subqueries: list, revised: list) -> list:
    next_id = max((s["id"] for s in subqueries), default=-1) + 1
    return [{"id": next_id + i, "query": r.query, "kind": r.kind, "coarse_intent": r.coarse_intent,
             "depends_on": r.depends_on, "spacy_hints": {}, "feasibility": None}
            for i, r in enumerate(revised)]


def _splice(subqueries: list, cursor: int, queries: list) -> list:
    """Replace the sub-query at cursor with N self-contained analytical sub-queries."""
    next_id = max((s["id"] for s in subqueries), default=-1) + 1
    new_items = [{"id": next_id + i, "query": q, "kind": "analytical", "coarse_intent": None,
                  "depends_on": None, "spacy_hints": {}, "feasibility": None}
                 for i, q in enumerate(queries)]
    return subqueries[:cursor] + new_items + subqueries[cursor + 1:]
