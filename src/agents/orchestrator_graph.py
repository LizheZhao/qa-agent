"""Agent orchestrator graph: plan -> per-sub-query agents -> re-plan on failure -> finalize.

Same workflow as pages/orchestrator_page.py, moved from an imperative page loop into one
LangGraph StateGraph. The route_next router is the loop: it sends each sub-query to the
coverage agent or to the analytical chain (dependency resolution -> NER clarification ->
readout/insights), the record node commits each result and advances the cursor, and the
replan node revises the remaining plan on failure, empty data or a dependency split.

Clarification interrupts pause the whole graph; the page resumes it with
Command(resume=selections). State must stay checkpointable, so the checkpointer uses
JsonPlusSerializer with pickle fallback (payload tables are DataFrames).
"""
import logging

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langchain_core.runnables import RunnableConfig

from src.model.planner import generate_plan
from src.model.filter_clarification_graph import _to_native
from src.agents.state import AgentState, CLEARED_SCRATCH, current_spec, spec_to_dict
from src.agents.coverage_agent import coverage_node
from src.agents.dependency_agent import resolve_dependency_node
from src.agents.clarification_agent import extract_ner_node, clarify_node, merge_and_tag, should_clarify
from src.agents.analytical_agent import run_analytical_node
from src.agents.replan import replan_node

logger = logging.getLogger(__name__)


def plan_node(state: AgentState, config: RunnableConfig) -> dict:
    cfg = config["configurable"]
    result = generate_plan(cfg["client_code"], cfg["model_group_id"], state["query"])
    subqueries = _to_native([spec_to_dict(enriched) for enriched in result.subqueries])
    logger.info(f"plan: {len(subqueries)} sub-queries, sequential={result.is_sequential}")
    return {"rephrased_query": result.plan.rephrased_query, "is_multi": result.plan.is_multi,
            "is_sequential": result.is_sequential, "subqueries": subqueries, "cursor": 0,
            "payloads": [], "contexts": {}, "replan_notes": [], "replan_count": 0, "attempts": {},
            **CLEARED_SCRATCH}


def record_node(state: AgentState) -> dict:
    """Commit the pending result, advance the cursor, clear the per-sub-query scratch."""
    payload = state["pending_payload"]
    return {"payloads": state["payloads"] + [payload],
            "contexts": {**state["contexts"], payload["index"]: state.get("pending_context") or ""},
            "cursor": state["cursor"] + 1,
            **CLEARED_SCRATCH}


def finalize_node(state: AgentState) -> dict:
    logger.info(f"run complete: {len(state.get('payloads', []))} sub-query results")
    return {}


def route_next(state: AgentState) -> str:
    """The main loop: pick the agent for the sub-query at the cursor, or finish."""
    if state["cursor"] >= len(state["subqueries"]):
        return "finalize"
    if current_spec(state)["kind"] == "coverage":
        return "coverage"
    return "resolve_dependency"


def after_resolve(state: AgentState) -> str:
    return "replan" if state.get("replan_trigger") else "extract_ner"


def after_analytical(state: AgentState) -> str:
    return "replan" if state.get("replan_trigger") else "record"


def after_replan(state: AgentState) -> str:
    # a drop leaves a rejection payload to commit; a split/revision re-enters the loop
    return "record" if state.get("pending_payload") else route_next(state)


def build_agent_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("plan", plan_node)
    graph.add_node("coverage", coverage_node)
    graph.add_node("resolve_dependency", resolve_dependency_node)
    graph.add_node("extract_ner", extract_ner_node)
    graph.add_node("clarify", clarify_node)
    graph.add_node("merge_and_tag", merge_and_tag)
    graph.add_node("run_analytical", run_analytical_node)
    graph.add_node("replan", replan_node)
    graph.add_node("record", record_node)
    graph.add_node("finalize", finalize_node)

    loop_targets = {"coverage": "coverage", "resolve_dependency": "resolve_dependency",
                    "finalize": "finalize"}
    graph.add_edge(START, "plan")
    graph.add_conditional_edges("plan", route_next, loop_targets)
    graph.add_conditional_edges("resolve_dependency", after_resolve,
                                {"replan": "replan", "extract_ner": "extract_ner"})
    graph.add_conditional_edges("extract_ner", should_clarify,
                                {"clarify": "clarify", "merge_and_tag": "merge_and_tag"})
    graph.add_edge("clarify", "merge_and_tag")
    graph.add_edge("merge_and_tag", "run_analytical")
    graph.add_conditional_edges("run_analytical", after_analytical,
                                {"replan": "replan", "record": "record"})
    graph.add_edge("coverage", "record")
    graph.add_conditional_edges("replan", after_replan, {**loop_targets, "record": "record"})
    graph.add_conditional_edges("record", route_next, loop_targets)
    graph.add_edge("finalize", END)

    return graph


def create_agent_app():
    """Compile the graph. pickle_fallback lets the checkpointer hold DataFrames."""
    checkpointer = MemorySaver(serde=JsonPlusSerializer(pickle_fallback=True))
    return build_agent_graph().compile(checkpointer=checkpointer)
