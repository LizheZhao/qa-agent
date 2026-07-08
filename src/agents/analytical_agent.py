"""Analytical/insights sub-agent: readout pipeline plus insight generation.

Port of _run_analytical_pipeline and _generate_insight from pages/orchestrator_page.py
into a graph node, with one addition: if the insight LLM denies on the visual context,
retry once with the raw readout context before surfacing the denial.

Failures (rejected readout, empty data) do not produce a payload here; they set
replan_trigger and the replan node decides what to do with the remaining plan.
"""
import logging

from langchain_core.runnables import RunnableConfig

from src.data.data_interface import ProcessIndicator
from src.model.filter_generator_refactor import generate_readoutdata
from src.model.readout import process_data, response_generate_with_denial
from src.model.common import PromptTemplates
from src.agents.state import AgentState, current_spec

logger = logging.getLogger(__name__)


def run_analytical_node(state: AgentState, config: RunnableConfig) -> dict:
    cfg = config["configurable"]
    client_code, model_group_id = cfg["client_code"], cfg["model_group_id"]
    llm_vendor = cfg.get("llm_vendor")
    spec = current_spec(state)
    query_to_run = state.get("filled_query") or spec["query"]

    try:
        process_indicator = ProcessIndicator.from_local(client_code, model_group_id)
        readout_data = generate_readoutdata(clientCode=client_code, modelgroupId=model_group_id,
                                            query=query_to_run, ner_results=state["ner_results"],
                                            spacy_filter=state["spacy_filter"],
                                            process_indicator=process_indicator)
    except ValueError as exc:
        return {"replan_trigger": {"type": "error", "reason": str(exc)}}

    (context_str, data, benchmark_data, benchmark_str, planner_data, spend_share_principle, pretext,
     pivot_biz_table, pretext_table, pretext_table_trend, match, principle_pretext, overall_view,
     readout_adj) = process_data(client_code, model_group_id, readout_data, None)

    # empty data: hand the rejection text to the replan node instead of answering
    if (data is None or data.empty) and (pivot_biz_table is None or pivot_biz_table.empty) \
            and (planner_data is None or planner_data.empty):
        rejection = readout_data.ner_filters.get("rejection_message")
        message = pretext if pretext else (rejection or PromptTemplates().rejection_response)
        return {"replan_trigger": {"type": "empty_data", "reason": message}}

    insight_query = "Generate insights with given context." if readout_adj else query_to_run
    all_config_texts = _prep_visual_context(
        client_code, model_group_id, data, planner_data, match, readout_data,
        pivot_biz_table, spend_share_principle, insight_query, pretext_table, pretext_table_trend)
    readout_source = all_config_texts if all_config_texts else context_str

    main_content = _generate_insight(insight_query, readout_source, llm_vendor,
                                     fallback_readout=context_str)
    principle_insights = _generate_insight("Summarize insights.", principle_pretext, llm_vendor) \
        if principle_pretext else ""

    if planner_data.empty:
        chunks = (f"Pretext: {pretext} \n\n"
                  f"Insights: {main_content} \n\n"
                  f"Principle insights: {principle_insights} \n\n"
                  f"Benchmark: {benchmark_str}")
    else:
        chunks = (f"Insights: {main_content} \n\n"
                  f"Pretext: {pretext} \n\n"
                  f"Principle insights: {principle_insights}  \n\n"
                  f"Benchmark: {benchmark_str}")

    payload = {"index": state["cursor"], "kind": "analytical", "query": query_to_run,
               "response": chunks, "tables": [pretext_table, pivot_biz_table]}
    return {"pending_payload": payload, "pending_context": context_str}


def _prep_visual_context(*args) -> str:
    # visual_config_utils imports the repo-root st_utils at module level, and that file
    # is missing from the repo; import lazily so the agent graph stays loadable without it
    import src.model.visual_config_utils as visual_configs
    return visual_configs._prep_visual_context(*args)


def _generate_insight(query: str, readout_source: str, llm_vendor: str,
                      fallback_readout: str = "") -> str:
    """response_generate_with_denial + the </think> strip, with one retry on denial."""
    result = response_generate_with_denial(query, readout_source, llm_vendor=llm_vendor)
    if result["is_denial"] and fallback_readout and fallback_readout != readout_source:
        logger.info("insight denied on visual context, retrying once with raw readout")
        result = response_generate_with_denial(query, fallback_readout, llm_vendor=llm_vendor)
    response = result["denial_reason"] if result["is_denial"] else result["answer"]
    if response and "</think>" in response:
        _, response = response.split("</think>", 1)
    return (response or "").strip()
