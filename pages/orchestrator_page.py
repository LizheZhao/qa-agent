"""Planner page (multi-page Streamlit, v1).

Drives the planner layer: decompose a query into sub-queries, route each as
coverage (answered directly) or analytical (existing clarification-graph + readout
pipeline), thread sequential dependencies via context extraction, and render each
sub-query under its own header.

st-main.py is intentionally untouched. This page reuses the clarification graph,
generate_readoutdata, process_data, and the coverage branch. The analytical insight
step mirrors insights_only: _prep_visual_context + response_generate_with_denial, and
the response chunk is displayed the same way. Rich chart rendering (the large st-main
chart layer) is deferred for v1; analytical sub-queries render the synthesized text
answer plus the key data tables.
"""
import os
import json

import streamlit as st
from langgraph.types import Command

import st_utils as utils
from src.utils import get_available_client_and_model_group
from src.render.streamlit_utils import new_thread_id
from src.model.filter_clarification_graph import create_app
from src.model.filter_generator_refactor import generate_readoutdata
from src.model.readout import process_data, response_generate_with_denial
from src.model.planner import (
    generate_plan, resolve_from_context,
    SubQuerySpec, SubQueryAnnotation, EnrichedSubQuery,
)
from src.model.coverage import answer_coverage
from src.model.common import PromptTemplates
import src.model.visual_config_utils as visual_configs
from src.data.data_interface import ProcessIndicator, NerData, FeasibilityConfig, SpacyConfig

st.set_page_config(page_title="Ask Genome - Planner", layout="wide")

SESSION_DEFAULTS = {
    "planner_query": None,
    "planner_result": None,
    "planner_active": 0,
    "planner_renders": None,
    "planner_context": None,
    "planner_await": False,
    "planner_fields": None,
    "planner_graph_result": None,
    "planner_thread": None,
    "planner_filled_query": None,
}
for _key, _default in SESSION_DEFAULTS.items():
    if _key not in st.session_state:
        st.session_state[_key] = _default

if st.session_state.get("graph_app") is None:
    st.session_state.graph_app = create_app()

# --------------------------------------------------------------------------- sidebar
with st.sidebar:
    client_model_group_options = get_available_client_and_model_group()
    selected = st.selectbox("Client Code", options=client_model_group_options.keys(),
                            key="planner_client_model_group")
    client_code, model_group_id = client_model_group_options[selected]
    st.session_state.client_code = client_code
    st.session_state.model_group_id = model_group_id
    os.environ["CLIENT_CODE"] = client_code
    os.environ["MODEL_GROUP_ID"] = str(model_group_id)
    st.markdown(f"Client Code: `{client_code}`")
    st.markdown(f"Model Group ID: `{model_group_id}`")

    st.divider()
    st.selectbox("LLM Vendor", ["openai", "gemini", "claude"], index=0, key="planner_llm_vendor")
    os.environ["EXTERNAL_VENDOR"] = st.session_state.planner_llm_vendor
    st.toggle("Use external llm for filter extraction", value=False, key="enterprise_ai_ner")
    st.toggle("LLM filter extraction priority", value=False, key="llm_priority")

utils._detect_and_reset_on_context_change(client_code=client_code, model_group_id=model_group_id, page_id=__file__)


def _reset_planner_state() -> None:
    st.session_state.planner_result = None
    st.session_state.planner_active = 0
    st.session_state.planner_renders = []
    st.session_state.planner_context = {}
    st.session_state.planner_await = False
    st.session_state.planner_fields = None
    st.session_state.planner_graph_result = None
    st.session_state.planner_thread = None
    st.session_state.planner_filled_query = None


def _graph_config(thread_id: str, coarse_intent) -> dict:
    return {"configurable": {
        "thread_id": thread_id,
        "client_code": st.session_state.client_code,
        "model_group_id": int(st.session_state.model_group_id),
        "enterprise_ai_ner": st.session_state.enterprise_ai_ner,
        "llm_priority": st.session_state.get("llm_priority", False),
        "coarse_intent": coarse_intent,
    }}


def _render_payload(payload: dict) -> None:
    label = f"Sub-query {payload['index'] + 1}: {payload['kind']}"
    with st.expander(label, expanded=True):
        st.caption(payload["query"])
        if payload.get("rejected"):
            st.warning(payload.get("message", "This sub-query could not be answered."))
            return
        if payload["kind"] == "analytical":
            # display the final response chunk the same way as insights_only
            with st.chat_message("assistant", avatar="./assets/ap-logo-margin.png"):
                st.markdown(payload.get("response", ""))
        elif payload.get("response"):
            st.write(payload["response"])
        for table in payload.get("tables", []):
            if table is not None and not table.empty:
                st.dataframe(table, use_container_width=True)


def _finish(payload: dict, context: str) -> None:
    """Store a completed sub-query payload + its downstream context, advance the cursor."""
    st.session_state.planner_renders.append(payload)
    st.session_state.planner_context[payload["index"]] = context
    st.session_state.planner_active += 1
    st.session_state.planner_await = False
    st.session_state.planner_fields = None
    st.session_state.planner_graph_result = None
    st.session_state.planner_thread = None
    st.session_state.planner_filled_query = None
    st.rerun()


def _process_coverage(idx: int, esq) -> None:
    with st.spinner("Looking up coverage..."):
        ner_data = NerData.from_local(client_code, int(model_group_id))
        feasibility = FeasibilityConfig.from_local(client_code, int(model_group_id))
        spacy_config = SpacyConfig.from_local(client_code, int(model_group_id))
        result = answer_coverage(esq.spec.query, esq.annotation.spacy_hints, spacy_config.core_filters,
                                 feasibility, ner_data.df_bi, ner_data.df_br)
    # store response + structured facts so a dependent resolution sees the reliable item list
    context = result["response"] + "\n\n" + json.dumps(result.get("facts", {}), default=str)
    payload = {"index": idx, "kind": "coverage", "query": esq.spec.query,
               "response": result["response"], "tables": [result.get("table")]}
    _finish(payload, context)


def _splice_subqueries(idx: int, queries: list) -> None:
    """Replace the template sub-query at idx with N self-contained analytical sub-queries."""
    result = st.session_state.planner_result
    existing_ids = [e.spec.id for e in result.subqueries]
    next_id = (max(existing_ids) + 1) if existing_ids else 0
    new_items = [EnrichedSubQuery(
        spec=SubQuerySpec(id=next_id + i, query=q, kind="analytical", coarse_intent=None, depends_on=None),
        annotation=SubQueryAnnotation()) for i, q in enumerate(queries)]
    result.subqueries[idx:idx + 1] = new_items


def _resolve_dependency(idx: int, esq):
    """Returns (query_to_run, reject_payload_or_None). For a multi-way split it splices the
    expanded sub-queries into the plan and reruns (does not return)."""
    dep = esq.spec.depends_on
    if dep is None:
        return esq.spec.query, None
    context = st.session_state.planner_context.get(dep)
    if not context:
        return None, {"index": idx, "kind": "analytical", "query": esq.spec.query, "rejected": True,
                      "message": f"This sub-query depends on sub-query {dep + 1}, which returned no data."}
    subqueries, resolved, reason = resolve_from_context(context, esq.spec.query)
    if not resolved or not subqueries:
        return None, {"index": idx, "kind": "analytical", "query": esq.spec.query, "rejected": True,
                      "message": f"Could not derive this sub-query from sub-query {dep + 1}: {reason}"}
    if len(subqueries) == 1:
        st.session_state.planner_filled_query = subqueries[0]
        return subqueries[0], None
    # split: fan out into one analytical sub-query per item, then re-process from this cursor
    _splice_subqueries(idx, subqueries)
    st.rerun()


def _run_analytical_pipeline(idx: int, query_to_run: str, esq) -> None:
    graph_result = st.session_state.planner_graph_result
    ner_results = graph_result["ner_results"]
    spacy_filter = graph_result["spacy_filter"]
    llm_vendor = st.session_state.planner_llm_vendor
    try:
        with st.spinner("Fetching data..."):
            process_indicator = ProcessIndicator.from_local(client_code, int(model_group_id))
            readout_data = generate_readoutdata(clientCode=client_code, modelgroupId=int(model_group_id),
                                                query=query_to_run, ner_results=ner_results,
                                                spacy_filter=spacy_filter, process_indicator=process_indicator)
    except ValueError as exc:
        _finish({"index": idx, "kind": "analytical", "query": query_to_run, "rejected": True,
                 "message": str(exc)}, "")
        return

    (context_str, data, benchmark_data, benchmark_str, planner_data, spend_share_principle, pretext,
     pivot_biz_table, pretext_table, pretext_table_trend, match, principle_pretext, overall_view,
     readout_adj) = process_data(client_code, model_group_id, readout_data, None)

    # empty data: show pretext / rejection text and give downstream dependents no context
    if (data is None or data.empty) and (pivot_biz_table is None or pivot_biz_table.empty) \
            and (planner_data is None or planner_data.empty):
        rejection = readout_data.ner_filters.get("rejection_message")
        message = pretext if pretext else (rejection or PromptTemplates().rejection_response)
        _finish({"index": idx, "kind": "analytical", "query": query_to_run, "rejected": True,
                 "message": message}, "")
        return

    # mirror insights_only: build visual context, synthesize with denial, assemble response chunks
    insight_query = "Generate insights with given context." if readout_adj else query_to_run
    with st.spinner("Generating insights..."):
        all_config_texts = visual_configs._prep_visual_context(
            client_code, int(model_group_id), data, planner_data, match, readout_data,
            pivot_biz_table, spend_share_principle, insight_query, pretext_table, pretext_table_trend)
        readout_source = all_config_texts if all_config_texts else context_str

        main_content = _generate_insight(insight_query, readout_source, llm_vendor)
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

    payload = {"index": idx, "kind": "analytical", "query": query_to_run, "response": chunks,
               "tables": [pretext_table, pivot_biz_table]}
    _finish(payload, context_str)


def _generate_insight(query: str, readout_source: str, llm_vendor: str) -> str:
    """response_generate_with_denial + the </think> strip, as in insights_only."""
    result = response_generate_with_denial(query, readout_source, llm_vendor=llm_vendor)
    response = result["denial_reason"] if result["is_denial"] else result["answer"]
    if response and "</think>" in response:
        _, response = response.split("</think>", 1)
    return (response or "").strip()


def _process_analytical(idx: int, esq) -> None:
    # Dependency resolution (once) before clarification.
    if not st.session_state.planner_await and st.session_state.planner_graph_result is None:
        query_to_run, reject = _resolve_dependency(idx, esq)
        if reject is not None:
            _finish(reject, "")
            return
        thread_id = new_thread_id()
        st.session_state.planner_thread = thread_id
        config = _graph_config(thread_id, esq.spec.coarse_intent)
        with st.spinner("Understanding sub-query..."):
            st.session_state.graph_app.invoke({"query": query_to_run}, config=config)
        graph_state = st.session_state.graph_app.get_state(config)
        if graph_state.next:
            st.session_state.planner_await = True
            st.session_state.planner_fields = graph_state.tasks[0].interrupts[0].value["clarification_fields"]
            st.session_state.planner_graph_result = graph_state.values
        else:
            st.session_state.planner_graph_result = graph_state.values
            _run_analytical_pipeline(idx, query_to_run, esq)
            return

    query_to_run = st.session_state.planner_filled_query or esq.spec.query

    # Clarification form.
    if st.session_state.planner_await and st.session_state.planner_fields:
        fields = st.session_state.planner_fields
        current = st.session_state.planner_graph_result.get("ner_results", {})
        st.markdown(f"#### Sub-query {idx + 1} needs confirmation")
        with st.form(f"planner_clarify_{idx}"):
            for field_name, options in fields.items():
                current_value = current.get(field_name, [])
                if isinstance(current_value, str):
                    current_value = [current_value]
                default = [o for o in options if o in current_value]
                st.multiselect(field_name.replace("_", " ").title(), options=options,
                               default=default, key=f"planner_clarify_{idx}_{field_name}")
            submitted = st.form_submit_button("Confirm selections")
        if submitted:
            selections = {f: st.session_state[f"planner_clarify_{idx}_{f}"] for f in fields}
            empty = [f for f, v in selections.items() if not v]
            if empty:
                st.error(f"Please select at least one option for: {', '.join(empty)}")
            else:
                config = _graph_config(st.session_state.planner_thread, esq.spec.coarse_intent)
                st.session_state.graph_app.invoke(Command(resume=selections), config=config)
                graph_state = st.session_state.graph_app.get_state(config)
                st.session_state.planner_graph_result = graph_state.values
                st.session_state.planner_await = False
                _run_analytical_pipeline(idx, query_to_run, esq)
        return


# --------------------------------------------------------------------------- main flow
st.title("Ask-Genome")
query = st.chat_input("Ask a question")
if query:
    _reset_planner_state()
    st.session_state.planner_query = query
    with st.spinner("Planning..."):
        st.session_state.planner_result = generate_plan(client_code, int(model_group_id), query)

result = st.session_state.planner_result
if result is not None:
    st.chat_message("user").write(st.session_state.planner_query)
    if result.plan.is_multi:
        st.caption(f"Rephrased: {result.plan.rephrased_query}")
        st.caption(f"{len(result.plan.subqueries)} sub-queries"
                   + (" (sequential)" if result.is_sequential else ""))

    for payload in st.session_state.planner_renders:
        _render_payload(payload)

    active = st.session_state.planner_active
    if active < len(result.subqueries):
        esq = result.subqueries[active]
        if esq.spec.kind == "coverage":
            _process_coverage(active, esq)
        else:
            _process_analytical(active, esq)
