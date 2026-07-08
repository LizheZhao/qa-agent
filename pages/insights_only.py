"""
pages/insights_only.py

Insights-only Streamlit page.  Mirrors the data pipeline of st-main_local_version.py
(same session-state keys, same NER → process_data → routing flow) but produces
text insights only — zero chart rendering.

Two insight methods, both always executed:
  Method 1 — Synthesised from raw chart data
              get_chart_set_inst(chart_configs)  →  response_generate_chart
  Method 2 — Summarised from per-chart insights (deprecated)
              get_insights_set_inst(insights)    →  response_generate_chart
"""

import os
import re
import csv
import json
import pandas as pd

import streamlit as st
import datetime

from langgraph.types import Command
from src.model.filter_generator_refactor import generate_readoutdata
from src.model.readout import process_data, response_generate_chart, response_generate_with_denial
from src.model.insight_report import DocumentRetriever
from src.model.roi_genome_slides import ROIGenomeDocumentRetriever
from src.model.filter_clarification_graph import create_app
from src.model.common import PromptTemplates
from src.render.streamlit_utils import new_thread_id
from src.data.data_interface import ProcessIndicator, ReadoutData
from src.utils import get_available_client_and_model_group
import src.model.readout_utils as readout_utils
import st_utils as utils
import viz_utils
import src.model.visual_config_utils as visual_configs
import src.insight_generation.contribution as contribution_utils

# ---------------------------------------------------------------------------
# Local helpers — "All" iteration logic for metric × level combinations
# ---------------------------------------------------------------------------
# save to json
def save_dict_to_jsonl(records: list[dict], filepath: str) -> None:
    with open(filepath, "a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# save response to csv
def write_dict_to_csv(data: dict, filepath: str) -> None:
    file_exists = os.path.exists(filepath)

    with open(filepath, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=data.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(data)


# save dataframes
def save_dataframes(
        df_activity_group: pd.DataFrame,
        df_measure_group: pd.DataFrame,
        df_measure: pd.DataFrame,
        query_id: str | int,
        test_version: str,
        output_folder: str = "output"
) -> None:
    folder = os.path.join(output_folder, test_version, f"query_{query_id}")
    os.makedirs(folder, exist_ok=True)

    df_activity_group.to_csv(os.path.join(folder, "df_activity_group.csv"), index=False)
    df_measure_group.to_csv(os.path.join(folder, "df_measure_group.csv"), index=False)
    df_measure.to_csv(os.path.join(folder, "df_measure.csv"), index=False)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Ask Genome – Insights",
    layout="wide",
    page_icon="./assets/ap-logo.png",
)

st.markdown("""
<style>
[data-testid="stMetric"] {
    background-color: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 15px 0;
    text-align: center;
    box-shadow: 0 2px 6px rgba(0,0,0,0.05);
    border: 1px solid #E2E8F0;
    box-shadow:
        0 1px 3px rgba(0, 0, 0, 0.08),
        0 4px 12px rgba(0, 0, 0, 0.05);
    backdrop-filter: blur(2px);
}
div[data-testid="stExpander"]>details{
  background:#fff;border:1px solid #E5E7EB;border-radius:12px;
  padding:8px 12px;box-shadow:0 1px 3px rgba(0,0,0,.04);
}
.sec-title{
  font-size:1.05rem; font-weight:700; color:#0F172A;
  margin:6px 0 8px; padding-left:10px; border-left:4px solid #2563EB;
}
</style>
""", unsafe_allow_html=True)

left, center, right = st.columns([1, 2, 1])

# ---------------------------------------------------------------------------
# Session state — mirrors st-main_local_version.py
# ---------------------------------------------------------------------------

for key in [
    "thinking_content", "main_content", "query", "readout",
    "selected_intention", "processed_data", "response_chunks",
    "intention_confirmed", "intention_required_confirmation",
    "multitask_pred", "spacy_filter", "process_indicator", "time_confirmed",
    "original_period", "original_time",
    # insights-only page extras
    "insight", "all_chart_configs", "all_insights", "resp1", "resp2", "charts_block"
]:
    if key not in st.session_state:
        if key in ["thinking_content", "main_content", "original_period", "original_time"]:
            st.session_state[key] = ""
        elif key in ["intention_confirmed", "intention_required_confirmation", "time_confirmed"]:
            st.session_state[key] = False
        elif key in ["insight", "all_chart_configs", "all_insights"]:
            st.session_state[key] = {}
        else:
            st.session_state[key] = None

SESSION_DEFAULTS = {
    "query": None,
    "thread_id": None,
    "client_code": None,
    "model_group_id": None,
    "graph_result": None,
    "awaiting_clarification": False,
    "clarification_fields": None,
    "ner_results": None,
    "field_status": None,
    "user_selections": None,
    "_thread_counter": 0,
    "final_filters": None,
    "readout_data": None,
    "process_indicator": None,
}

for key, default in SESSION_DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    commit = os.getenv("CI_COMMIT")
    st.markdown(f"Commit: `{commit}`")

    client_model_group_options = get_available_client_and_model_group()
    selected = st.selectbox("Client Code", client_model_group_options.keys(), on_change=utils.reset_query_and_data)
    client_code, model_group_id = client_model_group_options[selected]
    st.session_state.client_code = client_code
    st.session_state.model_group_id = model_group_id

    os.environ["CLIENT_CODE"] = client_code
    os.environ["MODEL_GROUP_ID"] = str(model_group_id)

    file_name = f"{st.session_state.client_code}_readout_data_{datetime.date.today().strftime('%m%d%Y')}.jsonl"
    st.session_state.data_dir = "/home/lzhao/Documents/ask-genome-data/"
    st.session_state.readout_data_file_name = file_name

    st.markdown(f"Client Code: `{client_code}`")
    st.markdown(f"Model Group ID: `{model_group_id}`")

    st.divider()
    llm_vendor = st.selectbox("Response LLM Vendor",
                              ["openai", "gemini", "claude"],
                              index=0,
                              on_change=utils.reset_query_and_data)
    os.environ["EXTERNAL_VENDOR"] = llm_vendor

    st.divider()
    st.toggle(
        "Use external llm for filter extraction",
        value=False,
        key="enterprise_ai_ner",
        help="When on, send every field to external llm used for response. Otherwise use deepseek",
        on_change=utils.reset_query_and_data
    )

    st.divider()
    st.toggle(
        "LLM filter extraction priority",
        value=False,
        key="llm_priority",
        help="When on, send every field to the LLM and let LLM override BERT for all fields.",
        on_change=utils.reset_query_and_data
    )

# ---------------------------------------------------------------------------
# Reset on page switch or client/model change
# ---------------------------------------------------------------------------

utils._detect_and_reset_on_context_change(
    client_code=client_code,
    model_group_id=model_group_id,
    page_id=__file__,
)

if st.session_state.get("graph_app") is None:
    st.session_state.graph_app = create_app()

# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------

with center:
    query = st.chat_input()

    if query:
        st.session_state.query = query
        st.session_state.readout = None
        st.session_state.processed_data = None
        st.session_state.response_chunks = None
        st.session_state.thinking_content = ""
        st.session_state.main_content = ""
        st.session_state.intention_confirmed = False
        st.session_state.intention_required_confirmation = False
        st.session_state.final_filters = None
        st.session_state.field_status = None
        st.session_state.awaiting_clarification = False
        st.session_state.clarification_fields = None
        st.session_state.graph_result = None
        # reset insight caches when a new query comes in
        st.session_state.insight = {}
        st.session_state.all_chart_configs = {}
        st.session_state.all_insights = {}
        st.session_state.resp1 = None
        st.session_state.resp2 = None

    if st.session_state.get("query"):
        st.chat_message("user", avatar=":material/person:").write(st.session_state.query)

    # ── Cache check — reuse saved readout if exact query exists in jsonl ──────
    if (st.session_state.query and st.session_state.readout is None
            and st.session_state.final_filters is None
            and not st.session_state.awaiting_clarification):
        try:
            cache_path = os.path.join(st.session_state.data_dir, st.session_state.readout_data_file_name)
            matched_record = None
            with open(cache_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    if record.get("query") == st.session_state.query:
                        matched_record = record  # last match wins

            if matched_record is not None:
                st.session_state.readout = ReadoutData(
                    ner_filters=matched_record["ner_filters"],
                    ner_results=matched_record["ner_results"],
                    df_activity_group=pd.DataFrame(matched_record["df_activity_group"]),
                    df_measure_group=pd.DataFrame(matched_record["df_measure_group"]),
                    df_measure=pd.DataFrame(matched_record["df_measure"]),
                )
                st.session_state.process_indicator = ProcessIndicator.from_local(
                    st.session_state.client_code, int(st.session_state.model_group_id)
                )
                if not st.session_state.original_period and not st.session_state.original_time:
                    st.session_state.original_period = st.session_state.readout.ner_filters["period_type"][0]
                    st.session_state.original_time = st.session_state.readout.ner_filters["time"]
        except Exception:
            pass

    # ── NER (langgraph clarification front-end, mirrors st-main.py) ───────────
    if (st.session_state.query and st.session_state.readout is None
            and st.session_state.final_filters is None
            and not st.session_state.awaiting_clarification):
        thread_id = new_thread_id()
        st.session_state.thread_id = thread_id
        config = {
            "configurable": {
                "thread_id": thread_id,
                "client_code": client_code,
                "model_group_id": int(model_group_id),
                "enterprise_ai_ner": st.session_state.enterprise_ai_ner,
                "llm_priority": st.session_state.get("llm_priority", False),
            }
        }
        st.session_state.config = config

        with st.spinner("Understanding query..."):
            result = st.session_state.graph_app.invoke({"query": st.session_state.query}, config=config)

        graph_state = st.session_state.graph_app.get_state(st.session_state.config)

        if graph_state.next:
            interrupt_value = graph_state.tasks[0].interrupts[0].value
            st.session_state.awaiting_clarification = True
            st.session_state.clarification_fields = interrupt_value["clarification_fields"]
            st.session_state.graph_result = graph_state.values
        else:
            st.session_state.final_filters = result["ner_results"]
            st.session_state.field_status = result["field_status"]
            st.session_state.graph_result = result

    # prompt clarification
    if st.session_state.awaiting_clarification and st.session_state.clarification_fields:
        fields = st.session_state.clarification_fields
        current_filters = st.session_state.graph_result.get("ner_results", {})

        st.markdown("### Some fields need confirmation")
        st.caption("The following fields were ambiguous. Please confirm or adjust.")

        with st.form("clarification_form"):
            for field_name, options in fields.items():
                current_value = current_filters.get(field_name, [])
                if isinstance(current_value, str):
                    current_value = [current_value]
                default_indices = [i for i, opt in enumerate(options) if opt in current_value]

                missing = [str(v) for v in current_value
                           if v not in options and v not in ("irrelevant", "all", "specific")]
                if missing:
                    st.caption(f":warning: LLM understands the request for "
                               f"**{field_name.replace('_', ' ')}** as *{', '.join(missing)}*, "
                               f"but it is not available in the data.")

                st.multiselect(
                    label=f"{field_name.replace('_', ' ').title()}",
                    options=options,
                    default=[options[i] for i in default_indices] if default_indices else [],
                    key=f"clarify_{field_name}",
                )

            submitted = st.form_submit_button("Confirm selections")

        if submitted:
            user_selections = {}
            for field_name in fields:
                user_selections[field_name] = st.session_state[f"clarify_{field_name}"]

            empty_fields = [f for f, v in user_selections.items() if not v]
            if empty_fields:
                st.error(f"Please select at least one option for: {', '.join(empty_fields)}")
            else:
                config = {
                    "configurable": {
                        "thread_id": st.session_state.thread_id,
                        "client_code": st.session_state.client_code,
                        "model_group_id": int(st.session_state.model_group_id),
                        "llm_priority": st.session_state.get("llm_priority", False),
                    }
                }
                result = st.session_state.graph_app.invoke(
                    Command(resume=user_selections), config=config,
                )
                st.session_state.final_filters = result["ner_results"]
                st.session_state.field_status = result["field_status"]
                st.session_state.awaiting_clarification = False
                st.session_state.clarification_fields = None
                st.rerun()

    # display confirmed filters
    if st.session_state.final_filters and st.session_state.field_status:
        st.markdown("**Filters ready for downstream pipeline:**")
        with st.expander("Raw filter output"):
            st.json({
                "ner_results": st.session_state.final_filters,
                "field_status": st.session_state.field_status,
            })

    # Fetch data
    if st.session_state.final_filters and st.session_state.field_status:
        with st.spinner("Fetching data..."):
            process_indicator = ProcessIndicator.from_local(
                st.session_state.client_code, int(st.session_state.model_group_id)
            )
            readout_data = generate_readoutdata(clientCode=st.session_state.client_code,
                                                modelgroupId=int(st.session_state.model_group_id),
                                                query=st.session_state.query,
                                                ner_results=st.session_state.final_filters,
                                                spacy_filter=st.session_state.graph_result["spacy_filter"],
                                                process_indicator=process_indicator)

            st.session_state.process_indicator = process_indicator
            st.session_state.readout = readout_data

            # saving verified filters
            try:
                res_dict = {"id": st.session_state.client_code,
                            "query": st.session_state.query,
                            "ner_results": st.session_state.final_filters,
                            "ner_filters": st.session_state.readout.ner_filters,
                            "df_activity_group": st.session_state.readout.df_activity_group.to_dict(orient='records'),
                            "df_measure_group": st.session_state.readout.df_measure_group.to_dict(orient='records'),
                            "df_measure": st.session_state.readout.df_measure.to_dict(orient='records'),
                            }

                save_dict_to_jsonl(records=[res_dict],
                                   filepath=os.path.join(st.session_state.data_dir,
                                                         st.session_state.readout_data_file_name))
            except Exception as e:
                pass

            # Track original time period (same as st-main_local_version.py)
            if not st.session_state.original_period and not st.session_state.original_time:
                st.session_state.original_period = st.session_state.readout.ner_filters["period_type"][0]
                st.session_state.original_time = st.session_state.readout.ner_filters["time"]

    # ── process_data ─────────────────────────────────────────────────────────
    if st.session_state.readout is not None and st.session_state.processed_data is None:
        (
            context_str, data, benchmark_data, benchmark_str,
            planner_data, spend_share_principle, pretext,
            pivot_biz_table, pretext_table, pretext_table_trend,
            match, principle_pretext, overall_view, readout_adj,
        ) = process_data(client_code, model_group_id, st.session_state.readout, None)

        st.session_state.processed_data = {
            "context_str": context_str,
            "data": data,
            "benchmark_data": benchmark_data,
            "benchmark_str": benchmark_str,
            "planner_data": planner_data,
            "spend_share_principle": spend_share_principle,
            "pretext": pretext,
            "pivot_biz_table": pivot_biz_table,
            "pretext_table": pretext_table,
            "pretext_table_trend": pretext_table_trend,
            "match": match,
            "principle_pretext": principle_pretext,
            "overall_view": overall_view,
            "readout_adj": readout_adj,
        }

    # ── Unpack ───────────────────────────────────────────────────────────────
    if st.session_state.processed_data is not None:
        context_str         = st.session_state.processed_data["context_str"]
        data                = st.session_state.processed_data["data"]
        benchmark_data      = st.session_state.processed_data["benchmark_data"]
        benchmark_str       = st.session_state.processed_data["benchmark_str"]
        planner_data        = st.session_state.processed_data["planner_data"]
        spend_share_principle = st.session_state.processed_data["spend_share_principle"]
        pretext             = st.session_state.processed_data["pretext"]
        pivot_biz_table     = st.session_state.processed_data["pivot_biz_table"]
        pretext_table       = st.session_state.processed_data["pretext_table"]
        pretext_table_trend = st.session_state.processed_data["pretext_table_trend"]
        match               = st.session_state.processed_data["match"]
        principle_pretext   = st.session_state.processed_data["principle_pretext"]
        overall_view        = st.session_state.processed_data["overall_view"]
        readout_adj         = st.session_state.processed_data["readout_adj"]
    else:
        st.stop()

tab1, tab2, tab3 = st.tabs(["Response", "Insights Report", "ROI Genome"])

with (tab1):
    # ---------------------------------------------------------------------------
    # Routing variables
    # ---------------------------------------------------------------------------

    intention, is_trend, is_specific, if_agg_exists, driver_col, metric_ls, main_metric_ls, data_type, trend_check = \
        viz_utils.viz_config(st.session_state.readout, match, pivot_biz_table)
    prompt_inst = readout_utils.load_prompt_instruction(client_code, model_group_id)

    row_height, min_height, max_height = 35, 100, 600

    # ---------------------------------------------------------------------------
    # Data table previews
    # ---------------------------------------------------------------------------

    if overall_view is not None and not overall_view.empty:
        st.markdown("### Overall View")
        rows = min(len(overall_view), 20)
        height = max(min_height, min(max_height, rows * row_height))
        with st.expander("Show data table"):
            st.dataframe(overall_view.reset_index(drop=True), use_container_width=True, height=height)

    if pivot_biz_table is not None and not pivot_biz_table.empty:
        st.markdown("### Aggregation View")
        rows = min(len(pivot_biz_table), 20)
        height = max(min_height, min(max_height, rows * row_height))
        with st.expander("Show data table"):
            st.dataframe(pivot_biz_table.reset_index(drop=True), use_container_width=True, height=height)

    if data is not None and not data.empty:
        st.markdown("### Detail View")
        rows = min(len(data), 20)
        height = max(min_height, min(max_height, rows * row_height))
        with st.expander("Show data table"):
            st.dataframe(data.reset_index(drop=True), use_container_width=True, height=height)

    if planner_data is not None and not planner_data.empty:
        core_dim = st.session_state.readout.ner_filters.get("core_dimension", {})
        sub_df = st.session_state.readout.df_measure

        inc_data = planner_data[planner_data['Increase/Decrease'] == 'Increase'].drop(columns=['Increase/Decrease'])
        dec_data = planner_data[planner_data['Increase/Decrease'] == 'Decrease'].drop(columns=['Increase/Decrease'])

        if not inc_data.empty:
            st.markdown("### Driver(s) with Spending Increase")
            with st.expander("Show data table"):
                st.dataframe(inc_data.reset_index(drop=True), use_container_width=True)
            # st.table(inc_data)

        if not dec_data.empty:
            st.markdown("### Driver(s) with Spending Decrease")
            with st.expander("Show data table"):
                st.dataframe(dec_data.reset_index(drop=True), use_container_width=True)
            # st.table(dec_data)

    # ---------------------------------------------------------------------------
    # Generating chart configs and readout
    # ---------------------------------------------------------------------------

    if readout_adj:
        insight_query = 'Generate insights with given context.'
    else:
        insight_query = query

    fallback_intention = any(x for x in ["planner", "spend", "activity", "cost per", "source of change"] if x in st.session_state.readout.ner_filters["intention"])
    if fallback_intention:
        all_config_texts = None
    elif "contribution" in st.session_state.readout.ner_filters["intention"]:
        all_configs = contribution_utils.build_contribution_tbl_for_insights(st.session_state.client_code,
                                                                                  st.session_state.model_group_id,
                                                                                  data,
                                                                                  pivot_biz_table,
                                                                                  st.session_state.readout)
        all_config_texts = visual_configs._generate_insights_prompt(all_configs, insight_query, "Contribution")
    else:
        all_config_texts = visual_configs._prep_visual_context(st.session_state.client_code,
                                                               st.session_state.model_group_id,
                                                               data,
                                                               planner_data,
                                                               match,
                                                               st.session_state.readout,
                                                               pivot_biz_table,
                                                               spend_share_principle,
                                                               insight_query,
                                                               pretext_table,
                                                               pretext_table_trend,
                                                               )

    # ---------------------------------------------------------------------------
    # Generating response
    # ---------------------------------------------------------------------------

    # Thinking and Response Rendering
    with st.spinner("Generating insights..."):
        if st.session_state.response_chunks is None:
            if not (data.empty and pivot_biz_table.empty and planner_data.empty):
                # Pick the source: all_config_texts if present, else context_str
                readout_source = all_config_texts if all_config_texts else context_str
                response_dict = response_generate_with_denial(
                    insight_query,
                    readout_source,
                    llm_vendor=llm_vendor,
                )
                is_denial = response_dict["is_denial"]
                denial_reason = response_dict["denial_reason"]
                full_response = response_dict["answer"]
                if is_denial:
                    response = denial_reason
                else:
                    response = full_response
                if "</think>" in response:
                    _, response = response.split("</think>", 1)
                main_content = response.strip()


                if principle_pretext:
                    principle_response = response_generate_with_denial(
                        "Summarize insights.",
                        principle_pretext,
                        llm_vendor=llm_vendor,
                    )
                    is_denial = principle_response["is_denial"]
                    denial_reason = principle_response["denial_reason"]
                    full_response = principle_response["answer"]
                    if is_denial:
                        response = denial_reason
                    else:
                        response = full_response
                    if "</think>" in response:
                        _, response = response.split("</think>", 1)
                    principle_insights = response.strip()
                else:
                    principle_insights = ""

                # Assemble final display
                if planner_data.empty:
                    st.session_state.response_chunks = (
                        f"Pretext: {pretext} \n\n"
                        f"Insights: {main_content} \n\n"
                        f"Principle insights: {principle_insights} \n\n"
                        f"Benchmark: {benchmark_str}"
                    )
                else:
                    st.session_state.response_chunks = (
                        f"Insights: {main_content} \n\n"
                        f"Pretext: {pretext} \n\n"
                        f"Principle insights: {principle_insights}  \n\n"
                        f"Benchmark: {benchmark_str}"
                    )
            else:
                if pretext:
                    st.session_state.response_chunks = pretext
                else:
                    rejection_message = readout_data.ner_filters.get("rejection_message")
                    st.session_state.response_chunks = rejection_message if rejection_message \
                        else PromptTemplates().rejection_response

        st.markdown("### Insights from Data")
        with st.expander("Show insights", expanded=False):
            with st.chat_message("assistant", avatar="./assets/ap-logo-margin.png"):
                st.markdown(st.session_state.response_chunks)

        st.markdown("### Readout from Data")
        with st.expander("Show readout", expanded=False):
            st.write(readout_source)

    if principle_pretext:
        st.markdown("### Principle text from Data")
        with st.expander("Show principle", expanded=False):
            st.write(principle_pretext)

    # ---------------------------------------------------------------------------
    # Benchmark context
    # ---------------------------------------------------------------------------

    if benchmark_data is not None and not benchmark_data.empty:
        st.markdown("### Benchmark Data")
        with st.expander("Show benchmark table"):
            st.table(benchmark_data)


# ---------------------------------------------------------------------------
# Insights Report document retrieval
# ---------------------------------------------------------------------------
with (tab2):
    text_for_rag = st.session_state.query
    use_query = True

    st.markdown("## 🔍 Insights Report Results")
    try:
        retriever = DocumentRetriever(client_code, collection_type="client-deck")
        retriever.retrieve_insights_documents(text_for_rag, st.session_state.readout.ner_filters, 5)
        if retriever.retrieved_docs:
            with st.spinner("Generating insights from decks..."):
                insight_query = (
                    f"You are given several documents from client report slides."
                    f"Generate insights with given context."
                    f"Write your response in markdown format."
                )
                insight_context = "\n\n ".join([doc["filename"]+ "\n " + doc["document"] for doc in retriever.retrieved_docs])
                insights_resp = response_generate_chart(
                    query=insight_query,
                    readout=insight_context,
                    llm_vendor=llm_vendor,
                )
                if "</think>" in insights_resp:
                    _, insights_resp = insights_resp.split("</think>", 1)
                insights_resp = insights_resp.strip()

                with st.expander("📈 Insights from report decks", expanded=True):
                    st.markdown(insights_resp)

            st.markdown("### 📄 Retrieved Documents")
            retriever.display_retrieved_documents(threshold=0.5)

        else:
            st.warning("No relevant documents found.")
    except Exception as e:
        st.warning(e)


# ---------------------------------------------------------------------------
# ROI GENOME document retrieval
# ---------------------------------------------------------------------------
with (tab3):

    if st.session_state.resp1:
        text_for_rag = st.session_state.resp1
        use_query = False
    else:
        text_for_rag = st.session_state.query
        use_query = True

    st.markdown("## 🔍 ROI Genome Results")
    try:
        retriever = ROIGenomeDocumentRetriever(client_code, collection_type="roi-genome-deck")
        retriever.retrieve_documents(text_for_rag, use_query, 5)
        if retriever.retrieved_docs:
            st.markdown("### 📄 Retrieved Documents")
            retriever.display_retrieved_documents(threshold=0.8)
        else:
            st.warning("No relevant documents found.")
    except Exception as e:
        st.warning(e)


# ---------------------------------------------------------------------------
# Save generated insights
# ---------------------------------------------------------------------------
# if st.session_state.resp1:
#     Data_dir = "/home/lzhao/Documents/ask-genome-data/"
#
#     res_file = f"{st.session_state.client_code}_test_insights.csv"
#     res = {"query": st.session_state.query, "intention": intention, "trend": trend_check,
#            "prompt1": st.session_state.charts_block,
#            "response1": st.session_state.resp1}
#
#     write_dict_to_csv(res, filepath=os.path.join(Data_dir, res_file))
#     res_df = pd.read_csv(os.path.join(Data_dir, res_file))
#     query_id = len(res_df)
#     save_dataframes(st.session_state.readout.df_activity_group,
#                     st.session_state.readout.df_measure_group,
#                     st.session_state.readout.df_measure,
#                     query_id=query_id,
#                     test_version=f"{st.session_state.client_code}_{st.session_state.model_group_id}",
#                     output_folder=Data_dir)

