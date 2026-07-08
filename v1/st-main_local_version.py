import os
import csv
import streamlit as st
import altair as alt
import json
from types import SimpleNamespace
import pandas as pd
import re
from streamlit_extras.stoggle import stoggle
import time

# from src.model.filter_generator import generate_ner_filter, generate_ner_filter_multi_turn, generate_ner_intention_multi_turn
from src.model.readout import stream_response, process_data, response_generate, response_generate_chart
from src.model.common import PromptTemplates
from src.data.data_interface import ProcessIndicator
from src.utils import get_available_client_and_model_group
import src.model.readout_utils as readout_utils
import st_utils as utils
from src.model.readout_utils import _recent_roi_spend,_trend_roi_spend, _trend_roi_other
import viz_utils
from streamlit_extras.stylable_container import stylable_container

Data_dir = "data/ask-genome-data/multi_turn/2604"
Data_dir = "/home/lzhao/Documents/ask-genome-data/"
readout_data_json_file = os.path.join(
        Data_dir,
        # f"stats_test_readout_data.json")
# f"level_test.jsonl")
# f"overall_notrend_roi_ner.jsonl")
# f"test_0812_viz_test.jsonl")
# "specific_q.jsonl")
# "hilsp_planner_readout.jsonl")
# "refresh_data.json")
# "Indeed_NER_results_processed.jsonl")
"hilsp_viz_ner_results.jsonl")
# "linkedin_queries_demo_ner_output.jsonl")
# "sample_queries_demo_ner_output.jsonl")

st.set_page_config(
    page_title='Ask Genome',
    layout='wide',
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

div[data-testid="stVegaLiteChart"],
div[data-testid="stAltairChart"]{
    background-color:  #F8FAFC;
    border-radius: 14px;
    padding: 16px; 
    margin-top: 12px; 
    border: 1px solid #E2E8F0;
    box-shadow:
        0 1px 3px rgba(0, 0, 0, 0.08),
        0 4px 12px rgba(0, 0, 0, 0.05);
}

div[data-testid="stExpander"]>details{
  background:#fff;border:1px solid #E5E7EB;border-radius:12px;
  padding:8px 12px;box-shadow:0 1px 3px rgba(0,0,0,.04);
}

div.stButton > button,
div.stDownloadButton > button{
  background: #F8FAFC;color:#0F172A;border:1px solid #E2E8F0;
  border-radius:10px;padding:8px 14px;font-weight:600;
  box-shadow:0 1px 3px rgba(0,0,0,.08);
  transition:filter .15s ease,transform .08s ease,box-shadow .2s ease;
}
div.stButton > button:hover,
div.stDownloadButton > button:hover{
  filter:brightness(1.05);transform:translateY(-1px);
  box-shadow:0 4px 10px rgba(37,99,235,.25);
}
div.stButton > button:disabled,
div.stDownloadButton > button:disabled{
  opacity:.6;cursor:not-allowed;transform:none;box-shadow:none;
}

.sec-title{
  font-size:1.05rem; font-weight:700; color:#0F172A;
  margin:6px 0 8px; padding-left:10px; border-left:4px solid #2563EB;
}

</style>
""", unsafe_allow_html=True)

left, center, right = st.columns([1, 2, 1])

# Init session states
for key in ["thinking_content", "main_content", "query", "readout",
            "selected_intention", "processed_data", "response_chunks", "intention_confirmed", "intention_required_confirmation", "multitask_pred", "spacy_filter", "process_indicator", "time_confirmed",
            "original_period", "original_time", "resp1", "resp2"]:
    if key not in st.session_state:
        if key in ["thinking_content", "main_content","original_period","original_time"]:
            st.session_state[key] = ""
        elif key in ["intention_confirmed", "intention_required_confirmation", "time_confirmed"]:
            st.session_state[key] = False
        else:
            st.session_state[key] = None

# Side panel
with st.sidebar:
    commit = os.getenv("CI_COMMIT")
    st.markdown(f"Commit: `{commit}`")

    # client_model_group_options = get_available_client_and_model_group()
    # selected = st.selectbox("Client Code", client_model_group_options.keys())
    # client_code, model_group_id = client_model_group_options[selected]
    client_code = '_RETDEMO'
    model_group_id = 111
    os.environ["CLIENT_CODE"] = client_code
    os.environ["MODEL_GROUP_ID"] = str(model_group_id)
    os.environ["model_vendor"] = "openai"

    st.markdown(f"Client Code: `{client_code}`")
    st.markdown(f"Model Group ID: `{model_group_id}`")

# ---------------------------------------------------------------------------
# Reset on page switch or client/model change
# ---------------------------------------------------------------------------

utils._detect_and_reset_on_context_change(
    client_code=client_code,
    model_group_id=model_group_id,
    page_id=__file__,
)

# Main panel
with center:
    # Reset session button
    # if st.button("Start New Question"):
    #     for key in ["thinking_content", "main_content", "query", "readout",
    #                 "selected_intention", "processed_data", "response_chunks", "intention_confirmed"]:
    #         if key in st.session_state:
    #             del st.session_state[key]
    #     st.stop()

    query = st.chat_input()

    if query:
        st.session_state.query = query
        st.session_state.readout = None
        # st.session_state.multitask_pred = None
        # st.session_state.spacy_filter = None
        # st.session_state.process_indicator = None
        st.session_state.processed_data = None
        st.session_state.response_chunks = None
        st.session_state.thinking_content = ""
        st.session_state.main_content = ""
        st.session_state.intention_confirmed = False
        st.session_state.intention_required_confirmation = False
        st.session_state.resp1 = None
        st.session_state.resp2 = None

    if st.session_state.get("query"):
        st.chat_message("user", avatar=":material/person:").write(st.session_state.query)
        if st.session_state.get("intention_confirmed") and st.session_state.get("intention_required_confirmation"):
            st.markdown(f"**Confirmed Intention:** `{st.session_state.selected_intention}`")

    if st.session_state.query and st.session_state.readout is None:
        # st.session_state.thinking_content = ""
        # st.session_state.main_content = ""
        # st.session_state.processed_data = None
        # st.session_state.response_chunks = None
        # st.session_state.intention_confirmed = False

        st.chat_message("assistant", avatar="./assets/ap-logo-margin.png").write("Thinking...")

        # process_indicator = ProcessIndicator.from_local(client_code, model_group_id)
        # readout_data = generate_ner_filter(client_code, model_group_id, st.session_state.query, process_indicator)
        # multitask_pred, spacy_filter = generate_ner_intention_multi_turn(client_code, model_group_id, st.session_state.query, process_indicator)
        with open(readout_data_json_file, "r") as f:
            for line in f:
                js_data = json.loads(line.strip())
                if query == js_data['query']:
                    data_instance = SimpleNamespace(**js_data)
                    data_instance.df_activity_group = pd.DataFrame.from_dict(js_data['df_activity_group'])
                    # if not data_instance.df_activity_group.empty:
                    #     data_instance.df_activity_group.loc[data_instance.df_activity_group['metric'] == 'margin roi', 'metric'] = 'roi'
                    data_instance.df_measure_group = pd.DataFrame.from_dict(js_data['df_measure_group'])
                    # if not data_instance.df_measure_group.empty:
                    #     data_instance.df_measure_group.loc[
                    #     data_instance.df_measure_group['metric'] == 'margin roi', 'metric'] = 'roi'
                    data_instance.df_measure = pd.DataFrame.from_dict(js_data['df_measure'])
                    # data_instance.df_measure.loc[
                    #     data_instance.df_measure['metric'] == 'margin roi', 'metric'] = 'roi'
                    # data_instance.ner_filters['metric'] = ['roi' if m == 'margin roi' else m for m in data_instance.ner_filters['metric']]
                    # data_instance.ner_filters['main_metric'] = ['roi' if m == 'margin roi' else m for m in
                    #                                        data_instance.ner_filters['main_metric']]
                    # data_instance.ner_filters = data_instance.ner_filter
                    readout_data = data_instance
                    break
        st.session_state.readout = readout_data
        if not st.session_state.original_period and not st.session_state.original_time:
            st.session_state.original_period = st.session_state.readout.ner_filters['period_type'][0]
            st.session_state.original_time = st.session_state.readout.ner_filters['time']
        # st.session_state.multitask_pred = multitask_pred
        # st.session_state.spacy_filter = spacy_filter
        # st.session_state.process_indicator = process_indicator

    # Intention confirmation
    # if st.session_state.readout is not None and not st.session_state.intention_confirmed:
    #     ori_intention = st.session_state.readout.ner_filters['intention'][0]
    #     probs = st.session_state.readout.ner_results["probabilities"]["intention"]
    #     ori_intention_prob = probs.get(ori_intention, 1.0)
    #
    #     if ori_intention_prob < 0.8:
    #         st.session_state.intention_required_confirmation = True
    #         with st.form("select_intention"):
    #             st.pills(
    #                 "Please confirm if the following reflects the intention of your question.",
    #                 options=list(probs.keys()),
    #                 default=ori_intention,
    #                 key="intention_selected"
    #             )
    #             submitted = st.form_submit_button("Confirm intention")
    #             if submitted:
    #                 selected_intention = st.session_state["intention_selected"]
    #                 # st.session_state.multitask_pred['intention']=[selected_intention]
    #                 st.session_state.readout.ner_filters['intention'][0] = selected_intention
    #                 # st.session_state.readout = generate_ner_filter_multi_turn(client_code, model_group_id, st.session_state.query, st.session_state.process_indicator, st.session_state.multitask_pred, st.session_state.spacy_filter)
    #                 st.session_state.selected_intention = selected_intention
    #                 st.session_state.intention_confirmed = True
    #                 st.session_state.processed_data = None
    #                 st.session_state.response_chunks = None
    #                 st.stop()
    #             else:
    #                 st.stop()
    #     else:
    #         st.session_state.selected_intention = ori_intention
    #         st.session_state.intention_confirmed = True
    #         st.session_state.intention_required_confirmation = False

    # Process data once intention is confirmed
    # if st.session_state.readout is not None and st.session_state.intention_confirmed and st.session_state.processed_data is None:
    if (st.session_state.readout is not None and st.session_state.processed_data is None):
        # if st.session_state.halo_confirmed and st.session_state.selected_halo_config.get('selected_from'):
        #     st.markdown(f"Current halo effect is from `{st.session_state.selected_halo_config.get('selected_from')}` to `{st.session_state.selected_halo_config.get('selected_to')}`")


        context_str, data, benchmark_data, benchmark_str, planner_data, spend_share_principle, pretext, pivot_biz_table, pretext_table, pretext_table_trend, match, principle_pretext, overall_view, _ = process_data(
            client_code, model_group_id, st.session_state.readout,
        )
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
            "overall_view": overall_view
        }

    # Unpack processed data
    if st.session_state.processed_data is not None:
        context_str = st.session_state.processed_data["context_str"]
        data = st.session_state.processed_data["data"]
        benchmark_data = st.session_state.processed_data["benchmark_data"]
        benchmark_str = st.session_state.processed_data["benchmark_str"]
        planner_data = st.session_state.processed_data["planner_data"]
        spend_share_principle = st.session_state.processed_data["spend_share_principle"]
        pretext = st.session_state.processed_data["pretext"]
        pivot_biz_table = st.session_state.processed_data["pivot_biz_table"]
        pretext_table = st.session_state.processed_data["pretext_table"]
        pretext_table_trend = st.session_state.processed_data["pretext_table_trend"]
        match = st.session_state.processed_data["match"]
        principle_pretext = st.session_state.processed_data["principle_pretext"]
        overall_view = st.session_state.processed_data["overall_view"]
    else:
        st.stop()

    # # Display confirmed query and intention
    # if st.session_state.query and st.session_state.intention_confirmed and not st.session_state.response_chunks:
    #     st.chat_message("user", avatar=":material/person:").write(st.session_state.query)
    #     st.markdown(f"**Confirmed intention:** `{st.session_state.selected_intention}`")

    # Thinking and Response Rendering
    with st.chat_message("assistant", avatar="./assets/ap-logo-margin.png"):
        with st.expander("Thinking...", expanded=False):
            placeholder_thinking = st.empty()
        placeholder_main = st.empty()

        if st.session_state.response_chunks is None:
            found_think = False
            if not (data.empty and pivot_biz_table.empty and planner_data.empty):
                for chunk in stream_response(st.session_state.query, context_str):
                    if not found_think:
                        st.session_state.thinking_content += chunk
                        if "</think>" in st.session_state.thinking_content:
                            found_think = True
                            before_think, after_think = st.session_state.thinking_content.split("</think>", 1)
                            st.session_state.thinking_content = before_think
                            st.session_state.main_content += after_think
                    else:
                        st.session_state.main_content += chunk

                    placeholder_thinking.write(st.session_state.thinking_content)
                    placeholder_main.write(st.session_state.main_content)
                if planner_data.empty:
                    st.session_state.response_chunks = f"{pretext} \n\n{st.session_state.main_content} \n\n{principle_pretext} \n\n{benchmark_str}"
                else:
                    st.session_state.response_chunks = f"{st.session_state.main_content} \n\n{pretext} \n\n{principle_pretext}  \n\n{benchmark_str}"
            else:
                if pretext:
                    st.session_state.response_chunks = pretext
                else:
                    rejection_message = st.session_state.readout.ner_filters.get("rejection_message")
                    st.session_state.response_chunks = rejection_message if rejection_message \
                        else PromptTemplates().rejection_response

        with placeholder_main:
            st.write(st.session_state.response_chunks)

intention, is_trend, is_specific, if_agg_exists, driver_col, metric_ls, main_metric_ls, data_type, trend_check = viz_utils.viz_config(st.session_state.readout, match, pivot_biz_table)
prompt_inst = readout_utils.load_prompt_instruction(client_code, model_group_id)
row_height = 35
min_height = 100
max_height = 600

if overall_view is not None and not overall_view.empty:
    st.markdown("### Overall View")
    rows = min(len(overall_view), 20)
    height = max(min_height, min(max_height, rows * row_height))
    with st.expander("Show data table"):
        st.dataframe(overall_view.reset_index(drop=True), use_container_width=True, height=height)

# Aggregation View
if pivot_biz_table is not None and not pivot_biz_table.empty:
    st.markdown("### Aggregation View")
    rows = min(len(pivot_biz_table), 20)
    height = max(min_height, min(max_height, rows * row_height))
    with st.expander("Show data table"):
        st.dataframe(pivot_biz_table.reset_index(drop=True), use_container_width=True, height=height)
    agg_data, outliers_agg = viz_utils.data_init(pivot_biz_table, metric_ls, driver_col=driver_col)

# Detail View
if data is not None and not data.empty:
    st.markdown("### Detail View")
    rows = min(len(data), 20)
    height = max(min_height, min(max_height, rows * row_height))
    with st.expander("Show data table"):
        st.dataframe(data.reset_index(drop=True), use_container_width=True, height=height)
    df = data.copy()
    if len(df)>1:
        if if_agg_exists and data_type == 'br':
            col_toggle, col_metric, col_level = st.columns([1, 2, 2])
            with col_toggle:
                # if if_agg_exists and data_type == 'br':
                st.markdown("""
                    <style>
                    .inline-toggle {
                        display: flex;
                        align-items: center;
                        gap: 6px;               /* spacing between items */
                        font-size: 14px;
                    }
                    .inline-toggle span {
                        margin: 0;
                        padding: 0;
                    }
                    </style>
                    """, unsafe_allow_html=True)
                # --- Inline Toggle Row ---
                st.markdown('<div class="inline-toggle">', unsafe_allow_html=True)
                # Build inline elements using st.columns with zero padding
                colA, colB, colC = st.columns([1, 0.5, 1])
                with colA:
                    st.markdown('<span>Detail</span>', unsafe_allow_html=True)
                with colB:
                    chart_mode = stoggle(
                        label="",
                        key="br_detail_level",
                        default_value=False,  # False = Detail
                        label_after=False,
                        inactive_color="#E5E7EB",
                        active_color="#2563EB",
                        track_color="#93C5FD",
                    )
                with colC:
                    st.markdown('<span>Aggregation</span>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                br_detail_level = "Aggregation View" if chart_mode else "Detail View"
        else:
            col_metric, col_level = st.columns([1,1])
            br_detail_level = 'Detail View'

        if br_detail_level == 'Aggregation View':
            df, outliers = viz_utils.data_init(pivot_biz_table, metric_ls, driver_col=driver_col)
        else:
            df, outliers = viz_utils.data_init(df, metric_ls, driver_col=driver_col)

        with col_metric:
            selected_metrics, metric_cols, metric_mappings, metric_labels, sel_m, m_cols = utils.get_selected_metric(df, metric_ls, main_metric_ls)
        # df, selected_levels = utils.get_selected_multi_levels(df, driver_col=driver_col)
        with col_level:
            df, selected_levels = utils.get_selected_multi_levels_new(df, driver_col=driver_col)

        df_plot, df_plot_cont_share, sorted_times = viz_utils.data_prepare(df, metric_cols, m_cols, driver_col)
        # KPI cards
        if pretext_table is not None and not pretext_table.empty and trend_check != 'yes' and intention != 'source of change':
            overall_m = metric_labels[0].title().title().replace("Roi", "ROI")
            with utils._section(f"Overall {overall_m}"):
                utils.most_recent_kpi_card_new(client_code, model_group_id, intention, pretext_table, selected_levels, sorted_times[-1])
        # overall trend
        if pretext_table_trend is not None and not pretext_table_trend.empty and trend_check == 'yes':
            overall_m = metric_labels[0].title().title().replace("Roi", "ROI")
            with utils._section(f"Overall {overall_m}"):
                col_kpi, col_trend = st.columns([1.5, 5], gap="small")
                with col_kpi:
                    utils.most_recent_kpi_card_new(client_code, model_group_id, intention, pretext_table, selected_levels, sorted_times[-1], 'column')
                with col_trend:
                    utils.get_pretext_trend_new(client_code,model_group_id, intention, pretext_table_trend, metric_labels, metric_cols, sorted_times, selected_levels)
                # col_kpi, col_trend = st.columns([1, 5], gap="small")
                # with col_kpi:
                #     utils.most_recent_kpi_card_new(pretext_table, selected_levels, 'column')
                # with col_trend:
                #     utils.get_pretext_trend_new(pretext_table_trend, metric_labels, metric_cols, sorted_times, selected_levels)
        if 'insight' not in st.session_state:
            st.session_state.insight = {}
        if data_type == 'bi':
            if intention in ['margin roi', 'performance']:
                ag_mapping = viz_utils.get_ag_level_mapping(df, st.session_state.readout, match, is_specific)
                recent_d, trend_d, resp_cost_d, _ = viz_utils.prepare_roi_data(df, client_code, model_group_id,
                                                                               st.session_state.readout.ner_filters,
                                                                               selected_levels, outliers,
                                                                               trend_check,
                                                                               selected_metrics, ag_mapping,
                                                                               is_specific, sorted_times, table='detail')
                # multi_roi_data = viz_utils.prepare_multiple_roi_data(recent_d, df, outliers, sorted_times)
                multi_roi_data = None

                if if_agg_exists:

                    recent_d_agg, trend_d_agg, resp_cost_d_agg, _ = viz_utils.prepare_roi_data(agg_data,
                                                                                               client_code,
                                                                                               model_group_id,
                                                                                               st.session_state.readout.ner_filters,
                                                                                               selected_levels,
                                                                                               outliers_agg,
                                                                                               trend_check,
                                                                                               selected_metrics,
                                                                                               ag_mapping,
                                                                                               is_specific, sorted_times,
                                                                                               table='agg')
                    # multi_roi_data_agg = viz_utils.prepare_multiple_roi_data(agg_data, df, outliers_agg, sorted_times)
                    multi_roi_data_agg = None
                else:
                    recent_d_agg, trend_d_agg, resp_cost_d_agg, multi_roi_data_agg = None, None, None, None

                roi_tabs = ["ROI & Spend Snapshot",
                            "ROI & Spend Trend" if is_trend else "ROI, Response & Cost Trends"]
                _roi_tabs = st.tabs(roi_tabs)
                tab1_chart_configs = {}
                tab1_insights = {}
                with _roi_tabs[0]:

                    if multi_roi_data:
                        left, right = st.columns([5, 1.5])
                        with left:
                            with utils._section("ROI and Spend"):
                                tab1_chart_configs, tab1_insights = utils.get_roi_recent_plot_new(recent_d, recent_d_agg, sorted_times[-1], is_specific,
                                                              ag_mapping, prompt_inst, llm_vendor=os.getenv("model_vendor"))

                        with right:
                            with utils._section("Short Term, Long Term and Total ROI"):
                                utils.get_multiple_roi_plot_new(multi_roi_data, multi_roi_data_agg, sorted_times[::-1],
                                                                is_specific, prompt_inst)
                    else:
                        with utils._section("ROI and Spend"):
                            tab1_chart_configs, tab1_insights = utils.get_roi_recent_plot_new(recent_d, recent_d_agg, sorted_times[-1], is_specific,
                                                          ag_mapping, prompt_inst, llm_vendor=os.getenv("model_vendor"))

                # left, right1, right2 = st.columns([2.5, 2.5, 1])
                tab2_chart_configs = {}
                tab2_insights = {}
                if trend_d or resp_cost_d:
                    with _roi_tabs[1]:
                        if trend_check == 'yes':
                            # with section("ROI & Spend Trend"):
                            # utils.get_roi_trend_plot_new(trend_d, trend_d_agg, sorted_times, is_specific, ag_mapping)
                            tab2_chart_configs, tab2_insights = utils.get_roi_trend_plot_new_multi_roi(trend_d, trend_d_agg, sorted_times, is_specific,
                                                                   ag_mapping, multi_roi_data, multi_roi_data_agg,
                                                                   selected_metrics[0], prompt_inst, llm_vendor=os.getenv("model_vendor"))
                        else:
                            # with section("ROI, Response & Cost Trends"):
                            tab2_chart_configs, tab2_insights = utils.get_roi_response_cost_plot_new(resp_cost_d, resp_cost_d_agg, sorted_times,
                                                                 is_specific,
                                                                 ag_mapping, prompt_inst, llm_vendor=os.getenv("model_vendor"))

                chart_configs = tab1_chart_configs | tab2_chart_configs
                charts_block = utils.get_chart_set_inst(chart_configs)
                if st.session_state.resp1 is None:
                    with st.spinner("Your insights are on the way..."):
                        insight_indiv_query = (
                            f"You are given several datasets about ROI. "
                            f"Provide structured insights from datasets and then answer the user's question: "
                            f"{st.session_state.query} "
                            f"Write your response in markdown format."
                        )
                        resp1 = response_generate_chart(query=insight_indiv_query, readout=charts_block, llm_vendor=os.getenv("model_vendor"))
                        st.session_state.resp1 = resp1
                utils.render_chart_insight_box("ROI Insights", st.session_state.resp1, height_px=500)


                # method 2: summarize insights for each chart
                chart_insights = tab1_insights | tab2_insights
                insights_block = utils.get_insights_set_inst(chart_insights)
                if st.session_state.resp2 is None:
                    with st.spinner("Your insights are on the way..."):
                        insight_summary_query = (
                            f"You are given several insights about ROI. "
                            f"Provide a structured summary of the insights and then answer the user's question: "
                            f"{st.session_state.query} "
                            f"Write your response in markdown format."
                        )
                        resp2 = response_generate_chart(query=insight_summary_query, readout=insights_block, llm_vendor=os.getenv("model_vendor"))
                        st.session_state.resp2 = resp2
                utils.render_chart_insight_box("ROI Insights Summary", st.session_state.resp2, height_px=500)

            elif intention == 'spending':
                tab_labels = ["Spend Snapshot", "Spend Trend"] if is_trend else ["Spend Snapshot"]
                if is_trend:
                    tab_snap, tab_trend = st.tabs(tab_labels)
                    with tab_snap:
                        utils.spend_snapshot(df, selected_levels, driver_col, metric_cols, sorted_times[::-1], st.session_state.readout, match, is_specific)
                    with tab_trend:
                        utils.spend_share_trend_by_level(data, df, selected_levels, metric_cols, st.session_state.readout, match, is_specific)
                else:
                    utils.spend_snapshot(df, selected_levels, driver_col, metric_cols, sorted_times[::-1], st.session_state.readout, match, is_specific)
                    utils.pct_change_heatmap(df, selected_levels, driver_col, m_cols, sorted_times[::-1], st.session_state.readout, match, is_specific, mode='change')
            else:
                tab_labels = [f"{intention.title()} Snapshot", f"{intention.title()} Trend"] if is_trend else [f"{intention.title()} Snapshot"]
                if is_trend:
                    tab_snap, tab_trend = st.tabs(tab_labels)
                    with tab_snap:
                        utils.other_bi_snapshot(df, selected_levels, outliers, driver_col, metric_cols, sorted_times[::-1],
                                             st.session_state.readout, match, is_specific)
                    with tab_trend:
                        utils.other_bi_trend_by_level(data, df, selected_levels, metric_cols,
                                                         st.session_state.readout, match, is_specific)

        else:
            if 'source of change' in selected_metrics:
                total_viz_soc = viz_utils.filter_bd_data(client_code, model_group_id, selected_levels,
                                                         ['sales', 'sourceofchange'])
                total_act_sales = viz_utils.filter_bd_data(client_code, model_group_id, selected_levels, ['sales'],
                                                           total='total')
                group_mapping = viz_utils.get_group_level_mapping(st.session_state.readout, match, is_specific)
                utils.get_plot_for_sourceofchange(total_viz_soc, total_act_sales, df_plot, outliers, metric_cols,
                                                  sorted_times[::-1], driver_col, br_detail_level, group_mapping)
            if 'contribution' in selected_metrics:
                tab_labels = ["Contribution Snapshot", "Contribution Trend" if is_trend else "Contribution Change"]
                tab_snap, tab_trend = st.tabs(tab_labels)
                with tab_snap:
                    tab1_chart_configs, tab1_insights = utils.get_contribution_pie(client_code, model_group_id, df_plot, df_plot_cont_share, sorted_times[::-1], selected_levels, metric_cols, m_cols, outliers, is_specific, st.session_state.readout, match, prompt_inst, driver_col = driver_col, llm_vendor= os.getenv("model_vendor"))
                with tab_trend:
                    # is_trend = True
                    kwargs = {'mode': 'change'} if not is_trend else {}
                    tab2_chart_configs, tab2_insights = utils._contribution_trend(br_detail_level, pivot_biz_table, data, selected_levels, is_specific, st.session_state.readout, match, prompt_inst, llm_vendor= os.getenv("model_vendor"),**kwargs)

                chart_configs = tab1_chart_configs | tab2_chart_configs
                charts_block = utils.get_chart_set_inst(chart_configs)
                if st.session_state.resp1 is None:
                    with st.spinner("Your insights are on the way..."):
                        resp = response_generate_chart(
                            query=(
                                f"You are given several datasets about Contribution. "
                                f"Provide structured insights from datasets and then answer the user's question: "
                                f"{st.session_state.query} "
                                f"Write your response in markdown format."
                            ),
                            readout=charts_block,
                            llm_vendor=os.getenv("model_vendor"),
                        )
                        st.session_state.resp1 = resp
                utils.render_chart_insight_box("Contribution Insights - combining prompts", st.session_state.resp1, height_px=500)

                chart_insights = tab1_insights | tab2_insights
                insights_block = utils.get_insights_set_inst(chart_insights)
                if st.session_state.resp2 is None:
                    with st.spinner("Your insights are on the way..."):
                        resp = response_generate_chart(
                            query=(
                                f"You are given several insights about Contribution. "
                                f"Provide a structured summary of the insights and then answer the user's question: "
                                f"{st.session_state.query} "
                                f"Write your response in markdown format."
                            ),
                            readout=insights_block,
                            llm_vendor=os.getenv("model_vendor"),
                        )
                        st.session_state.resp2 = resp
                utils.render_chart_insight_box(
                    "Contribution Insights - summarizing each chart insights",
                    st.session_state.resp2, height_px=500,
                )


# Benchmark Table
if benchmark_data is not None and not benchmark_data.empty:
    st.markdown("## Benchmark Data")
    st.table(benchmark_data)

# Planner Table
if planner_data is not None and not planner_data.empty:
    st.markdown("## Planner Data")

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
    if 'insight' not in st.session_state:
        st.session_state.insight = {}
    chart_configs, chart_configs_lvl = {}, {}
    core_dim = st.session_state.readout.ner_filters.get('core_dimension', {})
    sub_df = st.session_state.readout.df_measure
    levels = viz_utils.get_available_level(client_code, model_group_id, sub_df, core_dim)

    spend_share_data, _ = viz_utils.get_spend_share_principle_data(spend_share_principle)
    if not core_dim:
        utils.planner_overall_metric(client_code, model_group_id)
        spend_all_inst = utils.planner_spend_change_all_instruct(client_code, model_group_id, sub_df, core_dim)
        chart_configs['spend_change_all'] = {"query": "You are given a dataset containing spend % change by driver.",
                                             "instruction": spend_all_inst}
        bin_pct_inst = utils.planner_pct_bar_instruct(client_code, model_group_id, sub_df, levels, core_dim,
                                                   mode='pct_change')
        chart_configs["insights_pct_pct_change"] = {"query": "You are given a dataset grouped by Spend Change Group.",
                                                    "instruction": bin_pct_inst}
        bin_amt_inst = utils.planner_pct_bar_instruct(client_code, model_group_id, sub_df, levels, core_dim,
                                                   mode='spend_amount')
        chart_configs["insights_pct_spend_amount"] = {"query": "You are given a dataset grouped by Spend Change Group.",
                                                      "instruction": bin_amt_inst}
        with st.spinner("Your visuals and insights are one the way..."):
            start = time.time()
            insights = utils.generate_chart_insights_parallel_cached(chart_configs)
            elapsed = time.time() - start
        st.write(f"Insight generation time: {elapsed:.2f} seconds")
        left, right = st.columns([1, 2.5])
        with left:
            utils.get_spend_share_principle_plot(spend_share_data)
        with right:
            df_all, bar_all = utils.planner_spend_change_all(client_code, model_group_id, sub_df, core_dim)
            with st.expander("Show data table"):
                st.dataframe(df_all, use_container_width=True)
            st.altair_chart(bar_all, use_container_width=True)
            st.session_state.insight["spend_change_all"] = insights["spend_change_all"]
            saved = st.session_state.insight.get("spend_change_all")
            if saved:
                utils.render_chart_insight_box("Spend Incremental Insight", saved)
                # st.markdown("##### Spend Incremental Insight")
                # st.markdown(saved)

        left, right = st.columns([1, 1])
        with left:
            df_l, chart_l = utils.planner_pct_bar(client_code, model_group_id, sub_df, levels, core_dim,
                                                     mode='pct_change')
            with st.expander("Show data table"):
                st.dataframe(df_l, use_container_width=True)
            st.altair_chart(chart_l, use_container_width=True)
            st.session_state.insight["insights_pct_pct_change"] = insights["insights_pct_pct_change"]
            saved = st.session_state.insight.get("insights_pct_pct_change")
            if saved:
                utils.render_chart_insight_box("Spend Change Group Insight", saved)
                # st.markdown("##### Spend Change Group Insight")
                # st.markdown(saved)
        with right:
            df_r, chart_r = utils.planner_pct_bar(client_code, model_group_id, sub_df, levels, core_dim,
                                                     mode='spend_amount')
            with st.expander("Show data table"):
                st.dataframe(df_r, use_container_width=True)
            st.altair_chart(chart_r, use_container_width=True)
            st.session_state.insight["insights_pct_spend_amount"] = insights["insights_pct_spend_amount"]
            saved = st.session_state.insight.get("insights_pct_spend_amount")
            if saved:
                utils.render_chart_insight_box("Spend Amount Group Insight", saved)
                # st.markdown("##### Spend Amount Group Insight")
                # st.markdown(saved)

        lvl = st.radio("Select Spend Level", levels, horizontal=True)
        st.subheader(f"Spend Level: {lvl}")

        scatter_inst = utils.spend_kpi_scatter_insight(client_code, model_group_id, sub_df, [lvl], core_dim)
        chart_configs_lvl[f"spend_kpi_fair_share_{lvl}"] = {
            "query": "You are given a dataset containing spend and KPIs % change by driver.",
            "instruction": scatter_inst}
        diminish_inst = utils.plot_diminishing_return_instruct(client_code, model_group_id, sub_df, [lvl],
                                                                           core_dim)
        chart_configs_lvl[f"diminish_return_{lvl}"] = {
            "query": "You are given a dataset containing historical and forecast spend by driver.",
            "instruction": diminish_inst}
        with st.spinner("Your visuals and insights are one the way..."):
            start = time.time()
            insights_lvl = utils.generate_chart_insights_parallel_cached(chart_configs_lvl)
            elapsed = time.time() - start
        st.write(f"Insight generation time: {elapsed:.2f} seconds")
        # Each function is called with current level filtered
        left, right = st.columns([1, 2])
        with left:
            utils.get_planner_spend_level_plot(client_code, model_group_id, sub_df, [lvl], core_dim)
        with right:
            utils.spend_kpi_scatter(client_code, model_group_id, sub_df, [lvl], core_dim)
            st.session_state.insight[f"spend_kpi_fair_share_{lvl}"] = insights_lvl[f"spend_kpi_fair_share_{lvl}"]
            saved = st.session_state.insight.get(f"spend_kpi_fair_share_{lvl}")
            if saved:
                utils.render_chart_insight_box("Driver Performance Insight", saved)
                # st.markdown("##### Driver Performance Insight")
                # st.markdown(saved)
        diminish_df, diminish_chart = utils.plot_diminishing_return_by_spend_level(client_code, model_group_id,
                                                                                      sub_df, [lvl], core_dim)
        with st.expander("Show data table"):
            st.dataframe(diminish_df, use_container_width=True)
        st.altair_chart(diminish_chart, use_container_width=True)
        st.session_state.insight[f"diminish_return_{lvl}"] = insights_lvl[f"diminish_return_{lvl}"]
        saved = st.session_state.insight.get(f"diminish_return_{lvl}")
        if saved:
            utils.render_chart_insight_box("Diminishing Return Insight", saved)
            # st.markdown("##### Diminishing Return Insight")
            # st.markdown(saved)
    else:
        scatter_inst = utils.spend_kpi_scatter_insight(client_code, model_group_id, sub_df, levels, core_dim)
        chart_configs_lvl["spend_kpi_fair_share_levels"] = {
            "query": "You are given a dataset containing spend and KPIs % change by driver.",
            "instruction": scatter_inst}
        diminish_inst = utils.plot_diminishing_return_instruct(client_code, model_group_id, sub_df, levels,
                                                                           core_dim)
        chart_configs_lvl["diminish_return_levels"] = {
            "query": "You are given a dataset containing historical and forecast spend by driver.",
            "instruction": diminish_inst}
        with st.spinner("Your visuals and insights are one the way..."):
            start = time.time()
            insights_lvl = utils.generate_chart_insights_parallel_cached(chart_configs_lvl)
            elapsed = time.time() - start
        st.write(f"Insight generation time: {elapsed:.2f} seconds")
        utils.get_spend_share_principle_plot(spend_share_data)
        cols = st.columns([1, 1, 1])
        for i, lvl in enumerate(levels):
            with cols[i % 3]:
                utils.get_planner_spend_level_plot(client_code, model_group_id, sub_df, [lvl], core_dim)
        utils.spend_kpi_scatter(client_code, model_group_id, sub_df, levels, core_dim)
        st.session_state.insight["spend_kpi_fair_share_levels"] = insights_lvl["spend_kpi_fair_share_levels"]
        saved = st.session_state.insight.get("spend_kpi_fair_share_levels")
        if saved:
            utils.render_chart_insight_box("Driver Performance Insight", saved)
            # st.markdown("##### Driver Performance Insight")
            # st.markdown(saved)
        diminish_df, diminish_chart = utils.plot_diminishing_return_by_spend_level(client_code, model_group_id,
                                                                                      sub_df, levels, core_dim)
        with st.expander("Show data table"):
            st.dataframe(diminish_df, use_container_width=True)
        st.altair_chart(diminish_chart, use_container_width=True)
        st.session_state.insight["diminish_return_levels"] = insights_lvl["diminish_return_levels"]
        saved = st.session_state.insight.get(f"diminish_return_levels")
        if saved:
            utils.render_chart_insight_box("Diminishing Return Insight", saved)
            # st.markdown("##### Diminishing Return Insight")
            # st.markdown(saved)

# save generated response
if st.session_state.resp1 and st.session_state.resp2:
    # save response to csv
    def write_dict_to_csv(data: dict, filepath: str) -> None:
        file_exists = os.path.exists(filepath)

        with open(filepath, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=data.keys())

            if not file_exists:
                writer.writeheader()

            writer.writerow(data)


    res = {"query": st.session_state.query, "intention": intention, "trend": trend_check,
           "response1": st.session_state.resp1, "response2": st.session_state.resp2}
    write_dict_to_csv(res, filepath=os.path.join(Data_dir, "DEMO_query_list_response_generation_test_v2.csv"))
# if st.session_state.readout.ner_filters['trend'] == 'yes' and not (pivot_biz_table.empty and data.empty):
#
#     utils.time_period_clarif()

    #and not st.session_state.time_confirmed
