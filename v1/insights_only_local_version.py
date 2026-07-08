"""
v1/insights_only_local_version.py

Local-file twin of pages/insights_only.py.  Identical insights pipeline
(process_data -> chart configs -> response_generate_chart) but the NER /
langgraph front-end is removed: ReadoutData is loaded from a local jsonl
keyed by query, mirroring v1/st-main_local_version.py.

Readout jsonl layout (one record per line):
  {
    "query":            <str>,
    "ner_results":      <dict>,            # ner results
    "ner_filters":      <dict>,            # ner filter   (also accepts "ner_filter")
    "df_activity_group": <list[dict]>,     # df.to_dict(orient="records")
    "df_measure_group":  <list[dict]>,
    "df_measure":        <list[dict]>,
  }

File path is built from (DATA_DIR, client_code, model_group_id):
  {DATA_DIR}/{client_code}/{model_group_id}/{READOUT_JSONL_FILENAME}
"""

import os
import re
import csv
import json
import pandas as pd

import streamlit as st

from src.model.readout import process_data, response_generate_chart
from src.model.insight_report import DocumentRetriever
from src.model.roi_genome_slides import ROIGenomeDocumentRetriever
from src.data.data_interface import ProcessIndicator, ReadoutData
from src.utils import get_available_client_and_model_group
import src.model.readout_utils as readout_utils
import st_utils as utils
import viz_utils

from chart import (
    get_chart_set_inst,
    # BI build functions
    build_roi_snapshot_chart_configs,
    build_roi_trend_chart_configs,
    build_roi_response_cost_chart_configs,
    build_spend_snapshot_chart_configs,
    build_spend_trend_chart_configs,
    build_spend_heatmap_chart_configs,
    build_other_bi_snapshot_chart_configs,
    build_other_bi_trend_chart_configs,
    # BR build functions
    build_sourceofchange_chart_configs,
    build_contribution_snapshot_chart_configs,
    build_contribution_trend_chart_configs,
    # Planner data helpers (return data tuples, no st.* calls)
    spend_kpi_scatter_insight,
    planner_spend_change_all_instruct,
    planner_pct_bar_instruct,
    plot_diminishing_return_instruct,
)

# ---------------------------------------------------------------------------
# Local readout-data source
# ---------------------------------------------------------------------------

DATA_DIR = os.getenv("DATA_DIR", "/home/lzhao/Documents/ask-genome-data/")
os.environ["DATA_DIR"] = DATA_DIR
READOUT_JSONL_FILENAME = "readout_data.jsonl"


def _records_to_df(records) -> pd.DataFrame:
    return pd.DataFrame.from_records(records) if records else pd.DataFrame()


def load_readout_from_jsonl(data_dir: str, client_code: str, model_group_id, query: str):
    """Return a ReadoutData for the first jsonl record whose `query` matches, else None."""
    path = os.path.join(data_dir, client_code, str(model_group_id), READOUT_JSONL_FILENAME)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("query") != query:
                continue
            return ReadoutData(
                ner_filters=record.get("ner_filters", record.get("ner_filter", {})),
                ner_results=record.get("ner_results", record.get("ner_res", {})),
                df_activity_group=_records_to_df(record.get("df_activity_group")),
                df_measure_group=_records_to_df(record.get("df_measure_group")),
                df_measure=_records_to_df(record.get("df_measure")),
            )
    return None


# ---------------------------------------------------------------------------
# Local helpers — "All" iteration logic for metric × level combinations
# ---------------------------------------------------------------------------

def _select_metric_with_all(metric_ls, main_metric_ls, df):
    """Return the list of `selected_metrics` options to iterate over (no widgets).

    Each option is itself a list-of-strings (matching the shape build functions expect).

    - Contribution / spend combo  →  single option with the multi-metric list.
    - Cost-per-activity / activity / response  →  single option with [metric].
    - ROI / other BI  →  one option per metric present in df.
    """
    # Multi-metric combo (contribution / spend share) — no iteration
    if "contribution" in metric_ls or metric_ls[0] == "spend":
        defaults = metric_ls[: min(2, len(metric_ls))]
        return [defaults]

    # Single fixed metric — no iteration
    if metric_ls[0] in ["cost per activity", "activity", "response"]:
        return [[metric_ls[0]]]

    # ROI / other BI — iterate every metric present in the data
    main_metric_in_data = [m for m in main_metric_ls if any(m in col.lower() for col in df.columns)]
    if not main_metric_in_data:
        return []

    return [[m] for m in main_metric_in_data]


def _select_levels_with_all(df, driver_col):
    """Return the list of `selected_levels` dicts to iterate over (no widgets).

    - No level columns  →  `[{}]`.
    - Otherwise  →  one dict per level combo (iterates all).
    """
    tactics_idx = df.columns.tolist().index(driver_col) if driver_col in df.columns else None
    levels = df.columns.tolist()[:tactics_idx] if tactics_idx else []
    if "Marketing Focus" in levels:
        levels.remove("Marketing Focus")
    level_comb = df[levels].drop_duplicates().values.tolist() if levels else []

    if not levels:
        return [{}]
    return [dict(zip(levels, [str(v) for v in row])) for row in level_comb]


def _resolve_m_cols(metric_ls, df):
    """Compute (sel_m, m_cols) — these depend only on metric_ls, invariant across metric iterations."""
    if "contribution share" in metric_ls:
        sel_m = ["contribution share", "spend share"]
        m_cols = [
            col for col in df.columns
            if re.split(r"(Q[1-4]|H[1-2]|M(?:1[0-2]|[1-9])) \d{4}|\d{4}", col)[0].strip().lower() in sel_m
            and "% Change" not in col
        ]
    elif "share of spend" in metric_ls:
        sel_m = None
        m_cols = [col for col in df.columns if "Spend" in col and "Share" not in col and "% Change" in col]
    elif any(key in metric_ls for key in ["cost per activity", "activity", "response"]):
        sel_m = None
        m_cols = [
            col for col in df.columns
            if any(key in col for key in ["cost per activity", "activity", "response"]) and "% Change" in col
        ]
    else:
        sel_m, m_cols = None, None
    return sel_m, m_cols


def _combo_label(selected_metrics, level_dict):
    """Build a human-readable combo label used for namespacing chart_ids."""
    if isinstance(selected_metrics, list):
        metric_str = "+".join(selected_metrics)
    else:
        metric_str = str(selected_metrics)
    parts = [f"metric={metric_str}"]
    if level_dict:
        parts.extend(f"{k}={v}" for k, v in level_dict.items())
    else:
        parts.append("Overall")
    return " | ".join(parts)


def _build_configs_for_combo(
    *,
    data_type, intention, is_trend, is_specific, if_agg_exists, trend_check,
    df_combo, agg_data, outliers, outliers_agg,
    selected_metrics, selected_levels, metric_cols, m_cols,
    sorted_times, prompt_inst, br_detail_level,
    df_plot, df_plot_cont_share, data, pivot_biz_table,
    client_code, model_group_id, readout, match, driver_col,
    metric_ls,
):
    """Build chart_configs for ONE (metric, level) combo. Pure data — no st.* calls."""
    configs: dict = {}

    if data_type == "bi":
        if intention in ["margin roi", "performance"]:
            ag_mapping = viz_utils.get_ag_level_mapping(df_combo, readout, match, is_specific)
            recent_d, trend_d, resp_cost_d, _ = viz_utils.prepare_roi_data(
                df_combo, client_code, model_group_id,
                readout.ner_filters,
                selected_levels, outliers, trend_check,
                selected_metrics, ag_mapping, is_specific, sorted_times, table="detail",
            )
            if if_agg_exists:
                recent_d_agg, trend_d_agg, resp_cost_d_agg, _ = viz_utils.prepare_roi_data(
                    agg_data, client_code, model_group_id,
                    readout.ner_filters,
                    selected_levels, outliers_agg, trend_check,
                    selected_metrics, ag_mapping, is_specific, sorted_times, table="agg",
                )
            else:
                recent_d_agg = trend_d_agg = resp_cost_d_agg = None

            tab1 = build_roi_snapshot_chart_configs(
                recent_d, recent_d_agg, sorted_times[-1], is_specific,
                [], [], prompt_inst,
            )
            configs.update(tab1)

            if trend_d or resp_cost_d:
                if trend_check == "yes" and trend_d:
                    tab2 = build_roi_trend_chart_configs(
                        trend_d, trend_d_agg, sorted_times, is_specific,
                        add_long_term=False, prompt_inst=prompt_inst,
                        multi_roi_data=None, multi_roi_data_agg=None,
                    )
                elif resp_cost_d:
                    tab2 = build_roi_response_cost_chart_configs(
                        resp_cost_d, resp_cost_d_agg, sorted_times, is_specific,
                        ag_mapping, prompt_inst,
                    )
                else:
                    tab2 = {}
                configs.update(tab2)

        elif intention == "spending":
            configs.update(build_spend_snapshot_chart_configs(
                df_combo, selected_levels, driver_col, metric_cols,
                sorted_times[::-1], readout, match, is_specific,
            ))
            if is_trend:
                configs.update(build_spend_trend_chart_configs(
                    data, df_combo, selected_levels, metric_cols,
                    readout, match, is_specific,
                ))
            else:
                configs.update(build_spend_heatmap_chart_configs(
                    df_combo, selected_levels, driver_col, m_cols,
                    sorted_times[::-1], readout, match, is_specific,
                    mode="change",
                ))

        else:
            configs.update(build_other_bi_snapshot_chart_configs(
                df_combo, selected_levels, outliers, driver_col, metric_cols,
                sorted_times[::-1], readout, match, is_specific,
            ))
            if is_trend:
                configs.update(build_other_bi_trend_chart_configs(
                    data, df_combo, selected_levels, metric_cols,
                    readout, match, is_specific,
                ))

    else:
        if "source of change" in selected_metrics:
            total_viz_soc = viz_utils.filter_bd_data(
                client_code, model_group_id, selected_levels, ["sales", "sourceofchange"]
            )
            total_act_sales = viz_utils.filter_bd_data(
                client_code, model_group_id, selected_levels, ["sales"], total="total"
            )
            group_mapping = viz_utils.get_group_level_mapping(readout, match, is_specific)
            soc_result = build_sourceofchange_chart_configs(
                total_viz_soc, total_act_sales, df_plot, outliers,
                metric_cols, sorted_times[::-1], driver_col,
                br_detail_level, group_mapping,
            )
            soc_configs = soc_result[0] if isinstance(soc_result, tuple) else soc_result
            configs.update(soc_configs)

        if "contribution" in selected_metrics:
            select_time = viz_utils.get_time_align_dict(sorted_times[::-1])
            keys_list = list(select_time.keys())
            sel_time_left = select_time[keys_list[0]]
            sel_time_right = select_time[keys_list[min(1, len(keys_list) - 1)]]

            configs.update(build_contribution_snapshot_chart_configs(
                client_code, model_group_id, df_plot, df_plot_cont_share,
                sorted_times[::-1], selected_levels, metric_cols, m_cols,
                outliers, is_specific, readout, match,
                sel_time_left, sel_time_right, prompt_inst, driver_col,
            ))

            trend_kwargs = {} if is_trend else {"mode": "change"}
            cont_trend_result = build_contribution_trend_chart_configs(
                br_detail_level, pivot_biz_table, data,
                selected_levels, is_specific, readout, match, prompt_inst,
                **trend_kwargs,
            )
            cont_trend_configs = cont_trend_result[0] if isinstance(cont_trend_result, tuple) else cont_trend_result
            configs.update(cont_trend_configs)

    return configs


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Ask Genome – Insights (local)",
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
# Session state
# ---------------------------------------------------------------------------

for key in [
    "thinking_content", "main_content", "query", "readout",
    "selected_intention", "processed_data", "response_chunks",
    "intention_confirmed", "intention_required_confirmation",
    "multitask_pred", "spacy_filter", "process_indicator", "time_confirmed",
    "original_period", "original_time",
    # insights-only page extras
    "insight", "all_chart_configs", "all_insights", "resp1", "resp2",
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
    "client_code": None,
    "model_group_id": None,
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
    st.markdown(f"Data dir: `{DATA_DIR}`")

    client_model_group_options = get_available_client_and_model_group()
    selected = st.selectbox("Client Code", client_model_group_options.keys(), on_change=utils.reset_query_and_data)
    client_code, model_group_id = client_model_group_options[selected]
    st.session_state.client_code = client_code
    st.session_state.model_group_id = model_group_id

    os.environ["CLIENT_CODE"] = client_code
    os.environ["MODEL_GROUP_ID"] = str(model_group_id)

    st.markdown(f"Client Code: `{client_code}`")
    st.markdown(f"Model Group ID: `{model_group_id}`")

    st.divider()
    llm_vendor = st.selectbox("Response LLM Vendor",
                              ["openai", "gemini"],
                              index=0,
                              on_change=utils.reset_query_and_data)
    os.environ["EXTERNAL_VENDOR"] = llm_vendor

# ---------------------------------------------------------------------------
# Reset on page switch or client/model change
# ---------------------------------------------------------------------------

utils._detect_and_reset_on_context_change(
    client_code=client_code,
    model_group_id=model_group_id,
    page_id=__file__,
)

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
        # reset insight caches when a new query comes in
        st.session_state.insight = {}
        st.session_state.all_chart_configs = {}
        st.session_state.all_insights = {}
        st.session_state.resp1 = None
        st.session_state.resp2 = None

    if st.session_state.get("query"):
        st.chat_message("user", avatar=":material/person:").write(st.session_state.query)

    # ── Load readout from local jsonl (replaces NER pipeline) ─────────────────
    if st.session_state.query and st.session_state.readout is None:
        with st.spinner("Loading readout data from local file..."):
            readout_data = load_readout_from_jsonl(
                DATA_DIR, client_code, model_group_id, st.session_state.query
            )

        if readout_data is None:
            st.error(
                f"No readout record found for this query in "
                f"{os.path.join(DATA_DIR, client_code, str(model_group_id), READOUT_JSONL_FILENAME)}"
            )
            st.stop()

        process_indicator = ProcessIndicator.from_local(client_code, int(model_group_id))
        st.session_state.process_indicator = process_indicator
        st.session_state.readout = readout_data

        # Track original time period (same as v1/st-main_local_version.py)
        if not st.session_state.original_period and not st.session_state.original_time:
            st.session_state.original_period = st.session_state.readout.ner_filters["period_type"][0]
            st.session_state.original_time = st.session_state.readout.ner_filters["time"]

    # ── process_data ─────────────────────────────────────────────────────────
    if st.session_state.readout is not None and st.session_state.processed_data is None:
        (
            context_str, data, benchmark_data, benchmark_str,
            planner_data, spend_share_principle, pretext,
            pivot_biz_table, pretext_table, pretext_table_trend,
            match, principle_pretext, overall_view, _,
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
    # Data table previews (same as both st-main files)
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
        agg_data, outliers_agg = viz_utils.data_init(pivot_biz_table, metric_ls, driver_col=driver_col)
    else:
        agg_data, outliers_agg = None, None

    if data is not None and not data.empty:
        st.markdown("### Detail View")
        rows = min(len(data), 20)
        height = max(min_height, min(max_height, rows * row_height))
        with st.expander("Show data table"):
            st.dataframe(data.reset_index(drop=True), use_container_width=True, height=height)
        df = data.copy()

        if len(df) > 1:
            # Deterministic view: BR with agg available -> Aggregation; otherwise Detail.
            br_detail_level = "Aggregation View" if (if_agg_exists and data_type == "br") else "Detail View"

            if br_detail_level == "Aggregation View":
                df, outliers = viz_utils.data_init(pivot_biz_table, metric_ls, driver_col=driver_col)
            else:
                df, outliers = viz_utils.data_init(df, metric_ls, driver_col=driver_col)

            # Iterate every (metric, level) combo — no widgets
            metric_options = _select_metric_with_all(metric_ls, main_metric_ls, df)
            level_options = _select_levels_with_all(df, driver_col)

            # sel_m / m_cols are invariant across iterations (depend only on metric_ls)
            sel_m, m_cols = _resolve_m_cols(metric_ls, df)

            # ── Insight session state ─────────────────────────────────────────────
            if "insight" not in st.session_state:
                st.session_state.insight = {}

            # =====================================================================
            # Build chart configs — iterate (metric × level) combos
            # =====================================================================

            all_chart_configs: dict = {}
            combo_to_chart_ids: dict[str, list[str]] = {}

            # Context-aware topic label for insight query templates (invariant across combos)
            if data_type == "br" and "contribution" in metric_ls:
                insight_topic = "Contribution"
            elif intention in ["margin roi", "performance"]:
                insight_topic = "ROI"
            else:
                insight_topic = intention.title()

            for selected_metrics in metric_options:
                metric_cols, metric_mappings = viz_utils.get_metric_mapping(df, selected_metrics)
                metric_labels = list(metric_mappings.keys())

                for level_dict in level_options:
                    # Filter df to this combo
                    df_combo = df.copy()
                    for lvl, val in level_dict.items():
                        df_combo = df_combo[df_combo[lvl] == val]
                    if df_combo.empty:
                        continue

                    df_plot, df_plot_cont_share, sorted_times = viz_utils.data_prepare(
                        df_combo, metric_cols, m_cols, driver_col,
                    )

                    combo_configs = _build_configs_for_combo(
                        data_type=data_type, intention=intention,
                        is_trend=is_trend, is_specific=is_specific,
                        if_agg_exists=if_agg_exists, trend_check=trend_check,
                        df_combo=df_combo, agg_data=agg_data,
                        outliers=outliers, outliers_agg=outliers_agg,
                        selected_metrics=selected_metrics, selected_levels=level_dict,
                        metric_cols=metric_cols, m_cols=m_cols,
                        sorted_times=sorted_times, prompt_inst=prompt_inst,
                        br_detail_level=br_detail_level,
                        df_plot=df_plot, df_plot_cont_share=df_plot_cont_share,
                        data=data, pivot_biz_table=pivot_biz_table,
                        client_code=client_code, model_group_id=model_group_id,
                        readout=st.session_state.readout, match=match, driver_col=driver_col,
                        metric_ls=metric_ls,
                    )

                    combo_label = _combo_label(selected_metrics, level_dict)
                    for chart_id, cfg in combo_configs.items():
                        namespaced = f"{combo_label}::{chart_id}"
                        all_chart_configs[namespaced] = cfg
                        combo_to_chart_ids.setdefault(combo_label, []).append(namespaced)

            # =====================================================================
            # Generate per-chart insights (parallel) — cached in session state
            # =====================================================================

            if all_chart_configs and not st.session_state.all_chart_configs:
                st.session_state.all_chart_configs = all_chart_configs

            if not all_chart_configs:
                st.info("No chart configurations could be built for this query.")
                st.stop()

            # Rebuild combo_to_chart_ids from cached all_chart_configs to stay
            # consistent if the cache was populated on an earlier run (different widgets).
            combo_to_chart_ids = {}
            for namespaced_id in all_chart_configs.keys():
                if "::" in namespaced_id:
                    combo_label_key = namespaced_id.split("::", 1)[0]
                    combo_to_chart_ids.setdefault(combo_label_key, []).append(namespaced_id)

            # =====================================================================
            # Method 1 — Synthesised narrative from raw chart data
            # =====================================================================

            if st.session_state.resp1 is None:
                charts_block = get_chart_set_inst(all_chart_configs)
                insight_indiv_query = (
                    f"You are given several datasets about {insight_topic}. "
                    f"Provide structured insights from datasets and then answer the user's question: "
                    f"{st.session_state.query} "
                    f"Write your response in markdown format."
                )
                with st.spinner("Generating insights from raw data…"):
                    resp1 = response_generate_chart(
                        query=insight_indiv_query,
                        readout=charts_block,
                        llm_vendor=llm_vendor,
                    )
                    if "</think>" in resp1:
                        _, resp1 = resp1.split("</think>", 1)
                    st.session_state.resp1 = resp1.strip()
                    st.session_state.charts_block = charts_block

            st.markdown("### Insights from Data")
            with st.expander("Show insights", expanded=False):
                st.markdown(st.session_state.resp1)

            st.markdown("### Readout from Data")
            with st.expander("Show readout", expanded=False):
                st.write(st.session_state.charts_block)

    # ---------------------------------------------------------------------------
    # Planner insights (no chart rendering)
    # ---------------------------------------------------------------------------

    if planner_data is not None and not planner_data.empty:
        st.markdown("## Planner Insights")

        core_dim = st.session_state.readout.ner_filters.get("core_dimension", {})
        sub_df = st.session_state.readout.df_measure

        inc_data = planner_data[planner_data['Increase/Decrease'] == 'Increase'].drop(columns=['Increase/Decrease'])
        dec_data = planner_data[planner_data['Increase/Decrease'] == 'Decrease'].drop(columns=['Increase/Decrease'])

        if not inc_data.empty:
            st.markdown("### Driver(s) with Spending Increase")
            with st.expander("Show data table"):
                st.dataframe(inc_data.reset_index(drop=True), use_container_width=True)

        if not dec_data.empty:
            st.markdown("### Driver(s) with Spending Decrease")
            with st.expander("Show data table"):
                st.dataframe(dec_data.reset_index(drop=True), use_container_width=True)
        if 'insight' not in st.session_state:
            st.session_state.insight = {}
        chart_configs, chart_configs_lvl = {}, {}
        core_dim = st.session_state.readout.ner_filters.get('core_dimension', {})
        sub_df = pd.concat([st.session_state.readout.df_activity_group,
                            st.session_state.readout.df_measure_group,
                            st.session_state.readout.df_measure
                            ])

        spend_share_data, _ = viz_utils.get_spend_share_principle_data(spend_share_principle)

        # loop through all scenario id
        planner_chart_configs: dict = {}
        for scenario_id in spend_share_data['Scenario ID'].unique().tolist():
            planner_data_all = viz_utils.init_planner_data(client_code, model_group_id, sub_df, core_dim)

            planner_data_all, selected_kpi = utils._render_planner_filters(planner_data_all, scenario_id, planner_data)

            for kpi in selected_kpi:
                planner_data_filtered = planner_data_all[planner_data_all['KPI'] == kpi]
                levels = viz_utils.get_available_level(planner_data_all)

                if not core_dim:
                    overall_data = viz_utils.get_planner_overview_data(client_code, model_group_id, scenario_id, kpi)
                    spend_all_inst = planner_spend_change_all_instruct(planner_data_filtered)
                    planner_chart_configs[f"spend_change_all_{scenario_id}"] = {
                        "query": "You are given a dataset containing spend % change by driver.",
                        "instruction": spend_all_inst,
                        "data": "",
                    }
                    bin_pct_inst = planner_pct_bar_instruct(client_code, model_group_id, sub_df, levels, core_dim, overall_data, planner_data_filtered, mode="pct_change")
                    planner_chart_configs[f"insights_pct_pct_change_{scenario_id}"] = {
                        "query": "You are given a dataset grouped by Spend Change Group.",
                        "instruction": bin_pct_inst,
                        "data": "",
                    }
                    bin_amt_inst = planner_pct_bar_instruct(client_code, model_group_id, sub_df, levels, core_dim, overall_data, planner_data_filtered, mode="spend_amount")
                    planner_chart_configs[f"insights_pct_spend_amount_{scenario_id}"] = {
                        "query": "You are given a dataset grouped by Spend Change Group.",
                        "instruction": bin_amt_inst,
                        "data": "",
                    }

                    for lvl in levels:
                        scatter_inst = spend_kpi_scatter_insight(planner_data_filtered, [lvl])
                        planner_chart_configs[f"spend_kpi_fair_share_{lvl}_{scenario_id}"] = {
                            "query": "You are given a dataset containing spend and KPIs % change by driver.",
                            "instruction": scatter_inst,
                            "data": "",
                        }
                        diminish_inst = plot_diminishing_return_instruct(
                            client_code, model_group_id, planner_data_filtered, [lvl],
                        )
                        planner_chart_configs[f"diminish_return_{lvl}_{scenario_id}"] = {
                            "query": "You are given a dataset containing historical and forecast spend by driver.",
                            "instruction": diminish_inst,
                            "data": "",
                        }
                else:
                    scatter_inst = spend_kpi_scatter_insight(planner_data_filtered, levels)
                    planner_chart_configs[f"spend_kpi_fair_share_levels_{scenario_id}"] = {
                        "query": "You are given a dataset containing spend and KPIs % change by driver.",
                        "instruction": scatter_inst,
                        "data": "",
                    }
                    diminish_inst = plot_diminishing_return_instruct(client_code, model_group_id, planner_data_filtered, levels)
                    planner_chart_configs[f"diminish_return_levels_{scenario_id}"] = {
                        "query": "You are given a dataset containing historical and forecast spend by driver.",
                        "instruction": diminish_inst,
                        "data": "",
                    }

        if planner_chart_configs:
            planner_charts_block = get_chart_set_inst(planner_chart_configs)
            planner_insight_query = (
                f"You are given several datasets about planner spend allocation. "
                f"Provide structured insights from datasets and then answer the user's question: "
                f"{st.session_state.query} "
                f"Write your response in markdown format."
            )
            with st.spinner("Generating planner insights…"):
                planner_resp = response_generate_chart(
                    query=planner_insight_query,
                    readout=planner_charts_block,
                    llm_vendor=llm_vendor,
                )
                if "</think>" in planner_resp:
                    _, planner_resp = planner_resp.split("</think>", 1)
                planner_resp = planner_resp.strip()
                st.session_state.charts_block = planner_charts_block
                st.session_state.resp1 = planner_resp

            with st.expander("📈 Planner Insights", expanded=True):
                st.markdown(planner_resp)

            with st.expander("📊 Readout from Planner data", expanded=False):
                st.markdown(planner_charts_block)

    # ---------------------------------------------------------------------------
    # Benchmark context
    # ---------------------------------------------------------------------------

    if benchmark_data is not None and not benchmark_data.empty:
        st.markdown("## Benchmark Data")
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
            st.markdown("### 📄 Retrieved Documents")
            retriever.display_retrieved_documents(threshold=0.8)
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
if st.session_state.resp1:
    Data_dir = "/home/lzhao/Documents/ask-genome-data/"
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


    # save to json
    def save_dict_to_jsonl(records: list[dict], filepath: str) -> None:
        with open(filepath, "a", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")


    res_file = f"{st.session_state.client_code}_test_insights.csv"
    res = {"query": st.session_state.query, "intention": intention, "trend": trend_check,
           "prompt1": st.session_state.charts_block,
           "response1": st.session_state.resp1}

    write_dict_to_csv(res, filepath=os.path.join(Data_dir, res_file))
