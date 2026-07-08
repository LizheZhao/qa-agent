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

from chart import (
    get_chart_set_inst,
    # BI build functions
    build_roi_snapshot_chart_configs_for_insights,
    build_roi_trend_chart_configs_for_insights,
    build_spend_trend_chart_configs_for_insights,
    build_roi_response_cost_chart_configs_for_insights,
    build_spend_snapshot_chart_configs_for_insights,
    build_spend_heatmap_chart_configs,
    build_other_bi_snapshot_chart_configs,
    build_other_bi_trend_chart_configs,
    # BR build functions
    build_sourceofchange_chart_configs_for_insights,
    build_contribution_trend_chart_configs_for_insights,
    build_contribution_snapshot_chart_configs_for_insights,
    _sales_trend_data_for_insights,
    # Planner data helpers
    spend_kpi_scatter_insight_for_insights,
    plot_diminishing_return_instruct_for_insights,
    planner_spend_change_all_instruct_for_insights,
    planner_pct_bar_instruct_for_insights,
)


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
    """Build chart_configs for ONE (metric, level) combo."""
    configs: dict = {}

    if data_type == "bi":
        if intention in ["margin roi", "performance"]:
            ag_mapping = viz_utils.get_ag_level_mapping(df_combo, readout, match, is_specific)
            recent_d, trend_d, resp_cost_d = viz_utils.prepare_roi_data_for_insights(
                df_combo, client_code, model_group_id,
                readout.ner_filters,
                selected_levels, outliers, trend_check,
                selected_metrics, ag_mapping, is_specific, sorted_times, table="detail",
            )
            if if_agg_exists:
                recent_d_agg, trend_d_agg, resp_cost_d_agg = viz_utils.prepare_roi_data_for_insights(
                    agg_data, client_code, model_group_id,
                    readout.ner_filters,
                    selected_levels, outliers_agg, trend_check,
                    selected_metrics, ag_mapping, is_specific, sorted_times, table="agg",
                )
            else:
                recent_d_agg = trend_d_agg = resp_cost_d_agg = None

            tab1 = build_roi_snapshot_chart_configs_for_insights(
                recent_d, recent_d_agg, sorted_times[-1], is_specific,
                [], [], prompt_inst,
                client_code=client_code, model_group_id=model_group_id
            )
            configs.update(tab1)

            if trend_d or resp_cost_d:
                if trend_check == "yes" and trend_d:
                    tab2 = build_roi_trend_chart_configs_for_insights(
                        trend_d, trend_d_agg, sorted_times, is_specific,
                        add_long_term=False, prompt_inst=prompt_inst,
                        multi_roi_data=None, multi_roi_data_agg=None,
                        roi_type=selected_metrics[0],
                    )
                elif resp_cost_d:
                    tab2 = build_roi_response_cost_chart_configs_for_insights(
                        resp_cost_d, resp_cost_d_agg, sorted_times, is_specific,
                        ag_mapping, prompt_inst, roi_type=selected_metrics[0],
                        client_code=client_code, model_group_id=model_group_id
                    )
                else:
                    tab2 = {}
                configs.update(tab2)

        elif intention == "spending":
            configs.update(build_spend_snapshot_chart_configs_for_insights(
                df_combo, selected_levels, driver_col, metric_cols,
                sorted_times[::-1], readout, match, is_specific,
            ))
            if is_trend:
                configs.update(build_spend_trend_chart_configs_for_insights(
                    data, df_combo, selected_levels, metric_cols,
                    readout, match, is_specific,
                ))
            else:
                configs.update(build_spend_heatmap_chart_configs(
                    df_combo, selected_levels, driver_col, metric_cols,
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
            # total_viz_soc = viz_utils.filter_bd_data(
            #     client_code, model_group_id, selected_levels, ["sales", "sourceofchange"]
            # )
            total_viz_soc = viz_utils.get_soc_bd_data(client_code, model_group_id, selected_levels,
                                                      st.session_state.readout)
            total_act_sales = viz_utils.filter_bd_data(
                client_code, model_group_id, selected_levels, ["sales"], total="total"
            )
            group_mapping = viz_utils.get_group_level_mapping(readout, match, is_specific)
            soc_result = build_sourceofchange_chart_configs_for_insights(
                total_viz_soc, total_act_sales, df_plot, outliers,
                metric_cols, sorted_times[::-1], driver_col,
                br_detail_level, group_mapping, is_specific,
            )
            soc_configs = soc_result[0] if isinstance(soc_result, tuple) else soc_result
            configs.update(soc_configs)

        if "contribution" in selected_metrics:
            select_time = viz_utils.get_time_align_dict(sorted_times[::-1])
            keys_list = list(select_time.keys())
            sel_time_left = select_time[keys_list[0]]
            sel_time_right = select_time[keys_list[min(1, len(keys_list) - 1)]]

            cont_snap_configs = build_contribution_snapshot_chart_configs_for_insights(
                client_code, model_group_id, df_plot, df_plot_cont_share,
                sorted_times[::-1], selected_levels, metric_cols, m_cols,
                outliers, is_specific, readout, match,
                sel_time_left, sel_time_right, prompt_inst, driver_col,
            )
            configs.update(cont_snap_configs)

            mode = None if is_trend else "change"
            cont_trend_configs = build_contribution_trend_chart_configs_for_insights(
                br_detail_level, pivot_biz_table, data,
                selected_levels, is_specific, readout, match, prompt_inst,
                mode, client_code, model_group_id
            )

            configs.update(cont_trend_configs)

        if "sales" in selected_metrics:
            snapshot_cols = [c for c in df_combo.columns if "%" not in c]
            sales_snapshot_inst = prompt_inst[prompt_inst['intentionName'] == "Sales Snapshot"]['instruction'].values[0]
            configs["sales_snapshot"] = {
                "query": "You are given a dataset about sales.",
                "instruction": sales_snapshot_inst,
                "data": df_combo[snapshot_cols].to_json(orient="records"),
            }

            if is_trend:
                sales_trend_df = _sales_trend_data_for_insights(df_combo, driver_col)
                if not sales_trend_df.empty:
                    sales_trend_inst = \
                        prompt_inst[prompt_inst['intentionName'] == "Sales Trend"]['instruction'].values[0]
                    configs["sales_trend"] = {
                        "query": "You are given a dataset about sales over time.",
                        "instruction": sales_trend_inst,
                        "data": sales_trend_df.to_json(orient="records"),
                    }

    return configs


def _prep_pivot_biz_table(pivot_biz_table: pd.DataFrame, metric_ls: list, driver_col: list):
    if pivot_biz_table is not None and not pivot_biz_table.empty:
        agg_data, outliers_agg = viz_utils.data_init(pivot_biz_table, metric_ls, driver_col=driver_col)
    else:
        agg_data, outliers_agg = None, None
    return agg_data, outliers_agg


def _generate_insights_prompt(chart_configs: dict, query: str, insight_topic) -> str:
    """
    Generate insights from chart datasets
    Method 1: Insights from combined raw data
    Method 2: Insights from individual chart insight
    """
    ## -- Method 1 -- ##
    charts_block = get_chart_set_inst(chart_configs)
    insight_query = (
        f"You are given several datasets about {insight_topic}. "
        f"Provide structured insights from datasets and then answer the user's question: "
        f"{query} "
        f"Write your response in markdown format."
    )
    return insight_query + "\n" + charts_block


def _prep_visual_context(client_code: str, model_group_id: int, data: pd.DataFrame, planner_data: pd.DataFrame,
                         match: dict,
                         readout, pivot_biz_table: pd.DataFrame, spend_share_principle: pd.DataFrame, query: str,
                         pretext_table: pd.DataFrame, pretext_table_trend: pd.DataFrame) -> str:
    intention, is_trend, is_specific, if_agg_exists, driver_col, metric_ls, main_metric_ls, data_type, trend_check = \
        viz_utils.viz_config(readout, match, pivot_biz_table)
    prompt_inst = readout_utils.load_prompt_instruction(client_code, model_group_id)
    agg_data, outliers_agg = _prep_pivot_biz_table(pivot_biz_table, metric_ls, driver_col)

    all_chart_configs: dict = {}
    # Context-aware topic label for insight query templates (invariant across combos)
    if data_type == "br" and "contribution" in metric_ls:
        insight_topic = "Contribution"
    elif intention in ["margin roi", "performance"]:
        insight_topic = "ROI"
    else:
        insight_topic = intention.title()

    all_chart_configs: dict = {}

    if pretext_table is not None and not pretext_table.empty:
        pretext_table_config = {
            "query": f"You are given a overall view of {insight_topic}.",
            "instruction": "Provide a concise summary based on the dataset.",
            "data": pretext_table.to_json(orient="records"),
        }
        all_chart_configs["pretext_table_config"] = pretext_table_config

    if pretext_table_trend is not None and not pretext_table_trend.empty:
        pretext_table_trend_config = {
            "query": f"You are given a overall trend view of {insight_topic}.",
            "instruction": "Provide a concise summary based on the dataset.",
            "data": pretext_table_trend.to_json(orient="records"),
        }
        all_chart_configs["pretext_table_trend_config"] = pretext_table_trend_config

    if data is not None and not data.empty:
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

            # =====================================================================
            # Build chart configs — iterate (metric × level) combos
            # =====================================================================

            combo_to_chart_ids: dict[str, list[str]] = {}

            for selected_metrics in metric_options:
                metric_cols, metric_mappings = viz_utils.get_metric_mapping(df, selected_metrics)
                metric_labels = list(metric_mappings.keys())

                for level_dict in level_options:
                    # Filter df to this combo
                    df_combo = df.copy()
                    for lvl, val in level_dict.items():
                        df_combo = df_combo[df_combo[lvl] == val]
                    df_combo = df_combo.dropna(subset=metric_cols, how="all")
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
                        readout=readout, match=match, driver_col=driver_col,
                        metric_ls=metric_ls,
                    )

                    combo_label = _combo_label(selected_metrics, level_dict)
                    for chart_id, cfg in combo_configs.items():
                        namespaced = f"{combo_label}::{chart_id}"
                        all_chart_configs[namespaced] = cfg
                        combo_to_chart_ids.setdefault(combo_label, []).append(namespaced)

    if planner_data is not None and not planner_data.empty:

        core_dim = readout.ner_filters.get("core_dimension", {})
        sub_df = readout.df_measure

        inc_data = planner_data[planner_data['Increase/Decrease'] == 'Increase'].drop(columns=['Increase/Decrease'])
        dec_data = planner_data[planner_data['Increase/Decrease'] == 'Decrease'].drop(columns=['Increase/Decrease'])

        chart_configs, chart_configs_lvl = {}, {}
        core_dim = readout.ner_filters.get('core_dimension', {})
        sub_df = pd.concat([readout.df_activity_group,
                            readout.df_measure_group,
                            readout.df_measure
                            ])

        spend_share_data, _ = viz_utils.get_spend_share_principle_data(spend_share_principle)

        # loop through all scenario id
        planner_chart_configs: dict = {}
        planner_data_all = viz_utils.init_planner_data(client_code, model_group_id, sub_df, core_dim)
        for scenario_id in planner_data_all['Scenario ID'].unique().tolist():
            planner_data_scenario, selected_kpi = utils._render_planner_filters(planner_data=planner_data_all,
                                                                                scenario=scenario_id,
                                                                                planner_data_sub=planner_data)

            if not planner_data_scenario.empty:
                for kpi in selected_kpi:
                    planner_data_filtered = planner_data_scenario[planner_data_scenario['KPI'] == kpi]
                    levels = viz_utils.get_available_level(planner_data_scenario)

                    if not core_dim:
                        overall_data = viz_utils.get_planner_overview_data(client_code, model_group_id, scenario_id,
                                                                           kpi)
                        spend_all_config = planner_spend_change_all_instruct_for_insights(planner_data_filtered)
                        planner_chart_configs[f"spend_change_all_{scenario_id}"] = spend_all_config

                        bin_pct_config = planner_pct_bar_instruct_for_insights(client_code, model_group_id, sub_df,
                                                                               levels, core_dim, overall_data,
                                                                               planner_data_filtered, mode="pct_change")
                        planner_chart_configs[f"insights_pct_pct_change_{scenario_id}"] = bin_pct_config

                        bin_amt_config = planner_pct_bar_instruct_for_insights(client_code, model_group_id, sub_df,
                                                                               levels, core_dim, overall_data,
                                                                               planner_data_filtered,
                                                                               mode="spend_amount")
                        planner_chart_configs[f"insights_pct_spend_amount_{scenario_id}"] = bin_amt_config

                        for lvl in levels:
                            scatter_config = spend_kpi_scatter_insight_for_insights(planner_data_filtered, [lvl])
                            planner_chart_configs[f"spend_kpi_fair_share_{lvl}_{scenario_id}"] = scatter_config

                            diminish_config = plot_diminishing_return_instruct_for_insights(
                                client_code, model_group_id, planner_data_filtered, [lvl],
                            )
                            planner_chart_configs[f"diminish_return_{lvl}_{scenario_id}"] = diminish_config
                    else:
                        scatter_config = spend_kpi_scatter_insight_for_insights(planner_data_filtered, levels)
                        planner_chart_configs[f"spend_kpi_fair_share_levels_{scenario_id}"] = scatter_config

                        diminish_config = plot_diminishing_return_instruct_for_insights(client_code, model_group_id,
                                                                                        planner_data_filtered, levels)
                        planner_chart_configs[f"diminish_return_levels_{scenario_id}"] = diminish_config

        all_chart_configs.update({k: v for k, v in planner_chart_configs.items() if v})

    if all_chart_configs:
        chart_insight_context = _generate_insights_prompt(all_chart_configs, query, insight_topic)
    else:
        chart_insight_context = ""
    return chart_insight_context
