import os
import sys
import re
import numbers
import numpy as np
import pandas as pd
from itertools import product

import src.insight_generation.utils as utils
import src.model.readout_utils as readout_utils


def label_metric_level(df: pd.DataFrame, metric: str, time: str, q1=0.5, q2=0.75):
    metric_col = [x for x in df.columns.tolist() if (metric.lower() in x.lower()) and ("%" not in x) and (time in x) and ('level' not in x) and ('trend' not in x)]
    if metric_col:
        # df[metric_col[0]] = df[metric_col[0]].str.strip("%").astype(float).fillna(0) # assume cleaned before passed in
        l1 = df[metric_col[0]].quantile(q1)
        l2 = df[metric_col[0]].quantile(q2)

        def classify_metric(value):
            if pd.isna(value):
                return ""
            if value <= l1:
                return 'low'
            elif value <= l2:
                return 'moderate'
            else:
                return 'high'

        df[f'{metric} level {time}'] = df[metric_col[0]].apply(classify_metric)
    return df


def label_metric_level_with_dist(data: pd.DataFrame, quantile_df: pd.DataFrame, target_metrics: dict,
                                 target_time: str, group_keys: list) -> pd.DataFrame:
    """
    target_metrics: {kpiName: metric alias}
    """
    pivot_data = data.copy()

    # find the target metric column from

    for target_metric, alias in target_metrics.items():
        pattern = re.compile(rf"^{re.escape(alias)}\s+{re.escape(str(target_time))}", re.IGNORECASE)
        matches = [x for x in pivot_data.columns if pattern.match(x)]
        if not matches:
            continue
        target_col = matches[0]
        level_col = f'{alias} level {target_time}'

        q_df = quantile_df[
            (quantile_df['metric'].str.lower() == target_metric.lower())
            & (quantile_df['time_norm'].astype(str).str.contains(str(target_time)))
            ]
        if group_keys:
            q_lookup = q_df.set_index(group_keys)[['q1', 'q3']]
            global_threshold = None
        else:
            thresholds = q_df[['q1', 'q3']].drop_duplicates()
            q_lookup = None
            global_threshold = thresholds.iloc[0] if len(thresholds) == 1 else None
        print(target_metric, len(q_df))

        def _classify(row, target_col=target_col):
            raw = str(row[target_col])
            val = pd.to_numeric(raw.replace('%', '').replace(',', ''), errors='coerce')
            if "(MM)" in target_col:
                val *= 10**6
            if pd.isna(val):
                return np.nan
            if group_keys:
                key = row[group_keys[0]] if len(group_keys) == 1 else tuple(row[k] for k in group_keys)
                try:
                    q1, q3 = q_lookup.loc[key]
                except KeyError:
                    return np.nan
            elif global_threshold is not None:
                q1, q3 = global_threshold
            else:
                return np.nan
            if '%' in raw:
                val = val / 100
            if val < q1:
                return 'low'
            if val > q3:
                return 'high'
            return 'moderate'

        pivot_data[level_col] = pivot_data.apply(_classify, axis=1)

    return pivot_data


def build_roi_config_for_insights(client_code: str, model_group_id: int, query: str,
                                  data: pd.DataFrame, agg_data: pd.DataFrame,
                                  readout_data, metric_configs: pd.DataFrame) -> dict:
    """
    build contribution table for insights generation readout by time tag
    """
    # load prompt instructions for charting types
    prompt_instructions = readout_utils.load_insight_instructions(client_code, model_group_id, intention="margin roi")
    metric_dist = readout_utils.load_metric_distribution(client_code, model_group_id, file_type="bi")
    time_tag = readout_data.ner_filters.get("time_tag", "snapshot")
    level_type = readout_data.ner_filters.get("level")
    levels = level_type.get("general_level", []) + level_type.get("halo_level", [])
    driver_col = "Tactics"
    if metric_dist.empty or "time" not in metric_dist.columns:
        metric_dist = pd.DataFrame(columns=levels+["time", "metric", "q1", "q3", "time_norm"])
    else:
        metric_dist = utils.normalize_time(metric_dist) # add normalized time col for record matching
    
    # load metric info
    # TODO: load main metric, related metric from intention mapping
    roi_metrics = list(metric_configs[metric_configs["metricGroup"].str.lower().isin(["roi"])]["kpiName"].str.lower().values)
    spend_metrics = list(metric_configs[metric_configs["metricGroup"].str.lower().isin(["spend"])]["kpiName"].str.lower().values)
    response_metrics = list(metric_configs[metric_configs["metricGroup"].str.lower().isin(["response"])]["kpiName"].str.lower().values)
    cost_per_metrics = list(metric_configs[metric_configs["metricGroup"].str.lower().isin(["cost per activity"])]["kpiName"].str.lower().values)
    activity_metrics = list(metric_configs[metric_configs["metricGroup"].str.lower().isin(["activity"])]["kpiName"].str.lower().values)
    cost_per_conversion_metric = ["cost per conversion"]

    # process instructions
    instructions, tbl_naming = utils._instruction_maps(prompt_instructions, time_tag)

    roi_spend_tbl, roi_spend_agg_tbl = None, None
    roi_tbl, roi_agg_tbl = None, None
    response_cp_tbl, response_cp_agg_tbl = None, None
    other_related_tbl, other_related_agg_tbl = None, None
    cost_per_conversion_tbl, cost_per_conversion_agg_tbl = None, None
    overall_agg_tbl = None

    if time_tag not in ["snapshot", "period_over_period_change", "multiple_periods", "trend"]:
        raise Exception("time_tag must be 'snapshot', 'period_over_period_change', 'multiple_periods' or 'trend'")

    # the overall table is the same for every time_tag, so it is built once outside the dispatch
    overall_tbl = build_overall_tbl_for_insights(client_code, model_group_id, data, metric_dist,
                                                 roi_metrics, readout_data, metric_configs)
    if not agg_data.empty:
        overall_agg_tbl = build_overall_tbl_for_insights(client_code, model_group_id, agg_data, metric_dist,
                                                         roi_metrics, readout_data, metric_configs)

    if time_tag == "snapshot":
        roi_spend_tbl = build_roi_spend_tbl_for_insights(client_code, model_group_id, data, metric_dist,
                                                         roi_metrics, spend_metrics, driver_col, readout_data,
                                                         metric_configs)
        roi_tbl = build_detail_tbl_for_insights(client_code, model_group_id, data, metric_dist,
                                                roi_metrics, driver_col, readout_data,
                                                metric_configs)
        response_cp_tbl = build_roi_response_cp_tbl_for_insights(client_code, model_group_id, data, metric_dist,
                                                                 roi_metrics, response_metrics, cost_per_metrics,
                                                                 driver_col, readout_data,
                                                                 metric_configs)
        other_related_tbl = build_roi_other_relationship_tbl_for_insights(client_code, model_group_id, data,
                                                                          metric_dist,
                                                                          roi_metrics, activity_metrics, driver_col,
                                                                          readout_data, metric_configs)
        cost_per_conversion_tbl = build_detail_tbl_for_insights(client_code, model_group_id, data, metric_dist,
                                                                cost_per_conversion_metric, driver_col, readout_data,
                                                                metric_configs)

        if not agg_data.empty:
            roi_spend_agg_tbl = build_roi_spend_tbl_for_insights(client_code, model_group_id, agg_data, metric_dist,
                                                             roi_metrics, spend_metrics, driver_col, readout_data,
                                                             metric_configs)
            roi_agg_tbl = build_detail_tbl_for_insights(client_code, model_group_id, agg_data, metric_dist,
                                                    roi_metrics, driver_col, readout_data,
                                                    metric_configs)
            response_cp_agg_tbl = build_roi_response_cp_tbl_for_insights(client_code, model_group_id, agg_data, metric_dist,
                                                                     roi_metrics, response_metrics, cost_per_metrics,
                                                                     driver_col, readout_data,
                                                                     metric_configs)
            other_related_agg_tbl = build_roi_other_relationship_tbl_for_insights(client_code, model_group_id, agg_data,
                                                                              metric_dist,
                                                                              roi_metrics, activity_metrics, driver_col,
                                                                              readout_data, metric_configs)
            cost_per_conversion_agg_tbl = build_detail_tbl_for_insights(client_code, model_group_id, agg_data, metric_dist,
                                                                        cost_per_conversion_metric, driver_col,
                                                                        readout_data, metric_configs)
    elif time_tag == "period_over_period_change":
        roi_spend_tbl = build_roi_spend_tbl_for_insights(client_code, model_group_id, data, metric_dist,
                                                         roi_metrics, spend_metrics, driver_col, readout_data,
                                                         metric_configs)
        response_cp_tbl = build_roi_response_cp_tbl_for_insights(client_code, model_group_id, data, metric_dist,
                                                                 roi_metrics, response_metrics, cost_per_metrics,
                                                                 driver_col, readout_data,
                                                                 metric_configs)
        other_related_tbl = build_roi_other_relationship_tbl_for_insights(client_code, model_group_id, data,
                                                                          metric_dist,
                                                                          roi_metrics, activity_metrics, driver_col,
                                                                          readout_data, metric_configs)
        cost_per_conversion_tbl = build_detail_tbl_for_insights(client_code, model_group_id, data, metric_dist,
                                                                cost_per_conversion_metric, driver_col, readout_data,
                                                                metric_configs)

        if not agg_data.empty:
            roi_spend_agg_tbl = build_roi_spend_tbl_for_insights(client_code, model_group_id, agg_data, metric_dist,
                                                                 roi_metrics, spend_metrics, driver_col, readout_data,
                                                                 metric_configs)
            response_cp_agg_tbl = build_roi_response_cp_tbl_for_insights(client_code, model_group_id, agg_data,
                                                                         metric_dist,
                                                                         roi_metrics, response_metrics,
                                                                         cost_per_metrics,
                                                                         driver_col, readout_data,
                                                                         metric_configs)
            other_related_agg_tbl = build_roi_other_relationship_tbl_for_insights(client_code, model_group_id, agg_data,
                                                                                  metric_dist,
                                                                                  roi_metrics, activity_metrics,
                                                                                  driver_col,
                                                                                  readout_data,
                                                                                  metric_configs)
            cost_per_conversion_agg_tbl = build_detail_tbl_for_insights(client_code, model_group_id, agg_data,
                                                                        metric_dist,
                                                                        cost_per_conversion_metric, driver_col,
                                                                        readout_data, metric_configs)
    elif time_tag == "multiple_periods":
        roi_spend_tbl = build_roi_spend_tbl_for_insights(client_code, model_group_id, data, metric_dist,
                                                         roi_metrics, spend_metrics, driver_col, readout_data,
                                                         metric_configs)
        roi_tbl = build_detail_tbl_for_insights(client_code, model_group_id, data, metric_dist,
                                                roi_metrics, driver_col, readout_data,
                                                metric_configs)
        response_cp_tbl = build_roi_response_cp_tbl_for_insights(client_code, model_group_id, data, metric_dist,
                                                                 roi_metrics, response_metrics, cost_per_metrics,
                                                                 driver_col, readout_data,
                                                                 metric_configs)
        other_related_tbl = build_roi_other_relationship_tbl_for_insights(client_code, model_group_id, data,
                                                                          metric_dist,
                                                                          roi_metrics, activity_metrics, driver_col,
                                                                          readout_data, metric_configs)
        cost_per_conversion_tbl = build_detail_tbl_for_insights(client_code, model_group_id, data, metric_dist,
                                                                cost_per_conversion_metric, driver_col, readout_data,
                                                                metric_configs)
        if not agg_data.empty:
            roi_spend_agg_tbl = build_roi_spend_tbl_for_insights(client_code, model_group_id, agg_data, metric_dist,
                                                                 roi_metrics, spend_metrics, driver_col, readout_data,
                                                                 metric_configs)
            roi_agg_tbl = build_detail_tbl_for_insights(client_code, model_group_id, agg_data, metric_dist,
                                                        roi_metrics, driver_col, readout_data, metric_configs)
            response_cp_agg_tbl = build_roi_response_cp_tbl_for_insights(client_code, model_group_id, agg_data, metric_dist,
                                                                         roi_metrics, response_metrics, cost_per_metrics,
                                                                         driver_col, readout_data, metric_configs)
            other_related_agg_tbl = build_roi_other_relationship_tbl_for_insights(client_code, model_group_id, agg_data,
                                                                                  metric_dist, roi_metrics, activity_metrics,
                                                                                  driver_col, readout_data, metric_configs)
            cost_per_conversion_agg_tbl = build_detail_tbl_for_insights(client_code, model_group_id, agg_data,
                                                                        metric_dist, cost_per_conversion_metric, driver_col,
                                                                        readout_data, metric_configs)
    elif time_tag == "trend":
        roi_spend_tbl = build_roi_spend_tbl_for_insights(client_code, model_group_id, data, metric_dist,
                                                         roi_metrics, spend_metrics, driver_col, readout_data,
                                                         metric_configs)
        roi_tbl = build_detail_tbl_for_insights(client_code, model_group_id, data, metric_dist,
                                                roi_metrics, driver_col, readout_data, metric_configs)
        cost_per_conversion_tbl = build_detail_tbl_for_insights(client_code, model_group_id, data, metric_dist,
                                                                cost_per_conversion_metric, driver_col, readout_data,
                                                                metric_configs)

        if not agg_data.empty:
            roi_spend_agg_tbl = build_roi_spend_tbl_for_insights(client_code, model_group_id, agg_data, metric_dist,
                                                             roi_metrics, spend_metrics, driver_col, readout_data,
                                                             metric_configs)
            roi_agg_tbl = build_detail_tbl_for_insights(client_code, model_group_id, agg_data, metric_dist,
                                                        roi_metrics, driver_col, readout_data, metric_configs)
            cost_per_conversion_agg_tbl = build_detail_tbl_for_insights(client_code, model_group_id, agg_data,
                                                                        metric_dist, cost_per_conversion_metric,
                                                                        driver_col, readout_data, metric_configs)

    # build output config
    output_config = {}
    if overall_tbl is not None or overall_agg_tbl is not None:
        config_name = tbl_naming.get("overall", "Overall ROI by Business Driver")
        config_instruction = instructions.get("overall", "Provide insights based on given data.").format(query=query)
        config_data = overall_tbl if overall_tbl is not None else overall_agg_tbl
        if not config_data.empty:
            output_config[config_name] = {"instruction": config_instruction,
                                          "data": config_data.to_json(orient="records")}
    if roi_agg_tbl is not None:
        config_name = tbl_naming.get("roi_single", "ROI by {level} Tactics").format(level="aggregated")
        config_instruction = instructions.get("roi_single", "Provide insights based on given data.").format(level="aggregated")
        config_data = roi_agg_tbl
        output_config[config_name] = {"instruction": config_instruction, "data": config_data.to_json(orient="records")}
    if roi_tbl is not None:
        config_name = tbl_naming.get("roi_single", "ROI by {level} Tactics").format(level="detailed")
        config_instruction = instructions.get("roi_single", "Provide insights based on given data.").format(level="detailed")
        config_data = roi_tbl
        output_config[config_name] = {"instruction": config_instruction, "data": config_data.to_json(orient="records")}
    if roi_spend_tbl is not None or roi_spend_agg_tbl is not None:
        config_name = tbl_naming.get("roi_spend", "ROI vs Spend")
        config_data = roi_spend_agg_tbl if roi_spend_agg_tbl is not None else roi_spend_tbl
        level = "aggregated" if roi_spend_agg_tbl is not None else "detailed"
        config_instruction = instructions.get("roi_spend", "Provide insights based on given data.").format(level=level)
        output_config[config_name] = {"instruction": config_instruction, "data": config_data.to_json(orient="records")}
    # if response_cp_agg_tbl is not None:
    #     config_name = "ROI vs Response & Cost per by Tactics"
    #     config_instruction = instructions.get("roi_response_cp", "Provide insights based on given data.").format(level="aggregated")
    #     config_data = response_cp_agg_tbl
    #     output_config[config_name] = {"instruction": config_instruction, "data": config_data.to_json(orient="records")}
    if response_cp_tbl is not None:
        config_name = tbl_naming.get("roi_response_cp", "ROI vs Response & Cost per by Tactics")
        config_instruction = instructions.get("roi_response_cp", "Provide insights based on given data.").format(level="detailed")
        config_data = response_cp_tbl
        output_config[config_name] = {"instruction": config_instruction, "data": config_data.to_json(orient="records")}
    if other_related_agg_tbl is not None:
        related_metrics_str = ", ".join(activity_metrics)
        config_name = tbl_naming.get("other_related", "ROI vs {related_metrics_str} by {level} Tactics").format(level="aggregated", related_metrics_str=related_metrics_str)
        config_instruction = instructions.get("other_related", "Provide insights based on given data.").format(
            level="aggregated")
        config_data = other_related_agg_tbl
        output_config[config_name] = {"instruction": config_instruction, "data": config_data.to_json(orient="records")}
    if other_related_tbl is not None:
        related_metrics_str = ", ".join(activity_metrics)
        config_name = tbl_naming.get("other_related", "ROI vs {related_metrics_str} by {level} Tactics").format(level="detailed", related_metrics_str=related_metrics_str)
        config_instruction = instructions.get("other_related", "Provide insights based on given data.").format(
            level="detailed")
        config_data = other_related_tbl
        output_config[config_name] = {"instruction": config_instruction, "data": config_data.to_json(orient="records")}
    if cost_per_conversion_agg_tbl is not None:
        config_name = tbl_naming.get("other_single", "Cost Per Conversion by {level} Tactics").format(level="aggregated")
        config_instruction = instructions.get("other_single", "Provide insights based on given data.").format(level="aggregated", metric="cost per conversion")
        config_data = cost_per_conversion_agg_tbl
        output_config[config_name] = {"instruction": config_instruction, "data": config_data.to_json(orient="records")}
    if cost_per_conversion_tbl is not None:
        config_name = tbl_naming.get("other_single", "Cost Per Conversion by {level} Tactics").format(level="detailed")
        config_instruction = instructions.get("other_single", "Provide insights based on given data.").format(level="detailed", metric="cost per conversion")
        config_data = cost_per_conversion_tbl
        output_config[config_name] = {"instruction": config_instruction, "data": config_data.to_json(orient="records")}
    return output_config


def build_detail_tbl_for_insights(client_code: str, model_group_id: int, data: pd.DataFrame, metric_dist: pd.DataFrame,
                                  metrics: list, driver_col: str, readout_data, metric_configs: pd.DataFrame):
    time_tag = readout_data.ner_filters.get("time_tag")
    level_type = readout_data.ner_filters.get("level")
    levels = level_type.get("general_level", []) + level_type.get("halo_level", [])

    level_mapping, level_mapping_reverse, level_value_mapping, level_value_reverse = utils.get_level_rename_mapping(
        client_code, model_group_id)
    level_source_cols = utils.resolve_level_columns(levels, data, level_mapping_reverse)
    selected_levels = {display: data[display].unique().tolist() for display in level_source_cols}
    concat_df = utils.concat_readout_frames(readout_data)

    metrics_alias = utils._find_metric_alias(metric_configs, metrics)

    if not any(x for x in metrics if x in concat_df["metric"].unique()):
        return None

    if metric_dist.empty or "time" not in metric_dist.columns:
        metric_dist = pd.DataFrame(columns=levels+["time", "metric", "q1", "q3", "time_norm"])
    else:
        metric_dist = utils.normalize_time(metric_dist)

    sorted_times = utils.sorted_times_from_columns(data.columns)

    pairs = list(product(metrics_alias.values(), sorted_times))
    metric_cols = []
    for pair in pairs:
        target_metric, target_time = pair[0], pair[1]
        pattern = re.compile(rf"^{re.escape(target_metric)}\s+{re.escape(str(target_time))}", re.IGNORECASE)
        matches = [x for x in data.columns if pattern.match(x)]
        metric_cols.extend(matches)

    tbl = data[list(selected_levels.keys()) + [driver_col] + metric_cols]
    cleaned_df = utils.strip_sign_and_unit(tbl, metric_cols, char="$")

    if time_tag in ["snapshot", "multiple_periods"]:
        concat_df = utils.concat_readout_frames(readout_data)
        concat_df = concat_df.merge(metric_dist)
        concat_df = concat_df.drop_duplicates(subset=levels + ["time", "metric"])
        for source in level_source_cols.values():
            if source in concat_df.columns:
                concat_df[source] = concat_df[source].map(level_value_mapping).fillna(concat_df[source])
        concat_df = concat_df.rename(columns={source: display for display, source in level_source_cols.items()})
        quantile_df = concat_df[list(selected_levels.keys()) + ["metric", "time", "q1", "q3", "time_norm"]]
        cleaned_df = label_metric_level_with_dist(cleaned_df, quantile_df, metrics_alias, sorted_times[0],
                                                  list(selected_levels.keys()))
    if time_tag == "period_over_period_change":
        for metric in metrics_alias.values():
            cleaned_df = utils.calculate_metric_change(cleaned_df, metric, sorted_times)
        concat_df = utils.concat_readout_frames(readout_data)
        concat_df = concat_df.merge(metric_dist)
        concat_df = concat_df.drop_duplicates(subset=levels + ["time", "metric"])
        for source in level_source_cols.values():
            if source in concat_df.columns:
                concat_df[source] = concat_df[source].map(level_value_mapping).fillna(concat_df[source])
        concat_df = concat_df.rename(columns={source: display for display, source in level_source_cols.items()})
        quantile_df = concat_df[list(selected_levels.keys()) + ["metric", "time", "q1", "q3", "time_norm"]]
        cleaned_df = label_metric_level_with_dist(cleaned_df, quantile_df, metrics_alias, sorted_times[0],
                                                  list(selected_levels.keys()))
    elif time_tag == "trend":
        for metric in metrics_alias.values():
            cleaned_df = utils.label_metric_trend(
                cleaned_df, metric, sorted_times, title_metric=True
            )
        concat_df = utils.concat_readout_frames(readout_data)
        concat_df = concat_df.merge(metric_dist)
        concat_df = concat_df.drop_duplicates(subset=levels + ["time", "metric"])
        for source in level_source_cols.values():
            if source in concat_df.columns:
                concat_df[source] = concat_df[source].map(level_value_mapping).fillna(concat_df[source])
        concat_df = concat_df.rename(columns={source: display for display, source in level_source_cols.items()})
        quantile_df = concat_df[list(selected_levels.keys()) + ["metric", "time", "q1", "q3", "time_norm"]]
        cleaned_df = label_metric_level_with_dist(cleaned_df, quantile_df, metrics_alias, sorted_times[0],
                                                  list(selected_levels.keys()))
    else:
        pass

    # add sign and unit and change header
    # for col in cleaned_df.columns:
    #     if (col in metric_cols) and ("trend" not in col) and ("%" not in col):
    #         cleaned_df[col] = cleaned_df[col].apply(lambda x: f"${x}" if pd.notna(x) and isinstance(x, numbers.Number) else x)
    cleaned_df = utils._format_insight_metric_columns(
        cleaned_df, metrics_alias, metric_configs
    )
    return cleaned_df


def _overall_display_aliases(metric_configs: pd.DataFrame, metrics_alias: dict) -> dict:
    """Map each lowercase metric alias to the cased alias used in table headers."""
    cased = {str(alias).strip().lower(): str(alias).strip()
             for alias in metric_configs["alias"].dropna()}
    return {alias: cased.get(alias, alias) for alias in metrics_alias.values()}


def _load_overall_roi(client_code: str, model_group_id: int, roi_metrics: list,
                      levels: list, readout_data) -> pd.DataFrame:
    """Long-format ROI rows from total.csv, scoped to the drivers and periods of the filtered readout."""
    concat_df = utils.concat_readout_frames(readout_data)
    return utils.filter_total_data(client_code, model_group_id,
                                   target_metric=roi_metrics,
                                   target_level=levels,
                                   data=concat_df.copy(),
                                   total="total")


def _pivot_overall_roi(total_roi: pd.DataFrame, levels: list, metric_headers: dict) -> pd.DataFrame:
    """Pivot the long ROI rows into one '{metric} {time}' column per metric and period."""
    total_roi = total_roi.assign(metric_header=total_roi["metric"].str.lower().map(metric_headers))
    pivot = total_roi.pivot_table(index=levels + ["driver"],
                                  columns=["metric_header", "time_norm"],
                                  values="value",
                                  aggfunc="first").reset_index()
    pivot.columns = [f"{metric} {time}".strip() if time else str(metric)
                     for metric, time in pivot.columns]
    return pivot


def _select_overall_times(pivot: pd.DataFrame, total_roi: pd.DataFrame, headers: list,
                          levels: list, time_tag: str) -> tuple:
    """Keep the periods this time_tag calls for, newest first, adding % change for the comparison tags."""
    sorted_times = sorted(total_roi["time_norm"].unique(), key=utils.time_sort_key, reverse=True)
    compare = time_tag in ["snapshot", "period_over_period_change"]
    output_times = sorted_times[:2] if compare else sorted_times

    time_cols = []
    for header in headers:
        value_cols = [f"{header} {time}" for time in output_times]
        value_cols = [col for col in value_cols if col in pivot.columns]
        time_cols.extend(value_cols)
        if compare and len(value_cols) == 2:
            change_col = f"% Change in {header} {output_times[0]}"
            pivot = utils.add_percent_change(pivot, value_cols[0], value_cols[1], change_col)
            time_cols.append(change_col)

    return pivot[levels + ["driver"] + time_cols], output_times


def _rename_overall_display(client_code: str, model_group_id: int,
                            pivot: pd.DataFrame, levels: list) -> pd.DataFrame:
    """Apply the client's level renaming to the dimension columns and their values."""
    _, level_mapping_reverse, level_value_mapping, _ = utils.get_level_rename_mapping(
        client_code, model_group_id)
    pivot = pivot.rename(columns=level_mapping_reverse)
    pivot = pivot.rename(columns={"driver": "Business Driver"})

    id_cols = [level_mapping_reverse.get(level, level) for level in levels] + ["Business Driver"]
    for col in (col for col in id_cols if col in pivot.columns):
        pivot[col] = pivot[col].map(level_value_mapping).fillna(pivot[col])
    return pivot


def _overall_quantile_df(client_code: str, model_group_id: int, pivot: pd.DataFrame,
                         metric_dist: pd.DataFrame, levels: list, readout_data) -> tuple:
    """Build the q1/q3 lookup keyed on the pivot's display level columns, plus those column names."""
    _, level_mapping_reverse, level_value_mapping, _ = utils.get_level_rename_mapping(
        client_code, model_group_id)
    level_source_cols = utils.resolve_level_columns(levels, pivot, level_mapping_reverse)

    concat_df = utils.concat_readout_frames(readout_data)
    concat_df = concat_df.merge(metric_dist)
    concat_df = concat_df.drop_duplicates(subset=levels + ["time", "metric"])
    for source in level_source_cols.values():
        if source in concat_df.columns:
            concat_df[source] = concat_df[source].map(level_value_mapping).fillna(concat_df[source])
    concat_df = concat_df.rename(columns={source: display for display, source in level_source_cols.items()})

    group_keys = list(level_source_cols.keys())
    quantile_df = concat_df[group_keys + ["metric", "time", "q1", "q3", "time_norm"]]
    return quantile_df, group_keys


def build_overall_tbl_for_insights(client_code: str, model_group_id: int, data: pd.DataFrame,
                                   metric_dist: pd.DataFrame, roi_metrics: list, readout_data,
                                   metric_configs: pd.DataFrame) -> pd.DataFrame:
    """
    Build the overall ROI table: the roi metric group read from total.csv at business-driver
    granularity, restricted to the drivers surviving the readout's own filtering.

    `data` is unused. Unlike the sibling builders this table is rebuilt from total.csv rather
    than from the pivoted wide table; the argument is kept so all builders share a call shape.
    """
    time_tag = readout_data.ner_filters.get("time_tag")
    level_type = readout_data.ner_filters.get("level")
    levels = level_type.get("general_level", []) + level_type.get("halo_level", [])

    metrics_alias = utils._find_metric_alias(metric_configs, roi_metrics)
    display_aliases = _overall_display_aliases(metric_configs, metrics_alias)
    metric_headers = {kpi: display_aliases[alias] for kpi, alias in metrics_alias.items()}

    total_roi = _load_overall_roi(client_code, model_group_id, roi_metrics, levels, readout_data)
    if total_roi.empty:
        return pd.DataFrame()

    pivot = _pivot_overall_roi(total_roi, levels, metric_headers)
    pivot, output_times = _select_overall_times(
        pivot, total_roi, list(metric_headers.values()), levels, time_tag)
    pivot = _rename_overall_display(client_code, model_group_id, pivot, levels)

    if time_tag == "trend":
        for alias in metrics_alias.values():
            pivot = utils.label_metric_trend(pivot, alias, output_times, title_metric=True)

    quantile_df, group_keys = _overall_quantile_df(
        client_code, model_group_id, pivot, metric_dist, levels, readout_data)
    pivot = label_metric_level_with_dist(pivot, quantile_df, metrics_alias, output_times[0], group_keys)

    return utils._format_insight_metric_columns(pivot, metrics_alias, metric_configs)


def build_roi_spend_tbl_for_insights(client_code: str, model_group_id: int,
                                     data: pd.DataFrame, metric_dist: pd.DataFrame,
                                     metrics1: list, metrics2: list, driver_col: str,
                                     readout_data, metric_configs: pd.DataFrame):
    """
    build relationship tbl between roi and spend
    :param client_code:
    :param model_group_id:
    :param data:
    :param readout_data:
    :return:
    """
    time_tag = readout_data.ner_filters.get("time_tag")
    level_type = readout_data.ner_filters.get("level")
    levels = level_type.get("general_level", []) + level_type.get("halo_level", [])

    sorted_times = utils.sorted_times_from_columns(data.columns)

    level_mapping, level_mapping_reverse, level_value_mapping, level_value_reverse = utils.get_level_rename_mapping(
        client_code, model_group_id)

    metrics1_alias = utils._find_metric_alias(metric_configs, metrics1)
    metrics2_alias = utils._find_metric_alias(metric_configs, metrics2)
    metrics_alias = metrics1_alias | metrics2_alias

    level_source_cols = utils.resolve_level_columns(levels, data, level_mapping_reverse)
    selected_levels = {display: data[display].unique().tolist() for display in level_source_cols}

    # process metric distribution df
    concat_df = utils.concat_readout_frames(readout_data)
    concat_df = concat_df.merge(metric_dist)
    concat_df = concat_df.drop_duplicates(subset=levels + ["time", "metric"])
    for source in level_source_cols.values():
        if source in concat_df.columns:
            concat_df[source] = concat_df[source].map(level_value_mapping).fillna(concat_df[source])
    concat_df = concat_df.rename(columns={source: display for display, source in level_source_cols.items()})
    quantile_df = concat_df[list(selected_levels.keys()) + ["metric", "time", "q1", "q3", "time_norm"]]

    # if none of roi metric is in filtered data
    if not any(x for x in metrics1 if x in concat_df["metric"].unique()):
        return None
    metric1_cols = [x for x in data.columns.tolist() if any(m in x.lower() for m in metrics1_alias.values()) and ("%" not in x)]
    metric2_cols = [x for x in data.columns.tolist() if any(m in x.lower() for m in metrics2_alias.values()) and ("%" not in x)]

    res_df = data[list(selected_levels.keys()) + [driver_col] + metric1_cols + metric2_cols]
    cleaned_df = utils.strip_sign_and_unit(res_df, metric1_cols + metric2_cols, char="$")

    # tbl config by time tag: metric level, trend label, % change
    if time_tag == "snapshot":
        cleaned_df = label_metric_level_with_dist(cleaned_df, quantile_df, metrics_alias,
                                                  sorted_times[0], list(selected_levels.keys()))
    elif time_tag == "period_over_period_change":
        for metric in metrics_alias.values():
            cleaned_df = utils.calculate_metric_change(
                cleaned_df, metric=metric, sorted_times=sorted_times
            )
        cleaned_df = label_metric_level_with_dist(cleaned_df, quantile_df, metrics_alias,
                                                  sorted_times[0], list(selected_levels.keys()))
    elif time_tag == "multiple_periods":
        cleaned_df = label_metric_level_with_dist(cleaned_df, quantile_df, metrics_alias,
                                                  sorted_times[0], list(selected_levels.keys()))
    elif time_tag == "trend":
        for metric in metrics_alias.values():
            cleaned_df = utils.calculate_metric_change(
                cleaned_df, metric=metric, sorted_times=sorted_times[:2]
            )
            cleaned_df = utils.label_metric_trend(
                cleaned_df, metric=metric, sorted_times=sorted_times,
                title_metric=True
            )
        cleaned_df = label_metric_level_with_dist(cleaned_df, quantile_df, metrics_alias,
                                                  sorted_times[0], list(selected_levels.keys()))
    else:
        pass

    # add sign and unit and change header
    # for col in cleaned_df.columns:
    #     if any(x in col.lower() for x in metrics1 + metrics2) and ("level" not in col) and ("% Change" not in col):
    #         cleaned_df[col] = cleaned_df[col].apply(lambda x: f"${x:,.2f}" if pd.notna(x) and isinstance(x, numbers.Number) else x)
    #     if any(x in col.lower() for x in metrics1 + metrics2) and ("level" not in col) and ("% Change" in col):
    #         cleaned_df[col] = cleaned_df[col].apply(lambda x: f"{np.round(x * 100, 2)}%" if pd.notna(x) and isinstance(x, numbers.Number) else x)
    change_cols = [
        col for col in cleaned_df.columns
        if "% change" in str(col).casefold()
        and any(metric in str(col).casefold() for metric in metrics1 + metrics2)
    ]
    cleaned_df = utils._format_insight_metric_columns(cleaned_df, metrics_alias, metric_configs)
    return cleaned_df


def build_roi_response_cp_tbl_for_insights(client_code: str, model_group_id: int,
                                            data: pd.DataFrame, metric_dist: pd.DataFrame,
                                            metrics1: list, metrics2: list, metrics3: list, driver_col: str,
                                            readout_data, metric_configs: pd.DataFrame):
    """
    build relationship tbl between roi and related metric: no trend/change/level label
    :param client_code:
    :param model_group_id:
    :param data:
    :param readout_data:
    :return:
    """
    time_tag = readout_data.ner_filters.get("time_tag")
    level_type = readout_data.ner_filters.get("level")
    levels = level_type.get("general_level", []) + level_type.get("halo_level", [])

    metrics1_alias = utils._find_metric_alias(metric_configs, metrics1)
    metrics2_alias = utils._find_metric_alias(metric_configs, metrics2)
    metrics3_alias = utils._find_metric_alias(metric_configs, metrics3)
    metrics_alias = metrics1_alias | metrics2_alias | metrics3_alias

    sorted_times = utils.sorted_times_from_columns(data.columns)

    level_mapping, level_mapping_reverse, level_value_mapping, level_value_reverse = utils.get_level_rename_mapping(
        client_code, model_group_id)

    level_source_cols = utils.resolve_level_columns(levels, data, level_mapping_reverse)
    selected_levels = {display: data[display].unique().tolist() for display in level_source_cols}

    # process metric distribution df
    concat_df = utils.concat_readout_frames(readout_data)
    concat_df = concat_df.merge(metric_dist)
    concat_df = concat_df.drop_duplicates(subset=levels + ["time", "metric"])
    for source in level_source_cols.values():
        if source in concat_df.columns:
            concat_df[source] = concat_df[source].map(level_value_mapping).fillna(concat_df[source])
    concat_df = concat_df.rename(columns={source: display for display, source in level_source_cols.items()})
    quantile_df = concat_df[list(selected_levels.keys()) + ["metric", "time", "q1", "q3", "time_norm"]]

    # if none of roi metric is in filtered data
    if not any(x for x in metrics1 if x in concat_df["metric"].unique()):
        return None
    metric1_cols = [x for x in data.columns.tolist() if any(m in x.lower() for m in metrics1_alias.values()) and ("%" not in x)]
    metric2_cols = [x for x in data.columns.tolist() if any(m in x.lower() for m in metrics2_alias.values()) and ("%" not in x)]
    metric3_cols = [x for x in data.columns.tolist() if any(m in x.lower() for m in metrics3_alias.values()) and ("%" not in x)]

    res_df = data[list(selected_levels.keys()) + [driver_col] + metric1_cols + metric2_cols + metric3_cols]
    cleaned_df = utils.strip_sign_and_unit(
        res_df, metric1_cols + metric2_cols + metric3_cols, char="$"
    )

    # tbl config by time tag: metric level, trend label, % change
    if time_tag == "snapshot":
        cleaned_df = label_metric_level_with_dist(cleaned_df, quantile_df, metrics_alias,
                                                  sorted_times[0], list(selected_levels.keys()))
    elif time_tag == "period_over_period_change":
        for metric in metrics_alias.values():
            cleaned_df = utils.calculate_metric_change(
                cleaned_df, metric=metric, sorted_times=sorted_times
            )
        cleaned_df = label_metric_level_with_dist(cleaned_df, quantile_df, metrics_alias,
                                                  sorted_times[0], list(selected_levels.keys()))
    elif time_tag == "multiple_periods":
        cleaned_df = label_metric_level_with_dist(cleaned_df, quantile_df, metrics_alias,
                                                  sorted_times[0], list(selected_levels.keys()))
    elif time_tag == "trend":
        for metric in metrics_alias.values():
            cleaned_df = utils.calculate_metric_change(
                cleaned_df, metric=metric, sorted_times=sorted_times[:2]
            )
        cleaned_df = label_metric_level_with_dist(cleaned_df, quantile_df, metrics_alias,
                                                  sorted_times[0], list(selected_levels.keys()))

    # add sign and unit and change header
    # for col in cleaned_df.columns:
    #     if any(x in col.lower() for x in metrics1 + metrics3) and ("level" not in col) and (
    #             "% Change" not in col):
    #         cleaned_df[col] = cleaned_df[col].apply(lambda x: f"${x:,.2f}" if pd.notna(x)  and isinstance(x, numbers.Number) else x)
    #     if any(x in col.lower() for x in metrics2) and ("level" not in col) and (
    #             "% Change" not in col):
    #         cleaned_df[col] = cleaned_df[col].apply(lambda x: f"{x:,.2f}" if pd.notna(x)  and isinstance(x, numbers.Number) else x)
    #     if any(x in col.lower() for x in metrics1 + metrics2 + metrics3) and ("level" not in col) and (
    #             "% Change" in col):
    #         cleaned_df[col] = cleaned_df[col].apply(lambda x: f"{np.round(x * 100, 2)}%" if pd.notna(x)  and isinstance(x, numbers.Number) else x)
    change_cols = [
        col for col in cleaned_df.columns
        if "% change" in str(col).casefold()
        and any(metric in str(col).casefold() for metric in metrics1 + metrics2 + metrics3)
    ]
    cleaned_df = utils._format_insight_metric_columns(cleaned_df, metrics_alias, metric_configs)

    return cleaned_df


def build_roi_other_relationship_tbl_for_insights(client_code: str, model_group_id: int,
                                            data: pd.DataFrame, metric_dist: pd.DataFrame,
                                            metrics1: list, metrics2: list, driver_col: str,
                                            readout_data, metric_configs:pd.DataFrame):
    """
    build relationship tbl between roi and related metric: no trend/change/level label
    :param client_code:
    :param model_group_id:
    :param data:
    :param readout_data:
    :return:
    """
    time_tag = readout_data.ner_filters.get("time_tag")
    level_type = readout_data.ner_filters.get("level")
    levels = level_type.get("general_level", []) + level_type.get("halo_level", [])

    metrics1_alias = utils._find_metric_alias(metric_configs, metrics1)
    metrics2_alias = utils._find_metric_alias(metric_configs, metrics2)
    metrics_alias = metrics1_alias | metrics2_alias

    sorted_times = utils.sorted_times_from_columns(data.columns)

    level_mapping, level_mapping_reverse, level_value_mapping, level_value_reverse = utils.get_level_rename_mapping(
        client_code, model_group_id)

    level_source_cols = utils.resolve_level_columns(levels, data, level_mapping_reverse)
    selected_levels = {display: data[display].unique().tolist() for display in level_source_cols}

    # process metric distribution df
    concat_df = utils.concat_readout_frames(readout_data)
    concat_df = concat_df.merge(metric_dist)
    concat_df = concat_df.drop_duplicates(subset=levels + ["time", "metric"])
    for source in level_source_cols.values():
        if source in concat_df.columns:
            concat_df[source] = concat_df[source].map(level_value_mapping).fillna(concat_df[source])
    concat_df = concat_df.rename(columns={source: display for display, source in level_source_cols.items()})
    quantile_df = concat_df[list(selected_levels.keys()) + ["metric", "time", "q1", "q3", "time_norm"]]

    # if none of roi metric is in filtered data
    if not any(x for x in metrics1 if x in concat_df["metric"].unique()):
        return None
    metric1_cols = [x for x in data.columns.tolist() if any(m in x.lower() for m in metrics1_alias.values()) and ("%" not in x)]
    metric2_cols = [x for x in data.columns.tolist() if any(m in x.lower() for m in metrics2_alias.values()) and ("%" not in x)]

    res_df = data[list(selected_levels.keys()) + [driver_col] + metric1_cols + metric2_cols]
    cleaned_df = utils.strip_sign_and_unit(res_df, metric1_cols + metric2_cols, char="$")

    # tbl config by time tag: metric level, trend label, % change
    if time_tag == "snapshot":
        cleaned_df = label_metric_level_with_dist(cleaned_df, quantile_df, metrics_alias,
                                                  sorted_times[0], list(selected_levels.keys()))
    elif time_tag == "period_over_period_change":
        for metric in metrics_alias.values():
            cleaned_df = utils.calculate_metric_change(
                cleaned_df, metric=metric, sorted_times=sorted_times
            )
        cleaned_df = label_metric_level_with_dist(cleaned_df, quantile_df, metrics_alias,
                                                  sorted_times[0], list(selected_levels.keys()))
    elif time_tag == "multiple_periods":
        cleaned_df = label_metric_level_with_dist(cleaned_df, quantile_df, metrics_alias,
                                                  sorted_times[0], list(selected_levels.keys()))
    elif time_tag == "trend":
        for metric in metrics_alias.values():
            cleaned_df = utils.label_metric_trend(
                cleaned_df, metric=metric, sorted_times=sorted_times,
                title_metric=True
            )
        cleaned_df = label_metric_level_with_dist(cleaned_df, quantile_df, metrics_alias,
                                                  sorted_times[0], list(selected_levels.keys()))
    else:
        pass

    # add sign and unit and change header
    # for col in cleaned_df.columns:
    #     if any(x in col.lower() for x in metrics1) and ("level" not in col) and (
    #             "% Change" not in col):
    #         cleaned_df[col] = cleaned_df[col].apply(lambda x: f"${x:,.2f}" if pd.notna(x)  and isinstance(x, numbers.Number) else x)
    #     if any(x in col.lower() for x in metrics2) and ("level" not in col) and (
    #             "% Change" not in col):
    #         cleaned_df[col] = cleaned_df[col].apply(lambda x: f"{x:,.2f}" if pd.notna(x)  and isinstance(x, numbers.Number) else x)
    #     if any(x in col.lower() for x in metrics1 + metrics2) and ("level" not in col) and (
    #             "% Change" in col):
    #         cleaned_df[col] = cleaned_df[col].apply(lambda x: f"{np.round(x * 100, 2)}%" if pd.notna(x)  and isinstance(x, numbers.Number) else x)
    change_cols = [
        col for col in cleaned_df.columns
        if "% change" in str(col).casefold()
        and any(metric in str(col).casefold() for metric in metrics1 + metrics2)
    ]
    cleaned_df = utils._format_insight_metric_columns(cleaned_df, metrics_alias, metric_configs)

    return cleaned_df
