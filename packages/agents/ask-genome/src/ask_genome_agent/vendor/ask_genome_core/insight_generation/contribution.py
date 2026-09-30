import os
import sys
import re
import numbers
import numpy as np
import pandas as pd

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
                                 target_time: str, group_keys: list, metric_cols_to_keep: list = None) -> pd.DataFrame:
    driver_col = "Business Driver"
    keep_cols = group_keys + [driver_col]  # hierarchy + driver, in order
    level_cols = []
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

        def _classify(row, target_col=target_col):
            raw = str(row[target_col])
            val = pd.to_numeric(raw.replace('%', '').replace(',', ''), errors='coerce')
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
        if metric_cols_to_keep is None:
            keep_cols += [target_col, level_col]
        else:
            level_cols.append(level_col)

    if metric_cols_to_keep is not None:
        keep_cols += [col for col in metric_cols_to_keep if col in pivot_data.columns]
        keep_cols += level_cols

    pivot_data = pivot_data[keep_cols]
    return pivot_data


def build_contribution_config_for_insights(client_code: str, model_group_id: int, query: str,
                                           data: pd.DataFrame, agg_data: pd.DataFrame,
                                           readout_data, metric_configs: pd.DataFrame) -> dict:
    """
    build contribution table for insights generation readout by time tag
    """
    output_config = {}
    if data.empty:
        return output_config

    #load prompt instructions for charting types
    prompt_instructions = readout_utils.load_insight_instructions(client_code, model_group_id, intention="contribution")
    metric_dist = readout_utils.load_metric_distribution(client_code, model_group_id, file_type="br")
    time_tag = readout_data.ner_filters.get("time_tag", "snapshot")
    level_type = readout_data.ner_filters.get("level")
    levels = level_type.get("general_level", []) + level_type.get("halo_level", [])
    if metric_dist.empty or "time" not in metric_dist.columns:
        metric_dist = pd.DataFrame(columns=levels+["time", "metric", "q1", "q3", "time_norm"])
    else:
        metric_dist = utils.normalize_time(metric_dist)

    # process instructions
    instructions, tbl_naming = utils._instruction_maps(prompt_instructions, time_tag)

    # process metric dist
    overall_tbl, fair_share_tbl, detail_tbl = None, None, None
    overall_agg_tbl, fair_share_agg_tbl, detail_agg_tbl = None, None, None

    if time_tag in ["snapshot", "period_over_period_change"]:
        overall_tbl = build_overall_tbl_for_insights(
            client_code, model_group_id, data, metric_dist, readout_data, metric_configs
        )
        fair_share_tbl = build_fair_share_tbl_for_insights(
            client_code, model_group_id, data, metric_dist, readout_data, metric_configs
        )
        detail_tbl = build_detail_tbl_for_insights(
            client_code, model_group_id, data, metric_dist, readout_data, metric_configs
        )
        if not agg_data.empty:
            overall_agg_tbl = build_overall_tbl_for_insights(
                client_code, model_group_id, agg_data, metric_dist, readout_data, metric_configs
            )
            fair_share_agg_tbl = build_fair_share_tbl_for_insights(
                client_code, model_group_id, agg_data, metric_dist, readout_data, metric_configs
            )
            detail_agg_tbl = build_detail_tbl_for_insights(
                client_code, model_group_id, agg_data, metric_dist, readout_data, metric_configs
            )
    elif time_tag in ["multiple_periods", "trend"]:
        overall_tbl = build_overall_tbl_for_insights(
            client_code, model_group_id, data, metric_dist, readout_data, metric_configs
        )
        detail_tbl = build_detail_tbl_for_insights(
            client_code, model_group_id, data, metric_dist, readout_data, metric_configs
        )
        if not agg_data.empty:
            overall_agg_tbl = build_overall_tbl_for_insights(client_code, model_group_id, agg_data, metric_dist,
                                                             readout_data, metric_configs)
            detail_agg_tbl = build_detail_tbl_for_insights(client_code, model_group_id, agg_data, metric_dist,
                                                           readout_data, metric_configs)

    # build output config
    if overall_tbl is not None or overall_agg_tbl is not None:
        config_name = tbl_naming.get("overall", "Total marketing contribution")
        config_instruction = instructions.get("overall", "Provide insights based on given data.").format(query=query)
        config_data = overall_tbl if overall_tbl is not None else overall_agg_tbl
        output_config[config_name] = {"instruction": config_instruction,
                                      "data": config_data.to_json(orient="records")}
    if detail_agg_tbl is not None:
        config_name = tbl_naming.get("detail", "Contribution by Marketing Drivers - Aggregated").format(level="aggregated")
        config_instruction = instructions.get("detail", "Provide insights based on given data.").format(
            level="aggregated", query=query)
        config_data = detail_agg_tbl
        output_config[config_name] = {"instruction": config_instruction,
                                      "data": config_data.to_json(orient="records")}
    if detail_tbl is not None:
        config_name = tbl_naming.get("detail", "Contribution by Marketing Drivers - Aggregated").format(level="detailed")
        config_instruction = instructions.get("detail", "Provide insights based on given data.").format(
            level="detailed", query=query)
        config_data = detail_tbl
        output_config[config_name] = {"instruction": config_instruction,
                                      "data": config_data.to_json(orient="records")}
    if fair_share_tbl is not None or fair_share_agg_tbl is not None:
        config_name = tbl_naming.get("fair_share", "Marketing Contribution Fair Share Analysis - Contribution Share vs. Spend Share")
        config_instruction = instructions.get("fair_share", "Provide insights based on given data.").format(query=query)
        config_data = fair_share_agg_tbl if fair_share_agg_tbl is not None else fair_share_tbl
        output_config[config_name] = {"instruction": config_instruction, "data": config_data.to_json(orient="records")}
    return output_config


def build_overall_tbl_for_insights(client_code: str, model_group_id: int, data: pd.DataFrame,
                                   metric_dist: pd.DataFrame, readout_data,
                                   metric_configs: pd.DataFrame) -> pd.DataFrame:
    """
    build overall contribution pct/amt table
    :param client_code:
    :param model_group_id:
    :param data:
    :param readout_data:
    :return:
    """
    time_tag = readout_data.ner_filters.get("time_tag")
    level_type = readout_data.ner_filters.get("level")
    levels = level_type.get("general_level", []) + level_type.get("halo_level", [])
    concat_df = utils.concat_readout_frames(readout_data)
    drivers = concat_df["business_driver"].dropna().unique()
    level_mapping, level_mapping_reverse, level_value_mapping, level_value_reverse = utils.get_level_rename_mapping(
        client_code, model_group_id)

    # load metrics
    contribution_metrics = list(
        metric_configs[metric_configs["metricGroup"].str.lower().isin(["contribution"])]["kpiName"].str.lower().values)
    kpi_metrics = list(
        metric_configs[metric_configs["metricGroup"].str.lower().isin(["kpi"])]["kpiName"].str.lower().values)

    metrics_alias = utils._find_metric_alias(metric_configs, contribution_metrics + kpi_metrics)

    # Load from total_viz table
    cont_tmp = utils.filter_total_data(client_code, model_group_id,
                                       target_metric=contribution_metrics,
                                       target_level=levels,
                                       data=concat_df)
    sales_tmp = utils.filter_total_data(client_code, model_group_id,
                                        target_metric=kpi_metrics,
                                        target_level=levels,
                                        data=concat_df)

    if cont_tmp.empty or sales_tmp.empty:
        return pd.DataFrame()

    # 1. Merge contribution and sales on shared dimension keys.
    #    sales has no driver split (all 'overall'), so driver/metric are excluded from the join.
    keys = levels + ['time', 'period_type', 'time_norm']

    merged = cont_tmp.rename(columns={'value': 'Contribution'}).merge(
        sales_tmp[keys + ['value']].rename(columns={'value': 'sales_value'}),
        on=keys,
        how='left',
    )

    # 2. Compute contribution amount (pct treated as a percentage).
    merged['Sales'] = merged['Contribution'] / 100 * merged['sales_value']
    output_metrics = ['Contribution', 'Sales']

    # 3. Pivot years from rows to columns, keeping both pct and amt.
    pivot_keys = levels + ['driver']

    merged = merged.pivot_table(
        index=pivot_keys,
        columns='time_norm',
        values=output_metrics,
        aggfunc='first',
    ).reset_index()

    # 4. Flatten the MultiIndex columns -> contribution_pct_2024, contribution_amt_2025, etc.
    merged.columns = [
        f'{a} {b}' if b != '' else a for a, b in merged.columns
    ]

    # 5. Order columns by time

    def period_key(s):
        """
        sort time str by latest
        :param s:
        :return:
        """
        parts = str(s).split()
        if len(parts) == 1:  # '2024' -> year only
            return (int(parts[0]), 0)
        p1, p2 = parts  # 'Q1 2023' -> quarter + year or 'H1 2023', 'M1 2023'
        return (int(p2), int(p1[1:]))

    sorted_times = sorted(cont_tmp['time_norm'].unique(), key=period_key, reverse=True)
    comparison_time_tags = ["snapshot", "period_over_period_change"]
    if time_tag in comparison_time_tags:
        output_times = sorted_times[:2]
        time_cols = []
        for metric in output_metrics:
            value_cols = [f'{metric} {time}' for time in output_times]
            time_cols.extend(value_cols)
            if len(value_cols) == 2 and all(col in merged.columns for col in value_cols):
                change_col = f'% Change in {metric} {output_times[0]}'
                merged = utils.add_percent_change(
                    merged, value_cols[0], value_cols[1], change_col
                )
                time_cols.append(change_col)
    else:
        output_times = sorted_times
        time_cols = [f'{m} {t}' for t in output_times for m in output_metrics]

    merged = merged[pivot_keys + time_cols]

    # add sign and unit and change header
    merged = merged.rename(columns=level_mapping_reverse)
    merged = merged.rename(columns={"driver": "Business Driver"})

    if time_tag == "trend":
        for metric in metrics_alias.values():
            merged = utils.label_metric_trend(merged, metric, sorted_times)
            merged = utils.label_metric_trend(merged, metric, sorted_times)

    for col in merged.columns:
        merged[col] = merged[col].map(level_value_mapping).fillna(merged[col])
        # if '% Change' in col:
        #     merged[col] = merged[col].apply(
        #         lambda x: f"{round(x * 100, 2)}%" if pd.notna(x) and isinstance(x, numbers.Number) else x)
        # elif 'Percentage' in col and "level" not in col and "trend" not in col:
        #     merged[col] = merged[col].apply(lambda x: f"{round(x, 2)}%" if pd.notna(x) and isinstance(x, numbers.Number) else x)
        # elif "Amount" in col and "level" not in col and "trend" not in col:
        #     merged[col] = merged[col].apply(lambda x: f"{x:,.2f}" if pd.notna(x) and isinstance(x, numbers.Number) else x)

    value_cols = [col for col in time_cols if "% Change" not in col]
    change_cols = [col for col in time_cols if "% Change" in col]
    merged = utils._format_insight_metric_columns(
        merged, metrics_alias, metric_configs
    )

    merged.columns = merged.columns.str.replace("Sales", "Contribution Amount", regex=False)
    return merged


def build_fair_share_tbl_for_insights(client_code: str, model_group_id: int, data: pd.DataFrame,
                                      metric_dist: pd.DataFrame, readout_data,
                                      metric_configs: pd.DataFrame) -> pd.DataFrame:
    """
    build contribution/spend share table
    assume data have both contribution share and spend share
    :param client_code:
    :param model_group_id:
    :param data:
    :param readout_data:
    :return:
    """
    driver_col = 'Business Driver'  # assume all have contribution share and spend share
    time_tag = readout_data.ner_filters.get("time_tag")
    level_type = readout_data.ner_filters.get("level")
    levels = level_type.get("general_level", []) + level_type.get("halo_level", [])
    concat_df = utils.concat_readout_frames(readout_data)
    contribution_metric = [x for x in concat_df["metric"].unique() if
                           x in readout_utils._find_metrics_by_group(metric_configs, "contribution", alias=True)]
    spend_metric = [x for x in concat_df["metric"].unique() if
                    x in readout_utils._find_metrics_by_group(metric_configs, "spend", alias=True)]
    contribution_share_metric = [x for x in concat_df["metric"].unique() if
                                 x in readout_utils._find_metrics_by_group(metric_configs, "contribution share", alias=False)]
    spend_share_metric = [x for x in concat_df["metric"].unique() if
                          x in readout_utils._find_metrics_by_group(metric_configs, "spend share", alias=False)]

    metrics = contribution_metric + spend_metric  # assume all have contribution share and spend share
    contribution_alias = utils._find_metric_alias(metric_configs, contribution_metric)
    spend_alias = utils._find_metric_alias(metric_configs, spend_metric)
    contribution_share_alias = utils._find_metric_alias(metric_configs, contribution_share_metric)
    spend_share_alias = utils._find_metric_alias(metric_configs, spend_share_metric)

    metrics_alias = contribution_alias | spend_alias | contribution_share_alias | spend_share_alias

    contribution_cols = [x for x in data.columns.tolist() if any(m for m in contribution_share_alias.values() if m in x.lower()) and ("%" not in x)]
    spend_share_cols = [x for x in data.columns.tolist() if any(m for m in spend_share_alias.values() if m in x.lower()) and ("%" not in x)]
    metric_cols = contribution_cols + spend_share_cols

    level_mapping, level_mapping_reverse, level_value_mapping, level_value_reverse = utils.get_level_rename_mapping(
        client_code, model_group_id)
    level_source_cols = utils.resolve_level_columns(levels, data, level_mapping_reverse)
    selected_levels = {display: data[display].unique().tolist() for display in level_source_cols}

    sorted_times = utils.sorted_times_from_columns(data.columns)

    res_df = data[list(selected_levels.keys()) + [driver_col] + metric_cols]
    cleaned_df = utils.strip_sign_and_unit(res_df, metric_cols, char="%")
    cleaned_df = cleaned_df.dropna(subset=spend_share_cols, how="all")
    # 1. snapshot: add HML to most recent time contrib and spend share
    # 2. multi-periods: add trend analysis to contrib and spend share
    if time_tag == "snapshot":
        # process metric distribution df
        concat_df = concat_df.merge(metric_dist)
        concat_df = concat_df.drop_duplicates(subset=levels + ["time", "metric"])
        for source in level_source_cols.values():
            if source in concat_df.columns:
                concat_df[source] = concat_df[source].map(level_value_mapping).fillna(concat_df[source])
        concat_df = concat_df.rename(columns={source: display for display, source in level_source_cols.items()})
        quantile_df = concat_df[list(selected_levels.keys()) + ["metric", "time", "q1", "q3", "time_norm"]]

        # cleaned_df = label_metric_level(cleaned_df, "Contribution Share", sorted_times[-1])
        # cleaned_df = label_metric_level(cleaned_df, "Spend Share", sorted_times[-1])
        cleaned_df = label_metric_level_with_dist(cleaned_df, quantile_df, metrics_alias,
                                                  sorted_times[0], list(selected_levels.keys()),
                                                  metric_cols_to_keep=metric_cols)
    else:
        for metric in metrics_alias.values():
            cleaned_df = utils.label_metric_trend(cleaned_df, metric, sorted_times)
            cleaned_df = utils.label_metric_trend(cleaned_df, metric, sorted_times)

    # add sign and unit and change header
    # for col in cleaned_df.columns:
    #     if 'Share' in col and "level" not in col and "trend" not in col:
    #         cleaned_df[col] = cleaned_df[col].apply(lambda x: f"{round(x, 2)}%" if pd.notna(x) and isinstance(x, numbers.Number) else x)
    cleaned_df = utils._format_insight_metric_columns(
        cleaned_df, metrics_alias, metric_configs
    )
    return cleaned_df


def build_detail_tbl_for_insights(client_code: str, model_group_id: int, data: pd.DataFrame,
                                  metric_dist: pd.DataFrame, readout_data,
                                  metric_configs: pd.DataFrame) -> pd.DataFrame:
    """
    build detail contribution pct/amt table
    :param client_code:
    :param model_group_id:
    :param data:
    :param readout_data:
    :return:
    """
    driver_col = 'Business Driver'
    contribution_metrics = readout_utils._find_metrics_by_group(metric_configs, "Contribution", alias=True)
    sales_metrics = readout_utils._find_metrics_by_group(metric_configs, "KPI", alias=True)
    metrics = contribution_metrics + sales_metrics  # assume all have contribution share and spend share
    time_tag = readout_data.ner_filters.get("time_tag")
    level_type = readout_data.ner_filters.get("level")
    levels = level_type.get("general_level", []) + level_type.get("halo_level", [])

    period = r"(?:(?:Q[1-4]|H[1-2]|M(?:1[0-2]|[1-9])) )?\d{4}"

    contribution_metrics = list(
        metric_configs[metric_configs["metricGroup"].str.lower().isin(["contribution"])]["kpiName"].str.lower().values)
    kpi_metrics = list(
        metric_configs[metric_configs["metricGroup"].str.lower().isin(["kpi"])]["kpiName"].str.lower().values)
    contribution_share_metrics = list(
        metric_configs[metric_configs["metricGroup"].str.lower().isin(["contribution share"])]["kpiName"].str.lower().values)

    contribution_alias = utils._find_metric_alias(metric_configs, contribution_metrics)
    kpi_alias = utils._find_metric_alias(metric_configs, kpi_metrics)
    contribution_share_alias = utils._find_metric_alias(metric_configs, contribution_share_metrics)
    metrics_alias = contribution_alias | kpi_alias | contribution_share_alias

    contribution_cols = [x for x in data.columns.tolist() if
                         any(m for m in contribution_alias.values() if m in x.lower()) and ("%" not in x)]
    amount_cols = [x for x in data.columns.tolist() if
                         any(m for m in kpi_alias.values() if m in x.lower()) and ("%" not in x) and (x not in contribution_cols)]
    share_cols = [x for x in data.columns.tolist() if
                   any(m for m in contribution_share_alias.values() if m in x.lower()) and ("%" not in x)]
    contribution_change_cols = [x for x in data.columns.tolist() if
                         any(m for m in contribution_alias.values() if m in x.lower()) and ("%" in x)]
    amount_change_cols = [x for x in data.columns.tolist() if
                   any(m for m in kpi_alias.values() if m in x.lower()) and ("%" in x)]

    # column name and value mapping
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

    sorted_times = utils.sorted_times_from_columns(data.columns)

    if time_tag in ["snapshot", "period_over_period_change"]:
        comparison_times = sorted_times[:2]
        columns_by_lower = {str(col).lower(): col for col in data.columns}

        def comparison_block(metric_name) -> list:
            value_cols = [columns_by_lower.get(f'{metric_name} {time}'.lower()) for time in comparison_times]
            value_cols = [col for col in value_cols if col is not None]
            if len(comparison_times) == 2 and len(value_cols) == 2:
                change_col = columns_by_lower.get(
                    f'% Change in {metric_name} {comparison_times[0]}'.lower())
                if change_col is not None:
                    value_cols.append(change_col)
            return value_cols

        contribution_metric_cols = []
        amount_metric_cols = []
        for metric in contribution_alias.values():
            contribution_metric_cols += comparison_block(metric)
        for metric in kpi_alias.values():
            amount_metric_cols += comparison_block(metric)
        metric_cols = contribution_metric_cols + amount_metric_cols
        contribution_value_cols = [col for col in contribution_metric_cols if "% Change" not in col]
        amount_value_cols = [col for col in amount_metric_cols if "% Change" not in col]
    else:
        metric_cols = contribution_cols + amount_cols
        contribution_value_cols = contribution_cols
        amount_value_cols = amount_cols

    res_df = data[list(selected_levels.keys()) + [driver_col] + metric_cols]
    cleaned_df = utils.strip_sign_and_unit(
        res_df, contribution_value_cols, char="%"
    )
    cleaned_df = utils.strip_sign_and_unit(cleaned_df, amount_value_cols, char=",")

    # 1. snapshot: add HML to most recent time contrib and spend share
    # 2. multi-periods: add trend analysis to contrib and spend share
    if time_tag == "snapshot":
        # cleaned_df = label_metric_level(cleaned_df, "Contribution", sorted_times[-1])
        # cleaned_df = label_metric_level(cleaned_df, "Sales", sorted_times[-1])
        cleaned_df = label_metric_level_with_dist(cleaned_df, quantile_df,
                                                  contribution_alias | kpi_alias,
                                                  sorted_times[0], list(selected_levels.keys()),
                                                  metric_cols_to_keep=metric_cols)
    elif time_tag == "trend":
        # cleaned_df = label_metric_level(cleaned_df, "Contribution Share", sorted_times[-1])
        # cleaned_df = label_metric_level(cleaned_df, "Sales", sorted_times[-1])
        for metric in contribution_alias.values():
            cleaned_df = utils.label_metric_trend(cleaned_df, metric, sorted_times)
        # cleaned_df = label_metric_trend(cleaned_df, "Contribution Amount")
        # cleaned_df = cleaned_df[list(selected_levels.keys()) + [driver_col] + contribution_value_cols + ["Contribution Percentage trend"]]
    else:
        for metric in contribution_alias.values():
            cleaned_df = utils.label_metric_trend(cleaned_df, metric, sorted_times)
        for metric in kpi_alias.values():
            cleaned_df = utils.label_metric_trend(cleaned_df, metric, sorted_times)

    # add sign and unit and change header
    # for col in [x for x in contribution_value_cols + share_cols if x in cleaned_df.columns]:
    #     cleaned_df[col] = cleaned_df[col].apply(lambda x: f"{round(x, 2)}%" if pd.notna(x) and isinstance(x, numbers.Number)  else x)
    # for col in [x for x in amount_value_cols if x in cleaned_df.columns]:
    #     cleaned_df[col] = cleaned_df[col].apply(lambda x: f"{x:,.2f}" if pd.notna(x) and isinstance(x, numbers.Number)  else x)
    value_cols = [
        col for col in contribution_value_cols + share_cols + amount_value_cols
        if col in cleaned_df.columns
    ]
    cleaned_df = utils._format_insight_metric_columns(
        cleaned_df, metrics_alias, metric_configs
    )
    cleaned_df.columns = [
        re.sub("sales", "Contribution Amount", c, flags=re.IGNORECASE)
        if c in amount_change_cols + amount_cols else c
        for c in cleaned_df.columns
    ]
    return cleaned_df
