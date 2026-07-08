import os
import sys
import re
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import src.insight_generation.utils as utils
import src.model.readout_utils as readout_utils

def strip_sign_and_unit(df: pd.DataFrame, cols: list, char: str):
    """
    remove added sign and unit in tbl display
    :param df:
    :param cols:
    :param char:
    :return:
    """
    for col in cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(char, "", regex=False)
            .replace({"nan": np.nan, "None": np.nan, "": np.nan})
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


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


def label_metric_level_with_dist(data: pd.DataFrame, quantile_df: pd.DataFrame, target_metrics: list, target_time: str, group_keys: list) -> pd.DataFrame:
    driver_col = "Business Driver"
    keep_cols = group_keys + [driver_col]  # hierarchy + driver, in order
    pivot_data = data.copy()

    # find the target metric column from

    for target_metric in target_metrics:
        pattern = re.compile(rf"^{re.escape(target_metric)}\s+{re.escape(str(target_time))}$", re.IGNORECASE)
        matches = [x for x in pivot_data.columns if pattern.match(x)]
        if not matches:
            continue
        target_col = matches[0]
        level_col = f'{target_metric} level {target_time}'

        q_df = quantile_df[
            (quantile_df['metric'].str.lower() == target_metric.lower())
            & (quantile_df['time'].astype(str).str.contains(str(target_time)))
            ]
        q_lookup = q_df.set_index(group_keys)[['q1', 'q3']]

        def _classify(row, target_col=target_col):
            raw = str(row[target_col])
            val = pd.to_numeric(raw.replace('%', '').replace(',', ''), errors='coerce')
            if pd.isna(val):
                return np.nan
            try:
                q1, q3 = q_lookup.loc[tuple(row[k] for k in group_keys)]
            except KeyError:
                return np.nan
            if '%' in raw:
                val = val / 100
            if val < q1:
                return 'low'
            if val > q3:
                return 'high'
            return 'moderate'

        pivot_data[level_col] = pivot_data.apply(_classify, axis=1)
        keep_cols += [target_col, level_col]

    pivot_data = pivot_data[keep_cols]
    return pivot_data


def label_metric_trend(df: pd.DataFrame, metric: str, spike_ratio=4.0, volatile_ratio=1.5, flat_multiplier=0.06, sign_ratio=0.8):
    metric_cols = [x for x in df.columns.tolist() if (metric.lower() in x.lower()) and ("%" not in x) and ('level' not in x) and ('trend' not in x)]

    def classify_row(row):
        values = np.array(row[metric_cols], dtype=np.float64)
        values = values[~np.isnan(values)]
        if len(values) < 3:
            return ""

        mean = np.mean(values)
        std = np.std(values)
        all_mean_abs = np.mean(np.abs(values))
        max_deviation_ratio = np.max(np.abs(values - mean)) / (abs(mean) + 1e-6)

        ordered = values[::-1]
        rho, p_value = spearmanr(range(len(ordered)), ordered)

        flat_threshold = flat_multiplier * (all_mean_abs + 1e-6)
        volatile_threshold = 0.35 * (all_mean_abs + 1e-6)

        if max_deviation_ratio > spike_ratio:
            return 'spiking'
        if std > volatile_threshold:
            return 'volatile'
        if p_value < 0.05 and rho > 0:
            return 'consistently increasing'
        if p_value < 0.05 and rho < 0:
            return 'consistently decreasing'
        if std < flat_threshold:
            return 'flat'
        return 'mild fluctuated (increase)' if ordered[-1] > ordered[0] else 'mild fluctuated (decrease)'

    df[f'{metric} trend'] = df.apply(classify_row, axis=1)
    return df


def build_contribution_tbl_for_insights(client_code: str, model_group_id: int, data: pd.DataFrame, agg_data: pd.DataFrame, readout_data=None) -> dict:
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

    # process instructions
    if not prompt_instructions.empty:
        instructions = prompt_instructions[prompt_instructions["timeTag"] == time_tag].set_index("configName")["instruction"].to_dict()
    else:
        instructions = {}

    # process metric dist
    if metric_dist.empty:
        metric_dist = pd.DataFrame(columns=levels+["time", "metric", "q1", "q3"])

    overall_tbl, fair_share_tbl, detail_tbl = None, None, None
    overall_agg_tbl, fair_share_agg_tbl, detail_agg_tbl = None, None, None

    if time_tag in ["snapshot", "period_over_period_change"]:
        overall_tbl = build_overall_tbl_for_insights(client_code, model_group_id, data, metric_dist, readout_data)
        fair_share_tbl = build_fair_share_tbl_for_insights(client_code, model_group_id, data, metric_dist, readout_data)
        detail_tbl = build_detail_tbl_for_insights(client_code, model_group_id, data, metric_dist, readout_data)
        if not agg_data.empty:
            overall_agg_tbl = build_overall_tbl_for_insights(client_code, model_group_id, agg_data, metric_dist, readout_data)
            fair_share_agg_tbl = build_fair_share_tbl_for_insights(client_code, model_group_id, agg_data, metric_dist, readout_data)
            detail_agg_tbl = build_detail_tbl_for_insights(client_code, model_group_id, agg_data, metric_dist, readout_data)
    elif time_tag in ["multiple_periods", "trend"]:
        overall_tbl = build_overall_tbl_for_insights(client_code, model_group_id, data, metric_dist, readout_data)
        detail_tbl = build_detail_tbl_for_insights(client_code, model_group_id, data, metric_dist, readout_data)
        if not agg_data.empty:
            overall_agg_tbl = build_overall_tbl_for_insights(client_code, model_group_id, agg_data, metric_dist,
                                                             readout_data)
            detail_agg_tbl = build_detail_tbl_for_insights(client_code, model_group_id, agg_data, metric_dist,
                                                           readout_data)

    # build output config
    if overall_tbl is not None or overall_agg_tbl is not None:
        config_name = "Contribution by Marketing Drivers"
        config_instruction = instructions.get("overall", "Provide insights based on given data.")
        config_data = overall_tbl if overall_tbl is not None else overall_agg_tbl
        output_config[config_name] = {"instruction": config_instruction, "data": config_data.to_json(orient="records")}
    if detail_agg_tbl is not None:
        config_name = "Contribution by Aggregated Drivers"
        config_instruction = instructions.get("detail", "Provide insights based on given data.").format(level="aggregated")
        config_data = detail_tbl
        output_config[config_name] = {"instruction": config_instruction, "data": config_data.to_json(orient="records")}
    if detail_tbl is not None:
        config_name = "Contribution by Drivers"
        config_instruction = instructions.get("detail", "Provide insights based on given data.").format(level="granular")
        config_data = detail_tbl
        output_config[config_name] = {"instruction": config_instruction, "data": config_data.to_json(orient="records")}
    if fair_share_tbl is not None or fair_share_agg_tbl is not None:
        config_name = "Marketing Contribution Fair Share Analysis - Contribution Share vs. Spend Share"
        config_instruction = instructions.get("fair_share", "Provide insights based on given data.")
        config_data = fair_share_agg_tbl if fair_share_agg_tbl is not None else fair_share_tbl
        output_config[config_name] = {"instruction": config_instruction, "data": config_data.to_json(orient="records")}
    return output_config


def build_overall_tbl_for_insights(client_code: str, model_group_id: int, data: pd.DataFrame, metric_dist: pd.DataFrame, readout_data=None) -> pd.DataFrame:
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
    concat_df = pd.concat([readout_data.df_activity_group,
                           readout_data.df_measure_group,
                           readout_data.df_measure])
    drivers = concat_df["business_driver"].dropna().unique()
    level_mapping, level_mapping_reverse, level_value_mapping, level_value_reverse = utils.get_level_rename_mapping(
        client_code, model_group_id)

    # Load from total_viz table
    cont_tmp = utils.filter_total_data(client_code, model_group_id,
                                       target_metric=["contribution"],
                                       target_level=levels,
                                       data=concat_df)
    sales_tmp = utils.filter_total_data(client_code, model_group_id,
                                        target_metric=["sales"],
                                        target_level=levels,
                                        data=concat_df)

    if cont_tmp.empty or sales_tmp.empty:
        return pd.DataFrame()

    # 1. Merge contribution and sales on shared dimension keys.
    #    sales has no driver split (all 'overall'), so driver/metric are excluded from the join.
    keys = levels + ['time', 'period_type', 'time_norm']

    merged = cont_tmp.rename(columns={'value': 'Contribution Percentage'}).merge(
        sales_tmp[keys + ['value']].rename(columns={'value': 'sales_value'}),
        on=keys,
        how='left',
    )

    # 2. Compute contribution amount (pct treated as a percentage).
    merged['Contribution Amount'] = merged['Contribution Percentage'] / 100 * merged['sales_value']
    output_metrics = ['Contribution Percentage', 'Contribution Amount']

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
    if time_tag == "snapshot":
        output_times = [sorted_times[0]]
    else:
        output_times = sorted_times

    time_cols = [f'{m} {t}' for t in output_times for m in output_metrics]
    merged = merged[pivot_keys + time_cols]

    # add sign and unit and change header
    merged = merged.rename(columns=level_mapping_reverse)
    merged = merged.rename(columns={"driver": "Business Driver"})

    if time_tag == "trend":
        merged = label_metric_trend(merged, "Contribution Percentage")
        merged = label_metric_trend(merged, "Contribution Amount")

    for col in merged.columns:
        merged[col] = merged[col].map(level_value_mapping).fillna(merged[col])
        if 'Percentage' in col and "level" not in col and "trend" not in col:
            merged[col] = merged[col].apply(lambda x: f"{round(x, 2)}%" if pd.notna(x) else x)

    return merged


def build_fair_share_tbl_for_insights(client_code: str, model_group_id: int, data: pd.DataFrame, metric_dist: pd.DataFrame, readout_data=None) -> pd.DataFrame:
    """
    build contribution/spend share table
    assume data have both contribution share and spend share
    :param client_code:
    :param model_group_id:
    :param data:
    :param readout_data:
    :return:
    """
    driver_col = 'Business Driver'
    metrics = ["contribution share", "spend share"] # assume all have contribution share and spend share
    time_tag = readout_data.ner_filters.get("time_tag")
    level_type = readout_data.ner_filters.get("level")
    levels = level_type.get("general_level", []) + level_type.get("halo_level", [])

    level_mapping, level_mapping_reverse, level_value_mapping, level_value_reverse = utils.get_level_rename_mapping(client_code, model_group_id)
    if level_mapping_reverse:
        selected_levels =  {level_mapping_reverse[k]: data[level_mapping_reverse[k]].unique().tolist()
                            for k in levels if level_mapping_reverse[k] in data.columns}
    else:
        # for not mapped levels in formatted data
        selected_levels = {k: data[k].unique().tolist() for k in levels if k in data.columns}

    metric_cols = [x for x in data.columns.tolist() if any(m in x.lower() for m in metrics) and ("%" not in x)]
    sorted_times = sorted(set([re.search(r"(Q[1-4]|H[1-2]|M(?:1[0-2]|[1-9])) \d{4}|\d{4}", x).group()
                               for x in metric_cols
                               if re.search(r"(Q[1-4]|H[1-2]|M(?:1[0-2]|[1-9])) \d{4}|\d{4}", x)]))

    res_df = data[list(selected_levels.keys()) + [driver_col] + metric_cols]
    cleaned_df = strip_sign_and_unit(res_df, metric_cols, char="%")

    # 1. snapshot: add HML to most recent time contrib and spend share
    # 2. multi-periods: add trend analysis to contrib and spend share
    if time_tag == "snapshot":
        # process metric distribution df
        concat_df = pd.concat([readout_data.df_activity_group, readout_data.df_measure_group, readout_data.df_measure])
        concat_df = concat_df.merge(metric_dist)
        concat_df = concat_df.drop_duplicates(subset=levels + ["time", "metric"])
        for selected_level in selected_levels:
            concat_df[level_mapping[selected_level]] = concat_df[level_mapping[selected_level]].map(
                level_value_mapping).fillna(concat_df[level_mapping[selected_level]])
        concat_df = concat_df.rename(columns=level_mapping_reverse)
        quantile_df = concat_df[list(selected_levels.keys()) + ["metric", "time", "q1", "q3"]]

        # cleaned_df = label_metric_level(cleaned_df, "Contribution Share", sorted_times[-1])
        # cleaned_df = label_metric_level(cleaned_df, "Spend Share", sorted_times[-1])
        cleaned_df = label_metric_level_with_dist(cleaned_df, quantile_df,
                                                  ["Contribution Share", "Spend Share"],
                                                  sorted_times[-1], list(selected_levels.keys()))
    else:
        cleaned_df = label_metric_trend(cleaned_df, "Contribution Share")
        cleaned_df = label_metric_trend(cleaned_df, "Spend Share")

    # add sign and unit and change header
    for col in cleaned_df.columns:
        if 'Share' in col and "level" not in col and "trend" not in col:
            cleaned_df[col] = cleaned_df[col].apply(lambda x: f"{round(x, 2)}%" if pd.notna(x) else x)
    return cleaned_df


def build_detail_tbl_for_insights(client_code: str, model_group_id: int, data: pd.DataFrame, metric_dist: pd.DataFrame, readout_data=None) -> pd.DataFrame:
    """
    build detail contribution pct/amt table
    :param client_code:
    :param model_group_id:
    :param data:
    :param readout_data:
    :return:
    """
    driver_col = 'Business Driver'
    metrics = ["contribution", "contribution amount"]  # assume all have contribution share and spend share
    time_tag = readout_data.ner_filters.get("time_tag")
    level_type = readout_data.ner_filters.get("level")
    levels = level_type.get("general_level", []) + level_type.get("halo_level", [])

    period = r"(?:(?:Q[1-4]|H[1-2]|M(?:1[0-2]|[1-9])) )?\d{4}"

    contribution_cols = [c for c in data.columns if re.match(rf"^Contribution {period}$", c)]
    amount_cols = [c for c in data.columns if re.match(rf"^Contribution Amount {period}$", c)]
    share_cols = [c for c in data.columns if re.match(rf"^Contribution Share {period}$", c)]

    data = data.rename(columns={
        c: re.sub("Contribution", "Contribution Percentage", c, flags=re.IGNORECASE)
        for c in contribution_cols
    })
    # update contribution cols
    contribution_cols = [c for c in data.columns if re.match(rf"^Contribution Percentage {period}$", c)]

    # column name and value mapping
    level_mapping, level_mapping_reverse, level_value_mapping, level_value_reverse = utils.get_level_rename_mapping(
        client_code, model_group_id)
    if level_mapping_reverse:
        selected_levels = {level_mapping_reverse[k]: data[level_mapping_reverse[k]].unique().tolist()
                           for k in levels if level_mapping_reverse[k] in data.columns}
    else:
        # for not mapped levels in formatted data
        selected_levels = {k: data[k].unique().tolist() for k in levels if k in data.columns}

    # process metric distribution df
    concat_df = pd.concat([readout_data.df_activity_group, readout_data.df_measure_group, readout_data.df_measure])
    concat_df = concat_df.merge(metric_dist)
    concat_df = concat_df.drop_duplicates(subset=levels + ["time", "metric"])
    for selected_level in selected_levels:
        concat_df[level_mapping[selected_level]] = concat_df[level_mapping[selected_level]].map(
            level_value_mapping).fillna(concat_df[level_mapping[selected_level]])
    concat_df = concat_df.rename(columns=level_mapping_reverse)
    quantile_df = concat_df[list(selected_levels.keys()) + ["metric", "time", "q1", "q3"]]

    sorted_times = sorted(set([re.search(r"(Q[1-4]|H[1-2]|M(?:1[0-2]|[1-9])) \d{4}|\d{4}", x).group()
                               for x in contribution_cols + amount_cols + share_cols
                               if re.search(r"(Q[1-4]|H[1-2]|M(?:1[0-2]|[1-9])) \d{4}|\d{4}", x)]))

    res_df = data[list(selected_levels.keys()) + [driver_col] + contribution_cols + amount_cols]
    cleaned_df = strip_sign_and_unit(res_df, contribution_cols, char="%")
    cleaned_df = strip_sign_and_unit(cleaned_df, amount_cols, char=",")

    # 1. snapshot: add HML to most recent time contrib and spend share
    # 2. multi-periods: add trend analysis to contrib and spend share
    if time_tag == "snapshot":
        # cleaned_df = label_metric_level(cleaned_df, "Contribution", sorted_times[-1])
        # cleaned_df = label_metric_level(cleaned_df, "Sales", sorted_times[-1])
        cleaned_df = label_metric_level_with_dist(cleaned_df, quantile_df,
                                                  ["Contribution Percentage", "Contribution Amount"],
                                                  sorted_times[-1], list(selected_levels.keys()))
    elif time_tag == "trend":
        # cleaned_df = label_metric_level(cleaned_df, "Contribution Share", sorted_times[-1])
        # cleaned_df = label_metric_level(cleaned_df, "Sales", sorted_times[-1])
        cleaned_df = label_metric_trend(cleaned_df, "Contribution Percentage")
        cleaned_df = label_metric_trend(cleaned_df, "Contribution Amount")
    else:
        cleaned_df = label_metric_trend(cleaned_df, "Contribution Percentage")
        cleaned_df = label_metric_trend(cleaned_df, "Contribution Amount")

    # add sign and unit and change header
    for col in [x for x in contribution_cols + share_cols if x in cleaned_df.columns]:
        cleaned_df[col] = cleaned_df[col].apply(lambda x: f"{round(x, 2)}%" if pd.notna(x) else x)
    cleaned_df.columns = [
        re.sub("sales", "Contribution Amount", c, flags=re.IGNORECASE)
        if "sales" in c.lower() else c
        for c in cleaned_df.columns
    ]
    return cleaned_df
