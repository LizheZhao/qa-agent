import os
import sys
import re
import calendar
import numpy as np
import pandas as pd
import src.model.readout_utils as readout_utils


def normalize_time(df: pd.DataFrame):
    """
    normalize time for table display
    :param df:
    :return:
    """
    # normalize time
    df["time_norm"] = df["time"].str.replace(
        r'.*?(year|quarter|half|month)\s*(\d{0,2})\D*(\d{4}).*',
        lambda m: {
            'year': m.group(3),
            'quarter': f"Q{m.group(2)} {m.group(3)}",
            'half': f"H{m.group(2)} {m.group(3)}",
            'month': f"M{int(m.group(2) or 0)} {m.group(3)}"
        }[m.group(1)], regex=True)
    return df


def get_time_align_dict(sorted_time: list):
    """Map internal time tokens (e.g. 'M6 2023') to display labels (e.g. 'June 2023').
    Returns a dict of {display_label: internal_token}."""
    month_mapping = {f"M{i}": calendar.month_name[i] for i in range(1, 13)}
    month_mapping.update({f"Month {i}": calendar.month_name[i] for i in range(1, 13)})
    time_mapping = {}
    for t in sorted_time:
        time_mapping[re.sub(r"(M\d{1,2}|Month \d{1,2})", lambda m: month_mapping[m.group()], t)] = t
    if not time_mapping:
        time_mapping = {t:t for t in sorted_time}
    return time_mapping


def get_level_rename_mapping(client_code: str, model_group_id: int) -> tuple[dict, dict, dict, dict]:
    """
    Returns mappings between display and original level names and values.
    :param client_code:
    :param model_group_id:
    :return:
    """
    level_rename_df = readout_utils.load_level_rename_display(client_code, model_group_id)
    if level_rename_df.empty:
        level_mapping = dict()
        level_mapping_reverse = dict()
        level_value_mapping = dict()
        level_value_reverse = dict()
    else:
        level_mapping = (
            level_rename_df.loc[level_rename_df["mapped"].notna(), ["original", "mapped"]].set_index("mapped")[
                "original"].to_dict())
        level_mapping_reverse = (
            level_rename_df.loc[level_rename_df["mapped"].notna(), ["original", "mapped"]].set_index("original")[
                "mapped"].to_dict())
        level_value_mapping = (
            level_rename_df.loc[level_rename_df["mapped"].notna(), ["original_value", "mapped_value"]].set_index(
                "original_value")["mapped_value"].to_dict())
        level_value_reverse = (
            level_rename_df.loc[level_rename_df["mapped"].notna(), ["original_value", "mapped_value"]].set_index(
                "mapped_value")["original_value"].to_dict())
    return level_mapping, level_mapping_reverse, level_value_mapping, level_value_reverse


def filter_total_data(client_code: str, model_group_id: int, target_metric: list, target_level: list, data: pd.DataFrame):
    """Load, filter, and normalise business driver data by selected levels and metrics; return filtered df with normalised Time column."""
    df = readout_utils.load_total_viz_data(client_code, model_group_id)
    level_mapping, level_mapping_reverse, level_value_mapping, level_value_reverse = get_level_rename_mapping(client_code, model_group_id)

    df = df.merge(data[target_level + ["time"]],
                  left_on=target_level + ["time"],
                  right_on=target_level + ["time"],
                  how="right")

    if "sales" in target_metric:
        target_driver = ["overall"]
    else:
        target_driver = data["business_driver"].dropna().unique().tolist()

    # normalize time
    df["time_norm"] = df["time"].str.replace(
        r'.*?(year|quarter|half|month)\s*(\d{0,2})\D*(\d{4}).*',
        lambda m: {
            'year': m.group(3),
            'quarter': f"Q{m.group(2)} {m.group(3)}",
            'half': f"H{m.group(2)} {m.group(3)}",
            'month': f"M{int(m.group(2) or 0)} {m.group(3)}"
        }[m.group(1)], regex=True)

    if 'driver' not in data.columns:
        data.insert(loc=len(data.columns) - 4, column='driver', value='overall')

    metric_cols = [x for x in data.columns.tolist() if any(m in x.lower() for m in target_metric)]
    # sorted_times = sorted(set([re.search(r"(Q[1-4]|H[1-2]|M(?:1[0-2]|[1-9])) \d{4}|\d{4}", x).group() for x in metric_cols if
    #                            re.search(r"(Q[1-4]|H[1-2]|M(?:1[0-2]|[1-9])) \d{4}|\d{4}", x)]))

    filtered_df = df.copy()
    # filtered_df = filtered_df[filtered_df["time_norm"].isin(sorted_times)]
    filtered_df = filtered_df[filtered_df["driver"].isin(target_driver)]
    filtered_df = filtered_df[filtered_df["metric"].isin(target_metric)]

    return filtered_df