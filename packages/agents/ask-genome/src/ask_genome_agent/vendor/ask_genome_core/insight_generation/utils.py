# Vendored from ask-genome-core bb1e1637bca4:src/insight_generation/utils.py
# by scripts/sync_ask_genome_core.py. Do not edit by hand; change EDITS there.
import os
import sys
import re
import calendar
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import ask_genome_agent.vendor.ask_genome_core.model.readout_utils as readout_utils
import ask_genome_agent.vendor.ask_genome_core.insight_generation.contribution as contribution_utils
import ask_genome_agent.vendor.ask_genome_core.insight_generation.roi as roi_utils
import ask_genome_agent.vendor.ask_genome_core.insight_generation.spend as spend_utils
import ask_genome_agent.vendor.ask_genome_core.insight_generation.standard_metric as standard_metric_utils
import ask_genome_agent.vendor.ask_genome_core.model.visual_config_utils as visual_configs


MONTH_NUMBERS = {
    name.casefold(): i for i, name in enumerate(calendar.month_name) if name
}
MONTH_PATTERN = "|".join(calendar.month_name[1:])
TIME_PATTERN = re.compile(
    rf"(?:Q[1-4]|H[1-2]|M(?:1[0-2]|[1-9])|Month\s+(?:1[0-2]|[1-9])|"
    rf"{MONTH_PATTERN})\s+\d{{4}}|\b\d{{4}}\b",
    re.IGNORECASE,
)


def _instruction_maps(prompt_instructions, time_tag) -> tuple[dict, dict]:
    if prompt_instructions.empty:
        return {}, {}
    selected = prompt_instructions[prompt_instructions["timeTag"] == time_tag]
    if selected.empty:
        return {}, {}
    return (
        selected.set_index("configName")["instruction"].to_dict(),
        selected.set_index("configName")["sectionName"].to_dict(),
    )


def _find_metric_alias(metric_configs: pd.DataFrame, metrics: list) -> dict:
    res = metric_configs[metric_configs["kpiName"].str.lower().isin(metrics)].set_index("kpiName")["alias"].to_dict()
    return {k.lower(): v.lower() for k, v in res.items()}


def concat_readout_frames(readout_data, *, ignore_index: bool = False) -> pd.DataFrame:
    """Combine the activity and measure frames from a readout result."""
    return pd.concat(
        [
            readout_data.df_activity_group,
            readout_data.df_measure_group,
            readout_data.df_measure,
        ],
        ignore_index=ignore_index,
    )


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


def extract_time(value) -> str | None:
    """Extract a supported time label from a metric column name."""
    match = TIME_PATTERN.search(str(value))
    return match.group(0) if match else None


def time_sort_key(value) -> tuple[int, int]:
    """Return a chronological sort key for one supported time label."""
    value = str(value).strip()
    match = re.search(r"(\d{4})", value)
    if match is None:
        return 0, 0
    year = int(match.group(1))
    period = value[:match.start()].strip().casefold()
    if period.startswith("q"):
        return year, (int(period[1:]) - 1) * 3 + 1
    if period.startswith("h"):
        return year, (int(period[1:]) - 1) * 6 + 1
    if period.startswith("month"):
        return year, int(period.split()[-1])
    if re.fullmatch(r"m(?:1[0-2]|[1-9])", period):
        return year, int(period[1:])
    return year, MONTH_NUMBERS.get(period, 0)


def sorted_times_from_columns(columns) -> list[str]:
    """Return unique time labels from columns, ordered newest to oldest."""
    by_key = {}
    for column in columns:
        time = extract_time(column)
        if time:
            by_key.setdefault(time_sort_key(time), time)
    return [by_key[key] for key in sorted(by_key, reverse=True)]


def display_time(value) -> str:
    """Convert internal month labels to their full display name."""
    match = re.fullmatch(
        r"(?:M(1[0-2]|[1-9])|Month\s+(1[0-2]|[1-9]))\s+(\d{4})",
        str(value).strip(),
        re.IGNORECASE,
    )
    if match is None:
        return str(value)
    month_number = int(match.group(1) or match.group(2))
    return f"{calendar.month_name[month_number]} {match.group(3)}"


def find_column(df: pd.DataFrame, target_str: str, excluded_str: str=None) -> list:
    mask = df.columns.str.contains(target_str, case=False, regex=False)
    if excluded_str:
        mask &= ~df.columns.str.contains(excluded_str, case=False, regex=False)
    cols = list(df.columns[mask])
    return cols


def strip_sign_and_unit(
        df: pd.DataFrame, cols: list, char: str) -> pd.DataFrame:
    """Remove display signs and units from selected columns and coerce them to numbers."""
    _MULTIPLIERS = {"T": 1e12, "B": 1e9, "M": 1e6, "K": 1e3}
    special_char = f"[$%,{re.escape(char)}]"
    for col in set(cols):
        cleaned = (
            df[col]
            .astype(str)
            .str.replace(special_char, "", regex=True)
            .str.strip()
            .replace({"nan": np.nan, "None": np.nan, "": np.nan})
        )
        # split "1.2M" -> ("1.2", "M"); no suffix -> both NaN
        parts = cleaned.str.extract(r"^(.*?)\s*([TBMK])$", flags=re.IGNORECASE)
        base = cleaned.where(parts[1].isna(), parts[0])
        multiplier = parts[1].str.upper().map(_MULTIPLIERS).fillna(1.0)
        df[col] = pd.to_numeric(base, errors="coerce") * multiplier
    return df


def add_percent_change(
        data: pd.DataFrame, latest_col: str,
        previous_col: str, change_col: str) -> pd.DataFrame:
    """Add percentage change"""
    previous = data[previous_col].replace(0, np.nan)
    data[change_col] = (data[latest_col] - previous) / previous * 100
    return data


def calculate_metric_change(
        df: pd.DataFrame, metric: str, sorted_times: list,
        title_metric: bool = True) -> pd.DataFrame:
    sorted_times = sorted_times[::-1]
    prev, curr = None, None
    for target_time in sorted_times:
        pattern = re.compile(
            rf"^{re.escape(metric)}\s+{re.escape(str(target_time))}",
            re.IGNORECASE,
        )
        matches = [column for column in df.columns if pattern.match(column)]
        if matches:
            curr = matches[0]
        if prev and curr:
            display_metric = metric.title() if title_metric else metric
            change_col = f"% Change in {display_metric} {target_time}"
            df = add_percent_change(df, curr, prev, change_col)
        prev, curr = curr, None
    return df


def label_metric_trend(
        df: pd.DataFrame, metric: str, sorted_times: list,
        spike_ratio=4.0, volatile_ratio=1.5, flat_multiplier=0.06,
        sign_ratio=0.8, *, title_metric: bool = False) -> pd.DataFrame:
    """Add a trend label for a metric across the supplied time order."""
    metric_cols = []
    for time in sorted_times:
        pattern = re.compile(
            rf"^{re.escape(metric)}\s+{re.escape(str(time))}", re.IGNORECASE
        )
        metric_cols.extend(column for column in df.columns if pattern.match(column))

    if not metric_cols:
        return df

    def classify_row(row):
        values = np.array(row[metric_cols], dtype=np.float64)
        values = values[~np.isnan(values)]
        if len(values) < 3:
            return np.nan

        mean = np.mean(values)
        std = np.std(values)
        all_mean_abs = np.mean(np.abs(values))
        max_deviation_ratio = np.max(np.abs(values - mean)) / (abs(mean) + 1e-6)

        ordered = values
        rho, p_value = spearmanr(range(len(ordered)), ordered[::-1])

        flat_threshold = flat_multiplier * (all_mean_abs + 1e-6)
        volatile_threshold = 0.35 * (all_mean_abs + 1e-6)

        if max_deviation_ratio > spike_ratio:
            return "spiking"
        if std > volatile_threshold:
            return "volatile"
        if p_value < 0.05 and rho > 0:
            return "consistently increasing"
        if p_value < 0.05 and rho < 0:
            return "consistently decreasing"
        if std < flat_threshold:
            return "flat"
        return (
            "mild fluctuated (increase)"
            if ordered[0] > ordered[-1]
            else "mild fluctuated (decrease)"
        )

    trend_metric = metric.title() if title_metric else metric
    df[f"{trend_metric} trend"] = df.apply(classify_row, axis=1)
    return df


def get_metric_format(metric_configs: pd.DataFrame, m_metric: str):
    """Format config for one metric, matched on alias case-insensitively.

    Returns a json-safe dict (native python scalars, NaN -> None), or `default`
    if the metric is not found."""
    default = {'NumberDisplayDecimals': 2,
               'UsePowers': True,
               'Prefix': '',
               'Postfix': '',
               'Multiplier': 1,
               'IsPercent': False
              }
    FORMAT_COLS = list(default.keys())
    match = metric_configs["kpiName"].astype(str).str.strip().str.lower() == str(m_metric).strip().lower()
    if not match.any():
        return default
    row = metric_configs.loc[match, FORMAT_COLS].iloc[0]
    return {k: None if pd.isna(v) else v.item() if hasattr(v, "item") else v for k, v in row.items()}


def _format_cell(value, decimals, prefix="", postfix=""):
    """Format one populated cell. NA, blank and non-numeric pass through untouched."""
    if pd.isna(value) or (isinstance(value, str) and not value.strip()):
        return value
    try:
        num = float(value)
    except (TypeError, ValueError):
        return value
    return f"{prefix}{num:,.{decimals}f}{postfix}"


def _format_insight_metric_columns(data: pd.DataFrame, metrics: dict, metric_configs: pd.DataFrame) -> pd.DataFrame:
    """Format selected insight columns without reapplying upstream renaming and display_scale."""
    res = data.copy()

    for metric, alias in metrics.items():
        value_cols = find_column(res, metric, "% change")
        alias_value_cols = find_column(res, alias, "% change")

        fmt = get_metric_format(metric_configs, metric)
        for col in value_cols + alias_value_cols:
            decimals = int(fmt.get("NumberDisplayDecimals") or 0)
            prefix, postfix = fmt.get("Prefix") or "", fmt.get("Postfix") or ""
            res[col] = res[col].map(
                lambda v, d=decimals, p=prefix, s=postfix: _format_cell(v, d, p, s))

    change_cols = find_column(res, "% change", None)
    for col in change_cols:
        res[col] = data[col].map(lambda v: _format_cell(v, 2, postfix="%"))
    return res


def get_time_align_dict(sorted_time: list):
    """Map internal time tokens (e.g. 'M6 2023') to display labels (e.g. 'June 2023').
    Returns a dict of {display_label: internal_token}."""
    return {display_time(time): time for time in sorted_time}


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


def resolve_level_columns(levels: list, data: pd.DataFrame, level_mapping_reverse: dict) -> dict:
    """Match questionnaire level names to their columns in the formatted table.
    """
    by_lower = {str(col).lower(): col for col in data.columns}
    resolved = {}
    for level in levels:
        # a client with level_rename_display.csv has the column renamed to its mapped name
        display = (level_mapping_reverse or {}).get(level, level)
        match = by_lower.get(str(display).lower())
        if match is not None:
            resolved[match] = level
    if len(resolved) != len(levels):
        missing = [lv for lv in levels if lv not in resolved.values()]
        print(f"Levels {missing} are not in the formatted table, dropping them from the readout")
    return resolved


def filter_total_data(client_code: str, model_group_id: int, target_metric: list, target_level: list,
                      data: pd.DataFrame, total: str = "total_viz"):
    """Load, filter, and normalise business driver data by selected levels and metrics; return filtered df with normalised Time column."""
    if total == "total_viz":
        df = readout_utils.load_total_viz_data(client_code, model_group_id)
    else:
        try:
            df = readout_utils.load_total_data(client_code, model_group_id)
        except FileNotFoundError:
            df = pd.DataFrame()
    if df.empty:
        return df
    if 'driver' not in df.columns:
        df.insert(loc=len(df.columns) - 4, column='driver', value='overall')
    level_mapping, level_mapping_reverse, level_value_mapping, level_value_reverse = get_level_rename_mapping(client_code, model_group_id)
    targets = {m.lower() for m in target_metric}

    df = df.merge(data[target_level + ["time"]],
                  left_on=target_level + ["time"],
                  right_on=target_level + ["time"],
                  how="right")

    if "sales" in targets or "model output" in targets:
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

    filtered_df = df.copy()
    filtered_df = filtered_df[filtered_df["driver"].isin(target_driver)]
    filtered_df = filtered_df[filtered_df["metric"].str.lower().isin(targets)]

    return filtered_df


def _global_insight_instructions(client_code: str, model_group_id: int, intention: str,
                                 insight_topic: str, query: str) -> str:
    """Shared tone + response-structure instructions for the contribution/roi readouts.
    These live in insight_instruction.csv with timeTag='overall' and configName in
    ('system', 'response_structure'); returned joined in file order with the [intention] and
    [user query] placeholders filled. Empty when the rows are absent (no-op)."""
    inst_df = readout_utils.load_insight_instructions(client_code, model_group_id, intention=intention)
    if inst_df.empty or 'configName' not in inst_df.columns or 'timeTag' not in inst_df.columns:
        return ""
    rows = inst_df[(inst_df['timeTag'] == 'overall')
                   & (inst_df['configName'].isin(['system', 'response_structure']))]
    if rows.empty:
        return ""
    text = "\n\n".join(rows['instruction'].dropna().tolist())
    return text.replace('[intention]', insight_topic).replace('[user query]', query)


def build_readout(client_code: str, model_group_id: int, query: str, data: pd.DataFrame, agg_data: pd.DataFrame,
                  readout_data, metric_configs: pd.DataFrame) -> str :

    intention_list = readout_data.ner_filters["intention"]
    standard_intention = next(
        (intention for intention in intention_list if intention in ['activity', 'response', 'cost per', 'sales', 'source of change']),
        None,
    )
    if "margin roi" in intention_list:
        insight_topic = "ROI"
    else:
        insight_topic = intention_list[0].title()

    if "contribution" in intention_list:
        inst_intention = "contribution"
        all_configs = contribution_utils.build_contribution_config_for_insights(client_code,
                                                                                model_group_id,
                                                                                query,
                                                                                data,
                                                                                agg_data,
                                                                                readout_data,
                                                                                metric_configs)
    elif "margin roi" in intention_list or "performance" in intention_list:
        inst_intention = "margin roi"
        all_configs = roi_utils.build_roi_config_for_insights(client_code,
                                                              model_group_id,
                                                              query,
                                                              data,
                                                              agg_data,
                                                              readout_data,
                                                              metric_configs)
    elif "spending" in intention_list:
        insight_topic = "Spend"
        inst_intention = "spend"
        all_configs = spend_utils.build_spend_config_for_insights(
            client_code,
            model_group_id,
            query,
            data,
            agg_data,
            readout_data,
            metric_configs,
        )
    elif standard_intention:
        policy = standard_metric_utils.get_policy(standard_intention)
        insight_topic = policy.display_name
        inst_intention = standard_intention
        all_configs = standard_metric_utils.build_config_for_insights(
            client_code, model_group_id, query, data, agg_data,
            readout_data, metric_configs, intention=standard_intention,
        )
    else:
        raise Exception(f"Readout support not implemented for intention: {insight_topic}.")

    if all_configs:
        all_config_texts = visual_configs._generate_insights_prompt(all_configs, query, insight_topic)
        global_text = _global_insight_instructions(client_code, model_group_id, inst_intention, insight_topic, query)
        if global_text:
            all_config_texts = global_text + "\n\n" + all_config_texts
        return all_config_texts
    else:
        return ""
