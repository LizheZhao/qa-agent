import re

import numpy as np
import pandas as pd

import src.insight_generation.utils as utils
import src.model.readout_utils as readout_utils


DEFAULT_SECTIONS = {
    "overall": "Overall Spend",
    "allocation": "Spend allocation by {level} Tactics",
}


def _metric_key(value) -> str:
    return re.sub(r"\s+", " ", str(value).strip()).casefold()


def _format_spend_metric_columns(
        data: pd.DataFrame, metrics_alias: dict,
        metric_configs: pd.DataFrame) -> pd.DataFrame:
    ordered_aliases = dict(
        sorted(
            metrics_alias.items(),
            key=lambda item: max(len(item[0]), len(item[1])),
            reverse=True,
        )
    )
    return utils._format_insight_metric_columns(
        data, ordered_aliases, metric_configs
    )


def _metric_name_from_column(column, time: str) -> str:
    match = utils.TIME_PATTERN.search(str(column))
    name = str(column)[:match.start()].strip() if match else str(column)
    return re.sub(r"^%\s*change\s+in\s+", "", name, flags=re.I).strip()


def _find_metric_column(data, metrics, time, metrics_alias, *, change=False):
    alias_keys = {
        _metric_key(metrics_alias[metric])
        for metric in metrics
        if metric in metrics_alias
    }
    for column in data.columns:
        column_time = utils.extract_time(column)
        if (
                column_time is None
                or utils.time_sort_key(column_time) != utils.time_sort_key(time)
        ):
            continue
        is_change = bool(re.match(r"^%\s*change\s+in\s+", str(column), re.I))
        if is_change != change or " level " in str(column).casefold():
            continue
        metric_name = _metric_name_from_column(column, column_time)
        if _metric_key(metric_name) in alias_keys:
            return column
    return None


def _driver_and_dimensions(data):
    driver = next((c for c in data.columns if str(c).casefold() == "tactics"), None)
    if driver is None:
        return None, []
    return driver, data.columns[:data.columns.tolist().index(driver)].tolist()


def _parse_number(value):
    if pd.isna(value):
        return np.nan
    return pd.to_numeric(re.sub(r"[$,%\s]", "", str(value)), errors="coerce")


def _prepare_metric_distribution(client_code, model_group_id, metric_dist, readout_data):
    if metric_dist.empty or "time" not in metric_dist.columns:
        return pd.DataFrame()
    distribution = metric_dist.copy()
    scope = utils.concat_readout_frames(readout_data, ignore_index=True)
    levels = readout_data.ner_filters.get("level", {})
    levels = levels.get("general_level", []) + levels.get("halo_level", [])
    keys = [c for c in levels + ["time"] if c in distribution and c in scope]
    if keys and not scope.empty:
        distribution = distribution.merge(scope[keys].drop_duplicates(), on=keys, how="inner")
    if distribution.empty:
        return distribution
    distribution = utils.normalize_time(distribution)
    distribution = readout_utils.apply_level_mapping_display(client_code, model_group_id, distribution)
    distribution["_time_key"] = distribution["time_norm"].map(
        utils.time_sort_key
    )
    return distribution


def _add_latest_spend_level(
        data, spend_col, latest_time, dimensions, metric_dist, metric_names):
    if spend_col is None or metric_dist.empty:
        return data, None
    metric_name = _metric_name_from_column(spend_col, latest_time)
    metric_keys = {_metric_key(metric) for metric in metric_names}
    latest_time_key = utils.time_sort_key(latest_time)
    q_df = metric_dist[
        metric_dist["metric"].map(_metric_key).isin(metric_keys)
        & metric_dist["_time_key"].map(lambda value: value == latest_time_key)
    ].copy()
    q_df[["q1", "q3"]] = q_df[["q1", "q3"]].apply(pd.to_numeric, errors="coerce")
    q_df = q_df.dropna(subset=["q1", "q3"])
    if q_df.empty:
        return data, None

    group_keys = [c for c in dimensions if c in data and c in q_df]
    norm = lambda values: tuple("" if pd.isna(v) else str(v).strip().casefold() for v in values)
    if group_keys:
        thresholds = {
            norm(key if isinstance(key, tuple) else (key,)): (row.q1, row.q3)
            for key, row in q_df.groupby(group_keys, dropna=False)[["q1", "q3"]].first().iterrows()
        }
    else:
        unique = q_df[["q1", "q3"]].drop_duplicates()
        if len(unique) != 1:
            return data, None
        global_thresholds = tuple(unique.iloc[0])

    values = data[spend_col].map(_parse_number)
    if "(mm)" in str(spend_col).casefold():
        values *= 1_000_000

    def classify(row):
        value = values.loc[row.name]
        if pd.isna(value):
            return np.nan
        pair = thresholds.get(norm(row[group_keys].tolist())) if group_keys else global_thresholds
        if pair is None:
            return np.nan
        return "low" if value < pair[0] else "high" if value > pair[1] else "moderate"

    output = data.copy()
    level_col = f"{metric_name} level {latest_time}"
    output[level_col] = output.apply(classify, axis=1)
    return output, level_col


def build_spend_comparison_tbl_for_insights(
        data, metric_dist, spend_metrics, spend_share_metrics,
        metrics_alias, metric_configs):
    driver, dimensions = _driver_and_dimensions(data)
    times = utils.sorted_times_from_columns(data.columns)
    if data.empty or driver is None or not times:
        return pd.DataFrame()

    metric_cols, current_spend = [], None
    for metrics in [spend_metrics, spend_share_metrics]:
        block = [_find_metric_column(data, metrics, time, metrics_alias) for time in times[:2]]
        block = [column for column in block if column is not None]
        if metrics is spend_metrics and block:
            current_spend = block[0]
        metric_cols.extend(block)
        if len(times[:2]) == 2 and len(block) == 2:
            change = _find_metric_column(data, metrics, times[0], metrics_alias, change=True)
            if change is not None:
                metric_cols.append(change)
    if not metric_cols:
        return pd.DataFrame()

    output = data[dimensions + [driver] + metric_cols].copy()
    output = utils.strip_sign_and_unit(output, metric_cols, char="$")
    output, level_col = _add_latest_spend_level(
        output, current_spend, times[0], dimensions, metric_dist, spend_metrics
    )
    columns = dimensions + [driver] + metric_cols + ([level_col] if level_col else [])
    output = output[columns].dropna(subset=metric_cols, how="all")
    return _format_spend_metric_columns(output, metrics_alias, metric_configs)


def build_spend_history_tbl_for_insights(
        data, metric_dist, spend_metrics, spend_share_metrics,
        metrics_alias, metric_configs):
    driver, dimensions = _driver_and_dimensions(data)
    times = utils.sorted_times_from_columns(data.columns)
    if data.empty or driver is None or not times:
        return pd.DataFrame()

    metric_blocks = []
    for metrics in [spend_metrics, spend_share_metrics]:
        period_columns = [
            (time, _find_metric_column(data, metrics, time, metrics_alias))
            for time in times
        ]
        period_columns = [(time, column) for time, column in period_columns if column]
        if not period_columns:
            continue
        metric_blocks.append((metrics, period_columns))
    if not metric_blocks:
        return pd.DataFrame()

    value_cols = [
        column
        for _, period_columns in metric_blocks
        for _, column in period_columns
    ]
    output = data[dimensions + [driver] + value_cols].copy()
    output = utils.strip_sign_and_unit(output, value_cols, char="$")
    level_cols = []
    for metrics, period_columns in metric_blocks:
        latest_time, latest_col = period_columns[0]
        output, level_col = _add_latest_spend_level(
            output,
            latest_col,
            latest_time,
            dimensions,
            metric_dist,
            metrics,
        )
        if level_col:
            level_cols.append(level_col)
    output = output[dimensions + [driver] + value_cols + level_cols].dropna(
        subset=value_cols, how="all"
    )
    return _format_spend_metric_columns(output, metrics_alias, metric_configs)


def build_spend_trend_tbls_for_insights(
        data, metric_dist, spend_metrics, spend_share_metrics,
        metrics_alias, metric_configs):
    driver, dimensions = _driver_and_dimensions(data)
    times = utils.sorted_times_from_columns(data.columns)
    if data.empty or driver is None or len(times) < 3:
        return pd.DataFrame(), pd.DataFrame()

    metric_blocks = []
    for metrics in [spend_metrics, spend_share_metrics]:
        period_columns = [
            (time, _find_metric_column(data, metrics, time, metrics_alias))
            for time in times
        ]
        period_columns = [(time, column) for time, column in period_columns if column]
        if len(period_columns) < 3:
            continue
        metric_name = _metric_name_from_column(period_columns[0][1], period_columns[0][0])
        metric_blocks.append((metrics, metric_name, period_columns))

    if not metric_blocks:
        return pd.DataFrame(), pd.DataFrame()

    value_cols = [
        column
        for _, _, period_columns in metric_blocks
        for _, column in period_columns
    ]
    trend_cols, ordered_extras, level_cols = [], [], []
    wide_table = data[dimensions + [driver] + value_cols].copy()
    wide_table = utils.strip_sign_and_unit(wide_table, value_cols, char="$")
    numeric = wide_table.copy()
    for metrics, metric_name, period_columns in metric_blocks:
        trend_col = None
        if metrics is spend_metrics:
            metric_times = [time for time, _ in period_columns]
            numeric = utils.label_metric_trend(
                numeric, metric_name, metric_times, title_metric=True
            )
            trend_col = f"{metric_name.title()} trend"
            wide_table[trend_col] = numeric[trend_col]
            trend_cols.append(trend_col)

        latest_time, latest_col = period_columns[0]
        change_col = _find_metric_column(
            data, metrics, latest_time, metrics_alias, change=True
        )
        if change_col:
            wide_table[change_col] = data[change_col]
            wide_table = utils.strip_sign_and_unit(
                wide_table, [change_col], char="%"
            )
            ordered_extras.append(change_col)
        if trend_col:
            ordered_extras.append(trend_col)

        wide_table, level_col = _add_latest_spend_level(
            wide_table,
            latest_col,
            latest_time,
            dimensions,
            metric_dist,
            metrics,
        )
        if level_col:
            level_cols.append(level_col)

    label_table = wide_table[dimensions + [driver] + trend_cols + level_cols].copy()
    ordered_cols = dimensions + [driver] + value_cols + ordered_extras + level_cols
    wide_table = wide_table[ordered_cols].dropna(subset=value_cols, how="all")
    wide_table = _format_spend_metric_columns(
        wide_table, metrics_alias, metric_configs
    )
    return label_table, wide_table


def build_overall_spend_tbl_for_insights(
        client_code, model_group_id, readout_data, spend_metrics,
        times, time_tag, metrics_alias, metric_configs):
    try:
        total_data = readout_utils.load_total_data(client_code, model_group_id)
    except (FileNotFoundError, KeyError):
        return pd.DataFrame()
    required = {"metric", "time", "value"}
    if total_data.empty or not required.issubset(total_data.columns):
        return pd.DataFrame()

    scope = utils.concat_readout_frames(readout_data, ignore_index=True)
    levels = readout_data.ner_filters.get("level", {})
    levels = levels.get("general_level", []) + levels.get("halo_level", [])
    scope_keys = [c for c in levels + ["time"] if c in total_data and c in scope]
    if scope_keys and not scope.empty:
        total_data = total_data.merge(
            scope[scope_keys].drop_duplicates(), on=scope_keys, how="inner"
        )
    if "driver" in total_data:
        total_data = total_data[
            total_data["driver"].fillna("").astype(str).str.casefold().eq("overall")
        ]

    spend_keys = {_metric_key(metric) for metric in spend_metrics}
    total_data = total_data[total_data["metric"].map(_metric_key).isin(spend_keys)].copy()
    if total_data.empty:
        return pd.DataFrame()

    total_data = utils.normalize_time(total_data)
    total_data["_time_key"] = total_data["time_norm"].map(utils.time_sort_key)
    desired_times = times or readout_data.ner_filters.get("time", [])
    desired_labels = {
        utils.time_sort_key(time): utils.display_time(time)
        for time in desired_times
    }
    if desired_labels:
        total_data = total_data[total_data["_time_key"].isin(desired_labels)]
    if total_data.empty:
        return pd.DataFrame()

    total_data["time_label"] = total_data.apply(
        lambda row: desired_labels.get(
            row["_time_key"], utils.display_time(row["time_norm"])
        ),
        axis=1,
    )
    display_aliases = {
        _metric_key(row["kpiName"]): str(row["alias"]).strip()
        for _, row in metric_configs.iterrows()
        if _metric_key(row["kpiName"]) in spend_keys
        and pd.notna(row["alias"])
        and str(row["alias"]).strip()
    }
    total_data["metric_label"] = total_data["metric"].map(
        lambda metric: display_aliases.get(_metric_key(metric), str(metric).strip())
    )

    index_cols = [column for column in levels if column in total_data]
    if "driver" in total_data:
        index_cols.append("driver")
    if not index_cols:
        total_data["_overall"] = "Overall"
        index_cols = ["_overall"]

    pivot = total_data.pivot_table(
        index=index_cols,
        columns=["metric_label", "time_label"],
        values="value",
        aggfunc="first",
    ).reset_index()
    pivot.columns = [
        f"{metric} {time}".strip() if time else str(metric)
        for metric, time in pivot.columns
    ]

    ordered_times = sorted(
        total_data["time_label"].unique(),
        key=utils.time_sort_key,
        reverse=True,
    )
    if time_tag in {"snapshot", "period_over_period_change"}:
        ordered_times = ordered_times[:2]
    metric_labels = total_data["metric_label"].drop_duplicates().tolist()
    value_cols = [
        f"{metric} {time}"
        for metric in metric_labels
        for time in ordered_times
        if f"{metric} {time}" in pivot
    ]
    pivot = pivot[index_cols + value_cols]

    trend_cols = []
    if time_tag == "trend":
        for metric in metric_labels:
            pivot = utils.label_metric_trend(
                pivot, metric, ordered_times, title_metric=True
            )
            trend_col = f"{metric.title()} trend"
            if trend_col in pivot:
                trend_cols.append(trend_col)

    pivot = readout_utils.apply_level_mapping_display(client_code, model_group_id, pivot)
    pivot = pivot.rename(columns={"driver": "Business Driver", "_overall": "Business Driver"})
    id_cols = [column for column in pivot if column not in value_cols + trend_cols]
    removable = [
        column for column in id_cols
        if column != "Business Driver"
        and pivot[column].fillna("").astype(str).str.casefold().eq("overall").all()
    ]
    if removable:
        pivot = pivot.drop(columns=removable)
        id_cols = [column for column in id_cols if column not in removable]

    formatted = _format_spend_metric_columns(
        pivot[id_cols + value_cols], metrics_alias, metric_configs
    )
    formatted_value_cols = [column for column in formatted if column not in id_cols]
    if trend_cols:
        for column in trend_cols:
            formatted[column] = pivot[column].values
        return formatted[id_cols + trend_cols + formatted_value_cols]
    return formatted


def _format_template(template, **values):
    try:
        return str(template).format(**values)
    except (KeyError, IndexError, ValueError):
        return str(template)


def _add_config(output, name, instruction, data):
    if data is None or data.empty:
        return
    unique_name, suffix = name, 2
    while unique_name in output:
        unique_name = f"{name} ({suffix})"
        suffix += 1
    output[unique_name] = {
        "instruction": instruction,
        "data": data.to_json(orient="records"),
    }


def build_spend_config_for_insights(
        client_code, model_group_id, query, data, agg_data, readout_data,
        metric_configs: pd.DataFrame):
    output = {}
    if readout_data is None:
        return output

    time_tag = readout_data.ner_filters.get("time_tag", "snapshot")
    if time_tag not in ["snapshot", "period_over_period_change", "multiple_periods", "trend"]:
        raise ValueError(
            "time_tag must be 'snapshot', 'period_over_period_change', "
            "'multiple_periods' or 'trend'"
        )

    spend_metrics = list(
        readout_utils._find_metrics_by_group(metric_configs, "spend")
    )
    spend_share_metrics = list(
        readout_utils._find_metrics_by_group(metric_configs, "spend share")
    )
    if not spend_metrics:
        return output
    metrics_alias = utils._find_metric_alias(
        metric_configs, spend_metrics + spend_share_metrics
    )

    prompt_instructions = readout_utils.load_insight_instructions(
        client_code, model_group_id, intention="spending"
    )
    instructions, section_names = utils._instruction_maps(prompt_instructions, time_tag)
    metric_dist = _prepare_metric_distribution(
        client_code,
        model_group_id,
        readout_utils.load_metric_distribution(client_code, model_group_id, file_type="bi"),
        readout_data,
    )

    detail_data, aggregate_data = data, agg_data
    time_source = detail_data if not detail_data.empty else aggregate_data
    overall = build_overall_spend_tbl_for_insights(
        client_code,
        model_group_id,
        readout_data,
        spend_metrics,
        utils.sorted_times_from_columns(time_source.columns),
        time_tag,
        metrics_alias,
        metric_configs,
    )
    _add_config(
        output,
        section_names.get("overall", DEFAULT_SECTIONS["overall"]),
        _format_template(
            instructions.get("overall", "Provide insights based on given data."),
            query=query,
        ),
        overall,
    )

    for level, level_data in [("aggregated", aggregate_data), ("detailed", detail_data)]:
        if level_data.empty:
            continue
        if time_tag in {"snapshot", "period_over_period_change"}:
            table = build_spend_comparison_tbl_for_insights(
                level_data,
                metric_dist,
                spend_metrics,
                spend_share_metrics,
                metrics_alias,
                metric_configs,
            )
            _add_config(
                output,
                _format_template(
                    section_names.get("allocation", DEFAULT_SECTIONS["allocation"]),
                    level=level,
                ),
                _format_template(
                    instructions.get("allocation", "Provide insights based on given data."),
                    level=level,
                    query=query,
                ),
                table,
            )
        elif time_tag == "multiple_periods":
            table = build_spend_history_tbl_for_insights(
                level_data,
                metric_dist,
                spend_metrics,
                spend_share_metrics,
                metrics_alias,
                metric_configs,
            )
            _add_config(
                output,
                _format_template(
                    section_names.get("allocation", DEFAULT_SECTIONS["allocation"]),
                    level=level,
                ),
                _format_template(
                    instructions.get("allocation", "Provide insights based on given data."),
                    level=level,
                    query=query,
                ),
                table,
            )
        else:
            _, table = build_spend_trend_tbls_for_insights(
                level_data,
                metric_dist,
                spend_metrics,
                spend_share_metrics,
                metrics_alias,
                metric_configs,
            )
            _add_config(
                output,
                _format_template(
                    section_names.get("allocation", DEFAULT_SECTIONS["allocation"]),
                    level=level,
                ),
                _format_template(
                    instructions.get("allocation", "Provide insights based on given data."),
                    level=level,
                    query=query,
                ),
                table,
            )
    return output
