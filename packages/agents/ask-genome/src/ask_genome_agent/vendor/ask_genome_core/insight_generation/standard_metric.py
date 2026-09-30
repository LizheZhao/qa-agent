# Vendored from ask-genome-core bb1e1637bca4:src/insight_generation/standard_metric.py
# by scripts/sync_ask_genome_core.py. Do not edit by hand; change EDITS there.
from dataclasses import dataclass, field
import re

import numpy as np
import pandas as pd

import ask_genome_agent.vendor.ask_genome_core.insight_generation.utils as utils
import ask_genome_agent.vendor.ask_genome_core.model.readout_utils as readout_utils


@dataclass(frozen=True)
class MetricSpec:
    source: str
    kpi_name: str
    alias: str
    group: str


@dataclass(frozen=True)
class TableSpec:
    kind: str
    main_groups: tuple[str, ...]
    metrics: tuple[MetricSpec, ...]
    related_group: str | None = None
    captured_metrics: tuple[MetricSpec, ...] = ()


@dataclass(frozen=True)
class InsightPolicy:
    intention: str
    data_source: str
    display_name: str
    driver_name: str
    config_prefix: str
    add_change: bool = True
    add_trend: bool = True
    add_level: bool = True
    column_aliases: dict[str, tuple[str, ...]] = field(default_factory=dict)


POLICIES = {
    "activity": InsightPolicy("activity", "bi", "Activity", "Tactics", "activity"),
    "response": InsightPolicy("response", "bi", "Response", "Tactics", "response"),
    "cost per": InsightPolicy("cost per", "bi", "Cost per", "Tactics", "cost"),
    "sales": InsightPolicy(
        "sales", "br", "Sales", "Business Driver", "sales",
        column_aliases={"sales": ("Contribution Amount",)},
    ),
    "source of change": InsightPolicy(
        "source of change", "br", "Source of Change", "Business Driver",
        "source_of_change",
        add_change=False, add_trend=False, add_level=False,
    ),
}


def _key(value) -> str:
    return re.sub(r"\s+", " ", str(value).strip()).casefold()


def get_policy(intention: str) -> InsightPolicy:
    try:
        return POLICIES[intention]
    except KeyError as exc:
        raise ValueError(f"Unsupported standard insight intention: {intention}") from exc




def _metric_lookup(metric_configs: pd.DataFrame) -> dict[str, list[tuple[str, MetricSpec]]]:
    lookup: dict[str, list[tuple[str, MetricSpec]]] = {}
    for row in metric_configs.itertuples(index=False):
        kpi_name = str(row.kpiName).strip()
        alias = str(row.alias).strip() if pd.notna(row.alias) else kpi_name
        domain = str(getattr(row, "domain", "")).strip()
        metric = MetricSpec("", kpi_name, alias, str(row.metricGroup).strip())
        for name in (kpi_name, alias):
            lookup.setdefault(_key(name), []).append((domain, metric))
    return lookup


def _resolve_metrics(
        values, metric_configs: pd.DataFrame, data_source: str) -> tuple[MetricSpec, ...]:
    lookup = _metric_lookup(metric_configs)
    output = []
    for value in values:
        candidates = lookup.get(_key(value), [])
        match = next(
            (metric for domain, metric in candidates if _key(domain) == _key(data_source)),
            None,
        )
        if match is None:
            match = next(
                (metric for domain, metric in candidates if _key(domain) == "calculated"),
                None,
            )
        if match is not None:
            output.append(MetricSpec(str(value), match.kpi_name, match.alias, match.group))
    return tuple(dict.fromkeys(output))


def build_table_specs(
        ner_filters: dict, metric_configs: pd.DataFrame, policy: InsightPolicy) -> list[TableSpec]:
    """Create one direct table spec plus one spec per related metric group."""
    data_source = ner_filters.get("data")
    main_metrics = _resolve_metrics(
        ner_filters.get("main_metric", []), metric_configs, data_source
    )
    if not main_metrics:
        raise ValueError(f"{policy.display_name} insights require at least one main metric.")

    main_groups = tuple(dict.fromkeys(metric.group for metric in main_metrics))
    main_group_keys = {_key(group) for group in main_groups}
    main_metric_keys = {_key(metric.kpi_name) for metric in main_metrics}
    captured_metrics = tuple(
        metric for metric in _resolve_metrics(
            ner_filters.get("metric_captured", []), metric_configs, data_source
        )
        if _key(metric.kpi_name) not in main_metric_keys
    )
    captured_metric_keys = {_key(metric.kpi_name) for metric in captured_metrics}

    specs = [TableSpec("direct", main_groups, main_metrics)]
    related = {}
    for metric in _resolve_metrics(
            ner_filters.get("metric", []), metric_configs, data_source):
        group_key = _key(metric.group)
        if group_key not in main_group_keys and _key(metric.kpi_name) not in captured_metric_keys:
            related.setdefault(group_key, [metric.group, []])[1].append(metric)

    for related_group, metrics in related.values():
        specs.append(TableSpec(
            "related", main_groups, (*main_metrics, *metrics), related_group
        ))
    if captured_metrics:
        specs.append(TableSpec(
            "compound", main_groups, (*main_metrics, *captured_metrics),
            captured_metrics=captured_metrics,
        ))
    return specs


def _driver_and_dimensions(
        data: pd.DataFrame, policy: InsightPolicy) -> tuple[str, list[str]]:
    driver_key = _key(policy.driver_name)
    driver = next((column for column in data if _key(column) == driver_key), None)
    if driver is None:
        raise ValueError(f"{policy.display_name} insight data has no display driver column.")
    return driver, data.columns[:data.columns.get_loc(driver)].tolist()


def _prepare_metric_distribution(
        client_code: str, model_group_id: int, metric_dist: pd.DataFrame) -> pd.DataFrame:
    if metric_dist.empty or not {"metric", "time", "q1", "q3"}.issubset(metric_dist.columns):
        return pd.DataFrame()
    output = utils.normalize_time(metric_dist.copy())
    output = readout_utils.apply_level_mapping_display(client_code, model_group_id, output)
    output["_time_key"] = output["time_norm"].map(utils.time_sort_key)
    output[["q1", "q3"]] = output[["q1", "q3"]].apply(pd.to_numeric, errors="coerce")
    return output.dropna(subset=["q1", "q3"])


def _add_latest_level(
        data: pd.DataFrame, metric: MetricSpec, value_col: str, latest_time: str,
        dimensions: list[str], metric_dist: pd.DataFrame) -> str | None:
    if metric_dist.empty:
        return None
    metric_keys = {_key(metric.source), _key(metric.kpi_name), _key(metric.alias)}
    latest_key = utils.time_sort_key(latest_time)
    q_df = metric_dist[
        metric_dist["metric"].map(_key).isin(metric_keys)
        & metric_dist["_time_key"].map(lambda value: value == latest_key)
    ]
    if q_df.empty:
        return None

    group_keys = [column for column in dimensions if column in q_df]

    def normalized(values):
        return tuple(_key(value) for value in values)

    if group_keys:
        thresholds = {
            normalized(key if isinstance(key, tuple) else (key,)): (row.q1, row.q3)
            for key, row in q_df.groupby(group_keys, dropna=False)[["q1", "q3"]].first().iterrows()
        }
    else:
        dimension_columns = q_df.columns[:q_df.columns.get_loc("time")]
        overall = q_df[
            q_df[dimension_columns].apply(
                lambda column: column.map(_key).eq("overall")
            ).all(axis=1)
        ]
        candidates = overall if len(overall) == 1 else q_df
        unique = candidates[["q1", "q3"]].drop_duplicates()
        if len(unique) != 1:
            return None
        global_thresholds = tuple(unique.iloc[0])

    def classify(row):
        value = row[value_col]
        if pd.isna(value):
            return np.nan
        if "(mm)" in _key(value_col):
            value *= 1_000_000
        pair = thresholds.get(normalized(row[group_keys].tolist())) if group_keys else global_thresholds
        if pair is None:
            return np.nan
        return "low" if value < pair[0] else "high" if value > pair[1] else "moderate"

    level_col = f"{metric.alias} level {latest_time}"
    data[level_col] = data.apply(classify, axis=1)
    return level_col


def _format_table(
        data: pd.DataFrame, metric_columns: dict[MetricSpec, list[tuple[str, str]]],
        change_columns: list[str], metric_configs: pd.DataFrame) -> pd.DataFrame:
    output = data.copy()
    for metric, period_columns in metric_columns.items():
        metric_format = utils.get_metric_format(metric_configs, metric.kpi_name)
        decimals = int(metric_format.get("NumberDisplayDecimals") or 0)
        prefix = metric_format.get("Prefix") or ""
        postfix = metric_format.get("Postfix") or ""
        for _, column in period_columns:
            output[column] = output[column].map(
                lambda value, d=decimals, p=prefix, s=postfix:
                utils._format_cell(value, d, p, s)
            )
    for column in change_columns:
        output[column] = output[column].map(
            lambda value: utils._format_cell(value, 2, postfix="%")
        )
    return output


def _metric_column_names(metric: MetricSpec, policy: InsightPolicy) -> tuple[str, ...]:
    return (metric.alias, *policy.column_aliases.get(_key(metric.alias), ()))


def _metric_period_column(
        column_lookup: dict[str, str], metric: MetricSpec,
        policy: InsightPolicy, time: str) -> tuple[str | None, bool]:
    for name in _metric_column_names(metric, policy):
        for suffix in ("", " (MM)"):
            source = column_lookup.get(_key(f"{name} {time}{suffix}"))
            if source is not None:
                return source, bool(suffix)
    return None, False


def _present_metrics(
        table: pd.DataFrame, metrics: tuple[MetricSpec, ...],
        policy: InsightPolicy) -> tuple[MetricSpec, ...]:
    times = utils.sorted_times_from_columns(table.columns)
    column_lookup = {_key(column): column for column in table.columns}
    return tuple(
        metric for metric in metrics
        if any(
            _metric_period_column(column_lookup, metric, policy, time)[0] is not None
            for time in times
        )
    )


def build_table_for_insights(
        data: pd.DataFrame, table_spec: TableSpec, time_tag: str,
        metric_dist: pd.DataFrame, metric_configs: pd.DataFrame,
        policy: InsightPolicy) -> pd.DataFrame:
    times = utils.sorted_times_from_columns(data.columns)
    if time_tag == "snapshot":
        selected_times = times[:1]
    elif time_tag == "period_over_period_change":
        if len(times) < 2:
            raise ValueError(f"Period-over-period {policy.display_name} insights require two periods.")
        selected_times = times[:2]
    elif time_tag in {"multiple_periods", "trend"}:
        selected_times = times
    else:
        raise ValueError(f"Unsupported time_tag: {time_tag}")

    driver, dimensions = _driver_and_dimensions(data, policy)
    column_lookup = {_key(column): column for column in data.columns}
    source_columns = []
    rename_columns = {}
    metric_columns = {}
    for metric in table_spec.metrics:
        period_columns = []
        for time in selected_times:
            source, is_millions = _metric_period_column(
                column_lookup, metric, policy, time
            )
            if source is not None:
                target = f"{metric.alias} {time}{' (MM)' if is_millions else ''}"
                source_columns.append(source)
                rename_columns[source] = target
                period_columns.append((time, target))
        if period_columns:
            metric_columns[metric] = period_columns

    if not source_columns:
        return pd.DataFrame()
    output = data[dimensions + [driver] + source_columns].copy().rename(columns=rename_columns)
    value_columns = [column for columns in metric_columns.values() for _, column in columns]
    output = utils.strip_sign_and_unit(output, value_columns, char="")

    ordered_columns = dimensions + [driver]
    change_columns = []
    for metric, period_columns in metric_columns.items():
        columns_by_time = dict(period_columns)
        ordered_columns.extend(columns_by_time.values())
        latest_col = columns_by_time.get(selected_times[0])
        previous_col = columns_by_time.get(selected_times[1]) if len(selected_times) >= 2 else None
        if policy.add_change and time_tag in {"period_over_period_change", "trend"} and latest_col and previous_col:
            change_col = f"% Change in {metric.alias} {selected_times[0]}"
            output = utils.calculate_metric_change(
                output, metric.alias, selected_times[:2], title_metric=False
            )
            if change_col in output:
                change_columns.append(change_col)
                ordered_columns.append(change_col)
        if policy.add_trend and time_tag == "trend":
            metric_times = [time for time, _ in period_columns]
            output = utils.label_metric_trend(output, metric.alias, metric_times)
            trend_col = f"{metric.alias} trend"
            if trend_col in output:
                ordered_columns.append(trend_col)
        if policy.add_level and latest_col:
            level_col = _add_latest_level(
                output, metric, latest_col, selected_times[0], dimensions, metric_dist
            )
            if level_col:
                ordered_columns.append(level_col)

    output = output[ordered_columns].dropna(subset=value_columns, how="all")
    return _format_table(output, metric_columns, change_columns, metric_configs)


def _format_template(template, **values) -> str:
    try:
        return str(template).format(**values)
    except (KeyError, IndexError, ValueError):
        return str(template)


def _add_config(output: dict, name: str, instruction: str, data: pd.DataFrame) -> None:
    if data.empty:
        return
    unique_name, suffix = name, 2
    while unique_name in output:
        unique_name = f"{name} ({suffix})"
        suffix += 1
    output[unique_name] = {"instruction": instruction, "data": data.to_json(orient="records")}


def build_config_for_insights(
        client_code: str, model_group_id: int, query: str,
        data: pd.DataFrame, agg_data: pd.DataFrame, readout_data,
        metric_configs: pd.DataFrame, intention: str | None = None) -> dict:
    policy = get_policy(intention)
    table_specs = build_table_specs(readout_data.ner_filters, metric_configs, policy)
    time_tag = readout_data.ner_filters.get("time_tag", "snapshot")
    prompt_instructions = readout_utils.load_insight_instructions(
        client_code, model_group_id, intention=intention
    )
    general_instructions, general_sections = utils._instruction_maps(
        prompt_instructions, "overall"
    )
    instructions, section_names = utils._instruction_maps(prompt_instructions, time_tag)
    instructions = {**general_instructions, **instructions}
    section_names = {**general_sections, **section_names}
    metric_dist = pd.DataFrame()
    if policy.add_level:
        metric_dist = _prepare_metric_distribution(
            client_code,
            model_group_id,
            readout_utils.load_metric_distribution(
                client_code,
                model_group_id,
                file_type=readout_data.ner_filters.get("data"),
            ),
        )

    output = {}
    for table_spec in table_specs:
        config_suffix = "single" if table_spec.kind == "direct" else table_spec.kind
        config_key = f"{policy.config_prefix}_{config_suffix}"
        related_group = table_spec.related_group or ""
        default_name = {
            "direct": "{metric} by {level} {driver}",
            "related": "{metric} vs {related_metric_group} by {level} {driver}",
            "compound": "{metric} with mentioned metrics by {level} {driver}",
        }[table_spec.kind]
        for level, level_data in (("aggregated", agg_data), ("detailed", data)):
            if level_data.empty:
                continue
            table = build_table_for_insights(
                level_data, table_spec, time_tag, metric_dist, metric_configs, policy
            )
            present_metrics = _present_metrics(table, table_spec.metrics, policy)
            captured_metrics = tuple(
                metric for metric in present_metrics
                if metric in table_spec.captured_metrics
            )
            main_metrics = tuple(
                metric for metric in present_metrics
                if metric.group in table_spec.main_groups and metric not in captured_metrics
            )
            related_metrics = tuple(
                metric for metric in present_metrics
                if metric.group == table_spec.related_group
            )
            values = {
                "metric": policy.display_name,
                "main_metric_group": ", ".join(table_spec.main_groups),
                "main_metrics": ", ".join(metric.alias for metric in main_metrics),
                "related_metric_group": related_group,
                "related_metrics": ", ".join(metric.alias for metric in related_metrics),
                "mentioned_metric": ", ".join(metric.alias for metric in captured_metrics),
                "level": level,
                "driver": policy.driver_name,
                "query": query,
            }
            _add_config(
                output,
                _format_template(section_names.get(config_key, default_name), **values),
                _format_template(
                    instructions.get(config_key, "Provide insights based on given data."), **values
                ),
                table,
            )
    return output
