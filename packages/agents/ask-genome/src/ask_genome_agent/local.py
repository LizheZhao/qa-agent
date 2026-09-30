"""
Provides local, file-based access to client-specific Ask Genome configuration data.
Ported from ask-genome-core/src/data/local.py, but only includes functions needed by planning and
Question Understanding, not Postprocessing, Insights, or readout data.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, cast

import pandas as pd

logger = logging.getLogger(__name__)


def get_client_data_dir_path() -> str:
    return os.path.join(
        os.environ["ASK_GENOME_DATA_DIR"],
        os.environ["ASK_GENOME_CLIENT_CODE"],
        os.environ["ASK_GENOME_MODEL_GROUP_ID"],
    )


def get_data_path(key: str, override_filename: str | None = None) -> str:
    # Only the one key config.py's loaders actually use; the source's full map also covers
    # NerData/ReadoutData/Insights filenames, out of scope here.
    key_to_filename = {"feasibility_config": "feasibility_config.json"}
    filename = override_filename if override_filename is not None else key_to_filename[key]
    return os.path.join(get_client_data_dir_path(), filename)


def read_json(path: str) -> dict[str, Any]:
    with open(path) as file:
        return cast(dict[str, Any], json.load(file))


def get_client_data_config(
    client_code: str, model_group_id: int
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    data_path = get_client_data_dir_path()
    df_query_intentions = pd.read_csv(os.path.join(data_path, "map_dict.csv"), low_memory=False)
    df_data_levels = pd.read_csv(
        os.path.join(os.environ["ASK_GENOME_DATA_DIR"], "data_levels.csv"), low_memory=False
    )

    metric_info = cast(
        "dict[str, Any]",
        df_query_intentions.set_index("intentionName")
        .rename(columns={"relatedKPIs": "metric", "dataSource": "data"})
        .T.to_dict(),
    )
    for _, v in metric_info.items():
        v["metric"] = json.loads(v["metric"])
        v["mainMetric"] = [x.strip() for x in v["mainMetric"].split(",")]
        v["sortMetric"] = [x.strip() for x in v["sortMetric"].split(",")]

    df_data_levels = df_data_levels[
        (df_data_levels.clientCode == client_code) & (df_data_levels.modelGroupId == model_group_id)
    ]
    if len(df_data_levels) > 0:
        data_levels = cast(
            "dict[str, Any]",
            df_data_levels[
                ["biLevels", "brLevels", "brLevelsToIgnore", "spacyOverrideColumns"]
            ].to_dict("records")[0],
        )
    else:
        data_levels = {}
    defaults = {
        "biLevels": '["activity_group", "measure_group", "measure"]',
        "brLevels": '["business_driver", "business_driver_detail", "activity_group"]',
        "spacyOverrideColumns": (
            '["activity_group", "measure_group", "measure", "business_driver", '
            '"business_driver_detail"]'
        ),
        "brLevelsToIgnore": "[]",
    }
    for col, default_list in defaults.items():
        data_levels[col] = json.loads(data_levels.get(col, default_list))

    acronym_mapping_file = os.path.join(os.environ["ASK_GENOME_DATA_DIR"], "acronym_mapping.csv")
    if os.path.exists(acronym_mapping_file):
        acronym_df = pd.read_csv(acronym_mapping_file, low_memory=False)
        acronym_df = acronym_df[acronym_df.client.isin(["common", client_code.lower()])]
        acronym_mapping = cast(
            "dict[str, Any]", acronym_df.set_index("acronym")["definition"].to_dict()
        )
    else:
        logger.info("Couldn't find acronym file for %s %s", client_code, model_group_id)
        acronym_mapping = {}
    return metric_info, data_levels, acronym_mapping


def get_ner_postprocess_indicator(client_code: str, model_group_id: int) -> dict[str, Any]:
    file_path = os.path.join(get_client_data_dir_path(), "ner_postprocess_indicator.csv")
    if not os.path.exists(file_path):
        logger.info(
            "Couldn't find ner postprocess indicators for %s %s", client_code, model_group_id
        )
        return {}

    indicator_df = pd.read_csv(file_path, low_memory=False)
    indicator_df = indicator_df[indicator_df["enabled"] == 1]
    indicator = cast(
        "dict[str, Any]", indicator_df.set_index("indicator")["indicator_value"].to_dict()
    )
    for key in ("period_priority", "non_trend_report_length", "trend_report_length"):
        if key in indicator:
            indicator[key] = [x.strip() for x in indicator[key].split(",")]
    if "report_yoy" in indicator:
        indicator["report_yoy"] = bool(indicator.get("report_yoy", 0))
    return indicator


def get_spacy_config(
    client_code: str, model_group_id: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    data_path = get_client_data_dir_path()
    core_dimension_file = os.path.join(data_path, "core_dimensions.jsonl")
    core_dimension_filter_file = os.path.join(data_path, "core_dimension_filter.json")
    if os.path.exists(core_dimension_file) and os.path.exists(core_dimension_filter_file):
        core_dimension_filter = read_json(core_dimension_filter_file)
        core_dimensions = cast(
            "list[dict[str, Any]]",
            pd.read_json(path_or_buf=core_dimension_file, lines=True).to_dict("records"),
        )
    else:
        logger.info("Couldn't find spacy files for %s %s", client_code, model_group_id)
        core_dimension_filter, core_dimensions = {}, []
    return core_dimension_filter, core_dimensions


def get_term_variants(client_code: str, model_group_id: int) -> pd.DataFrame:
    variant_file = os.path.join(os.environ["ASK_GENOME_DATA_DIR"], "variants.csv")
    if os.path.exists(variant_file):
        variants = pd.read_csv(variant_file)
        return variants[variants.client.isin(["common", client_code.lower()])]
    logger.info("Couldn't find term variant files for %s %s", client_code, model_group_id)
    return pd.DataFrame(columns=["term", "variants", "client"])


def get_feasibility_config(client_code: str, model_group_id: int) -> dict[str, Any]:
    file_path = get_data_path("feasibility_config")
    if not os.path.isfile(file_path):
        logger.info("Couldn't find feasibility config for %s %s", client_code, model_group_id)
        return {}
    return read_json(file_path)


def get_client_data(client_code: str, model_group_id: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ported verbatim from ask-genome-core/src/data/local.py:get_client_data.

    Both frames are needed: which one a question reads is decided per intention, not per client,
    so any br-sourced intention (contribution, sales, source of change) has nothing to filter
    without br.csv.
    """

    data_path = get_client_data_dir_path()
    df_bi = pd.read_csv(os.path.join(data_path, "bi.csv"), low_memory=False)
    df_br = pd.read_csv(os.path.join(data_path, "br.csv"), low_memory=False)
    return df_bi, df_br


# Display columns, and what a metric with no formatting row is shown as.
_FORMAT_COLUMNS = (
    "MetricName",
    "NumberDisplayDecimals",
    "UsePowers",
    "Prefix",
    "Postfix",
    "Multiplier",
    "IsPercent",
)
# A calculated metric has no formatting row of its own; it is a percentage.
_CALCULATED_FORMAT: dict[str, Any] = {
    "NumberDisplayDecimals": 2,
    "UsePowers": 0,
    "Prefix": "",
    "Postfix": "%",
    "Multiplier": 1,
    "IsPercent": 0,
}
_METRIC_CONFIG_COLUMNS = [
    "code",
    "metricGroup",
    "kpiName",
    "domain",
    "alias",
    *_FORMAT_COLUMNS[1:],
]


def get_metric_configs(client_code: str, model_group_id: int) -> pd.DataFrame:
    """metric_mapping.csv joined to the shared standardized metrics and this client's formatting.

    Ported from readout_utils.get_metric_configs and _standardize_metric_configs, with two fixes.

    Two names per group, kept apart as the multi-turn source now keeps them. `code` is what
    map_dict.csv uses ('cost_per_activity'), so filtering and ranking look groups up by it.
    `metricGroup` is the standardized name ('Cost per Activity'), which the GPT readout matches
    on lower-cased. The older source looked groups up by the name, so multi-word groups resolved
    to nothing and LINKEDIN's 'cost per' intention could never match a row.

    A missing metric_formatting.json. The source selects its display columns out of an empty
    frame and raises KeyError, which is five of the six onboarded clients. Here the columns are
    filled with defaults instead, so a client without the file gets unformatted numbers rather
    than no answer.
    """

    mapping_path = os.path.join(get_client_data_dir_path(), "metric_mapping.csv")
    standardized_path = os.path.join(os.environ["ASK_GENOME_DATA_DIR"], "standardized_metrics.csv")
    for name, path in (
        ("metric_mapping.csv", mapping_path),
        ("standardized_metrics.csv", standardized_path),
    ):
        if not os.path.exists(path):
            logger.warning("%s not found at %s", name, path)
            return pd.DataFrame(columns=_METRIC_CONFIG_COLUMNS)

    merged = pd.read_csv(mapping_path, low_memory=False).merge(
        _metric_formatting(), how="left", left_on="kpiName", right_on="MetricName"
    )
    merged = merged.merge(
        pd.read_csv(standardized_path, low_memory=False),
        how="left",
        left_on="standardizedMetricId",
        right_on="id",
    ).rename(columns={"metricName": "metricGroup"})

    return _standardize_metric_configs(merged)


def _metric_formatting() -> pd.DataFrame:
    """This client's display settings, or an empty frame with the right columns."""

    path = os.path.join(get_client_data_dir_path(), "metric_formatting.json")
    if not os.path.exists(path):
        return pd.DataFrame(columns=list(_FORMAT_COLUMNS))
    with open(path) as file:
        records = json.load(file)
    formatting = pd.DataFrame(records)
    for column in _FORMAT_COLUMNS:
        if column not in formatting.columns:
            formatting[column] = pd.NA
    return formatting[list(_FORMAT_COLUMNS)]


def _standardize_metric_configs(out: pd.DataFrame) -> pd.DataFrame:
    """Ported from readout_utils._standardize_metric_configs.

    Fills formatting blanks for calculated metrics and settles what each metric is called:
    MetricName falls back to alias then kpiName, and alias back to MetricName, so neither is
    ever blank whichever of the three files named it.
    """

    def blank(series: pd.Series) -> pd.Series:
        return series.isna() | series.astype(str).str.strip().eq("")

    calculated = out["domain"].astype(str).str.lower() == "calculated"
    for column, value in _CALCULATED_FORMAT.items():
        current = out[column] if column in out.columns else pd.Series(pd.NA, index=out.index)
        out[column] = current.mask(calculated & blank(current), value)

    name = out["MetricName"].mask(blank(out["MetricName"]))
    alias = out["alias"].mask(blank(out["alias"]))
    out["MetricName"] = name.fillna(alias).fillna(out["kpiName"])
    out["alias"] = alias.fillna(out["MetricName"])

    out["NumberDisplayDecimals"] = (
        pd.to_numeric(out["NumberDisplayDecimals"], errors="coerce").fillna(2).astype(int)
    )
    for column in ("UsePowers", "Multiplier", "IsPercent"):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    for column in ("Prefix", "Postfix"):
        out[column] = out[column].fillna("").astype(str).str.strip()
    return out


def get_questionnaire(
    client_code: str, model_group_id: int
) -> tuple[dict[str, Any], dict[str, list[str]]]:
    """Ported verbatim from ask-genome-core/src/data/local.py:get_questionnaire.

    Returns (questionnaire, level_type):
      questionnaire: {dimension: {"field": [...], "context": ..., "prompt": ...}}
      level_type: {level_type_name: [field, ...]} built from the CSV's optional "level" column.
    """

    file_path = os.path.join(get_client_data_dir_path(), "questionnaire.csv")
    questionnaire_df = pd.read_csv(file_path, low_memory=False)
    indexed = questionnaire_df.fillna("").set_index("dimension")[["field", "context", "prompt"]]
    questionnaire = cast("dict[str, Any]", indexed.T.to_dict())
    for v in questionnaire.values():
        v["field"] = [x.strip() for x in v["field"].split(",")]

    if "level" not in questionnaire_df.columns:
        questionnaire_df["level"] = ""
    else:
        questionnaire_df["level"] = questionnaire_df["level"].fillna("")
    level_type_df = questionnaire_df[questionnaire_df["level"] != ""][["field", "level"]]
    level_type: dict[str, list[str]] = {}
    for _, row in level_type_df.iterrows():
        levels = [x.strip() for x in row["level"].split(",")]
        fields = [x.strip() for x in row["field"].split(",")]
        for lvl, f in zip(levels, fields, strict=False):
            level_type.setdefault(lvl, []).append(f)
    return questionnaire, level_type


def get_learning_examples(client_code: str, model_group_id: int) -> pd.DataFrame:
    """Ported verbatim from ask-genome-core/src/data/local.py:get_learning_examples."""

    file_path = os.path.join(get_client_data_dir_path(), "in_context_examples.csv")
    if not os.path.exists(file_path):
        logger.info("Couldn't find in-context examples for %s %s", client_code, model_group_id)
        return pd.DataFrame(columns=["query", "intention"])
    return pd.read_csv(file_path, low_memory=False)
