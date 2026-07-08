"""Offline builder for feasibility_config.json (run at data refresh).

Computes, per data_source (bi/br): the metric list, per-(metric, period_type)
time ranges, and marginal coverage per dimension value for the columns that
questionnaire.csv classifies as level/tagging. No value cross-product is stored;
joint co-occurrence is answered live via groupby at query time.

Usage:
    python scripts/build_feasibility_config.py --client-code COLGUS --model-group-id 1
"""
import os
import sys
import json
import logging
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.data_interface import NerData          # noqa: E402
from src.data.local import get_data_path              # noqa: E402

logger = logging.getLogger(__name__)

METRIC_COL = "metric"
PERIOD_TYPE_COL = "period_type"
START_COL = "start"
END_COL = "end"

# level_type buckets to enumerate (level + tagging categories from questionnaire.csv)
COVERAGE_LEVEL_TYPES = ["general_level", "general_tagging", "halo_level", "halo_tagging"]
HIGH_CARDINALITY_WARN = 500


def _to_native(value):
    return value.item() if hasattr(value, "item") else value


def _safe_min(series):
    series = series.dropna()
    return None if series.empty else _to_native(series.min())


def _safe_max(series):
    series = series.dropna()
    return None if series.empty else _to_native(series.max())


def _period_ranges(df) -> dict:
    """Per period_type -> {start, end, count}; count = number of distinct periods."""
    ranges = {}
    if PERIOD_TYPE_COL not in df.columns:
        return ranges
    for period_type, sub in df.groupby(PERIOD_TYPE_COL):
        entry = {}
        if END_COL in sub.columns:
            entry["count"] = int(sub[END_COL].nunique())
            entry["end"] = _safe_max(sub[END_COL])
        else:
            entry["count"] = int(len(sub))
        if START_COL in sub.columns:
            entry["start"] = _safe_min(sub[START_COL])
        ranges[str(period_type)] = entry
    return ranges


def _metrics_present(df) -> list:
    if METRIC_COL not in df.columns:
        return []
    return sorted(df[METRIC_COL].dropna().astype(str).unique().tolist())


def build_data_source(df, level_type: dict) -> dict:
    metrics = _metrics_present(df)

    metric_time_ranges = {}
    if METRIC_COL in df.columns:
        for metric, sub in df.groupby(METRIC_COL):
            metric_time_ranges[str(metric)] = _period_ranges(sub)

    coverage_cols = []
    for level in COVERAGE_LEVEL_TYPES:
        coverage_cols.extend(level_type.get(level, []))

    dimensions = {}
    for col in coverage_cols:
        if col not in df.columns:
            logger.warning("coverage column '%s' not in data, skipping", col)
            continue
        category = next((lt for lt in COVERAGE_LEVEL_TYPES if col in level_type.get(lt, [])), None)
        values = df[col].dropna().astype(str).unique().tolist()
        if len(values) > HIGH_CARDINALITY_WARN:
            logger.warning("column '%s' has %d distinct values (high cardinality)", col, len(values))
        col_values = {}
        for value in values:
            sub = df[df[col].astype(str) == value]
            col_values[value] = {
                "period_coverage": _period_ranges(sub),
                "metrics_present": _metrics_present(sub),
            }
        dimensions[col] = {"category": category, "values": col_values}

    return {"metrics": metrics, "metric_time_ranges": metric_time_ranges, "dimensions": dimensions}


def build_feasibility_config(client_code: str, model_group_id: int, built_at: str) -> dict:
    ner_data = NerData.from_local(client_code, model_group_id)
    return {
        "client_code": client_code,
        "model_group_id": model_group_id,
        "built_at": built_at,
        "data_sources": {
            "bi": build_data_source(ner_data.df_bi, ner_data.level_type),
            "br": build_data_source(ner_data.df_br, ner_data.level_type),
        },
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Build feasibility_config.json for a client/model group.")
    parser.add_argument("--client-code", required=True)
    parser.add_argument("--model-group-id", required=True)
    parser.add_argument("--built-at", default="")
    args = parser.parse_args()

    # if not os.getenv("DATA_DIR"):
    #     raise RuntimeError("DATA_DIR env var is required")
    os.environ["DATA_DIR"] = "/datascience6/data/ask-genome-data/"
    os.environ["CLIENT_CODE"] = args.client_code
    os.environ["MODEL_GROUP_ID"] = str(args.model_group_id)

    built_at = args.built_at or datetime.now().strftime("%Y-%m-%d")
    config = build_feasibility_config(args.client_code, int(args.model_group_id), built_at)

    out_path = get_data_path("feasibility_config")
    with open(out_path, "w") as f:
        json.dump(config, f, indent=2)
    logger.info("feasibility config written to %s", out_path)


if __name__ == "__main__":
    main()
