"""Describes a filtered table in plain text so the response model knows what the rows cover.

The table has no meaning on its own: a row's hierarchy columns say which granularity it reports at,
and the client's level and tagging columns say how it is cut. This reads those off the frame and
lists them with their distinct values, so the model can say what the answer covers and what it
leaves out. It reads only the frame, the client's level_type and the reporting levels.
"""

from __future__ import annotations

import pandas as pd

from ask_genome_agent.nodes.question_understanding.support.levels import POSSIBLE_LEVELS

_MAX_VALUES = 8


def describe_dataframe(
    frame: pd.DataFrame, level_type: dict[str, list[str]], data_levels: list[str]
) -> str:
    if frame.empty:
        return "The table has no rows."

    lines = [f"{len(frame)} rows, {len(frame.columns)} columns."]

    if "time" in frame.columns and "period_type" in frame.columns:
        period_types = ", ".join(str(kind) for kind in frame["period_type"].dropna().unique())
        lines.append(f"Time: {frame['time'].min()} to {frame['time'].max()} ({period_types}).")

    if "metric" in frame.columns:
        lines.append(f"Metrics: {_values(frame, 'metric')}.")

    if "value" in frame.columns:
        value = pd.to_numeric(frame["value"], errors="coerce")
        lines.append(
            f"Value: min {value.min():g}, max {value.max():g}, {int(value.isna().sum())} missing."
        )

    lines.append(f"Reporting levels: {', '.join(data_levels) or 'none'}.")
    if "record_level" in frame.columns:
        counts = frame["record_level"].value_counts()
        per_level = ", ".join(f"{name} {count}" for name, count in counts.items())
        lines.append(f"Rows per level: {per_level}.")

    hierarchy = [column for column in POSSIBLE_LEVELS if column in frame.columns]
    for column in hierarchy:
        lines.append(f"Hierarchy {column}: {_values(frame, column)}.")

    for group, columns in level_type.items():
        for column in columns:
            if column in frame.columns and column not in hierarchy:
                lines.append(f"{group} {column}: {_values(frame, column)}.")

    return "\n".join(lines)


def _values(frame: pd.DataFrame, column: str) -> str:
    """Distinct count, then the first few values."""

    distinct = frame[column].dropna().unique().tolist()
    shown = ", ".join(str(value) for value in distinct[:_MAX_VALUES])
    more = f", +{len(distinct) - _MAX_VALUES} more" if len(distinct) > _MAX_VALUES else ""
    return f"{len(distinct)} distinct ({shown}{more})"
