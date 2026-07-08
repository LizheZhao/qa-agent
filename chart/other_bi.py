"""
chart/other_bi.py
Other Business Intelligence chart functions following the build/render separation pattern.

Pattern
-------
- build_*_chart_configs(...)  — pure Python/pandas, builds chart_configs dict, ZERO st.* calls
- render_*_charts(...)        — pure Streamlit rendering, accepts pre-computed insights dict, ZERO LLM calls
- Orchestrator (keeps current name) — calls build → generate_chart_insights_parallel_cached → render
                                      → returns (chart_configs, insights)
"""

import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
import src.model.readout_utils as readout_utils
import viz_utils as viz_utils
from chart.common import (
    render_chart_insight_box,
    generate_chart_insights_parallel_cached,
    _section,
    table_element,
)
import time


# ---------------------------------------------------------------------------
# Private chart-building helpers
# ---------------------------------------------------------------------------

def _other_bi_trend_table_str(df):
    """Table-text summary of the other-BI trend frame. '' when empty/insufficient. No st.* calls."""
    if df.empty or df["TimeLabel"].nunique() <= 1:
        return ""
    df = df.drop(["Metric", "Time", "Value", "group"], axis=1, errors="ignore").copy()
    if df["Tactics"].nunique() < 3:
        return ""
    _, table_str, _, _ = readout_utils.table_to_text(pd.DataFrame(), df, take_head=False)
    return table_str


def _build_other_bi_trend_chart(df, m_name, title, sorted_labels):
    """
    Pure-data version of _other_bi_trend_chart.
    Returns chart | None.  Use _other_bi_trend_table_str(df) for the table text.
    No st.* calls.
    """
    if df.empty or df["TimeLabel"].nunique() <= 1:
        return None

    latest_time = sorted_labels[-1]
    df_latest = df[df["TimeLabel"] == latest_time]
    if not df_latest.empty:
        tactics_order = df_latest.sort_values("Value", ascending=False)["Tactics"].tolist()
    else:
        tactics_order = df["Tactics"].unique().tolist()

    reversed_time_order = sorted_labels[::-1]
    df = df.drop(["Metric", "Time", "ValueFormatted", "group"], axis=1, errors="ignore").copy()

    if df["Tactics"].nunique() < 3:
        return None

    base = alt.Chart(df).encode(
        x=alt.X(
            "Tactics:N",
            title="Tactics",
            sort=tactics_order,
            axis=alt.Axis(labelAngle=0),
        ),
        xOffset=alt.XOffset("TimeLabel:N", sort=reversed_time_order),
    )

    bar = base.mark_bar(opacity=0.8).encode(
        y=alt.Y(
            "Value:Q",
            title=m_name,
            axis=alt.Axis(titleColor="steelblue"),
        ),
        color=alt.Color(
            "TimeLabel:N",
            sort=reversed_time_order,
            legend=alt.Legend(title="Time"),
            scale=alt.Scale(scheme="blues"),
        ),
        tooltip=["Tactics", "TimeLabel", "Value"],
    )

    chart = (
        alt.layer(bar)
        .resolve_scale(y="independent")
        .properties(title=title, width=1200, height=400)
    )
    return chart


def _render_other_bi_trend_chart(df_original, m_name, title, sorted_labels):
    """
    Rendering version of _other_bi_trend_chart — includes st.expander for the
    data table.  Returns (chart | None, table_str | None).
    """
    if df_original.empty or df_original["TimeLabel"].nunique() <= 1:
        return None, None

    latest_time = sorted_labels[-1]
    df_latest = df_original[df_original["TimeLabel"] == latest_time]
    if not df_latest.empty:
        tactics_order = df_latest.sort_values("Value", ascending=False)["Tactics"].tolist()
    else:
        tactics_order = df_original["Tactics"].unique().tolist()

    reversed_time_order = sorted_labels[::-1]
    df = df_original.drop(
        ["Metric", "Time", "group"], axis=1, errors="ignore"
    ).copy()

    with st.expander("Show data table"):
        st.dataframe(df.drop('Value', axis=1), use_container_width=True)

    if df["Tactics"].nunique() < 3:
        return None, None

    _, table_str, _, _ = readout_utils.table_to_text(pd.DataFrame(), df, take_head=False)

    base = alt.Chart(df).encode(
        x=alt.X(
            "Tactics:N",
            title="Tactics",
            sort=tactics_order,
            axis=alt.Axis(labelAngle=0),
        ),
        xOffset=alt.XOffset("TimeLabel:N", sort=reversed_time_order),
    )

    bar = base.mark_bar(opacity=0.8).encode(
        y=alt.Y(
            "Value:Q",
            title=m_name,
            axis=alt.Axis(titleColor="steelblue"),
        ),
        color=alt.Color(
            "TimeLabel:N",
            sort=reversed_time_order,
            legend=alt.Legend(title="Time"),
            scale=alt.Scale(scheme="blues"),
        ),
        tooltip=["Tactics", "TimeLabel", "Value"],
    )

    chart = (
        alt.layer(bar)
        .resolve_scale(y="independent")
        .properties(title=title, width=1200, height=400)
    )
    return chart, table_str


# ===========================================================================
# other_bi_snapshot
# ===========================================================================

def build_other_bi_snapshot_chart_configs(
    df,
    selected_levels,
    outliers,
    driver_col,
    metric_cols,
    sorted_time,
    readout,
    match,
    is_specific,
):
    """
    build chart_configs dict for the other BI snapshot view.
    """
    chart_configs = {}
    m_name = readout.ner_filters["main_metric"][0]
    select_time = viz_utils.get_time_align_dict(sorted_time)

    df_group, outlier_str, _ = viz_utils.other_bi_snapshot_data(
        df, m_name, outliers, selected_levels, driver_col, metric_cols,
        readout, match, is_specific, mode="recent",
    )
    groups = df_group["group"].dropna().unique().tolist()

    for g in groups:
        df_long = (
            df_group.drop("group", axis=1)
            if g == "All"
            else df_group[df_group["group"] == g]
        )
        if not df_long.empty:
            df_long = df_long.copy()
            df_long["Axis"] = df_long["Metric Type"].apply(
                lambda x: "left" if m_name in x and "change" not in x else "right"
            )
            df_left = df_long[df_long["Axis"] == "left"]
            tactics_order = (
                df_left.sort_values("Value", ascending=False)["Tactics"].tolist()
                if not df_left.empty
                else df_long["Tactics"].unique().tolist()
            )
            if len(tactics_order) >= 3:
                chart_configs[f"{m_name}_snapshot_{g}"] = {
                    "query": f"You are given a dataset about most recent {g}.",
                    "instruction": "Provide a clear, concise, business-style insight summary.",
                    "data": df_long.to_json(orient="records"),
                }

    return chart_configs


def render_other_bi_snapshot(
    df,
    selected_levels,
    outliers,
    driver_col,
    metric_cols,
    sorted_time,
    readout,
    match,
    is_specific,
    insights,
):
    """
    Pure Streamlit rendering of the other BI snapshot view.
    Accepts pre-computed insights dict. ZERO LLM calls.
    """
    m_name = readout.ner_filters["main_metric"][0]
    select_time = viz_utils.get_time_align_dict(sorted_time)

    df_group, outlier_str, _ = viz_utils.other_bi_snapshot_data(
        df, m_name, outliers, selected_levels, driver_col, metric_cols,
        readout, match, is_specific, mode="recent",
    )
    groups = df_group["group"].dropna().unique().tolist()

    for g in groups:
        g_title = f"({g})" if g != "All" else ""
        df_long = (
            df_group.drop("group", axis=1)
            if g == "All"
            else df_group[df_group["group"] == g]
        )
        if not df_long.empty:
            with st.expander("Show data table"):
                st.dataframe(df_long.drop('Value', axis =1), use_container_width=True)

            df_long = df_long.copy()
            df_long["Axis"] = df_long["Metric Type"].apply(
                lambda x: "left" if m_name in x and "change" not in x else "right"
            )
            df_left = df_long[df_long["Axis"] == "left"]
            df_right = df_long[df_long["Axis"] == "right"]
            tactics_order = (
                df_left.sort_values("Value", ascending=False)["Tactics"].tolist()
                if not df_left.empty
                else df_long["Tactics"].unique().tolist()
            )

            bar_left = alt.Chart(df_left).mark_bar(opacity=0.8, color="steelblue").encode(
                x=alt.X(
                    "Tactics:N",
                    sort=tactics_order,
                    title="Tactics",
                    axis=alt.Axis(labelAngle=0),
                ),
                y=alt.Y(
                    "Value:Q",
                    axis=alt.Axis(title=f"{m_name.title()}", titleColor="steelblue"),
                ),
                xOffset="Metric Type:N",
                tooltip=["Tactics", "Metric Type", "Value"],
            )
            line_right = alt.Chart(df_right).mark_line(point=True, color="orange").encode(
                x=alt.X(
                    "Tactics:N",
                    sort=tactics_order,
                    title="Tactics",
                    axis=alt.Axis(labelAngle=0),
                ),
                y=alt.Y(
                    "Value:Q",
                    axis=alt.Axis(
                        title=f"{m_name.title()} % Change", titleColor="orange"
                    ),
                ),
                tooltip=["Tactics", "Metric Type", "Value"],
            )
            dual_chart = (
                alt.layer(bar_left, line_right)
                .resolve_scale(y="independent")
                .properties(
                    title=(
                        f"{m_name.title()} and % Change in "
                        f"{next(iter(select_time))} {g_title}"
                    ),
                    width=900,
                    height=450,
                )
            )
            if len(tactics_order) >= 3:
                st.altair_chart(dual_chart, use_container_width=True)
                st.markdown(outlier_str)
                saved = insights.get(f"{m_name}_snapshot_{g}", "")
                if saved:
                    render_chart_insight_box(
                        f"{m_name.title()} Snapshot Insight", saved
                    )


def other_bi_snapshot(
    df,
    selected_levels,
    outliers,
    driver_col,
    metric_cols,
    sorted_time,
    readout,
    match,
    is_specific,
    llm_vendor,
):
    """
    Orchestrator for the other BI snapshot view.
    build → generate insights → render → returns (chart_configs, insights).
    """
    chart_configs = build_other_bi_snapshot_chart_configs(
        df, selected_levels, outliers, driver_col, metric_cols,
        sorted_time, readout, match, is_specific,
    )
    with st.spinner("Your visuals and insights are on the way..."):
        start = time.time()
        insights = generate_chart_insights_parallel_cached(chart_configs, llm_vendor)
        elapsed = time.time() - start
    render_other_bi_snapshot(
        df, selected_levels, outliers, driver_col, metric_cols,
        sorted_time, readout, match, is_specific, insights,
    )
    return chart_configs, insights


# ===========================================================================
# other_bi_trend_by_level
# ===========================================================================

def build_other_bi_trend_chart_configs(
    data, df, selected_levels, metric_cols, readout, match, is_specific
):
    """
    build chart_configs dict for the other BI trend view.
    """
    chart_configs = {}
    m_name = readout.ner_filters["main_metric"][0]
    df_combined, sorted_labels, _ = viz_utils.other_bi_trend_data(
        data, df, selected_levels, metric_cols, readout, match, is_specific
    )
    groups = df_combined["group"].dropna().unique().tolist()
    tables_for_llm = []
    has_chart = False

    if "All" not in groups:
        for g in groups:
            df_group = df_combined[df_combined["group"] == g]
            chart = _build_other_bi_trend_chart(
                df_group, m_name, f"{m_name.title()} — {g}", sorted_labels
            )
            table_str = _other_bi_trend_table_str(df_group)
            if chart is not None and table_str:
                has_chart = True
                # Per-group key for individual insight (used in render)
                chart_configs[f"{m_name}_trend_{g}"] = {
                    "query": f"You are given a dataset about {g}.",
                    "instruction": "Provide a clear, concise, business-style insight summary.",
                    "data": table_str,
                }
    else:
        levels = ["high", "moderate", "low"]
        for lvl in levels:
            sub = df_combined[df_combined[f"{m_name} level"] == lvl]
            chart = _build_other_bi_trend_chart(
                sub, m_name, f"{m_name.title()} — {lvl.title()}", sorted_labels
            )
            table_str = _other_bi_trend_table_str(sub)
            if chart is not None and table_str:
                has_chart = True
                tables_for_llm.append(f"{lvl.title()} data\n{table_str}")

        if has_chart and tables_for_llm:
            chart_configs[f"{m_name}_trend_by_level"] = {
                "query": f"You are given a dataset about {m_name.title()} trend by level.",
                "instruction": "Provide a clear, concise, business-style insight summary.",
                "data": str(tables_for_llm),
            }

    return chart_configs


def render_other_bi_trend(
    data, df, selected_levels, metric_cols, readout, match, is_specific, insights
):
    """
    Pure Streamlit rendering of the other BI trend view.
    Accepts pre-computed insights dict. ZERO LLM calls.
    """
    m_name = readout.ner_filters["main_metric"][0]
    df_combined, sorted_labels, _ = viz_utils.other_bi_trend_data(
        data, df, selected_levels, metric_cols, readout, match, is_specific
    )
    groups = df_combined["group"].dropna().unique().tolist()
    has_chart = False

    if "All" not in groups:
        for g in groups:
            df_group = df_combined[df_combined["group"] == g]
            ch, table_str = _render_other_bi_trend_chart(
                df_group, m_name, f"{m_name.title()} — {g}", sorted_labels
            )
            if ch is not None:
                c1, c2, c3 = st.columns([1, 3, 1])
                with c2:
                    st.altair_chart(ch, use_container_width=False)
                saved = insights.get(f"{m_name}_trend_{g}", "")
                if saved:
                    render_chart_insight_box(
                        f"{m_name.title()} — {g} Trend Insight", saved
                    )
    else:
        levels = ["high", "moderate", "low"]
        for lvl in levels:
            sub = df_combined[df_combined[f"{m_name} level"] == lvl]
            ch, table_str = _render_other_bi_trend_chart(
                sub, m_name, f"{m_name.title()} — {lvl.title()}", sorted_labels
            )
            if ch is not None:
                c1, c2, c3 = st.columns([1, 3, 1])
                has_chart = True
                with c2:
                    st.altair_chart(ch, use_container_width=False)

        if has_chart:
            saved = insights.get(f"{m_name}_trend_by_level", "")
            if saved:
                render_chart_insight_box(
                    f"{m_name.title()} Trend By Level Insight", saved
                )


def other_bi_trend_by_level(
    data, df, selected_levels, metric_cols, readout, match, is_specific, llm_vendor
):
    """
    Orchestrator for the other BI trend view.
    build → generate insights → render → returns (chart_configs, insights).
    """
    chart_configs = build_other_bi_trend_chart_configs(
        data, df, selected_levels, metric_cols, readout, match, is_specific
    )
    with st.spinner("Your visuals and insights are on the way..."):
        start = time.time()
        insights = generate_chart_insights_parallel_cached(chart_configs, llm_vendor)
        elapsed = time.time() - start
    render_other_bi_trend(
        data, df, selected_levels, metric_cols, readout, match, is_specific, insights
    )
    return chart_configs, insights
