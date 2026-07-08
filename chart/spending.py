"""
chart/spending.py
Spend-related chart functions following the build/render separation pattern.

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

def _spend_share_trend_table_str(df):
    """Table-text summary of the spend-share trend frame. '' when empty. No st.* calls."""
    if df.empty or df["Tactics"].nunique() <= 1:
        return ""
    df = df.drop(["Metric", "Value", "Time", "group"], axis=1, errors="ignore")
    _, table_str, _, _ = readout_utils.table_to_text(pd.DataFrame(), df, take_head=False)
    return table_str


def _build_spend_share_trend_chart(df, title, sorted_labels):
    """
    Pure-data version of _spend_share_trend_chart.
    Returns chart | None.  Use _spend_share_trend_table_str(df) for the table text.
    """
    if df.empty or df["Tactics"].nunique() <= 1:
        return None

    df = df.drop(["Metric", "ValueFormatted", "Time", "group"], axis=1, errors="ignore")

    time_order_map = {t: i for i, t in enumerate(sorted_labels)}
    df = df.copy()
    df["time_order"] = df["TimeLabel"].map(time_order_map)

    latest_time = sorted_labels[-1]
    df_latest = df[(df["Metric Type"] == "Spend") & (df["TimeLabel"] == latest_time)]
    if not df_latest.empty:
        tactics_order = df_latest.sort_values("Value", ascending=False)["Tactics"].tolist()
    else:
        tactics_order = df["Tactics"].unique().tolist()

    reversed_time_order = sorted_labels[::-1]
    base = alt.Chart(df).encode(
        x=alt.X(
            "Tactics:N",
            title="Tactics",
            axis=alt.Axis(labelAngle=0),
            sort=tactics_order,
        ),
        xOffset=alt.XOffset("TimeLabel:N", sort=reversed_time_order),
    )

    bar = base.transform_filter(
        alt.FieldOneOfPredicate(field="Metric Type", oneOf=["Share of Spend"])
    ).mark_bar(opacity=0.8).encode(
        y=alt.Y(
            "Value:Q",
            title="Share of Spend (%)",
            axis=alt.Axis(titleColor="orange"),
        ),
        color=alt.Color(
            "TimeLabel:N",
            sort=reversed_time_order,
            legend=alt.Legend(title="Time"),
            scale=alt.Scale(scheme="blues"),
        ),
        tooltip=["Tactics", "TimeLabel", "Value"],
    )

    line = base.transform_filter(
        alt.FieldOneOfPredicate(field="Metric Type", oneOf=["Spend"])
    ).mark_line(point={"filled": True, "size": 70}).encode(
        y=alt.Y(
            "Value:Q",
            title="Spend (MM)",
            axis=alt.Axis(titleColor="steelblue"),
        ),
        color=alt.value("#1f77b4"),
        detail="Tactics:N",
        order=alt.Order("time_order:Q"),
        tooltip=["Tactics", "TimeLabel", "Value"],
    )

    chart = (
        alt.layer(bar, line)
        .resolve_scale(y="independent")
        .properties(title=title, width=1200, height=400)
    )
    return chart


def _render_spend_share_trend_chart(df_original, title, sorted_labels):
    """
    Rendering version of _spend_share_trend_chart — includes st.expander for
    the data table and returns (chart | None, table_str | None).
    """
    if df_original.empty or df_original["Tactics"].nunique() <= 1:
        return None, None

    df = df_original.drop(
        ["Metric", "Time", "group"], axis=1, errors="ignore"
    ).copy()

    with st.expander("Show data table"):
        df_display = df.drop('Value', axis = 1)
        st.dataframe(df_display, use_container_width=True)

    _, table_str, _, _ = readout_utils.table_to_text(pd.DataFrame(), df, take_head=False)

    time_order_map = {t: i for i, t in enumerate(sorted_labels)}
    df["time_order"] = df["TimeLabel"].map(time_order_map)

    latest_time = sorted_labels[-1]
    df_latest = df[(df["Metric Type"] == "Spend") & (df["TimeLabel"] == latest_time)]
    if not df_latest.empty:
        tactics_order = df_latest.sort_values("Value", ascending=False)["Tactics"].tolist()
    else:
        tactics_order = df["Tactics"].unique().tolist()

    reversed_time_order = sorted_labels[::-1]
    base = alt.Chart(df).encode(
        x=alt.X(
            "Tactics:N",
            title="Tactics",
            axis=alt.Axis(labelAngle=0),
            sort=tactics_order,
        ),
        xOffset=alt.XOffset("TimeLabel:N", sort=reversed_time_order),
    )

    bar = base.transform_filter(
        alt.FieldOneOfPredicate(field="Metric Type", oneOf=["Share of Spend"])
    ).mark_bar(opacity=0.8).encode(
        y=alt.Y(
            "Value:Q",
            title="Share of Spend (%)",
            axis=alt.Axis(titleColor="orange"),
        ),
        color=alt.Color(
            "TimeLabel:N",
            sort=reversed_time_order,
            legend=alt.Legend(title="Time"),
            scale=alt.Scale(scheme="blues"),
        ),
        tooltip=["Tactics", "TimeLabel", "Value"],
    )

    line = base.transform_filter(
        alt.FieldOneOfPredicate(field="Metric Type", oneOf=["Spend"])
    ).mark_line(point={"filled": True, "size": 70}).encode(
        y=alt.Y(
            "Value:Q",
            title="Spend (MM)",
            axis=alt.Axis(titleColor="steelblue"),
        ),
        color=alt.value("#1f77b4"),
        detail="Tactics:N",
        order=alt.Order("time_order:Q"),
        tooltip=["Tactics", "TimeLabel", "Value"],
    )

    chart = (
        alt.layer(bar, line)
        .resolve_scale(y="independent")
        .properties(title=title, width=1200, height=400)
    )
    return chart, table_str


# ===========================================================================
# spend_snapshot
# ===========================================================================

def build_spend_snapshot_chart_configs(
    df, selected_levels, driver_col, metric_cols, sorted_time, readout, match, is_specific
):
    """
    build chart_configs dict for the spend snapshot view.
    """
    chart_configs = {}
    select_time = viz_utils.get_time_align_dict(sorted_time)

    df_group, _ = viz_utils.spend_snapshot_data(
        df, selected_levels, driver_col, metric_cols, readout, match, is_specific,
        mode="recent", type="value",
    )
    df_group_change, _ = viz_utils.spend_snapshot_data(
        df, selected_levels, driver_col, metric_cols, readout, match, is_specific,
        mode="recent", type="change",
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
                lambda x: "left" if "Spend" in x and "Change" not in x else "right"
            )
            df_left = df_long[df_long["Axis"] == "left"]
            tactics_order = (
                df_left.sort_values("Value", ascending=False)["Tactics"].tolist()
                if not df_left.empty
                else df_long["Tactics"].unique().tolist()
            )
            if len(tactics_order) >= 3:
                _, table_str, _, _ = readout_utils.table_to_text(
                    pd.DataFrame(), df_long, take_head=False
                )
                df_long = df_long.drop('Value', axis = 1)
                chart_configs[f"spend_snapshot_{g}"] = {
                    "query": "You are given a dataset about most recent spend.",
                    "instruction": "Provide a clear, concise, business-style insight summary.",
                    "data": df_long.to_json(orient="records"),
                }

        df_long_change = (
            df_group_change.drop("group", axis=1)
            if g == "All"
            else df_group_change[df_group_change["group"] == g]
        )
        if not df_long_change.empty:
            df_long_change = df_long_change.copy()
            df_long_change["Axis"] = df_long_change["Metric Type"].apply(
                lambda x: "left" if "Share" in x and "Change" not in x else "right"
            )
            df_left_change = df_long_change[df_long_change["Axis"] == "left"]
            tactics_order_change = (
                df_left_change.sort_values("Value", ascending=False)["Tactics"].tolist()
                if not df_left_change.empty
                else df_long_change["Tactics"].unique().tolist()
            )
            if len(tactics_order_change) >= 3:
                df_long_change = df_long_change.drop('Value', axis=1)
                chart_configs[f"spend_snapshot_change_{g}"] = {
                    "query": "You are given a dataset about most recent spend change.",
                    "instruction": "Provide a clear, concise, business-style insight summary.",
                    "data": df_long_change.to_json(orient="records"),
                }

    return chart_configs


def build_spend_snapshot_chart_configs_for_insights(
    df, selected_levels, driver_col, metric_cols, sorted_time, readout, match, is_specific
):
    """
    build chart_configs dict for the spend snapshot view.
    """
    chart_configs = {}
    select_time = viz_utils.get_time_align_dict(sorted_time)

    df_group, _ = viz_utils.spend_snapshot_data_for_insights(
        df, selected_levels, driver_col, metric_cols, readout, match, is_specific,
        mode="recent", type="value",
    )
    df_group_change, _ = viz_utils.spend_snapshot_data_for_insights(
        df, selected_levels, driver_col, metric_cols, readout, match, is_specific,
        mode="recent", type="change",
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
                lambda x: "left" if "Spend" in x and "Change" not in x else "right"
            )
            df_left = df_long[df_long["Axis"] == "left"]
            tactics_order = (
                df_left.sort_values("Value", ascending=False)["Tactics"].tolist()
                if not df_left.empty
                else df_long["Tactics"].unique().tolist()
            )
            if len(tactics_order):
                _, table_str, _, _ = readout_utils.table_to_text(
                    pd.DataFrame(), df_long, take_head=False
                )
                df_long = df_long.drop('Value', axis = 1)
                chart_configs[f"spend_snapshot_{g}"] = {
                    "query": "You are given a dataset about most recent spend.",
                    "instruction": "Provide a clear, concise, business-style insight summary.",
                    "data": df_long.to_json(orient="records"),
                }

        df_long_change = (
            df_group_change.drop("group", axis=1)
            if g == "All"
            else df_group_change[df_group_change["group"] == g]
        )
        if not df_long_change.empty:
            df_long_change = df_long_change.copy()
            df_long_change["Axis"] = df_long_change["Metric Type"].apply(
                lambda x: "left" if "Share" in x and "Change" not in x else "right"
            )
            df_left_change = df_long_change[df_long_change["Axis"] == "left"]
            tactics_order_change = (
                df_left_change.sort_values("Value", ascending=False)["Tactics"].tolist()
                if not df_left_change.empty
                else df_long_change["Tactics"].unique().tolist()
            )
            if len(tactics_order_change):
                df_long_change = df_long_change.drop('Value', axis=1)
                chart_configs[f"spend_snapshot_change_{g}"] = {
                    "query": "You are given a dataset about most recent spend change.",
                    "instruction": "Provide a clear, concise, business-style insight summary.",
                    "data": df_long_change.to_json(orient="records"),
                }

    return chart_configs


def render_spend_snapshot(
    df, selected_levels, driver_col, metric_cols, sorted_time, readout, match, is_specific,
    insights,
):
    """
    Pure Streamlit rendering of the spend snapshot view.
    Accepts pre-computed insights dict. ZERO LLM calls.
    """
    select_time = viz_utils.get_time_align_dict(sorted_time)

    df_group, _ = viz_utils.spend_snapshot_data(
        df, selected_levels, driver_col, metric_cols, readout, match, is_specific,
        mode="recent", type="value",
    )
    df_group_change, _ = viz_utils.spend_snapshot_data(
        df, selected_levels, driver_col, metric_cols, readout, match, is_specific,
        mode="recent", type="change",
    )

    groups = df_group["group"].dropna().unique().tolist()

    for g in groups:
        g_title = f"({g})" if g != "All" else ""

        # --- value chart ---
        df_long = (
            df_group.drop("group", axis=1)
            if g == "All"
            else df_group[df_group["group"] == g]
        )
        if not df_long.empty:
            with st.expander("Show data table"):
                df_long_display = df_long.drop('Value', axis = 1)
                st.dataframe(df_long_display, use_container_width=True)

            df_long = df_long.copy()
            df_long["Axis"] = df_long["Metric Type"].apply(
                lambda x: "left" if "Spend" in x and "Change" not in x else "right"
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
                y=alt.Y("Value:Q", axis=alt.Axis(title="Spend (MM)", titleColor="steelblue")),
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
                    axis=alt.Axis(title="Spend % Change", titleColor="orange"),
                ),
                tooltip=["Tactics", "Metric Type", "Value"],
            )
            dual_chart = (
                alt.layer(bar_left, line_right)
                .resolve_scale(y="independent")
                .properties(
                    title=f"Spend and % Change in {next(iter(select_time))} {g_title}",
                    width=900,
                    height=450,
                )
            )
            if len(tactics_order) >= 3:
                st.altair_chart(dual_chart, use_container_width=True)
                saved = insights.get(f"spend_snapshot_{g}", "")
                if saved:
                    render_chart_insight_box("Spend Snapshot Insight", saved)

        # --- change chart ---
        df_long_change = (
            df_group_change.drop("group", axis=1)
            if g == "All"
            else df_group_change[df_group_change["group"] == g]
        )
        if not df_long_change.empty:
            with st.expander("Show data table"):
                df_long_change_display = df_long_change.drop('Value', axis=1)
                st.dataframe(df_long_change_display, use_container_width=True)

            df_long_change = df_long_change.copy()
            df_long_change["Axis"] = df_long_change["Metric Type"].apply(
                lambda x: "left" if "Share" in x and "Change" not in x else "right"
            )
            df_left_change = df_long_change[df_long_change["Axis"] == "left"]
            df_right_change = df_long_change[df_long_change["Axis"] == "right"]
            tactics_order_change = (
                df_left_change.sort_values("Value", ascending=False)["Tactics"].tolist()
                if not df_left_change.empty
                else df_long_change["Tactics"].unique().tolist()
            )

            bar_left_change = alt.Chart(df_left_change).mark_bar(
                opacity=0.8, color="steelblue"
            ).encode(
                x=alt.X(
                    "Tactics:N",
                    sort=tactics_order_change,
                    title="Tactics",
                    axis=alt.Axis(labelAngle=0),
                ),
                y=alt.Y(
                    "Value:Q",
                    axis=alt.Axis(title="Share of Spend", titleColor="steelblue"),
                ),
                xOffset="Metric Type:N",
                tooltip=["Tactics", "Metric Type", "Value"],
            )
            line_right_change = alt.Chart(df_right_change).mark_line(
                point=True, color="orange"
            ).encode(
                x=alt.X(
                    "Tactics:N",
                    sort=tactics_order_change,
                    title="Tactics",
                    axis=alt.Axis(labelAngle=0),
                ),
                y=alt.Y(
                    "Value:Q",
                    axis=alt.Axis(
                        title="Share of Spend % Change", titleColor="orange"
                    ),
                ),
                tooltip=["Tactics", "Metric Type", "Value"],
            )
            dual_chart_change = (
                alt.layer(bar_left_change, line_right_change)
                .resolve_scale(y="independent")
                .properties(
                    title=(
                        f"Share of Spend and % Change in "
                        f"{next(iter(select_time))} {g_title}"
                    ),
                    width=900,
                    height=450,
                )
            )
            if len(tactics_order_change) >= 3:
                st.altair_chart(dual_chart_change, use_container_width=True)
                saved = insights.get(f"spend_snapshot_change_{g}", "")
                if saved:
                    render_chart_insight_box("Spend Share Snapshot Insight", saved)


def spend_snapshot(
    df, selected_levels, driver_col, metric_cols, sorted_time, readout, match, is_specific,
    llm_vendor,
):
    """
    Orchestrator for the spend snapshot view.
    build → generate insights → render → returns (chart_configs, insights).
    """
    chart_configs = build_spend_snapshot_chart_configs(
        df, selected_levels, driver_col, metric_cols, sorted_time, readout, match, is_specific
    )
    with st.spinner("Your visuals and insights are on the way..."):
        start = time.time()
        insights = generate_chart_insights_parallel_cached(chart_configs, llm_vendor)
        elapsed = time.time() - start
    render_spend_snapshot(
        df, selected_levels, driver_col, metric_cols, sorted_time, readout, match, is_specific,
        insights,
    )
    return chart_configs, insights


# ===========================================================================
# spend_share_trend_by_level
# ===========================================================================

def build_spend_trend_chart_configs(
    data, df, selected_levels, metric_cols, readout, match, is_specific
):
    """
    build chart_configs dict for the spend share trend view.
    """
    chart_configs = {}
    df_combined, sorted_labels, _ = viz_utils.spend_share_trend_data(
        data, df, selected_levels, metric_cols, readout, match, is_specific
    )
    groups = df_combined["group"].dropna().unique().tolist()
    tables_for_llm = []
    has_chart = False

    if "All" not in groups:
        for g in groups:
            df_group = df_combined[df_combined["group"] == g]
            chart = _build_spend_share_trend_chart(
                df_group, f"Spend vs Share of Spend — {g}", sorted_labels
            )
            if chart is not None:
                has_chart = True
                table_str = _spend_share_trend_table_str(df_group)
                if table_str:
                    tables_for_llm.append(f"{g} data\n{table_str}")
    else:
        levels = ["high", "moderate", "low"]
        for lvl in levels:
            sub = df_combined[df_combined["spend level"] == lvl]
            chart = _build_spend_share_trend_chart(
                sub, f"Spend vs Share of Spend — {lvl.title()}", sorted_labels
            )
            if chart is not None:
                has_chart = True
                table_str = _spend_share_trend_table_str(sub)
                if table_str:
                    tables_for_llm.append(f"{lvl.title()} Level data\n{table_str}")

    if has_chart and tables_for_llm:
        combined_table_str = "\n\n".join(tables_for_llm)
        chart_configs["spend_trend"] = {
            "query": "You are given spend and share of spend trend data.",
            "instruction": "Provide a clear, concise, business-style insight summary.",
            "data": combined_table_str,
        }

    return chart_configs


def render_spend_trend(
    data, df, selected_levels, metric_cols, readout, match, is_specific, insights
):
    """
    Pure Streamlit rendering of the spend share trend view.
    Accepts pre-computed insights dict. ZERO LLM calls.
    """
    df_combined, sorted_labels, _ = viz_utils.spend_share_trend_data(
        data, df, selected_levels, metric_cols, readout, match, is_specific
    )
    groups = df_combined["group"].dropna().unique().tolist()
    has_chart = False

    if "All" not in groups:
        for g in groups:
            df_group = df_combined[df_combined["group"] == g]
            ch, table_str = _render_spend_share_trend_chart(
                df_group, f"Spend vs Share of Spend — {g}", sorted_labels
            )
            if ch is not None:
                has_chart = True
                c1, c2, c3 = st.columns([1, 3, 1])
                with c2:
                    st.altair_chart(ch, use_container_width=False)
            else:
                st.write(f"No data available for spend level: {g}")
    else:
        levels = ["high", "moderate", "low"]
        for lvl in levels:
            sub = df_combined[df_combined["spend level"] == lvl]
            ch, table_str = _render_spend_share_trend_chart(
                sub, f"Spend vs Share of Spend — {lvl.title()}", sorted_labels
            )
            if ch is not None:
                has_chart = True
                c1, c2, c3 = st.columns([1, 3, 1])
                with c2:
                    st.altair_chart(ch, use_container_width=False)
            else:
                st.write(f"No data available for spend level: {lvl}")

    if has_chart:
        saved = insights.get("spend_trend", "")
        if saved:
            render_chart_insight_box("Spend Trend Insight", saved)


def spend_share_trend_by_level(
    data, df, selected_levels, metric_cols, readout, match, is_specific, llm_vendor
):
    """
    Orchestrator for the spend share trend view.
    build → generate insights → render → returns (chart_configs, insights).
    """
    chart_configs = build_spend_trend_chart_configs(
        data, df, selected_levels, metric_cols, readout, match, is_specific
    )
    with st.spinner("Your visuals and insights are on the way..."):
        start = time.time()
        insights = generate_chart_insights_parallel_cached(chart_configs, llm_vendor)
        elapsed = time.time() - start
    render_spend_trend(
        data, df, selected_levels, metric_cols, readout, match, is_specific, insights
    )
    return chart_configs, insights

def build_spend_trend_chart_configs_for_insights(
    data, df, selected_levels, metric_cols, readout, match, is_specific
):
    """
    build chart_configs dict for the spend share trend view.
    """
    chart_configs = {}
    df_combined, sorted_labels, _ = viz_utils.spend_share_trend_data(
        data, df, selected_levels, metric_cols, readout, match, is_specific
    )
    groups = df_combined["group"].dropna().unique().tolist()
    tables_for_llm = []

    if "All" not in groups:
        for g in groups:
            df_group = df_combined[df_combined["group"] == g]
            table_str = _spend_share_trend_table_str(df_group)
            if table_str:
                tables_for_llm.append(f"{g} data\n{table_str}")
    else:
        levels = ["high", "moderate", "low"]
        for lvl in levels:
            sub = df_combined[df_combined["spend level"] == lvl]
            table_str = _spend_share_trend_table_str(sub)
            if table_str:
                tables_for_llm.append(f"{lvl.title()} Level data\n{table_str}")

    combined_table_str = "\n\n".join(tables_for_llm)
    chart_configs["spend_trend"] = {
        "query": "You are given spend and share of spend trend data.",
        "instruction": "Provide a clear, concise, business-style insight summary.",
        "data": combined_table_str,
    }

    return chart_configs

# ===========================================================================
# pct_change_heatmap
# ===========================================================================

def build_spend_heatmap_chart_configs(
    df,
    selected_levels,
    driver_col,
    metric_cols,
    sorted_time,
    readout,
    match,
    is_specific,
    mode="change",
    time_col="Time",
    tactic_col="Tactics",
    val_col="Value",
):
    """
    Python/pandas: build chart_configs dict for the spend % change heatmap.
    """
    chart_configs = {}
    select_time = viz_utils.get_time_align_dict(sorted_time)

    df_group, _ = viz_utils.pct_heatmap_data(
        df, selected_levels, driver_col, metric_cols, readout, match, is_specific, mode=mode
    )
    groups = df_group["group"].dropna().unique().tolist()

    for g in groups:
        df_sub = (
            df_group.drop("group", axis=1)
            if g == "All"
            else df_group[df_group["group"] == g]
        )
        if df_sub[time_col].nunique() > 1:
            _, table_str, _, _ = readout_utils.table_to_text(
                pd.DataFrame(), df_sub, take_head=False
            )
            df_sub = df_sub.drop("Value", axis = 1)
            chart_configs[f"spend_change_heatmap_{g}"] = {
                "query": "You are given a dataset about spend change.",
                "instruction": "Provide a clear, concise, business-style insight summary.",
                "data": df_sub.to_json(orient="records"),
            }

    return chart_configs


def render_spend_heatmap(
    df,
    selected_levels,
    driver_col,
    m_cols,
    sorted_time,
    readout,
    match,
    is_specific,
    mode="change",
    time_col="Time",
    tactic_col="Tactics",
    val_col="Value",
    insights=None,
):
    """
    Pure Streamlit rendering of the spend % change heatmap.
    Accepts pre-computed insights dict. ZERO LLM calls.
    """
    if insights is None:
        insights = {}

    select_time = viz_utils.get_time_align_dict(sorted_time)
    df_group, _ = viz_utils.pct_heatmap_data(
        df, selected_levels, driver_col, m_cols, readout, match, is_specific, mode=mode
    )
    groups = df_group["group"].dropna().unique().tolist()

    for g in groups:
        df_sub = (
            df_group.drop("group", axis=1)
            if g == "All"
            else df_group[df_group["group"] == g]
        )
        height = max(300, 40 * df_sub[tactic_col].nunique())
        width = max(600, 120 * df_sub[time_col].nunique())

        if df_sub[time_col].nunique() > 1:
            with st.expander("Show data table"):
                st.dataframe(df_sub, use_container_width=True)

            chart = (
                alt.Chart(df_sub)
                .mark_rect()
                .encode(
                    x=alt.X(f"{time_col}:N", title="Time", axis=alt.Axis(labelAngle=0)),
                    y=alt.Y(f"{tactic_col}:N", title="Tactic", sort="-x"),
                    color=alt.Color(
                        f"{val_col}:Q",
                        scale=alt.Scale(scheme="redblue", domainMid=0),
                        legend=alt.Legend(title="Spend % Change"),
                    ),
                    tooltip=[
                        tactic_col,
                        time_col,
                        alt.Tooltip(val_col, title="Spend % Change", format=".1f"),
                    ],
                )
                .properties(
                    width=width,
                    height=height,
                    title=f"Spend % Change in {next(iter(select_time))} ({g})",
                )
            )
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.altair_chart(chart, use_container_width=False)
                saved = insights.get(f"spend_change_heatmap_{g}", "")
                if saved:
                    render_chart_insight_box("Spend Change Heatmap Insight", saved)


def pct_change_heatmap(
    df,
    selected_levels,
    driver_col,
    m_cols,
    sorted_time,
    readout,
    match,
    is_specific,
    mode="change",
    time_col="Time",
    tactic_col="Tactics",
    val_col="Value",
    llm_vendor=None,
):
    """
    Orchestrator for the spend % change heatmap.
    build → generate insights → render → returns (chart_configs, insights).
    """
    chart_configs = build_spend_heatmap_chart_configs(
        df, selected_levels, driver_col, m_cols, sorted_time, readout, match, is_specific,
        mode=mode, time_col=time_col, tactic_col=tactic_col, val_col=val_col,
    )
    insights = {}
    if llm_vendor and chart_configs:
        with st.spinner("Your visuals and insights are on the way..."):
            start = time.time()
            insights = generate_chart_insights_parallel_cached(chart_configs, llm_vendor)
            elapsed = time.time() - start
    render_spend_heatmap(
        df, selected_levels, driver_col, m_cols, sorted_time, readout, match, is_specific,
        mode=mode, time_col=time_col, tactic_col=tactic_col, val_col=val_col,
        insights=insights,
    )
    return chart_configs, insights
