"""
chart/roi.py
ROI / performance chart functions following the build/render separation pattern.

Pattern
-------
- build_*_chart_configs(...)   — pure Python/pandas, builds chart_configs dict, ZERO st.* calls
- render_*_charts(...)         — pure Streamlit rendering, accepts pre-computed insights dict, ZERO LLM calls
- Orchestrator (keeps current name) — calls build → generate_chart_insights_parallel_cached → render → returns (chart_configs, insights)
"""

import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.colors as mcolors
from streamlit_elements import elements, mui, dashboard
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
# Utility
# ---------------------------------------------------------------------------

def fmt_percent(v):
    try:
        return f"{float(v):.1f}%"
    except Exception:
        return str(v)


# ---------------------------------------------------------------------------
# Private chart helpers
# ---------------------------------------------------------------------------

def _most_recent_bar_plot_str():
    """Insight instruction for the most-recent bar plot (data-independent constant)."""
    return """Provide a clear, concise, business-style insight summary."""


def _most_recent_bar_plot(df, bar_col1, bar_col2, tactic_col, bar_name1, bar_name2,
                          title, height=400, angle=-80, y_axis_upper=[-1, -1], mode='value'):
    expander_data = df[['Tactics', bar_col1, bar_col2]]
    expander_data = readout_utils.add_unit_sign(expander_data, 1, period_suffixes = ["% Change"])
    # expander_data = viz_utils.replace_with_labels(expander_data, [bar_col1, bar_col2])
    d = expander_data.to_json(orient="records")

    df_melt = df.melt(
        id_vars=[tactic_col],
        value_vars=[bar_col1, bar_col2],
        var_name="Metric",
        value_name="Value"
    )

    sort_order = (
        df[[tactic_col, bar_col1]]
        .sort_values(bar_col1, ascending=False)[tactic_col]
        .tolist()
    )

    d_min = min(0, df_melt["Value"].min())
    d_max = max(0, df_melt["Value"].max())

    value_left_scale = alt.Scale(domain=[0, y_axis_upper[0]]) if y_axis_upper[0] > 0 else alt.Scale()
    value_right_scale = alt.Scale(domain=[0, y_axis_upper[1]]) if y_axis_upper[1] > 0 else alt.Scale()

    bars_1 = (
        alt.Chart(df_melt.query("Metric == @bar_col1"))
        .mark_bar(color="#1f77b4", opacity=0.85)
        .encode(
            x=alt.X(f"{tactic_col}:N",
                    sort=sort_order,
                    axis=alt.Axis(labelFontSize=10, labelAngle=angle)),
            xOffset=alt.X("Metric:N", sort=[bar_col1, bar_col2]),
            y=alt.Y("Value:Q", title=bar_name1, axis=alt.Axis(grid=False),
                    scale=alt.Scale(domain=(d_min, d_max), nice=False)) if mode == 'change'
            else alt.Y("Value:Q", title=bar_name1, axis=alt.Axis(grid=False), scale=value_left_scale),
            tooltip=[tactic_col, "Metric", "Value"],
        )
    )

    bars_2 = (
        alt.Chart(df_melt.query("Metric == @bar_col2"))
        .mark_bar(color="#FFD230", opacity=0.85)
        .encode(
            x=alt.X(f"{tactic_col}:N", sort=sort_order),
            xOffset=alt.X("Metric:N", sort=[bar_col1, bar_col2]),
            y=alt.Y("Value:Q", title=bar_name2,
                    axis=alt.Axis(grid=False),
                    scale=alt.Scale(domain=(d_min, d_max), nice=False)) if mode == 'change'
            else alt.Y("Value:Q", title=bar_name2,
                       axis=alt.Axis(grid=False), scale=value_right_scale),
            tooltip=[tactic_col, "Metric", "Value"],
        )
    )

    chart = (
        alt.layer(bars_1, bars_2)
        .resolve_scale(y="independent")
        .properties(
            height=height,
            title={
                "text": title,
                "subtitle": [f"Blue bar = {bar_name1}, Yellow bar = {bar_name2} (Sorted by {bar_name1} desc)"],
                "subtitleColor": "gray",
                "subtitleFontSize": 12,
                "anchor": "start",
            },
            padding={"left": 5, "right": 20, "top": 5, "bottom": 5}
        )
    )
    chart = chart.configure(background="transparent").configure_view(stroke=None, fill="transparent")

    return chart, d


def _most_recent_bar_plot_for_insights(df, bar_col1, bar_col2, tactic_col, bar_name1, bar_name2,
                          title, height=400, angle=-80, y_axis_upper=[-1, -1], mode='value', client_code=None, model_group_id=None):
    expander_data = df[['Tactics', bar_col1, bar_col2]]
    expander_data = readout_utils.add_unit_sign(expander_data, 1, period_suffixes=["% Change"], client_code=client_code, model_group_id=model_group_id)
    d = expander_data.to_json(orient="records")

    return d


def _roi_spend_scatter_config(recent_data, spend_col, roi_col, time, table='detail'):
    """Insight config dict for the ROI-vs-spend scatter. No st.* calls."""
    level_col = [c for c in recent_data.columns if 'roi' in c and 'level' in c][0]
    expander_data = recent_data[['Tactics', spend_col, roi_col, level_col]]
    chart_title = f"ROI vs Spend (Scatter Plot) - {time}"
    _, table_str, _, _ = readout_utils.table_to_text(pd.DataFrame(), expander_data, take_head=False)
    instruct = f"""Provide a clear, concise, business-style insight summary.
            Data: {table_str}"""
    return {chart_title: {"query": f"You are given a dataset about {chart_title}.",
                          "instruction": instruct}}


def _roi_spend_scatter(recent_data, spend_col, roi_col, time, table='detail'):
    level_col = [c for c in recent_data.columns if 'roi' in c and 'level' in c][0]

    top3 = recent_data.nlargest(3, roi_col)
    bottom3 = recent_data.nsmallest(3, roi_col)
    label_data = pd.concat([top3, bottom3])

    labels = (
        alt.Chart(label_data)
        .mark_text(align='left', dx=5, dy=5, fontSize=12)
        .encode(
            x=f"{spend_col}:Q",
            y=f"{roi_col}:Q",
            text="Tactics:N"
        )
    )
    scatter = (
        alt.Chart(recent_data)
        .mark_circle(size=80, opacity=0.8)
        .encode(
            x=alt.X(f"{spend_col}:Q", title="Spend(MM)", axis=alt.Axis(tickCount=5, grid=True)),
            y=alt.Y(f"{roi_col}:Q", title="ROI", axis=alt.Axis(tickCount=5, grid=True)),
            color=alt.Color(f"{level_col}:N", title="ROI Level",
                            scale=alt.Scale(domain=['excellent', 'good', 'justifiable', 'inefficient'],
                                            range=["#2ca02c", "#1f77b4", "#FFD230", "#ff7f0e"])
                            ) if (recent_data[level_col] != "").any() else alt.Color(),
            tooltip=[
                alt.Tooltip('Tactics', title="Tactic"),
                alt.Tooltip(spend_col, title="Spend(MM)"),
                alt.Tooltip(roi_col, title="ROI")
            ]
        ))
    chart_title = f"ROI vs Spend (Scatter Plot) - {time}"
    chart = (scatter + labels).properties(
        height=400,
        title={
            "text": chart_title,
            "subtitle": ["Performance distribution across ROI levels by channel (hover dots for details)"],
            "subtitleColor": "gray",
            "subtitleFontSize": 12,
            "anchor": "start"},
        padding={"left": 5, "right": 20, "top": 5, "bottom": 5})
    chart = chart.configure(background="transparent").configure_view(stroke=None, fill="transparent")

    return chart, chart_title


def _roi_spend_scatter_for_insights(recent_data, spend_col, roi_col, time, table='detail'):
    level_col = [c for c in recent_data.columns if 'roi' in c and 'level' in c][0]

    top3 = recent_data.nlargest(3, roi_col)
    bottom3 = recent_data.nsmallest(3, roi_col)
    label_data = pd.concat([top3, bottom3])

    # labels = (
    #     alt.Chart(label_data)
    #     .mark_text(align='left', dx=5, dy=5, fontSize=12)
    #     .encode(
    #         x=f"{spend_col}:Q",
    #         y=f"{roi_col}:Q",
    #         text="Tactics:N"
    #     )
    # )
    # scatter = (
    #     alt.Chart(recent_data)
    #     .mark_circle(size=80, opacity=0.8)
    #     .encode(
    #         x=alt.X(f"{spend_col}:Q", title="Spend(MM)", axis=alt.Axis(tickCount=5, grid=True)),
    #         y=alt.Y(f"{roi_col}:Q", title="ROI", axis=alt.Axis(tickCount=5, grid=True)),
    #         color=alt.Color(f"{level_col}:N", title="ROI Level",
    #                         scale=alt.Scale(domain=['excellent', 'good', 'justifiable', 'inefficient'],
    #                                         range=["#2ca02c", "#1f77b4", "#FFD230", "#ff7f0e"])
    #                         ) if (recent_data[level_col] != "").any() else alt.Color(),
    #         tooltip=[
    #             alt.Tooltip('Tactics', title="Tactic"),
    #             alt.Tooltip(spend_col, title="Spend(MM)"),
    #             alt.Tooltip(roi_col, title="ROI")
    #         ]
    #     ))
    # chart_title = f"ROI vs Spend (Scatter Plot) - {time}"
    # chart = (scatter + labels).properties(
    #     height=400,
    #     title={
    #         "text": chart_title,
    #         "subtitle": ["Performance distribution across ROI levels by channel (hover dots for details)"],
    #         "subtitleColor": "gray",
    #         "subtitleFontSize": 12,
    #         "anchor": "start"},
    #     padding={"left": 5, "right": 20, "top": 5, "bottom": 5})
    # chart = chart.configure(background="transparent").configure_view(stroke=None, fill="transparent")

    return None, ""


def _roi_spend_change_scatter_config(recent_change_data, spend_change_col, roi_change_col, time, table='detail'):
    """Insight config dict for the ROI-change-vs-spend-change scatter. No st.* calls."""
    level_col = [c for c in recent_change_data.columns if 'roi' in c and 'level' in c][0]
    expander_data = recent_change_data[['Tactics', spend_change_col, roi_change_col, level_col]]
    chart_title = f"ROI Change vs Spend Change (Scatter Plot) - {time}"
    _, table_str, _, _ = readout_utils.table_to_text(pd.DataFrame(), expander_data, take_head=False)
    instruct = f"""Provide a clear, concise, business-style insight summary.
            Data: {table_str}"""
    return {chart_title: {"query": f"You are given a dataset about {chart_title}.",
                          "instruction": instruct}}


def _roi_spend_change_scatter(recent_change_data, spend_change_col, roi_change_col, time, table='detail'):
    level_col = [c for c in recent_change_data.columns if 'roi' in c and 'level' in c][0]

    x_min = min(0, recent_change_data[spend_change_col].min()) - 30
    x_max = max(0, recent_change_data[spend_change_col].max()) + 30
    y_min = min(0, recent_change_data[roi_change_col].min()) - 30
    y_max = max(0, recent_change_data[roi_change_col].max()) + 30

    quad_df = pd.DataFrame({
        "x": [x_max - 10, x_min + 10, x_min + 10, x_max - 10],
        "y": [y_max - 10, y_max - 10, y_min + 10, y_min + 10],
        "label": ["Q1 - ROI ↑ Spend ↑: Positive Growth", "Q2 - ROI ↑ Spend ↓: Efficiency Gain",
                  "Q3 - ROI ↓ Spend ↓: Declining", "Q4 - ROI ↓ Spend ↑: Mixed Performance"]
    })

    quad_text = (
        alt.Chart(quad_df)
        .mark_text(fontSize=12, opacity=0.7)
        .encode(
            x="x:Q", y="y:Q", text="label:N"))

    top3 = recent_change_data.nlargest(3, roi_change_col)
    bottom3 = recent_change_data.nsmallest(3, roi_change_col)
    label_data = pd.concat([top3, bottom3])

    labels = (
        alt.Chart(label_data)
        .mark_text(align='left', dx=5, dy=5, fontSize=12)
        .encode(
            x=f"{spend_change_col}:Q",
            y=f"{roi_change_col}:Q",
            text="Tactics:N"
        ))
    chart_title = f"ROI Change vs Spend Change (Scatter Plot) - {time}"
    base = (
        alt.Chart(recent_change_data)
        .mark_circle(size=80, opacity=0.8)
        .encode(
            x=alt.X(f"{spend_change_col}:Q", title="Spend Change %", axis=alt.Axis(tickCount=5, grid=True),
                    scale=alt.Scale(domain=[x_min, x_max])),
            y=alt.Y(f"{roi_change_col}:Q", title="ROI Change %", axis=alt.Axis(tickCount=5, grid=True),
                    scale=alt.Scale(domain=[y_min, y_max])),
            color=alt.Color(f"{level_col}:N", legend=None,
                            scale=alt.Scale(domain=['excellent', 'good', 'justifiable', 'inefficient'],
                                            range=["#2ca02c", "#1f77b4", "#FFD230", "#ff7f0e"])
                            ) if (recent_change_data[level_col] != "").any() else alt.Color(),
            tooltip=[
                alt.Tooltip('Tactics', title="Tactic"),
                alt.Tooltip(spend_change_col, title="Spend Change %"),
                alt.Tooltip(roi_change_col, title="ROI Change %")
            ]
        )
        .properties(
            height=400,
            title={
                "text": chart_title,
                "subtitle": ["Four-quadrant performance momentum analysis by channel (hover dots for details)"],
                "subtitleColor": "gray",
                "subtitleFontSize": 12,
                "anchor": "start"
            })
    )

    hline = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="gray", strokeWidth=1.5, opacity=0.5).encode(y="y:Q")
    vline = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="gray", strokeWidth=1.5, opacity=0.5).encode(x="x:Q")
    chart = base + hline + vline + quad_text + labels
    chart = chart.configure(background="transparent").configure_view(stroke=None, fill="transparent")
    chart = chart.properties(
        padding={"left": 5, "right": 20, "top": 5, "bottom": 5}
    )

    return chart, chart_title


def _roi_spend_change_scatter_for_insights(recent_change_data, spend_change_col, roi_change_col, time, table='detail'):
    level_col = [c for c in recent_change_data.columns if 'roi' in c and 'level' in c][0]

    x_min = min(0, recent_change_data[spend_change_col].min()) - 30
    x_max = max(0, recent_change_data[spend_change_col].max()) + 30
    y_min = min(0, recent_change_data[roi_change_col].min()) - 30
    y_max = max(0, recent_change_data[roi_change_col].max()) + 30

    # quad_df = pd.DataFrame({
    #     "x": [x_max - 10, x_min + 10, x_min + 10, x_max - 10],
    #     "y": [y_max - 10, y_max - 10, y_min + 10, y_min + 10],
    #     "label": ["Q1 - ROI ↑ Spend ↑: Positive Growth", "Q2 - ROI ↑ Spend ↓: Efficiency Gain",
    #               "Q3 - ROI ↓ Spend ↓: Declining", "Q4 - ROI ↓ Spend ↑: Mixed Performance"]
    # })
    #
    # quad_text = (
    #     alt.Chart(quad_df)
    #     .mark_text(fontSize=12, opacity=0.7)
    #     .encode(
    #         x="x:Q", y="y:Q", text="label:N"))
    #
    # top3 = recent_change_data.nlargest(3, roi_change_col)
    # bottom3 = recent_change_data.nsmallest(3, roi_change_col)
    # label_data = pd.concat([top3, bottom3])
    #
    # labels = (
    #     alt.Chart(label_data)
    #     .mark_text(align='left', dx=5, dy=5, fontSize=12)
    #     .encode(
    #         x=f"{spend_change_col}:Q",
    #         y=f"{roi_change_col}:Q",
    #         text="Tactics:N"
    #     ))
    # chart_title = f"ROI Change vs Spend Change (Scatter Plot) - {time}"
    # base = (
    #     alt.Chart(recent_change_data)
    #     .mark_circle(size=80, opacity=0.8)
    #     .encode(
    #         x=alt.X(f"{spend_change_col}:Q", title="Spend Change %", axis=alt.Axis(tickCount=5, grid=True),
    #                 scale=alt.Scale(domain=[x_min, x_max])),
    #         y=alt.Y(f"{roi_change_col}:Q", title="ROI Change %", axis=alt.Axis(tickCount=5, grid=True),
    #                 scale=alt.Scale(domain=[y_min, y_max])),
    #         color=alt.Color(f"{level_col}:N", legend=None,
    #                         scale=alt.Scale(domain=['excellent', 'good', 'justifiable', 'inefficient'],
    #                                         range=["#2ca02c", "#1f77b4", "#FFD230", "#ff7f0e"])
    #                         ) if (recent_change_data[level_col] != "").any() else alt.Color(),
    #         tooltip=[
    #             alt.Tooltip('Tactics', title="Tactic"),
    #             alt.Tooltip(spend_change_col, title="Spend Change %"),
    #             alt.Tooltip(roi_change_col, title="ROI Change %")
    #         ]
    #     )
    #     .properties(
    #         height=400,
    #         title={
    #             "text": chart_title,
    #             "subtitle": ["Four-quadrant performance momentum analysis by channel (hover dots for details)"],
    #             "subtitleColor": "gray",
    #             "subtitleFontSize": 12,
    #             "anchor": "start"
    #         })
    # )
    #
    # hline = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="gray", strokeWidth=1.5, opacity=0.5).encode(y="y:Q")
    # vline = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="gray", strokeWidth=1.5, opacity=0.5).encode(x="x:Q")
    # chart = base + hline + vline + quad_text + labels
    # chart = chart.configure(background="transparent").configure_view(stroke=None, fill="transparent")
    # chart = chart.properties(
    #     padding={"left": 5, "right": 20, "top": 5, "bottom": 5}
    # )

    return None, ""


def _roi_spend_trend_multi_str():
    """Insight instruction for the ROI-spend trend (data-independent constant)."""
    return """Provide a clear, concise, business-style insight summary."""


def _roi_spend_trend_multi(trend_data, bar_values, line_values, times, width=400, height=250,
                           y_axis_upper=[-1, -1], title="", mode='layer', add_long_term=False):
    level_col = [c for c in trend_data.columns if 'roi' in c and 'level' in c][0]
    tactics_idx = trend_data.columns.get_loc('Tactics')
    index_cols = (list(trend_data.columns[:tactics_idx + 1]) + [level_col]
                  if mode == 'layer'
                  else list(trend_data.columns[:tactics_idx + 1]))
    existing_cols = [x for x in bar_values[0] + [m for x in line_values for m in x] if x in trend_data["Metric"].values]
    expander_data = trend_data.pivot(index=index_cols, columns='Metric',
                                     values='ValueFormatted').reset_index()[index_cols + existing_cols]

    chart_ls = []
    trend_data['time_order'] = pd.Categorical(trend_data['Time'], categories=times, ordered=True).codes
    roi_levels = trend_data[level_col].unique().tolist()
    layers = ['excellent', 'good', 'justifiable', 'inefficient'] if mode == 'layer' else ['all']
    for l in layers:
        if mode == 'layer' and l not in roi_levels:
            continue
        elif mode == 'layer':
            data = trend_data[trend_data[level_col] == l]
        else:
            data = trend_data
        sorted_tactics = (
            data[
                (data["Time"] == times[-1]) & (
                    data['Metric'].isin(bar_values[0]))]
            .sort_values("Value", ascending=False)["Tactics"]
            .tolist()
        )

        roi_scale = alt.Scale(domain=[0, y_axis_upper[1]]) if y_axis_upper[1] > 0 else alt.Scale()
        spend_scale = alt.Scale(domain=[0, y_axis_upper[0]]) if y_axis_upper[0] > 0 else alt.Scale()

        tooltip = [
            alt.Tooltip("Tactics", title="Tactic"),
            alt.Tooltip("Metric", title="Metric"),
            alt.Tooltip("TimeLabel", title="Time"),
            alt.Tooltip("Value", title="Value")
        ]

        base = alt.Chart(data).encode(
            x=alt.X(
                'Tactics:N',
                title='Tactics',
                axis=alt.Axis(labelAngle=0),
                sort=sorted_tactics
            ),
            xOffset=alt.XOffset('Time:N', sort=times[::-1]),
            tooltip=tooltip,
        )

        bar = base.transform_filter(
            alt.FieldOneOfPredicate(field="Metric", oneOf=bar_values[0])
        ).mark_bar(opacity=0.7).encode(
            y=alt.Y(
                "Value:Q",
                axis=alt.Axis(title="Spend"), scale=spend_scale),
            color=alt.Color("TimeLabel:N", legend=alt.Legend(title="Time"),
                            scale=alt.Scale(scheme="blues"), sort=times[::-1]))

        cond_expr = " || ".join([f"datum.Metric == '{m}'" for m in line_values[0]])

        line_base = base.transform_filter(
            alt.FieldOneOfPredicate(field="Metric", oneOf=[m for x in line_values for m in x])
        ).transform_calculate(
            ROI_type=f"({cond_expr}) ? 'Short-term ROI' : 'Long-term ROI'"
        )
        line = line_base.transform_filter(
            alt.FieldOneOfPredicate(field="Metric", oneOf=([m for x in line_values for m in x]))
        ).mark_line(point={"filled": True, "size": 80}).encode(
            y=alt.Y("Value:Q", axis=alt.Axis(title="ROI"), scale=roi_scale),
            color=alt.Color(
                "ROI_type:N",
                scale=alt.Scale(
                    domain=["Short-term ROI", "Long-term ROI"],
                    range=["#1f77b4", "#ff7f0e"],
                ),
                legend=alt.Legend(title="ROI Type")
            ) if add_long_term else alt.value("#1f77b4"),
            detail="Tactics:N",
            order=alt.Order('time_order:Q'),
        )

        chart = alt.layer(bar, line).resolve_scale(y="independent", color="independent")
        chart = chart.properties(
            width="container",
            height=height,
            title={
                "text": f"{l.capitalize()} ROI Level Performance" if mode == 'layer' else "ROI Performance" + title,
                "subtitle": ["Individual tactic performance with ROI trends and spend allocation"],
                "subtitleColor": "gray",
                "subtitleFontSize": 12,
                "anchor": "start"
            },
            padding={"left": 10, "right": 30, "top": 10, "bottom": 10}
        )
        chart = (
            chart
            .configure(background="transparent")
            .configure_view(stroke=None, fill="transparent")
        )
        chart_ls.append(chart)
    return chart_ls, expander_data


def _roi_spend_trend_multi_for_insights(trend_data, bar_values, line_values, times, width=400, height=250,
                           y_axis_upper=[-1, -1], title="", mode='layer', add_long_term=False, roi_type='roi'):
    level_col = [c for c in trend_data.columns if roi_type in c and 'level' in c][0]
    tactics_idx = trend_data.columns.get_loc('Tactics')
    index_cols = (list(trend_data.columns[:tactics_idx + 1]) + [level_col]
                  if mode == 'layer'
                  else list(trend_data.columns[:tactics_idx + 1]))
    existing_cols = [x for x in bar_values[0] + [m for x in line_values for m in x] if x in trend_data["Metric"].values]
    expander_data = trend_data.pivot(index=index_cols, columns='Metric',
                                     values='ValueFormatted').reset_index()[index_cols + existing_cols]
    return expander_data


def _roi_resp_cost_table_str(expander_data):
    """Table text for ROI/Response/Cost bar data. No st.* calls."""
    _, table_str, _, _ = readout_utils.table_to_text(pd.DataFrame(), expander_data, take_head=False)
    return table_str


def _roi_resp_cost_bar(df_plot, time, height=300, y_axis_upper=[-1, -1], title="", mode='layer'):
    level_col = [c for c in df_plot.columns if 'roi' in c and 'level' in c][0]
    tactics_idx = df_plot.columns.get_loc('Tactics')

    idx_ls = list(df_plot.columns[:tactics_idx + 1]) + [level_col]
    expander_data = df_plot.pivot(index=idx_ls, columns='Metric',values='Value').reset_index()
    expander_data = readout_utils.add_unit_sign(expander_data, len(idx_ls), period_suffixes=['% Change'])
    if mode != 'layer' and level_col in expander_data.columns:
        expander_data = expander_data.drop(columns=[level_col])

    chart_ls = []

    def get_shades(color):
        cmap = mpl.colormaps[color]
        return [mcolors.rgb2hex(cmap(x)) for x in np.linspace(0.4, 0.8, 3)]

    labels = ['ROI', 'Response', 'Cost']
    color_map = {'excellent': 'Greens', 'good': 'Blues', 'justifiable': 'Wistia', 'inefficient': 'Oranges'}

    roi_levels = df_plot[level_col].unique().tolist()

    left_scale = alt.Scale(domain=[0, y_axis_upper[0]]) if y_axis_upper[0] > 0 else alt.Scale()
    right_scale = alt.Scale(domain=[0, y_axis_upper[1]]) if y_axis_upper[1] > 0 else alt.Scale()

    layers = ['excellent', 'good', 'justifiable', 'inefficient']
    has_layer = df_plot[level_col].astype(str).str.lower().isin(layers).any()
    if not has_layer:
        mode = 'all'
        layers = ['all']
    for l in layers:
        if mode == 'layer' and l not in roi_levels:
            continue
        elif mode == 'layer':
            df_plot_sub = df_plot[df_plot[level_col] == l]
        else:
            df_plot_sub = df_plot

        color_scale = alt.Scale(
            domain=labels,
            range=get_shades(color_map[l] if l in color_map else color_map['good'])
        )

        sort_order = (
            df_plot_sub[df_plot_sub['metric_label'] == 'ROI']
            .sort_values('Value', ascending=False)['Tactics']
            .tolist()
        )

        x = alt.X("Tactics:N", title="Tactics", axis=alt.Axis(labelAngle=-20), sort=sort_order)
        xoff = alt.X("metric_label:N", title=None,
                     sort=labels, scale=alt.Scale(domain=labels))

        bars_roi_left = (
            alt.Chart(df_plot_sub[df_plot_sub["metric_label"] == "ROI"])
            .mark_bar()
            .encode(
                x=x,
                xOffset=xoff,
                y=alt.Y("Value:Q", title="ROI", axis=alt.Axis(grid=True, tickCount=4), scale=left_scale),
                color=alt.Color("metric_label:N", title="Metric", scale=color_scale),
                tooltip=[
                    alt.Tooltip("Tactics:N", title="Tactics"),
                    alt.Tooltip("metric_label:N", title="Metric"),
                    alt.Tooltip("Value:N", title="Value"),
                ],
            )
        )

        bars_resp_cost_right = (
            alt.Chart(df_plot_sub[df_plot_sub["metric_label"].isin(["Response", "Cost"])])
            .mark_bar()
            .encode(
                x=x,
                xOffset=xoff,
                y=alt.Y(
                    "Value:Q",
                    title="Response/Cost",
                    axis=alt.Axis(grid=False),
                    scale=right_scale
                ),
                color=alt.Color("metric_label:N", title="Metric", scale=color_scale),
                tooltip=[
                    alt.Tooltip("Tactics:N", title="Tactics"),
                    alt.Tooltip("metric_label:N", title="Metric"),
                    alt.Tooltip("Value:N", title="Value"),
                ],
            )
        )

        chart = alt.layer(bars_roi_left, bars_resp_cost_right) \
            .resolve_scale(y="independent") \
            .properties(title=f"{l.capitalize()} ROI - {time}" if mode == 'layer' else f"ROI - {time}{title}",
                        height=height) \
            .configure_view(stroke="black")
        chart = chart.configure(background="transparent")
        chart_ls.append(chart)

    return chart_ls, expander_data


def _roi_resp_cost_bar_for_insights(df_plot, time, height=300, y_axis_upper=[-1, -1], title="", mode='layer', roi_type='roi', client_code=None, model_group_id=None):
    try:
        level_col = [c for c in df_plot.columns if roi_type in c and 'level' in c][0]
        tactics_idx = df_plot.columns.get_loc('Tactics')
        idx_ls = list(df_plot.columns[:tactics_idx + 1]) + [level_col]
        expander_data = df_plot.pivot(index=idx_ls, columns='Metric', values='Value').reset_index()
        expander_data = readout_utils.add_unit_sign(expander_data, len(idx_ls), period_suffixes=['% Change'], client_code=client_code, model_group_id=model_group_id)
        if mode != 'layer' and level_col in expander_data.columns:
            expander_data = expander_data.drop(columns=[level_col])
    except:
        expander_data = df_plot
    return expander_data


def _roi_resp_cost_change_bar(df_plot_change, time, height=300, y_axis_upper=[-1, -1], title="", mode='layer'):
    level_col = [c for c in df_plot_change.columns if 'roi' in c and 'level' in c][0]
    tactics_idx = df_plot_change.columns.get_loc('Tactics')

    idx_ls = list(df_plot_change.columns[:tactics_idx + 1]) + [level_col]

    expander_data = df_plot_change.pivot(index=idx_ls, columns='Metric', values='Value').reset_index()
    expander_data = readout_utils.add_unit_sign(expander_data, len(idx_ls),
                                                period_suffixes=['% Change'])

    if mode != 'layer' and level_col in expander_data.columns:
        expander_data = expander_data.drop(columns=[level_col])

    chart_ls = []

    def get_shades(color):
        cmap = mpl.colormaps[color]
        return [mcolors.rgb2hex(cmap(x)) for x in np.linspace(0.4, 0.8, 3)]

    change_labels = ['ROI Change %', 'Response Change %', 'Cost Change %']
    color_map = {'excellent': 'Greens', 'good': 'Blues', 'justifiable': 'Wistia', 'inefficient': 'Oranges'}

    roi_levels = df_plot_change[level_col].unique().tolist()

    layers = ['excellent', 'good', 'justifiable', 'inefficient']
    has_layer = df_plot_change[level_col].astype(str).str.lower().isin(layers).any()
    if not has_layer:
        mode = 'all'
        layers = ['all']

    for l in layers:
        if mode == 'layer' and l not in roi_levels:
            df_plot_sub = df_plot_change
        elif mode == 'layer':
            df_plot_sub = df_plot_change[df_plot_change[level_col] == l]
        else:
            df_plot_sub = df_plot_change

        color_scale = alt.Scale(
            domain=change_labels,
            range=get_shades(color_map[l] if l in color_map else color_map['good'])
        )
        sort_order = (
            df_plot_sub[df_plot_sub['metric_label'] == 'ROI Change %']
            .sort_values('Value', ascending=False)['Tactics']
            .tolist()
        )
        chart = (
            alt.Chart(df_plot_sub)
            .mark_bar()
            .encode(
                x=alt.X(f"Tactics:N", title="Tactics", axis=alt.Axis(labelAngle=-20), sort=sort_order),
                xOffset=alt.X("metric_label:N", title=None, sort=change_labels),
                y=alt.Y("Value:Q", title="% Change", axis=alt.Axis(grid=True, tickCount=4)),
                color=alt.Color("metric_label:N", title="Metric", scale=color_scale),
                tooltip=[
                    alt.Tooltip(f"Tactics:N", title="Tactics"),
                    alt.Tooltip("metric_label:N", title="Metric"),
                    alt.Tooltip("Value:N", title="Value"),
                ],
            )
        ).properties(
            title=f"{l.capitalize()} ROI - {time} Performance Change" if mode == 'layer'
            else f"ROI - {time} Performance Change{title}",
            height=height
        ).configure_view(stroke="black")

        chart = chart.configure(background="transparent")
        chart_ls.append(chart)

    return chart_ls, expander_data


def _roi_resp_cost_change_bar_for_insights(df_plot_change, time, height=300, y_axis_upper=[-1, -1], title="",
                                           mode='layer', roi_type="roi",
                                           client_code=None, model_group_id=None):
    try:
        level_col = [c for c in df_plot_change.columns if roi_type in c and 'level' in c][0]
        tactics_idx = df_plot_change.columns.get_loc('Tactics')
        idx_ls = list(df_plot_change.columns[:tactics_idx + 1]) + [level_col]

        expander_data = df_plot_change.pivot(index=idx_ls, columns='Metric', values='Value').reset_index()
        expander_data = readout_utils.add_unit_sign(expander_data, len(idx_ls),
                                                    period_suffixes=['% Change'],
                                                    client_code=client_code, model_group_id=model_group_id)

        if mode != 'layer' and level_col in expander_data.columns:
            expander_data = expander_data.drop(columns=[level_col])
    except:
        expander_data = df_plot_change
    return expander_data



def multiple_roi_stacked_bar(data, short_col, long_col, total_col, time, x_upper_bound=-1, title=""):
    with st.expander("Show data table"):
        tactics_idx = data.columns.get_loc("Tactics")
        expander_data = data[data['Metric'].isin([short_col, long_col, total_col])].pivot(
            index=list(data.columns[:tactics_idx + 1]), columns='Metric',
            values='Value').reset_index()
        st.dataframe(expander_data, use_container_width=True)

    df2 = data[data['Metric'].isin([short_col, long_col, total_col])]
    df2["show_seg"] = (df2['Value'] > 0.8) | (df2['Metric'] == total_col)
    long = df2[df2['Metric'].isin([short_col, long_col])]

    type_map = {
        short_col: "Short-term ROI",
        long_col: "Long-term ROI"
    }
    long["roi_type"] = long["Metric"].map(type_map)
    long["roi_order"] = long["roi_type"].map({"Short-term ROI": 0, "Long-term ROI": 1})

    tactics_sort = df2[df2['Metric'] == short_col].sort_values('Value', ascending=False)['Tactics'].tolist()
    color_scale = alt.Scale(
        domain=["Short-term ROI", "Long-term ROI"],
        range=["#E0B300", "#ff7f0e"]
    )

    x_scale = alt.Scale(domain=[0, x_upper_bound]) if x_upper_bound > 0 else alt.Scale()

    bars = (
        alt.Chart(long, title=f"Short-term ROI vs Long-term ROI {(' - ' + time)}{title}")
        .mark_bar()
        .encode(
            y=alt.Y(f"Tactics:N", title="Tactics", sort=tactics_sort),
            x=alt.X("Value:Q", title="ROI (Total = Short-term + Long-term)", stack="zero", scale=x_scale),
            color=alt.Color("roi_type:N", title="Type", scale=color_scale),
            order=alt.Order("roi_order:Q"),
            tooltip=[
                alt.Tooltip(f"Tactics:N", title="Tactic"),
                alt.Tooltip("roi_type:N", title="Type"),
                alt.Tooltip("Value:Q", title="Value", format=".1f")
            ]
        )
    )
    seg_labels = (
        alt.Chart(long)
        .transform_filter(alt.datum.show_seg)
        .mark_text(align='right', baseline='middle', dx=-4)
        .encode(
            y=alt.Y(f"Tactics:N", sort=tactics_sort),
            x=alt.X("Value:Q", stack="zero"),
            order=alt.Order("roi_order:Q"),
            detail="roi_type:N",
            text=alt.Text("Value:Q", format=".1f"),
        )
    )

    total_df = df2[df2['Metric'] == total_col]
    total_labels = (
        alt.Chart(total_df)
        .mark_text(align="left", dx=4)
        .encode(
            y=alt.Y(f"Tactics:N", sort=tactics_sort),
            x=alt.X("Value:Q"),
            text=alt.Text("Value:Q", format=".1f"),
        )
    )

    chart = (bars + seg_labels + total_labels).properties(
        height=800, padding={"left": 5, "right": 20, "top": 10, "bottom": 10})
    chart = chart.configure(background="transparent").configure_view(fill="transparent")

    return chart


# ---------------------------------------------------------------------------
# Segmentation summary helpers
# (these live here in roi.py; chart.common re-exports them for other modules)
# ---------------------------------------------------------------------------

def build_level_stats(df_long, roi_n, spend_n, roi_n_1, spend_n_1, roi_level_col) -> dict:
    needed_metrics = [roi_n, spend_n, roi_n_1, spend_n_1]
    df = df_long[df_long["Metric"].isin(needed_metrics)].copy()

    if df.empty:
        return {}

    wide = (
        df.pivot_table(
            index=["Tactics", roi_level_col],
            columns="Metric",
            values="Value"
        )
        .reset_index()
    )

    level_stats = {}
    level_order = ["excellent", "good", "justifiable", "inefficient"]

    for level in level_order:
        df_level = wide[wide[roi_level_col] == level].copy()
        if df_level.empty:
            level_stats[level] = {
                "avg_roi": 0.0,
                "spend": 0.0,
                "count": 0,
                "roi_change": "0.0%",
            }
            continue

        spend_level = df_level[spend_n].sum()
        if spend_level > 0:
            avg_roi_2024 = float((df_level[roi_n] * df_level[spend_n]).sum() / spend_level)
        else:
            avg_roi_2024 = float(df_level[roi_n].mean())

        spend_level_n_1 = df_level[spend_n_1].sum()
        if spend_level_n_1 > 0:
            avg_roi_2023 = float((df_level[roi_n_1] * df_level[spend_n_1]).sum() / spend_level_n_1)
        else:
            avg_roi_2023 = float(df_level[roi_n_1].mean())

        if avg_roi_2023 and not np.isnan(avg_roi_2023) and avg_roi_2023 != 0:
            roi_change_pct = (avg_roi_2024 / avg_roi_2023 - 1.0) * 100.0
        else:
            roi_change_pct = 0.0

        roi_change_str = f"{roi_change_pct:.1f}%"

        level_stats[level] = {
            "avg_roi": avg_roi_2024,
            "spend": spend_level,
            "count": int(df_level.shape[0]),
            "roi_change": roi_change_str,
        }

    return level_stats


def _roi_level_card(level_name, stats, palette):
    avg_roi = stats.get("avg_roi", 0.0) or 0.0
    spend = stats.get("spend", 0.0) or 0.0
    count = int(stats.get("count", 0) or 0)
    delta_raw = stats.get("roi_change", None)
    delta_value = float(str(delta_raw).strip().replace("%", ""))

    arrow = "▲" if delta_value > 0 else ("▼" if delta_value < 0 else "●")
    delta_color = palette["up"] if delta_value > 0 else (
        palette["down"] if delta_value < 0 else palette["neutral"]
    )

    with mui.Paper(
        elevation=0,
        sx={
            "borderRadius": 2,
            "border": f"1px solid {palette['border']}",
            "backgroundColor": palette["bg"],
            "p": 1.5,
            "display": "flex",
            "flexDirection": "column",
            "gap": "4px",
        },
    ):
        mui.Typography(
            level_name.capitalize(),
            variant="subtitle2",
            sx={"fontWeight": 700, "color": palette["text"]},
        )
        mui.Typography(
            f"{avg_roi:.2f}x",
            variant="h5",
            sx={"fontWeight": 700},
        )
        with mui.Stack(direction="row", spacing=1, sx={"mt": 0.5}):
            mui.Chip(
                label=f"Spend {spend:.1f} (MM)",
                size="small",
                variant="outlined",
                sx={"borderColor": palette["border"]},
            )
            mui.Chip(
                label=f"{count} tactics",
                size="small",
                variant="outlined",
                sx={"borderColor": palette["border"]},
            )
        mui.Typography(
            f"{arrow} {abs(delta_value):.1f}%",
            variant="body2",
            sx={"mt": 0.5, "color": delta_color},
        )


def render_roi_segmentation_summary(level_stats, time):
    palettes = {
        "excellent": {
            "bg": "#E5F4E8",
            "border": "#1F7A1F",
            "text": "#195F19",
            "up": "#0B8043",
            "down": "#B31412",
            "neutral": "#5F6368",
        },
        "good": {
            "bg": "#E7F0FB",
            "border": "#155A8A",
            "text": "#124669",
            "up": "#185ABC",
            "down": "#B31412",
            "neutral": "#5F6368",
        },
        "justifiable": {
            "bg": "#FFF8D9",
            "border": "#C19000",
            "text": "#7A5A00",
            "up": "#188038",
            "down": "#B31412",
            "neutral": "#5F6368",
        },
        "inefficient": {
            "bg": "#FFE8D9",
            "border": "#CC5E00",
            "text": "#9A4500",
            "up": "#188038",
            "down": "#B31412",
            "neutral": "#5F6368",
        },
    }

    with elements("roi_segmentation_summary"):
        with mui.Paper(
            key="roi_seg_panel",
            elevation=1,
            sx={
                "p": 2,
                "display": "flex",
                "flexDirection": "column",
                "gap": "10px",
            },
        ):
            mui.Typography(
                f"ROI Segmentation - {time}",
                variant="h6",
                sx={"mb": 0.5},
            )
            mui.Typography(
                "Average ROI & spend distribution by efficiency tier",
                variant="body2",
                sx={"color": "text.secondary", "mb": 1},
            )
            with mui.Grid(
                container=True,
                spacing=2,
                sx={"justifyContent": "space-between"}
            ):
                order = ["excellent", "good", "justifiable", "inefficient"]
                for name in order:
                    stats = level_stats.get(name, {}) or {}
                    with mui.Grid(item=True, xs=3):
                        _roi_level_card(name, stats, palettes[name])


# ---------------------------------------------------------------------------
# Group 1 — Snapshot
# ---------------------------------------------------------------------------

def build_roi_snapshot_chart_configs(d, agg_d, time_sel, is_specific,
                                     chart_mode_agg_ls, chart_mode_ls, prompt_inst):
    """Pure Python/pandas: build chart_configs dict for the ROI snapshot view.
    ZERO st.* calls.
    """
    chart_configs = {}
    if not agg_d and not d:
        return chart_configs

    if agg_d:
        recent_data, cols_dict, outliers = agg_d['data'], agg_d['chart_cols'], agg_d['outlier_str']
    else:
        recent_data, cols_dict, outliers = pd.DataFrame(), dict(), dict()

    if d:
        recent_data_detail, cols_dict, outliers_detail = d['data'], d['chart_cols'], d['outlier_str']
    else:
        recent_data_detail, cols_dict, outliers_detail = pd.DataFrame(), dict(), dict()

    roi_value_col, roi_change_col, spend_value_col, spend_change_col = (
        cols_dict['roi'], cols_dict['roi_change'], cols_dict['spend'], cols_dict['spend_change']
    )
    time_label = list(viz_utils.get_time_align_dict([time_sel]).keys())[0]
    inst_value = prompt_inst[prompt_inst['intentionName'] == "ROI-Spend latest"]['instruction'].values[0]
    inst_change = prompt_inst[prompt_inst['intentionName'] == "ROI-Spend trend test"]['instruction'].values[0]

    roi_upper = spend_upper = -1
    for data in [recent_data, recent_data_detail]:
        if not data.empty and len(data) > 2:
            roi_upper = max(roi_upper, data[roi_value_col].max())
            spend_upper = max(spend_upper, data[spend_value_col].max())

    if not recent_data.empty:
        for st_col, data, scatter_f, spend_col, roi_col, outlier_str, mode in [
            (None, recent_data, _roi_spend_scatter, spend_value_col, roi_value_col, outliers.get('value', ''), 'value'),
            (None, recent_data, _roi_spend_change_scatter, spend_change_col, roi_change_col, outliers.get('change', ''), 'change'),
        ]:
            if not (spend_col in data.columns and roi_col in data.columns):
                continue
            title = (f"ROI vs Spend {(' - ' + time_label)}" if mode == 'value'
                     else f"ROI Change vs Spend Change - {time_label}")
            name_suffix = " Change %" if mode == 'change' else ""
            # chart_mode_agg = chart_mode_agg_ls[1] if mode == 'change' else chart_mode_agg_ls[0]
            _, d_json = _most_recent_bar_plot(
                data, spend_col, roi_col, "Tactics",
                "Spend(MM)" + name_suffix, "ROI" + name_suffix,
                title, y_axis_upper=[spend_upper, roi_upper], mode=mode)
            agg_recent_key = f"agg_{title}"
            chart_configs[agg_recent_key] = {
                "query": f"You are given a dataset about {title}.",
                "instruction": inst_value if mode == 'value' else inst_change,
                "data": d_json,
            }

    if not recent_data_detail.empty:
        for st_col, data, scatter_f, spend_col, roi_col, outlier_str, mode in [
            (None, recent_data_detail, _roi_spend_scatter, spend_value_col, roi_value_col, outliers_detail.get('value', ''), 'value'),
            (None, recent_data_detail, _roi_spend_change_scatter, spend_change_col, roi_change_col, outliers_detail.get('change', ''), 'change'),
        ]:
            if not (spend_col in data.columns and roi_col in data.columns):
                continue

            bar_title = "ROI vs Spend" if mode == 'value' else "ROI Change vs Spend Change"
            name_suffix = " Change %" if mode == 'change' else ""
            if not is_specific:
                title_st_col = f"{bar_title} {(' - ' + time_label)}"
                # chart_mode = chart_mode_ls[1] if mode == 'change' else chart_mode_ls[0]
                _, d_json = _most_recent_bar_plot(
                    data, spend_col, roi_col, "Tactics",
                    "Spend(MM)" + name_suffix, "ROI" + name_suffix,
                    title_st_col, y_axis_upper=[spend_upper, roi_upper], mode=mode)
                detail_recent_key = f"specific_{title_st_col}"
                chart_configs[detail_recent_key] = {
                    "query": f"You are given a dataset about {title_st_col}.",
                    "instruction": inst_value if mode == 'value' else inst_change,
                    "data": d_json,
                }
            else:
                group_col = 'group'
                groups = data['group'].unique()
                for i, grp in enumerate(groups):
                    data_group = data[data[group_col] == grp]
                    title_gp = f"{bar_title} {(' - ' + time_label)}{(' - ' + grp) if grp else ''}"
                    _, d_json = _most_recent_bar_plot(
                        data_group, spend_col, roi_col, "Tactics",
                        "Spend(MM)" + name_suffix, "ROI" + name_suffix,
                        title_gp, 300, -20,
                        y_axis_upper=[spend_upper, roi_upper], mode=mode)
                    group_recent_key = f"group_{title_gp}"
                    chart_configs[group_recent_key] = {
                        "query": f"You are given a dataset about {title_gp}.",
                        "instruction": inst_value if mode == 'value' else inst_change,
                        "data": d_json,
                    }
                # chart_configs[f"agg_group_specific_{bar_title} {(' - ' + time_label)}"] = {
                #     "query": f"You are given a dataset about {bar_title} {(' - ' + time_label)}.",
                #     "instruction": inst_value if mode == 'value' else inst_change,
                #     "data": data[['Tactics', spend_col, roi_col, group_col]].to_json(orient="records"),
                # }

    return chart_configs


def build_roi_snapshot_chart_configs_for_insights(d, agg_d, time_sel, is_specific,
                                     chart_mode_agg_ls, chart_mode_ls, prompt_inst, client_code, model_group_id):
    """
    build chart_configs dict for the ROI snapshot view.
    """
    chart_configs = {}
    if not agg_d and not d:
        return chart_configs

    if agg_d:
        recent_data, cols_dict, outliers = agg_d['data'], agg_d['chart_cols'], agg_d['outlier_str']
    else:
        recent_data, cols_dict, outliers = pd.DataFrame(), dict(), dict()

    if d:
        recent_data_detail, cols_dict, outliers_detail = d['data'], d['chart_cols'], d['outlier_str']
    else:
        recent_data_detail, cols_dict, outliers_detail = pd.DataFrame(), dict(), dict()

    roi_value_col, roi_change_col, spend_value_col, spend_change_col, activity_value_col, activity_change_col = (
        cols_dict['roi'], cols_dict['roi_change'], cols_dict['spend'], cols_dict['spend_change'], cols_dict['activity'], cols_dict['activity_change']
    )
    time_label = list(viz_utils.get_time_align_dict([time_sel]).keys())[0]
    inst_value = prompt_inst[prompt_inst['intentionName'] == "ROI-Spend latest"]['instruction'].values[0]
    inst_change = prompt_inst[prompt_inst['intentionName'] == "ROI-Spend trend test"]['instruction'].values[0]

    roi_upper = spend_upper = activity_upper = -1
    for data in [recent_data, recent_data_detail]:
        if not data.empty and len(data) > 2:
            roi_upper = max(roi_upper, data[roi_value_col].max())
            spend_upper = max(spend_upper, data[spend_value_col].max())
            activity_upper = max(activity_upper, data[activity_value_col].max())

    if not recent_data.empty:
        for st_col, data, scatter_f, spend_col, roi_col, activity_col, outlier_str, mode in [
            (None, recent_data, _roi_spend_scatter_for_insights, spend_value_col, roi_value_col, activity_value_col, outliers.get('value', ''), 'value'),
            (None, recent_data, _roi_spend_change_scatter_for_insights, spend_change_col, roi_change_col, activity_change_col, outliers.get('change', ''), 'change'),
        ]:
            if not (spend_col in data.columns and roi_col in data.columns):
                continue
            else:
                title = (f"ROI vs Spend {(' - ' + time_label)}" if mode == 'value'
                         else f"ROI Change vs Spend Change - {time_label}")
                name_suffix = " Change %" if mode == 'change' else ""
                # chart_mode_agg = chart_mode_agg_ls[1] if mode == 'change' else chart_mode_agg_ls[0]
                d_json = _most_recent_bar_plot_for_insights(
                    data, spend_col, roi_col, "Tactics",
                    "Spend(MM)" + name_suffix, "ROI" + name_suffix,
                    title, y_axis_upper=[spend_upper, roi_upper], mode=mode, client_code=client_code, model_group_id=model_group_id)
                agg_recent_key = f"agg_{title}"
                chart_configs[agg_recent_key] = {
                    "query": f"You are given a dataset about {title}.",
                    "instruction": inst_value if mode == 'value' else inst_change,
                    "data": d_json,
                }
            if not (activity_col in data.columns and roi_col in data.columns):
                continue
            else:
                title = (f"ROI vs Activity {(' - ' + time_label)}" if mode == 'value'
                         else f"ROI Change vs Activity Change - {time_label}")
                name_suffix = " Change %" if mode == 'change' else ""
                # chart_mode_agg = chart_mode_agg_ls[1] if mode == 'change' else chart_mode_agg_ls[0]
                d_json = _most_recent_bar_plot_for_insights(
                    data, activity_col, roi_col, "Tactics",
                    "Activity" + name_suffix, "ROI" + name_suffix,
                    title, y_axis_upper=[activity_upper, roi_upper], mode=mode, client_code=client_code,
                    model_group_id=model_group_id)
                agg_recent_key = f"agg_{title}"
                chart_configs[agg_recent_key] = {
                    "query": f"You are given a dataset about {title}.",
                    "instruction": inst_value if mode == 'value' else inst_change,
                    "data": d_json,
                }

    if not recent_data_detail.empty:
        for st_col, data, scatter_f, spend_col, roi_col, activity_col, outlier_str, mode in [
            (None, recent_data_detail, _roi_spend_scatter_for_insights, spend_value_col, roi_value_col, activity_value_col, outliers_detail.get('value', ''), 'value'),
            (None, recent_data_detail, _roi_spend_change_scatter_for_insights, spend_change_col, roi_change_col, activity_change_col, outliers_detail.get('change', ''), 'change'),
        ]:
            if not (spend_col in data.columns and roi_col in data.columns):
                continue
            else:
                bar_title = "ROI vs Spend" if mode == 'value' else "ROI Change vs Spend Change"
                name_suffix = " Change %" if mode == 'change' else ""
                if not is_specific:
                    title_st_col = f"{bar_title} {(' - ' + time_label)}"
                    # chart_mode = chart_mode_ls[1] if mode == 'change' else chart_mode_ls[0]
                    d_json = _most_recent_bar_plot_for_insights(
                        data, spend_col, roi_col, "Tactics",
                        "Spend(MM)" + name_suffix, "ROI" + name_suffix,
                        title_st_col, y_axis_upper=[spend_upper, roi_upper], mode=mode,
                        client_code=client_code, model_group_id=model_group_id)
                    detail_recent_key = f"specific_{title_st_col}"
                    chart_configs[detail_recent_key] = {
                        "query": f"You are given a dataset about {title_st_col}.",
                        "instruction": inst_value if mode == 'value' else inst_change,
                        "data": d_json,
                    }
                else:
                    group_col = 'group'
                    groups = data['group'].unique()
                    for i, grp in enumerate(groups):
                        data_group = data[data[group_col] == grp]
                        title_gp = f"{bar_title} {(' - ' + time_label)}{(' - ' + grp) if grp else ''}"
                        d_json = _most_recent_bar_plot_for_insights(
                            data_group, spend_col, roi_col, "Tactics",
                            "Spend(MM)" + name_suffix, "ROI" + name_suffix,
                            title_gp, 300, -20,
                            y_axis_upper=[spend_upper, roi_upper], mode=mode,
                            client_code=client_code, model_group_id=model_group_id)
                        group_recent_key = f"group_{title_gp}"
                        chart_configs[group_recent_key] = {
                            "query": f"You are given a dataset about {title_gp}.",
                            "instruction": inst_value if mode == 'value' else inst_change,
                            "data": d_json,
                        }
                    # chart_configs[f"agg_group_specific_{bar_title} {(' - ' + time_label)}"] = {
                    #     "query": f"You are given a dataset about {bar_title} {(' - ' + time_label)}.",
                    #     "instruction": inst_value if mode == 'value' else inst_change,
                    #     "data": data[['Tactics', spend_col, roi_col, group_col]].to_json(orient="records"),
                    # }
            if not (activity_col in data.columns and roi_col in data.columns):
                continue
            else:
                bar_title = "ROI vs Activity" if mode == 'value' else "ROI Change vs Activity Change"
                name_suffix = " Change %" if mode == 'change' else ""
                if not is_specific:
                    title_st_col = f"{bar_title} {(' - ' + time_label)}"
                    # chart_mode = chart_mode_ls[1] if mode == 'change' else chart_mode_ls[0]
                    d_json = _most_recent_bar_plot_for_insights(
                        data, activity_col, roi_col, "Tactics",
                        "Activity" + name_suffix, "ROI" + name_suffix,
                        title_st_col, y_axis_upper=[activity_upper, roi_upper], mode=mode,
                        client_code=client_code, model_group_id=model_group_id)
                    detail_recent_key = f"specific_{title_st_col}"
                    chart_configs[detail_recent_key] = {
                        "query": f"You are given a dataset about {title_st_col}.",
                        "instruction": inst_value if mode == 'value' else inst_change,
                        "data": d_json,
                    }
                else:
                    group_col = 'group'
                    groups = data['group'].unique()
                    for i, grp in enumerate(groups):
                        data_group = data[data[group_col] == grp]
                        title_gp = f"{bar_title} {(' - ' + time_label)}{(' - ' + grp) if grp else ''}"
                        d_json = _most_recent_bar_plot_for_insights(
                            data_group, activity_col, roi_col, "Tactics",
                            "Activity" + name_suffix, "ROI" + name_suffix,
                            title_gp, 300, -20,
                            y_axis_upper=[activity_upper, roi_upper], mode=mode,
                            client_code=client_code, model_group_id=model_group_id)
                        group_recent_key = f"group_{title_gp}"
                        chart_configs[group_recent_key] = {
                            "query": f"You are given a dataset about {title_gp}.",
                            "instruction": inst_value if mode == 'value' else inst_change,
                            "data": d_json,
                        }
    return chart_configs


def render_roi_snapshot_charts(d, agg_d, time_sel, is_specific,
                               chart_mode_agg_ls, chart_mode_ls, insights):
    """Pure Streamlit rendering of the ROI snapshot view.
    Accepts pre-computed insights dict. ZERO LLM calls.
    """
    if not agg_d and not d:
        return

    if agg_d:
        recent_data, cols_dict, outliers = agg_d['data'], agg_d['chart_cols'], agg_d['outlier_str']
    else:
        recent_data, cols_dict, outliers = pd.DataFrame(), dict(), dict()

    if d:
        recent_data_detail, cols_dict, outliers_detail = d['data'], d['chart_cols'], d['outlier_str']
    else:
        recent_data_detail, cols_dict, outliers_detail = pd.DataFrame(), dict(), dict()

    roi_value_col, roi_change_col, spend_value_col, spend_change_col = (
        cols_dict['roi'], cols_dict['roi_change'], cols_dict['spend'], cols_dict['spend_change']
    )
    time_label = list(viz_utils.get_time_align_dict([time_sel]).keys())[0]

    roi_upper = spend_upper = -1
    for data in [recent_data, recent_data_detail]:
        if not data.empty and len(data) > 2:
            roi_upper = max(roi_upper, data[roi_value_col].max())
            spend_upper = max(spend_upper, data[spend_value_col].max())

    if not recent_data.empty:
        st.markdown("#### Aggregation View")
        if len(recent_data) > 2:
            left, right = st.columns([1, 1])
            for st_col, data, scatter_f, spend_col, roi_col, outlier_str, mode, chart_mode_agg_ in [
                (left, recent_data, _roi_spend_scatter, spend_value_col, roi_value_col,
                 outliers.get('value', ''), 'value', chart_mode_agg_ls[0]),
                (right, recent_data, _roi_spend_change_scatter, spend_change_col, roi_change_col,
                 outliers.get('change', ''), 'change', chart_mode_agg_ls[1]),
            ]:
                with st_col:
                    if not (spend_col in data.columns and roi_col in data.columns):
                        continue
                    title = (f"ROI vs Spend {(' - ' + time_label)}" if mode == 'value'
                             else f"ROI Change vs Spend Change - {time_label}")
                    name_suffix = " Change %" if mode == 'change' else ""
                    if chart_mode_agg_:
                        bar, _ = _most_recent_bar_plot(
                            data, spend_col, roi_col, "Tactics",
                            "Spend(MM)" + name_suffix, "ROI" + name_suffix,
                            title, y_axis_upper=[spend_upper, roi_upper], mode=mode)
                        st.altair_chart(bar, use_container_width=True)
                        st.session_state.insight[f"agg_{title}"] = insights.get(f"agg_{title}")
                        saved = st.session_state.insight.get(f"agg_{title}")
                        if saved:
                            render_chart_insight_box("Recent ROI Insight", saved)
                    else:
                        chart, config_key = scatter_f(data, spend_col, roi_col, time_label, table='agg')
                        st.altair_chart(chart, use_container_width=True)
                        st.session_state.insight[config_key] = insights.get(config_key)
                        saved = st.session_state.insight.get(config_key)
                        if saved:
                            render_chart_insight_box("Recent ROI Insight", saved)
        # table display
        table_display = readout_utils.add_unit_sign(recent_data, metric_col_idx=1, period_suffixes=['% Change'])
        table_element(table_display.sort_values(spend_value_col, ascending=False),
                      "", "roi_snapshot_table_agg")
        st.markdown(outliers.get('value', '') + "\n" + outliers.get('change', ''))

    if not recent_data_detail.empty:
        st.markdown("#### Detail View")
        if len(recent_data_detail) > 2:
            left, right = st.columns([1, 1])
            for st_col, data, scatter_f, spend_col, roi_col, outlier_str, mode, chart_mode_ in [
                (left, recent_data_detail, _roi_spend_scatter, spend_value_col, roi_value_col,
                 outliers_detail.get('value', ''), 'value', chart_mode_ls[0]),
                (right, recent_data_detail, _roi_spend_change_scatter, spend_change_col, roi_change_col,
                 outliers_detail.get('change', ''), 'change', chart_mode_ls[1]),
            ]:
                if not (spend_col in data.columns and roi_col in data.columns):
                    continue

                bar_title = "ROI vs Spend" if mode == 'value' else "ROI Change vs Spend Change"
                name_suffix = " Change %" if mode == 'change' else ""
                with st_col:
                    if not is_specific:
                        title_st_col = f"{bar_title} {(' - ' + time_label)}"
                        if chart_mode_:
                            bar, _ = _most_recent_bar_plot(
                                data, spend_col, roi_col, "Tactics",
                                "Spend(MM)" + name_suffix, "ROI" + name_suffix,
                                title_st_col, y_axis_upper=[spend_upper, roi_upper], mode=mode)
                            st.altair_chart(bar, use_container_width=True)
                            st.session_state.insight[f"specific_{title_st_col}"] = insights.get(f"specific_{title_st_col}")
                            saved = st.session_state.insight.get(f"specific_{title_st_col}")
                            if saved:
                                render_chart_insight_box("Recent ROI Insight", saved)
                        else:
                            chart, config_key = scatter_f(data, spend_col, roi_col, time_label)
                            st.altair_chart(chart, use_container_width=True)
                            st.session_state.insight[config_key] = insights.get(config_key)
                            saved = st.session_state.insight.get(config_key)
                            if saved:
                                render_chart_insight_box("Recent ROI Insight", saved)
                    else:
                        group_col = 'group'
                        groups = data['group'].unique()
                        large_groups = data.groupby(group_col)['Tactics'].count().min() > 4
                        cols = st.columns([1, 1]) if len(groups) > 1 and not large_groups else st.columns([1])
                        for i, grp in enumerate(groups):
                            data_group = data[data[group_col] == grp]
                            with cols[i % 2] if len(groups) > 1 and not large_groups else cols[0]:
                                title_gp = f"{bar_title} {(' - ' + time_label)}{(' - ' + grp) if grp else ''}"
                                bar, _ = _most_recent_bar_plot(
                                    data_group, spend_col, roi_col, "Tactics",
                                    "Spend(MM)" + name_suffix, "ROI" + name_suffix,
                                    title_gp, 300, -20,
                                    y_axis_upper=[spend_upper, roi_upper], mode=mode)
                                st.altair_chart(bar, use_container_width=True)
                                st.session_state.insight[f"group_{title_gp}"] = insights.get(f"group_{title_gp}")
                                saved = st.session_state.insight.get(f"group_{title_gp}")
                                if saved:
                                    render_chart_insight_box(f"Recent ROI Insight - {grp}", saved)

                        scatter_f(data, spend_col, roi_col, time_label)
        # table
        # table_display = viz_utils.replace_with_labels(recent_data_detail, [roi_value_col, roi_change_col, spend_value_col, spend_change_col])
        table_display = readout_utils.add_unit_sign(recent_data_detail, metric_col_idx=1, period_suffixes=['% Change'])
        table_element(
            table_display.sort_values(spend_value_col, ascending=False),
            "", "roi_snapshot_table")
        st.markdown(outliers_detail.get('value', '') + "\n" + outliers_detail.get('change', ''))


def get_roi_recent_plot_new(d, agg_d, time_sel, is_specific, ag_mapping, prompt_inst, llm_vendor):
    """Orchestrator: ROI snapshot (get roi & spend snapshot for overall roi questions).

    1. Render Bar/Scatter toggles → populate chart_mode_agg_ls, chart_mode_ls
    2. build_roi_snapshot_chart_configs(...)
    3. generate_chart_insights_parallel_cached(...)
    4. render_roi_snapshot_charts(...)
    Returns (chart_configs, insights).
    """
    chart_mode_agg_ls, chart_mode_ls = [False, False], [False, False]
    if not agg_d and not d:
        return

    if agg_d:
        recent_data, cols_dict, outliers = agg_d['data'], agg_d['chart_cols'], agg_d['outlier_str']
    else:
        recent_data, cols_dict, outliers = pd.DataFrame(), dict(), dict()

    if d:
        recent_data_detail, cols_dict, outliers_detail = d['data'], d['chart_cols'], d['outlier_str']
    else:
        recent_data_detail, cols_dict, outliers_detail = pd.DataFrame(), dict(), dict()

    roi_value_col, roi_change_col, spend_value_col, spend_change_col = (
        cols_dict['roi'], cols_dict['roi_change'], cols_dict['spend'], cols_dict['spend_change']
    )

    # --- Render toggles (populates chart_mode_agg_ls / chart_mode_ls) ---
    if not recent_data.empty:
        if len(recent_data) > 2:
            left, right = st.columns([1, 1])
            for st_col, data, scatter_f, spend_col, roi_col, outlier_str, mode in [
                (left, recent_data, _roi_spend_scatter, spend_value_col, roi_value_col,
                 outliers.get('value', ''), 'value'),
                (right, recent_data, _roi_spend_change_scatter, spend_change_col, roi_change_col,
                 outliers.get('change', ''), 'change'),
            ]:
                with st_col:
                    if not (spend_col in data.columns and roi_col in data.columns):
                        continue
                    chart_mode_agg = st.toggle("View: Bar ⟷ Scatter", value=True, key=f"view_mode_{mode}")
                    if mode == 'value':
                        chart_mode_agg_ls[0] = chart_mode_agg
                    else:
                        chart_mode_agg_ls[1] = chart_mode_agg

    if not recent_data_detail.empty:
        if len(recent_data_detail) > 2:
            left, right = st.columns([1, 1])
            for st_col, data, scatter_f, spend_col, roi_col, outlier_str, mode in [
                (left, recent_data_detail, _roi_spend_scatter, spend_value_col, roi_value_col,
                 outliers_detail.get('value', ''), 'value'),
                (right, recent_data_detail, _roi_spend_change_scatter, spend_change_col, roi_change_col,
                 outliers_detail.get('change', ''), 'change'),
            ]:
                if not (spend_col in data.columns and roi_col in data.columns):
                    continue
                with st_col:
                    if not is_specific:
                        chart_mode = st.toggle("View: Bar ⟷ Scatter", value=True, key=f"view_mode_{mode}")
                        if mode == 'value':
                            chart_mode_ls[0] = chart_mode
                        else:
                            chart_mode_ls[1] = chart_mode

    # --- Build configs ---
    chart_configs = build_roi_snapshot_chart_configs(
        d, agg_d, time_sel, is_specific, chart_mode_agg_ls, chart_mode_ls, prompt_inst)

    # --- Generate insights ---
    with st.spinner("Your visuals and insights are on the way..."):
        start = time.time()
        chart_configs_by_group = {k: v for k, v in chart_configs.items() if not k.startswith('agg_group')}
        insights = generate_chart_insights_parallel_cached(chart_configs_by_group, llm_vendor)
        elapsed = time.time() - start
    st.write(f"Insight generation time: {elapsed:.2f} seconds")

    # --- Render ---
    render_roi_snapshot_charts(d, agg_d, time_sel, is_specific,
                               chart_mode_agg_ls, chart_mode_ls, insights)

    return chart_configs, insights


# ---------------------------------------------------------------------------
# Group 2 — Trend
# ---------------------------------------------------------------------------

def build_roi_trend_chart_configs(trend_d, trend_d_agg, times, is_specific, add_long_term,
                                  prompt_inst, multi_roi_data=None, multi_roi_data_agg=None,
                                  roi_type='roi'):
    """
    build chart_configs dict for the ROI trend view
    """
    roi_level_col = f'{roi_type} level'
    chart_configs = {}

    if trend_d_agg:
        trend_data_agg = viz_utils.get_multi_roi_trend_data(trend_d_agg, multi_roi_data_agg, roi_level_col)
    else:
        trend_data_agg = pd.DataFrame()
    if trend_d:
        trend_data = viz_utils.get_multi_roi_trend_data(trend_d, multi_roi_data, roi_level_col)
    else:
        trend_data = pd.DataFrame()

    chart_cols = trend_d['chart_cols'] if trend_d else trend_d_agg['chart_cols']
    bar_spend_cols = [chart_cols['spend']]
    line_roi_cols = [chart_cols['roi']]

    if add_long_term:
        line_roi_cols += [multi_roi_data['chart_cols']['long_term_roi']]

    inst_trend = prompt_inst[prompt_inst['intentionName'] == "ROI-Spend trend"]['instruction'].values[0]

    roi_upper = spend_upper = -1
    for data in [trend_data_agg, trend_data]:
        if not data.empty and len(data) > 2:
            if not add_long_term:
                roi_upper = max(roi_upper, data[data['Metric'].isin(line_roi_cols[0])]['Value'].max())
            else:
                roi_upper = max(roi_upper, data[data['Metric'].isin(line_roi_cols[1] + line_roi_cols[0])]['Value'].max())
            spend_upper = max(spend_upper, data[data['Metric'].isin(bar_spend_cols[0])]['Value'].max())

    if not trend_data_agg.empty:
        _, d = _roi_spend_trend_multi(
            trend_data_agg, bar_spend_cols, line_roi_cols, times,
            y_axis_upper=[spend_upper, roi_upper], mode='all', add_long_term=add_long_term)
        trend_key = "agg_trend_roi"
        chart_configs[trend_key] = {
            "query": "You are given a dataset about ROI trend by level.",
            "instruction": inst_trend,
            "data": d.to_json(orient="records"),
        }

    if not trend_data.empty:
        if not is_specific:
            _, d = _roi_spend_trend_multi(
                trend_data, bar_spend_cols, line_roi_cols, times,
                y_axis_upper=[spend_upper, roi_upper], mode='layer', add_long_term=add_long_term)
            trend_key = "detail_trend_roi"
            chart_configs[trend_key] = {
                "query": "You are given a dataset about ROI trend by level.",
                "instruction": inst_trend,
                "data": d.to_json(orient="records"),
            }
        else:
            group_col = 'group'
            groups = data['group'].unique()
            large_groups = data.groupby(group_col)['Tactics'].count().min() > 4
            cols = st.columns([1, 1]) if len(groups) > 1 and not large_groups else st.columns([1])
            for i, grp in enumerate(groups):
                trend_data_grp = trend_data[trend_data[group_col] == grp]
                with cols[i % 2] if len(groups) > 1 and not large_groups else cols[0]:
                    _, d = _roi_spend_trend_multi(trend_data_grp, bar_spend_cols,
                                                                              line_roi_cols, times,
                                                                              y_axis_upper=[spend_upper, roi_upper],
                                                                              width=200,
                                                                              height=300,
                                                                              title=f' - {grp}' if grp else '',
                                                                              mode='all', add_long_term=add_long_term)
                    trend_key = f"group_{grp}_trend_roi"
                    chart_configs[trend_key] = {"query": f"You are given a dataset about ROI trend by level.",
                                                "instruction": inst_trend,
                                                "data": d.to_json(orient="records")}
            # tactics_idx = trend_data.columns.get_loc('Tactics')
            # index_cols = list(trend_data.columns[:tactics_idx + 1]) + [group_col]
            # expander_data = trend_data.pivot(index=index_cols, columns='Metric',
            #                                  values='Value').reset_index()[
            #     index_cols + bar_spend_cols[0] + [m for x in line_roi_cols for m in x]]
            # chart_configs["detail_trend_roi"] = {
            #     "query": "You are given a dataset about ROI trend by level.",
            #     "instruction": inst_trend,
            #     "data": expander_data.to_json(orient="records"),
            # }

    return chart_configs


def build_roi_trend_chart_configs_for_insights(trend_d, trend_d_agg, times, is_specific, add_long_term,
                                  prompt_inst, multi_roi_data=None, multi_roi_data_agg=None,
                                  roi_type='roi'):
    """
    build chart_configs dict for the ROI trend view
    """
    roi_level_col = f'{roi_type} level'
    chart_configs = {}

    if trend_d_agg:
        trend_data_agg = viz_utils.get_multi_roi_trend_data(trend_d_agg, multi_roi_data_agg, roi_level_col)
    else:
        trend_data_agg = pd.DataFrame()
    if trend_d:
        trend_data = viz_utils.get_multi_roi_trend_data(trend_d, multi_roi_data, roi_level_col)
    else:
        trend_data = pd.DataFrame()

    chart_cols = trend_d['chart_cols'] if trend_d else trend_d_agg['chart_cols']
    bar_spend_cols = [chart_cols['spend']]
    line_roi_cols = [chart_cols['roi']]

    if add_long_term:
        line_roi_cols += [multi_roi_data['chart_cols']['long_term_roi']]

    inst_trend = prompt_inst[prompt_inst['intentionName'] == "ROI-Spend trend"]['instruction'].values[0]

    roi_upper = spend_upper = -1
    for data in [trend_data_agg, trend_data]:
        if not data.empty and len(data) > 2:
            if not add_long_term:
                roi_upper = max(roi_upper, data[data['Metric'].isin(line_roi_cols[0])]['Value'].max())
            else:
                roi_upper = max(roi_upper, data[data['Metric'].isin(line_roi_cols[1] + line_roi_cols[0])]['Value'].max())
            spend_upper = max(spend_upper, data[data['Metric'].isin(bar_spend_cols[0])]['Value'].max())

    if not trend_data_agg.empty:
        d = _roi_spend_trend_multi_for_insights(
            trend_data_agg, bar_spend_cols, line_roi_cols, times,
            y_axis_upper=[spend_upper, roi_upper], mode='all', add_long_term=add_long_term, roi_type=roi_type)
        trend_key = "agg_trend_roi"
        chart_configs[trend_key] = {
            "query": "You are given a dataset about ROI trend by level.",
            "instruction": inst_trend,
            "data": d.to_json(orient="records"),
        }

    if not trend_data.empty:
        if not is_specific:
            d = _roi_spend_trend_multi_for_insights(
                trend_data, bar_spend_cols, line_roi_cols, times,
                y_axis_upper=[spend_upper, roi_upper], mode='layer', add_long_term=add_long_term, roi_type=roi_type)
            trend_key = "detail_trend_roi"
            chart_configs[trend_key] = {
                "query": "You are given a dataset about ROI trend by level.",
                "instruction": inst_trend,
                "data": d.to_json(orient="records"),
            }
        else:
            group_col = 'group'
            groups = data['group'].unique()
            large_groups = data.groupby(group_col)['Tactics'].count().min() > 4
            for i, grp in enumerate(groups):
                trend_data_grp = trend_data[trend_data[group_col] == grp]
                d = _roi_spend_trend_multi_for_insights(trend_data_grp, bar_spend_cols,
                                                                            line_roi_cols, times,
                                                                            y_axis_upper=[spend_upper, roi_upper],
                                                                            width=200,
                                                                            height=300,
                                                                            title=f' - {grp}' if grp else '',
                                                                            mode='all', add_long_term=add_long_term,
                                                                            roi_type=roi_type)
                trend_key = f"group_{grp}_trend_roi"
                chart_configs[trend_key] = {"query": f"You are given a dataset about ROI trend by level.",
                                            "instruction": inst_trend,
                                            "data": d.to_json(orient="records")}
            # tactics_idx = trend_data.columns.get_loc('Tactics')
            # index_cols = list(trend_data.columns[:tactics_idx + 1]) + [group_col]
            # expander_data = trend_data.pivot(index=index_cols, columns='Metric',
            #                                  values='Value').reset_index()[
            #     index_cols + bar_spend_cols[0] + [m for x in line_roi_cols for m in x]]
            # chart_configs["detail_trend_roi"] = {
            #     "query": "You are given a dataset about ROI trend by level.",
            #     "instruction": inst_trend,
            #     "data": expander_data.to_json(orient="records"),
            # }

    return chart_configs


def render_roi_trend_charts(trend_d, trend_d_agg, times, is_specific,
                            multi_roi_data, multi_roi_data_agg, roi_type, add_long_term,
                            bar_spend_cols, line_roi_cols, roi_upper, spend_upper, insights):
    """Pure Streamlit rendering of the ROI trend view.
    Accepts pre-computed insights dict. ZERO LLM calls.
    """
    roi_level_col = f'{roi_type} level'

    if trend_d_agg:
        trend_data_agg = viz_utils.get_multi_roi_trend_data(trend_d_agg, multi_roi_data_agg, roi_level_col)
    else:
        trend_data_agg = pd.DataFrame()
    if trend_d:
        trend_data = viz_utils.get_multi_roi_trend_data(trend_d, multi_roi_data, roi_level_col)
    else:
        trend_data = pd.DataFrame()

    if not trend_data_agg.empty:
        st.markdown("#### Aggregation View")
        chart_ls_agg, expander_data = _roi_spend_trend_multi(
            trend_data_agg, bar_spend_cols, line_roi_cols, times,
            y_axis_upper=[spend_upper, roi_upper], mode='all', add_long_term=add_long_term)
        with st.expander("Show data table"):
            st.dataframe(expander_data, use_container_width=True)
        st.altair_chart(chart_ls_agg[0], use_container_width=True)
        st.session_state.insight["agg_trend_roi"] = insights.get("agg_trend_roi")
        saved = st.session_state.insight.get("agg_trend_roi")
        if saved:
            render_chart_insight_box("ROI Trend Insight", saved)

    if not trend_data.empty:
        st.markdown("#### Detail View")
        if not is_specific:
            col = f"{roi_type} level"
            mode = ('all' if col in trend_data.columns
                    and trend_data[col].astype(str).str.strip().eq("").all()
                    else 'layer')
            chart_ls, expander_data = _roi_spend_trend_multi(
                trend_data, bar_spend_cols, line_roi_cols, times,
                y_axis_upper=[spend_upper, roi_upper], mode=mode, add_long_term=add_long_term)
            with st.expander("Show data table"):
                st.dataframe(expander_data, use_container_width=True)
            for chart in chart_ls:
                st.altair_chart(chart, use_container_width=True)
            st.session_state.insight["detail_trend_roi"] = insights.get("detail_trend_roi")
            saved = st.session_state.insight.get("detail_trend_roi")
            if saved:
                render_chart_insight_box("ROI Trend Insight", saved)
        else:
            group_col = 'group'
            groups = trend_data['group'].unique()
            large_groups = trend_data.groupby(group_col)['Tactics'].count().min() > 4
            cols = st.columns([1, 1]) if len(groups) > 1 and not large_groups else st.columns([1])
            for i, grp in enumerate(groups):
                trend_data_grp = trend_data[trend_data[group_col] == grp]
                with cols[i % 2] if len(groups) > 1 and not large_groups else cols[0]:
                    chart_ls_detail, expander_data_grp = _roi_spend_trend_multi(
                        trend_data_grp, bar_spend_cols, line_roi_cols, times,
                        y_axis_upper=[spend_upper, roi_upper], width=200, height=300,
                        title=f' - {grp}' if grp else '', mode='all', add_long_term=add_long_term)
                    with st.expander("Show data table"):
                        st.dataframe(expander_data_grp, use_container_width=True)
                    st.altair_chart(chart_ls_detail[0], use_container_width=True)
                    st.session_state.insight[f"group_{grp}_trend_roi"] = insights.get(f"group_{grp}_trend_roi")
                    saved = st.session_state.insight.get(f"group_{grp}_trend_roi")
                    if saved:
                        render_chart_insight_box(f"ROI Trend Insight - {grp}", saved)


def get_roi_trend_plot_new_multi_roi(trend_d, trend_d_agg, times, is_specific, ag_mapping,
                                     multi_roi_data=None, multi_roi_data_agg=None,
                                     roi_type='roi', prompt_inst=pd.DataFrame(),
                                     llm_vendor='deepseek'):
    """Orchestrator: ROI trend (get roi & spend trend for overall trend).

    1. Segmentation summary + toggle rendering
    2. build_roi_trend_chart_configs(...)
    3. generate_chart_insights_parallel_cached(...)
    4. Compute axis bounds
    5. render_roi_trend_charts(...)
    Returns (chart_configs, insights).
    """
    roi_level_col = f'{roi_type} level'

    if trend_d_agg:
        trend_data_agg = viz_utils.get_multi_roi_trend_data(trend_d_agg, multi_roi_data_agg, roi_level_col)
    else:
        trend_data_agg = pd.DataFrame()
    if trend_d:
        trend_data = viz_utils.get_multi_roi_trend_data(trend_d, multi_roi_data, roi_level_col)
    else:
        trend_data = pd.DataFrame()

    chart_cols = trend_d['chart_cols'] if trend_d else trend_d_agg['chart_cols']
    bar_spend_cols = [chart_cols['spend']]
    line_roi_cols = [chart_cols['roi']]

    # --- Segmentation summary card ---
    if not is_specific:
        summary_data = trend_data[trend_data[roi_level_col].isin(['excellent', 'good', 'justifiable', 'inefficient'])]
        level_stats = build_level_stats(
            summary_data, line_roi_cols[0][0], bar_spend_cols[0][0],
            line_roi_cols[0][1], bar_spend_cols[0][1], roi_level_col)
        time_label = list(viz_utils.get_time_align_dict([times[-1]]).keys())[0]
        if level_stats:
            render_roi_segmentation_summary(level_stats, time_label)

    # --- Toggle ---
    add_long_term = False
    if multi_roi_data:
        add_long_term = st.toggle(
            "Show Long-term ROI",
            default_value=False,
            label_after=True,
            inactive_color="#D3D3D3",
            active_color="#1f77b4",
            track_color="#ADD8E6"
        )
    if add_long_term:
        line_roi_cols += [multi_roi_data['chart_cols']['long_term_roi']]

    # --- Build configs ---
    chart_configs = build_roi_trend_chart_configs(
        trend_d, trend_d_agg, times, is_specific, add_long_term,
        prompt_inst, multi_roi_data, multi_roi_data_agg, roi_type)

    # --- Generate insights ---
    with st.spinner("Your visuals and insights are on the way..."):
        start = time.time()
        chart_configs_by_group = {k: v for k, v in chart_configs.items() if not k.startswith('agg_group')}
        insights = generate_chart_insights_parallel_cached(chart_configs_by_group, llm_vendor)
        elapsed = time.time() - start
    st.write(f"Insight generation time: {elapsed:.2f} seconds")

    # --- Compute axis bounds ---
    roi_upper = spend_upper = -1
    for data in [trend_data_agg, trend_data]:
        if not data.empty and len(data) > 2:
            if not add_long_term:
                roi_upper = max(roi_upper, data[data['Metric'].isin(line_roi_cols[0])]['Value'].max())
            else:
                roi_upper = max(roi_upper, data[data['Metric'].isin(line_roi_cols[1] + line_roi_cols[0])]['Value'].max())
            spend_upper = max(spend_upper, data[data['Metric'].isin(bar_spend_cols[0])]['Value'].max())

    # --- Render ---
    render_roi_trend_charts(
        trend_d, trend_d_agg, times, is_specific,
        multi_roi_data, multi_roi_data_agg, roi_type, add_long_term,
        bar_spend_cols, line_roi_cols, roi_upper, spend_upper, insights)

    return chart_configs, insights


# ---------------------------------------------------------------------------
# Group 3 — Response / Cost
# ---------------------------------------------------------------------------

def build_roi_response_cost_chart_configs(resp_cost_d, resp_cost_d_agg, times, is_specific,
                                          ag_mapping, prompt_inst):
    """Pure Python/pandas: build chart_configs dict for the ROI response/cost view.
    ZERO st.* calls.
    """
    chart_configs = {}
    if not resp_cost_d_agg and not resp_cost_d:
        return chart_configs

    if resp_cost_d_agg:
        df_plot_agg, df_plot_change_agg, chart_cols, outliers_agg = (
            resp_cost_d_agg['data'], resp_cost_d_agg['change_data'],
            resp_cost_d_agg['chart_cols'], resp_cost_d_agg['outlier_str'])
    else:
        df_plot_agg, df_plot_change_agg, chart_cols, outliers_agg = (
            pd.DataFrame(), pd.DataFrame(), dict(), dict())

    if resp_cost_d:
        df_plot, df_plot_change, chart_cols, outliers = (
            resp_cost_d['data'], resp_cost_d['change_data'],
            resp_cost_d['chart_cols'], resp_cost_d['outlier_str'])
    else:
        df_plot, df_plot_change, chart_cols, outliers = (
            pd.DataFrame(), pd.DataFrame(), dict(), dict())

    inst_cost_response = prompt_inst[
        prompt_inst['intentionName'] == "ROI-Response-Cost Per Trend"]['instruction'].values[0]

    time_label = list(viz_utils.get_time_align_dict([times[-1]]).keys())[0]

    for agg_data, data, outlier_str_agg, outlier_str, header, metric_cols, _bar_f, mode in [
        (df_plot_agg, df_plot, outliers_agg.get('value', ''), outliers.get('value', ''),
         "ROI, Response, Cost & Activity", chart_cols['metric'], _roi_resp_cost_bar, 'value'),
        (df_plot_change_agg, df_plot_change, outliers_agg.get('change', ''), outliers.get('change', ''),
         "Performance Change", chart_cols['metric_change'], _roi_resp_cost_change_bar, 'change'),
    ]:
        if data.empty and agg_data.empty:
            continue

        left_upper = right_upper = -1
        for _d in [agg_data, data]:
            if not _d.empty and len(_d) > 2:
                left_upper = max(left_upper, _d[_d['metric_label'].isin(['ROI'])]['Value'].max())
                right_upper = max(right_upper, _d[_d['metric_label'].isin(['Response', 'Cost'])]['Value'].max())

        if not agg_data.empty:
            _, d = _bar_f(
                agg_data, time_label, y_axis_upper=[left_upper, right_upper], mode='all', roi_type=roi_type)
            chart_configs[f"agg_{header}"] = {
                "query": f"You are given a dataset about {header}.",
                "instruction": inst_cost_response,
                "data": d.to_json(orient="records"),
            }

        if not data.empty:
            if not is_specific:
                _, d = _bar_f(
                    data, time_label, y_axis_upper=[left_upper, right_upper], mode='layer', roi_type=roi_type)
                chart_configs[f"detail_{header}"] = {
                    "query": f"You are given a dataset about {header}.",
                    "instruction": inst_cost_response,
                    "data": d.to_json(orient="records"),
                }
            else:
                group_col = 'group'
                groups = data['group'].unique()
                for i, grp in enumerate(groups):
                    df_plot_grp = data[data[group_col] == grp]
                    _, d = _bar_f(
                        df_plot_grp, time_label,
                        title=f' - {grp}' if grp else '',
                        y_axis_upper=[left_upper, right_upper], mode='all')
                    chart_configs[f"group_{grp}_{header}"] = {
                        "query": f"You are given a dataset about {header}.",
                        "instruction": inst_cost_response,
                        "data": d.to_json(orient="records"),
                    }
                # tactics_idx = data.columns.get_loc('Tactics')
                # expander_data = data.pivot(
                #     index=list(data.columns[:tactics_idx + 1]) + [group_col],
                #     columns='Metric', values='Value').reset_index()
                # chart_configs[f"agg_group_detail_{header}_{mode}"] = {
                #     "query": f"You are given a dataset about {header}.",
                #     "instruction": inst_cost_response,
                #     "data": expander_data.to_json(orient="records"),
                # }

    return chart_configs


def build_roi_response_cost_chart_configs_for_insights(resp_cost_d, resp_cost_d_agg, times, is_specific,
                                                       ag_mapping, prompt_inst, roi_type="roi",
                                                       client_code=None, model_group_id=None):
    """
    build chart_configs dict for the ROI response/cost view.
    """
    chart_configs = {}
    if not resp_cost_d_agg and not resp_cost_d:
        return chart_configs

    if resp_cost_d_agg:
        df_plot_agg, df_plot_change_agg, chart_cols, outliers_agg = (
            resp_cost_d_agg['data'], resp_cost_d_agg['change_data'],
            resp_cost_d_agg['chart_cols'], resp_cost_d_agg['outlier_str'])
    else:
        df_plot_agg, df_plot_change_agg, chart_cols, outliers_agg = (
            pd.DataFrame(), pd.DataFrame(), dict(), dict())

    if resp_cost_d:
        df_plot, df_plot_change, chart_cols, outliers = (
            resp_cost_d['data'], resp_cost_d['change_data'],
            resp_cost_d['chart_cols'], resp_cost_d['outlier_str'])
    else:
        df_plot, df_plot_change, chart_cols, outliers = (
            pd.DataFrame(), pd.DataFrame(), dict(), dict())

    inst_cost_response = prompt_inst[
        prompt_inst['intentionName'] == "ROI-Response-Cost Per Trend"]['instruction'].values[0]

    time_label = list(viz_utils.get_time_align_dict([times[-1]]).keys())[0]

    for agg_data, data, outlier_str_agg, outlier_str, header, metric_cols, _bar_f, mode in [
        (df_plot_agg, df_plot, outliers_agg.get('value', ''), outliers.get('value', ''),
         "ROI, Response, Cost & Activity", chart_cols['metric'], _roi_resp_cost_bar_for_insights, 'value'),
        (df_plot_change_agg, df_plot_change, outliers_agg.get('change', ''), outliers.get('change', ''),
         "Performance Change", chart_cols['metric_change'], _roi_resp_cost_change_bar_for_insights, 'change'),
    ]:
        if data.empty and agg_data.empty:
            continue

        left_upper = right_upper = -1
        for _d in [agg_data, data]:
            if not _d.empty and len(_d) > 2:
                left_upper = max(left_upper, _d[_d['metric_label'].isin(['ROI'])]['Value'].max())
                right_upper = max(right_upper, _d[_d['metric_label'].isin(['Response', 'Cost'])]['Value'].max())

        if not agg_data.empty:
            d = _bar_f(
                agg_data, time_label, y_axis_upper=[left_upper, right_upper], mode='all', roi_type=roi_type,
                client_code=client_code, model_group_id=model_group_id)
            chart_configs[f"agg_{header}"] = {
                "query": f"You are given a dataset about {header}.",
                "instruction": inst_cost_response,
                "data": d.to_json(orient="records"),
            }

        if not data.empty:
            if not is_specific:
                d = _bar_f(
                    data, time_label, y_axis_upper=[left_upper, right_upper], mode='layer',
                    client_code=client_code, model_group_id=model_group_id)
                chart_configs[f"detail_{header}"] = {
                    "query": f"You are given a dataset about {header}.",
                    "instruction": inst_cost_response,
                    "data": d.to_json(orient="records"),
                }
            else:
                group_col = 'group'
                groups = data['group'].unique()
                for i, grp in enumerate(groups):
                    df_plot_grp = data[data[group_col] == grp]
                    d = _bar_f(
                        df_plot_grp, time_label,
                        title=f' - {grp}' if grp else '',
                        y_axis_upper=[left_upper, right_upper], mode='all',
                        client_code=client_code, model_group_id=model_group_id)
                    chart_configs[f"group_{grp}_{header}"] = {
                        "query": f"You are given a dataset about {header}.",
                        "instruction": inst_cost_response,
                        "data": d.to_json(orient="records"),
                    }
                # tactics_idx = data.columns.get_loc('Tactics')
                # expander_data = data.pivot(
                #     index=list(data.columns[:tactics_idx + 1]) + [group_col],
                #     columns='Metric', values='Value').reset_index()
                # chart_configs[f"agg_group_detail_{header}_{mode}"] = {
                #     "query": f"You are given a dataset about {header}.",
                #     "instruction": inst_cost_response,
                #     "data": expander_data.to_json(orient="records"),
                # }

    return chart_configs


def render_roi_response_cost_charts(resp_cost_d, resp_cost_d_agg, times, is_specific, insights):
    """Pure Streamlit rendering of the ROI response/cost view.
    Accepts pre-computed insights dict. ZERO LLM calls.
    """
    if not resp_cost_d_agg and not resp_cost_d:
        return

    if resp_cost_d_agg:
        df_plot_agg, df_plot_change_agg, chart_cols, outliers_agg = (
            resp_cost_d_agg['data'], resp_cost_d_agg['change_data'],
            resp_cost_d_agg['chart_cols'], resp_cost_d_agg['outlier_str'])
    else:
        df_plot_agg, df_plot_change_agg, chart_cols, outliers_agg = (
            pd.DataFrame(), pd.DataFrame(), dict(), dict())

    if resp_cost_d:
        df_plot, df_plot_change, chart_cols, outliers = (
            resp_cost_d['data'], resp_cost_d['change_data'],
            resp_cost_d['chart_cols'], resp_cost_d['outlier_str'])
    else:
        df_plot, df_plot_change, chart_cols, outliers = (
            pd.DataFrame(), pd.DataFrame(), dict(), dict())

    col1, col2 = st.columns(2)
    time_label = list(viz_utils.get_time_align_dict([times[-1]]).keys())[0]

    for col, agg_data, data, outlier_str_agg, outlier_str, header, metric_cols, _bar_f, mode in [
        (col1, df_plot_agg, df_plot, outliers_agg.get('value', ''), outliers.get('value', ''),
         "ROI, Response & Cost", chart_cols['metric'], _roi_resp_cost_bar, 'value'),
        (col2, df_plot_change_agg, df_plot_change, outliers_agg.get('change', ''), outliers.get('change', ''),
         "Performance Change", chart_cols['metric_change'], _roi_resp_cost_change_bar, 'change'),
    ]:
        with col:
            if data.empty and agg_data.empty:
                continue

            left_upper = right_upper = -1
            for _d in [agg_data, data]:
                if not _d.empty and len(_d) > 2:
                    left_upper = max(left_upper, _d[_d['metric_label'].isin(['ROI'])]['Value'].max())
                    right_upper = max(right_upper, _d[_d['metric_label'].isin(['Response', 'Cost'])]['Value'].max())

            with _section(header):
                if not agg_data.empty:
                    st.markdown("#### Aggregation View")
                    if agg_data['Tactics'].nunique() > 2:
                        chart_ls_agg, expander_data = _bar_f(
                            agg_data, time_label, y_axis_upper=[left_upper, right_upper], mode='all')
                        with st.expander("Show data table"):
                            st.dataframe(expander_data, use_container_width=True)
                        st.altair_chart(chart_ls_agg[0], use_container_width=True)
                        st.markdown(outlier_str_agg)
                        st.session_state.insight[f"agg_{header}"] = insights.get(f"agg_{header}")
                        saved = st.session_state.insight.get(f"agg_{header}")
                        if saved:
                            render_chart_insight_box(f"{header} Insight", saved)
                    else:
                        table_display = agg_data.pivot(index="Tactics", columns="Metric", values="Value").reset_index()[['Tactics'] + metric_cols]
                        table_display = readout_utils.add_unit_sign(table_display, 1, period_suffixes=["% Change"])
                        table_element(
                            table_display,
                            "", f"cost_response_agg_{mode}")

                if not data.empty:
                    st.markdown("#### Detail View")

                    if data['Tactics'].nunique() > 2:
                        if not is_specific:
                            chart_ls, expander_data = _bar_f(
                                data, time_label, y_axis_upper=[left_upper, right_upper], mode='layer')
                            with st.expander("Show data table"):
                                st.dataframe(expander_data, use_container_width=True)
                            for chart in chart_ls:
                                st.altair_chart(chart, use_container_width=True)
                            st.session_state.insight[f"detail_{header}"] = insights.get(f"detail_{header}")
                            saved = st.session_state.insight.get(f"detail_{header}")
                            if saved:
                                render_chart_insight_box(f"{header} Insight", saved)
                        else:
                            group_col = 'group'
                            groups = data['group'].unique()
                            large_groups = data.groupby(group_col)['Tactics'].count().min() > 4
                            cols = st.columns([1, 1]) if len(groups) > 1 and not large_groups else st.columns([1])
                            for i, grp in enumerate(groups):
                                df_plot_grp = data[data[group_col] == grp]
                                with cols[i % 2] if len(groups) > 1 and not large_groups else cols[0]:
                                    chart_ls_grp, expander_data_grp = _bar_f(
                                        df_plot_grp, time_label,
                                        title=f' - {grp}' if grp else '',
                                        y_axis_upper=[left_upper, right_upper], mode='all')
                                    with st.expander("Show data table"):
                                        st.dataframe(expander_data_grp, use_container_width=True)
                                    st.altair_chart(chart_ls_grp[0], use_container_width=True)
                                    st.session_state.insight[f"group_{grp}_{header}"] = insights.get(f"group_{grp}_{header}")
                                    saved = st.session_state.insight.get(f"group_{grp}_{header}")
                                    if saved:
                                        render_chart_insight_box(f"{header} Insight - {grp}", saved)
                        st.markdown(outlier_str)
                    else:
                        table_display = \
                        data.pivot(index="Tactics", columns="Metric", values="Value").reset_index()[
                            ['Tactics'] + metric_cols]
                        table_display = readout_utils.add_unit_sign(table_display, 1,
                                                                    period_suffixes=["% Change"])
                        table_element(
                            table_display,
                            "", f"cost_response_detail_{mode}")


def get_roi_response_cost_plot_new(resp_cost_d, resp_cost_d_agg, times, is_specific,
                                   ag_mapping, prompt_inst, llm_vendor):
    """Orchestrator: ROI response/cost snapshot.

    1. build_roi_response_cost_chart_configs(...)
    2. generate_chart_insights_parallel_cached(...)
    3. render_roi_response_cost_charts(...)
    Returns (chart_configs, insights).
    """
    if not resp_cost_d_agg and not resp_cost_d:
        return

    # --- Build configs ---
    chart_configs = build_roi_response_cost_chart_configs(
        resp_cost_d, resp_cost_d_agg, times, is_specific, ag_mapping, prompt_inst)

    # --- Generate insights ---
    with st.spinner("Your visuals and insights are on the way..."):
        start = time.time()
        chart_configs_by_group = {k: v for k, v in chart_configs.items() if not k.startswith('agg_group')}
        insights = generate_chart_insights_parallel_cached(chart_configs_by_group, llm_vendor)
        elapsed = time.time() - start

    # --- Render ---
    render_roi_response_cost_charts(resp_cost_d, resp_cost_d_agg, times, is_specific, insights)

    return chart_configs, insights


# ---------------------------------------------------------------------------
# Multiple ROI stacked view
# ---------------------------------------------------------------------------

def get_multiple_roi_plot_new(multi_roi_data, multi_roi_data_agg, sorted_time, is_specific, prompt_inst):
    level_col = [c for c in multi_roi_data['data'].columns if 'roi' in c and 'level' in c][0]
    group_col = level_col if not is_specific else 'group'
    short_cols = multi_roi_data['chart_cols']['roi']
    long_cols = multi_roi_data['chart_cols']['long_term_roi']
    total_cols = multi_roi_data['chart_cols']['total_roi']
    sorted_time = [t for t in sorted_time if any(t in col for col in long_cols)]
    select_time = viz_utils.get_time_align_dict(sorted_time)
    time_label_dict = {v: k for k, v in select_time.items()}

    for d, table in [(multi_roi_data_agg, "Aggregation"), (multi_roi_data, "Detail")]:
        if not d:
            continue

        sel_time = sorted_time[0]
        st.markdown(f"#### {table} View")

        data = d['data'].copy()
        outlier_str = d['outlier_str']

        short_col = [x for x in short_cols if sel_time in x][0]
        long_col = [x for x in long_cols if sel_time in x][0]
        total_col = [x for x in total_cols if sel_time in x][0]

        x_scale = data[data['Metric'] == total_col]['Value'].max()
        data[group_col] = 'All'
        groups = data[group_col].unique()
        for i, grp in enumerate(groups):
            if grp and grp not in ['high outlier', 'low outlier']:
                data_grp = data[(data[group_col] == grp) & (data['Value'] > 0)]
                chart = multiple_roi_stacked_bar(
                    data_grp, short_col, long_col, total_col,
                    time_label_dict[sel_time], x_scale,
                    title=f' - {grp}' if grp and grp != 'All' else '')
                st.altair_chart(chart, use_container_width=True)

        st.markdown(outlier_str)
    return


# ---------------------------------------------------------------------------
# Nivo helpers
# ---------------------------------------------------------------------------

def get_nivo_grouped_bar_data(
    df: pd.DataFrame,
    tactic_col: str,
    bar_col1: str,
    bar_col2: str,
    bar_name1: str,
    bar_name2: str,
    title: str,
    mode: str = "value",
    angle: int = -40,
):
    expander_data = df[[tactic_col, bar_col1, bar_col2]].copy()
    _, table_str, _, _ = readout_utils.table_to_text(pd.DataFrame(), expander_data, take_head=False)
    instruct = f"""Provide a clear, concise, business-style insight summary.
Data: {table_str}"""

    sort_order = (
        df[[tactic_col, bar_col1]]
        .sort_values(bar_col1, ascending=False)[tactic_col]
        .tolist()
    )

    df_melt = df.melt(
        id_vars=[tactic_col],
        value_vars=[bar_col1, bar_col2],
        var_name="Metric",
        value_name="Value",
    )

    d_min = min(0, df_melt["Value"].min())
    d_max = max(0, df_melt["Value"].max())

    if mode == "change":
        y_min = d_min
        y_max = d_max
    else:
        y_min = 0
        y_max = "auto"

    df_sorted = df.set_index(tactic_col).loc[sort_order].reset_index()

    nivo_data = []
    for _, row in df_sorted.iterrows():
        nivo_data.append(
            {
                tactic_col: str(row[tactic_col]),
                bar_col1: float(row[bar_col1]) if pd.notna(row[bar_col1]) else 0.0,
                bar_col2: float(row[bar_col2]) if pd.notna(row[bar_col2]) else 0.0,
            }
        )

    config = {
        "title": title,
        "subtitle": f"Blue bar = {bar_name1}, Yellow bar = {bar_name2} (Sorted by {bar_name1} desc)",
        "tactic_col": tactic_col,
        "keys": [bar_col1, bar_col2],
        "bar_labels": [bar_name1, bar_name2],
        "angle": angle,
        "y_min": y_min,
        "y_max": y_max,
    }

    return expander_data, instruct, nivo_data, config


def build_roi_snapshot_table(recent_data, roi_value_col, roi_change_col, spend_value_col, spend_change_col):
    change_cols = [roi_change_col, spend_change_col]
    has_change = all(c in recent_data.columns for c in change_cols)

    display_cols = (
        ['Tactics', roi_value_col, roi_change_col, spend_value_col, spend_change_col]
        if has_change else
        ['Tactics', roi_value_col, spend_value_col]
    )

    table_display = recent_data[display_cols].copy()
    table_display[[roi_value_col, spend_value_col]] = table_display[[roi_value_col, spend_value_col]].map(
        lambda x: f"${x}" if pd.notna(x) else x)

    if has_change:
        table_display[change_cols] = table_display[change_cols].map(
            lambda x: f"{x}%" if pd.notna(x) else x)

    return table_display


