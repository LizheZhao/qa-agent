import re
import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import src.model.readout_utils as readout_utils
import viz_utils as viz_utils
from chart.common import (
    render_chart_insight_box, generate_chart_insights_parallel_cached, _section, table_element
)
import time


# ---------------------------------------------------------------------------
# Shared helpers (waterfall / soc)
# ---------------------------------------------------------------------------

def transform_total_time(row):
    if row['period_type'] == 'fiscal year':
        time_label = re.sub(r"^fiscal year\s+(\d{4})$", r"\1", row['time'], flags=re.IGNORECASE)
    elif row['period_type'] == 'fiscal month':
        time_label = re.sub(r"^fiscal month\s+(\d+)\s+(\d{4})$", r"M\1 \2", row['time'], flags=re.IGNORECASE)
    elif row['period_type'] == 'fiscal quarter':
        time_label = re.sub(r"^fiscal quarter\s+(\d+)\s+(\d{4})$", r"Q\1 \2", row['time'], flags=re.IGNORECASE)
    else:
        time_label = row['time']
    return time_label


def get_waterfall(soc_table, periods, sales_ls, data_level_label):
    p1, p2 = periods
    s1_p, s2_p, s1_a, s2_a = sales_ls

    with st.expander("Show data table"):
        expander_data = soc_table.reset_index(drop=True)
        rename_dict = {'soc_pct': 'source of change %', 'soc_volume': 'contribution to change'}
        expander_data = readout_utils.add_unit_sign(expander_data.rename(rename_dict, axis=1), period_suffixes=["% Change"])

        st.dataframe(expander_data, use_container_width=True)

    x = [p1] + [d.capitalize() for d in soc_table['driver'].tolist()] + [p2]
    measures = ['relative'] + ['relative'] * (len(soc_table)) + ['total']
    y = [s1_a] + soc_table['soc_volume'].tolist() + [s2_a]

    text = [f"{v/1e6:,.1f}M" for v in y]
    text[-1] += f"<br>({(s2_a - s1_a)/s1_a * 100:.1f}%)"
    text[0] = "<span style='font-size:15px'><b>" + text[0] + "</b></span>"
    text[-1] = "<span style='font-size:15px'><b>" + text[-1] + "</b></span>"
    hover_templates = []
    hover_templates.append((
        f"%{{x}}<br>"
        f"Total Sales: {s1_a:,.1f}<br>"
        "<extra></extra>"
    ))
    for idx, row in soc_table.iterrows():
        hover_templates.append((
            f"%{{x}}<br>"
            f"Source of Change: {row['soc_volume']:,.1f}<br>"
            f"Source of Change%: {row['soc_pct']:,.1f}%<br>"
            "<extra></extra>"
        ))
    hover_templates.append((
        f"%{{x}}<br>"
        f"Total Sales: {s2_a:,.1f}<br>"
        "<extra></extra>"
    ))
    fig = go.Waterfall(
        name="SOC",
        orientation="v",
        measure=measures,
        x=x,
        y=y,
        text=text,
        textposition="outside",
        hovertemplate=hover_templates,
        connector={"line": {"width": 1}},
    )
    x_array = [str(p1)] + [str(v).capitalize() for v in soc_table['driver']] + [str(p2)]
    return fig, x_array


def soc_fig_update(waterfall, title, category_array):
    fig = go.Figure(waterfall)
    fig.update_layout(
        title=title,
        yaxis_title="Sales",
        waterfallgap=0.2,
        margin=dict(l=40, r=40, t=60, b=40),
    )
    fig.update_layout(
        xaxis=dict(
            type="category",
            categoryorder="array",
            categoryarray=category_array
        )
    )
    return fig


# ---------------------------------------------------------------------------
# Source of Change — build / render / orchestrator
# ---------------------------------------------------------------------------

def build_sourceofchange_chart_configs(
    total_viz_soc, total_act_sales, df_plot, outliers, metric_cols,
    sorted_times, driver_col, br_detail_level, group_mapping, is_specific=False,
):
    """
    Pure data preparation: derives period_start/period_end, computes SOC
    tables for both levels, and returns a chart_configs dict ready for
    generate_chart_insights_parallel_cached.

    Returns
    -------
    chart_configs : dict
    period_start  : str
    period_end    : str
    bd_level_soc  : DataFrame
    soc_table     : DataFrame
    sales_ls_bd   : list
    sales_ls_det  : list
    total_table_str : str
    """
    time_mapping = viz_utils.get_time_align_dict(sorted_times)
    # Use the first (most-recent) mapped period as default
    soc_period = list(time_mapping.values())[0]

    all_time_period = sorted(total_viz_soc['time_norm'].unique())
    period_end = re.search(
        r"(Q[1-4]|H[1-2]|M(?:1[0-2]|[1-9])) \d{4}|\d{4}", soc_period
    ).group()
    period_start = all_time_period[all_time_period.index(period_end) - 1]

    # BD-level SOC
    bd_level_soc, sales_ls_bd, _ = viz_utils.get_soc_data(
        total_viz_soc, total_act_sales, period_start, period_end,
        df_plot, driver_col, data_level='overall'
    )
    expander_data = pd.DataFrame(
        {'Period': [period_start, period_end], 'Sales': [sales_ls_bd[2], sales_ls_bd[3]]}
    )
    _, total_table_str, _, _ = readout_utils.table_to_text(
        pd.DataFrame(), expander_data, take_head=False
    )

    # Agg/detail level SOC
    soc_table, sales_ls_det, _ = viz_utils.get_soc_data(
        total_viz_soc, total_act_sales, period_start, period_end,
        df_plot, driver_col, data_level='agg/detail'
    )

    rename_dict = {'soc_pct': 'source of change %', 'soc_volume': 'contribution to change'}
    soc_table_display = readout_utils.add_unit_sign(soc_table.rename(rename_dict, axis=1), period_suffixes=["% Change"])
    bd_level_soc_display = readout_utils.add_unit_sign(bd_level_soc.rename(rename_dict, axis=1), period_suffixes=["% Change"])

    chart_configs = {}
    if not is_specific:
        # BD-level insight config
        _, table_str_bd, _, _ = readout_utils.table_to_text(
            pd.DataFrame(), bd_level_soc_display, take_head=False
        )
        instruct_bd = (
            f"Provide a clear, concise, business-style insight summary.\n"
            f"Total data: {total_table_str}\n"
            f"Data: {table_str_bd}"
        )
        chart_configs["soc_bd"] = {
            "query": "You are given a dataset about source of change.",
            "instruction": instruct_bd,
            "data": table_str_bd,
        }

    # Detail-level insight config
    _, table_str_det, _, _ = readout_utils.table_to_text(
        pd.DataFrame(), soc_table_display, take_head=False
    )
    instruct_det = (
        f"Provide a clear, concise, business-style insight summary.\n"
        f"Total data: {total_table_str}\n"
        f"Data: {table_str_det}"
    )
    chart_configs["soc_detail"] = {
        "query": "You are given a dataset about source of change.",
        "instruction": instruct_det,
        "data": table_str_det,
    }

    return (
        chart_configs,
        period_start,
        period_end,
        bd_level_soc,
        soc_table,
        sales_ls_bd,
        sales_ls_det,
        total_table_str,
    )


def build_sourceofchange_chart_configs_for_insights(
    total_viz_soc, total_act_sales, df_plot, outliers, metric_cols,
    sorted_times, driver_col, br_detail_level, group_mapping, is_specific=False,
    client_code=None, model_group_id=None,
):
    """
    Pure data preparation: derives period_start/period_end, computes SOC
    tables for both levels, and returns a chart_configs dict ready for
    insights generation

    Returns
    -------
    chart_configs : dict
    period_start  : str
    period_end    : str
    bd_level_soc  : DataFrame
    soc_table     : DataFrame
    sales_ls_bd   : list
    sales_ls_det  : list
    total_table_str : str
    """
    time_mapping = viz_utils.get_time_align_dict(sorted_times)
    # Use the first (most-recent) mapped period as default
    soc_period = list(time_mapping.values())[0]

    all_time_period = sorted(total_viz_soc['time_norm'].unique())
    period_end = re.search(
        r"(Q[1-4]|H[1-2]|M(?:1[0-2]|[1-9])) \d{4}|\d{4}", soc_period
    ).group()
    period_start = all_time_period[all_time_period.index(period_end) - 1]

    # BD-level SOC
    bd_level_soc, sales_ls_bd, _ = viz_utils.get_soc_data(
        total_viz_soc, total_act_sales, period_start, period_end,
        df_plot, driver_col, data_level='overall', charting=False,
    )
    expander_data = pd.DataFrame(
        {'Period': [period_start, period_end], 'Sales': [sales_ls_bd[2], sales_ls_bd[3]]}
    )
    _, total_table_str, _, _ = readout_utils.table_to_text(
        pd.DataFrame(), expander_data, take_head=False
    )

    # Agg/detail level SOC
    soc_table, sales_ls_det, _ = viz_utils.get_soc_data(
        total_viz_soc, total_act_sales, period_start, period_end,
        df_plot, driver_col, data_level='agg/detail', charting=False,
    )

    rename_dict = {'soc_pct': 'source of change %', 'soc_volume': 'contribution to change'}
    soc_table_display = readout_utils.add_unit_sign(soc_table.rename(rename_dict, axis=1), period_suffixes=["% Change"], client_code=client_code, model_group_id=model_group_id)
    bd_level_soc_display = readout_utils.add_unit_sign(bd_level_soc.rename(rename_dict, axis=1), period_suffixes=["% Change"], client_code=client_code, model_group_id=model_group_id)

    chart_configs = {}
    if not is_specific:
        # BD-level insight config
        _, table_str_bd, _, _ = readout_utils.table_to_text(
            pd.DataFrame(), bd_level_soc_display, take_head=False
        )
        instruct_bd = (
            f"Provide a clear, concise, business-style insight summary.\n"
            f"Total data: {total_table_str}\n"
            f"Data: {table_str_bd}"
        )
        chart_configs["soc_bd"] = {
            "query": "You are given a dataset about source of change.",
            "instruction": instruct_bd,
            "data": table_str_bd,
        }

    # Detail-level insight config
    _, table_str_det, _, _ = readout_utils.table_to_text(
        pd.DataFrame(), soc_table_display, take_head=False
    )
    instruct_det = (
        f"Provide a clear, concise, business-style insight summary.\n"
        f"Total data: {total_table_str}\n"
        f"Data: {table_str_det}"
    )
    chart_configs["soc_detail"] = {
        "query": "You are given a dataset about source of change.",
        "instruction": instruct_det,
        "data": table_str_det,
    }

    return (
        chart_configs,
        period_start,
        period_end,
        bd_level_soc,
        soc_table,
        sales_ls_bd,
        sales_ls_det,
        total_table_str,
    )


def render_sourceofchange(
    total_viz_soc, total_act_sales, df_plot, outliers, metric_cols,
    sorted_times, driver_col, br_detail_level, group_mapping, insights
):
    """
    Pure Streamlit rendering.  Accepts pre-computed insights dict from
    generate_chart_insights_parallel_cached.
    """
    time_mapping = viz_utils.get_time_align_dict(sorted_times)
    soc_period = st.selectbox(
        "Select source of change period:", time_mapping.keys(), index=0, key="soc_period"
    )
    soc_period = time_mapping[soc_period]

    all_time_period = sorted(total_viz_soc['time_norm'].unique())
    period_end = re.search(
        r"(Q[1-4]|H[1-2]|M(?:1[0-2]|[1-9])) \d{4}|\d{4}", soc_period
    ).group()
    period_start = all_time_period[all_time_period.index(period_end) - 1]

    # --- BD-level ---
    bd_level_soc, sales_ls_bd, _ = viz_utils.get_soc_data(
        total_viz_soc, total_act_sales, period_start, period_end,
        df_plot, driver_col, data_level='overall'
    )
    expander_data = pd.DataFrame(
        {'Period': [period_start, period_end], 'Sales': [sales_ls_bd[2], sales_ls_bd[3]]}
    )
    # expander_data['Sales'] = expander_data['Sales'].apply(viz_utils.fmt_dollar)
    expander_data = readout_utils.add_unit_sign(expander_data, period_suffixes=['% Change'])
    with st.expander("Show sales table"):
        st.dataframe(expander_data, use_container_width=True)

    fig_bd, x_array = get_waterfall(bd_level_soc, [period_start, period_end], sales_ls_bd, "Business Drivers")
    fig_bd = soc_fig_update(
        fig_bd,
        f"Sales Source of Change by Business Drivers: {period_start} → {period_end}",
        x_array,
    )
    st.plotly_chart(fig_bd)
    if insights and insights.get("soc_bd"):
        render_chart_insight_box("Source of Change — Business Drivers", insights["soc_bd"])

    # --- Agg/detail level ---
    soc_table, sales_ls_det, _ = viz_utils.get_soc_data(
        total_viz_soc, total_act_sales, period_start, period_end,
        df_plot, driver_col, data_level='agg/detail'
    )
    fig, x_array = get_waterfall(
        soc_table, [period_start, period_end], sales_ls_det, br_detail_level + " Drivers"
    )
    fig = soc_fig_update(
        fig,
        f"Sales Source of Change by Business Drivers: {period_start} → {period_end}",
        x_array,
    )
    st.plotly_chart(fig)
    if insights and insights.get("soc_detail"):
        render_chart_insight_box("Source of Change — Detail", insights["soc_detail"])


def get_plot_for_sourceofchange(
    total_viz_soc, total_act_sales, df_plot, outliers, metric_cols,
    sorted_times, driver_col, br_detail_level, group_mapping,
    llm_vendor='deepseek'
):
    """
    Orchestrator: build → generate insights in parallel → render.

    Returns (chart_configs, insights).
    """
    (
        chart_configs,
        period_start, period_end,
        bd_level_soc, soc_table,
        sales_ls_bd, sales_ls_det,
        total_table_str,
    ) = build_sourceofchange_chart_configs(
        total_viz_soc, total_act_sales, df_plot, outliers, metric_cols,
        sorted_times, driver_col, br_detail_level, group_mapping
    )

    with st.spinner("Your visuals and insights are on the way..."):
        start = time.time()
        insights = generate_chart_insights_parallel_cached(chart_configs, llm_vendor)
        elapsed = time.time() - start
    st.write(f"Insight generation time: {elapsed:.2f} seconds")

    render_sourceofchange(
        total_viz_soc, total_act_sales, df_plot, outliers, metric_cols,
        sorted_times, driver_col, br_detail_level, group_mapping, insights
    )

    return chart_configs, insights


# ---------------------------------------------------------------------------
# Contribution snapshot helpers (private)
# ---------------------------------------------------------------------------

def _render_pie_table_str(plot_df_tmp):
    """Table text for the contribution pie data. '' when empty. No st.* calls."""
    if plot_df_tmp is None or plot_df_tmp.empty:
        return ""
    _, total_str, _, _ = readout_utils.table_to_text(pd.DataFrame(), plot_df_tmp, take_head=False)
    return total_str


def _render_cont_bars_table_str(bars_df_tmp):
    """Table text for the contribution-bars data. '' when empty/insufficient. No st.* calls."""
    if bars_df_tmp is None or len(bars_df_tmp) < 3:
        return ""
    _, cont_str, _, _ = readout_utils.table_to_text(pd.DataFrame(), bars_df_tmp, take_head=False)
    return cont_str


def _render_sales_bars_table_str(sales_bars_df_tmp):
    """Table text for the sales-bars data. '' when empty/insufficient. No st.* calls."""
    if sales_bars_df_tmp is None or len(sales_bars_df_tmp) < 3:
        return ""
    _, sales_str, _, _ = readout_utils.table_to_text(pd.DataFrame(), sales_bars_df_tmp, take_head=False)
    return sales_str


def _render_fair_share_scatter_table_str(plot_df):
    """Table text for the fair-share scatter data. '' when empty/insufficient. No st.* calls."""
    if plot_df is None or len(plot_df) < 3:
        return ""
    _, table_str, _, _ = readout_utils.table_to_text(pd.DataFrame(), plot_df, take_head=False)
    return table_str


def _contribution_trend_table_str(df_level):
    """Table text for one contribution-trend level frame. '' when empty. No st.* calls."""
    if df_level is None or df_level.empty:
        return ""
    _, table_str, _, _ = readout_utils.table_to_text(pd.DataFrame(), df_level, take_head=False)
    return table_str


def _render_pie_data(
    client_code, model_group_id, sorted_times, selected_levels, detail_data,
    sel_time, sel_time_dict, metric_cols, outliers, readout, match, is_specific,
    driver_col='Business Driver'
):
    bd_data = viz_utils.get_contribution_bd_data(
        client_code, model_group_id, sorted_times, selected_levels, readout=readout,
    )

    plot_df, _, _, _, _, _, _ = viz_utils.contribution_pie_bar_data(
        bd_data, detail_data, sel_time, metric_cols, outliers, readout, match, is_specific,
        driver_col=driver_col
    )
    if plot_df.empty:
        return None, None

    plot_df_tmp = plot_df.copy()
    plot_df_tmp = plot_df_tmp.drop(['fraction', 'label'], axis=1)
    plot_df_tmp['value'] = plot_df_tmp['value'].apply(viz_utils.fmt_pct)

    base = alt.Chart(plot_df).encode(
        theta=alt.Theta("fraction:Q", stack=True),
        color=alt.Color("driver:N", legend=alt.Legend(title="Driver")),
        tooltip=[
            alt.Tooltip("driver:N", title="Driver"),
            alt.Tooltip("value:Q", title="Contribution", format=".1f"),
        ],
    )
    pie = base.mark_arc(innerRadius=40, outerRadius=80)
    labels = base.mark_text(radius=100, size=12, fill='black').encode(
        theta=alt.Theta("fraction:Q", stack=True),
        text=alt.Text("label:N"),
    )
    chart = (pie + labels).properties(
        height=200, width=150,
        padding={"left": 10, "right": 30, "top": 10, "bottom": 10},
    ).configure(background="transparent").configure_view(fill="transparent")
    return chart, plot_df_tmp


def _render_pie_data_for_insights(
    client_code, model_group_id, sorted_times, selected_levels, detail_data,
    sel_time, sel_time_dict, metric_cols, outliers, readout, match, is_specific,
    driver_col='Business Driver'
):
    bd_data = viz_utils.get_contribution_bd_data(
        client_code, model_group_id, sorted_times, selected_levels, readout=readout,
    )

    plot_df, _, _, _, _, _, _ = viz_utils.contribution_pie_bar_data(
        bd_data, detail_data, sel_time, metric_cols, outliers, readout, match, is_specific,
        driver_col=driver_col
    )
    if plot_df.empty:
        return None, None

    plot_df_tmp = plot_df.copy()
    plot_df_tmp = plot_df_tmp.drop(['fraction', 'label'], axis=1)
    plot_df_tmp['value'] = plot_df_tmp['value'].apply(viz_utils.fmt_pct)
    return plot_df_tmp


def _render_pie(
    client_code, model_group_id, sorted_times, selected_levels, detail_data,
    sel_time, sel_time_dict, metric_cols, outliers, readout, match, is_specific,
    driver_col='Business Driver'
):
    chart, plot_df_tmp = _render_pie_data(
        client_code, model_group_id, sorted_times, selected_levels, detail_data,
        sel_time, sel_time_dict, metric_cols, outliers, readout, match, is_specific,
        driver_col=driver_col,
    )
    if chart:
        text = (
            f"Contribution by Business Driver — "
            f"{next((k for k, v in sel_time_dict.items() if v == sel_time), None)}"
        )
        st.markdown(
            f"<h3 style='font-size:{16}px; font-weight:600; margin-bottom:6px;'>{text}</h3>",
            unsafe_allow_html=True,
        )
        if plot_df_tmp is not None and not plot_df_tmp.empty:
            with st.expander("Show data table"):
                st.dataframe(plot_df_tmp, use_container_width=True)
        st.altair_chart(chart, use_container_width=True)


def _render_cont_bars_data(
    client_code, model_group_id, sorted_times, selected_levels, detail_data,
    sel_time_l, sel_time_r, sel_time_dict, metric_cols, outliers, readout, match, is_specific,
    driver_col='Business Driver'
):
    bd_data = viz_utils.get_contribution_bd_data(
        client_code, model_group_id, sorted_times, selected_levels, readout=readout,
    )
    _, bars_df_l, _, outlier_str, _, _, _ = viz_utils.contribution_pie_bar_data(
        bd_data, detail_data, sel_time_l, metric_cols, outliers, readout, match, is_specific,
        driver_col=driver_col,
    )
    _, bars_df_r, _, _, _, _, _ = viz_utils.contribution_pie_bar_data(
        bd_data, detail_data, sel_time_r, metric_cols, outliers, readout, match, is_specific,
        driver_col=driver_col,
    )
    if sel_time_l != sel_time_r:
        bars_df_l["Period"] = "Current"
        bars_df_r["Period"] = "Previous"
        bars_df = pd.concat([bars_df_l, bars_df_r])
    else:
        bars_df = bars_df_l

    if bars_df.empty:
        return outlier_str, None, None

    bars = (
        alt.Chart(bars_df)
        .mark_bar()
        .encode(
            y=alt.Y("group:N", title=None, sort="-x"),
            x=alt.X("Value:Q", title="Contribution (%)", stack=None),
            yOffset=alt.YOffset("Period:N"),
            color=alt.Color("Period:N", title="Period"),
            tooltip=[
                alt.Tooltip("group:N", title="Driver"),
                alt.Tooltip("Period:N", title="Period"),
                alt.Tooltip("Value:Q", title="Contribution (%)", format=".1f"),
            ],
        )
        .properties(
            height=max(600, 25 * bars_df["group"].nunique()),
            padding={"left": 10, "right": 30, "top": 10, "bottom": 10},
        )
    )
    bars_df_tmp = bars_df.copy()
    if (bars_df_tmp['group'] == bars_df_tmp[driver_col]).all():
        bars_df_tmp = bars_df_tmp.drop(columns=['group'])

    bars_df_tmp = bars_df_tmp.drop('Value', axis=1)

    # bars_df_tmp['Value'] = bars_df_tmp['Value'].astype(str) + '%'

    if len(bars_df) >= 3:
        bars = bars.configure(background="transparent").configure_view(fill="transparent")
    else:
        bars = None
        # bars_df_tmp = None
    return outlier_str, bars, bars_df_tmp


def _render_cont_bars_data_for_insights(
    client_code, model_group_id, sorted_times, selected_levels, detail_data,
    sel_time_l, sel_time_r, sel_time_dict, metric_cols, outliers, readout, match, is_specific,
    driver_col='Business Driver'
):
    bd_data = viz_utils.get_contribution_bd_data(
        client_code, model_group_id, sorted_times, selected_levels, readout=readout,
    )
    _, bars_df_l, _, outlier_str, _, _, _ = viz_utils.contribution_pie_bar_data(
        bd_data, detail_data, sel_time_l, metric_cols, outliers, readout, match, is_specific,
        driver_col=driver_col,
    )
    _, bars_df_r, _, _, _, _, _ = viz_utils.contribution_pie_bar_data(
        bd_data, detail_data, sel_time_r, metric_cols, outliers, readout, match, is_specific,
        driver_col=driver_col,
    )
    if sel_time_l != sel_time_r:
        bars_df_l["Period"] = "Current"
        bars_df_r["Period"] = "Previous"
        bars_df = pd.concat([bars_df_l, bars_df_r])
    else:
        bars_df = bars_df_l

    if bars_df.empty:
        return None

    bars_df_tmp = bars_df.copy()
    if (bars_df_tmp['group'] == bars_df_tmp[driver_col]).all():
        bars_df_tmp = bars_df_tmp.drop(columns=['group'])
    bars_df_tmp = bars_df_tmp.drop('Value', axis=1)
    return bars_df_tmp


def _render_cont_bars(
    client_code, model_group_id, sorted_times, selected_levels, detail_data,
    sel_time_l, sel_time_r, sel_time_dict, metric_cols, outliers, readout, match, is_specific,
    driver_col='Business Driver'
):
    outlier_str, bars, bars_df_tmp = _render_cont_bars_data(
        client_code, model_group_id, sorted_times, selected_levels, detail_data,
        sel_time_l, sel_time_r, sel_time_dict, metric_cols, outliers, readout,
        match, is_specific, driver_col,
    )
    if bars:
        text = (
            f"Contribution % by Marketing driver — "
            f"{next((k for k, v in sel_time_dict.items() if v == sel_time_l), None)} "
            f"vs {next((k for k, v in sel_time_dict.items() if v == sel_time_r), None)}"
        )
        st.markdown(
            f"<h3 style='font-size:{16}px; font-weight:600; margin-bottom:6px;'>{text}</h3>",
            unsafe_allow_html=True,
        )
        # bars_df_tmp['Value'] = bars_df_tmp['Value'].astype(str) + "%"
        if bars_df_tmp is not None and not bars_df_tmp.empty:
            with st.expander("Show data table"):
                st.dataframe(bars_df_tmp, use_container_width=True)
        bars = bars.configure(background="transparent").configure_view(fill="transparent")
        st.altair_chart(bars, use_container_width=True)
        st.markdown(outlier_str)


def _render_sales_bars_data(
    client_code, model_group_id, sorted_times, selected_levels, detail_data,
    sel_time_l, sel_time_r, sel_time_dict, metric_cols, outliers, readout, match, is_specific,
    driver_col='Business Driver'
):
    bd_data = viz_utils.get_contribution_bd_data(
        client_code, model_group_id, sorted_times, selected_levels, readout=readout,
    )
    _, _, sales_bars_df_l, _, _, _, _ = viz_utils.contribution_pie_bar_data(
        bd_data, detail_data, sel_time_l, metric_cols, outliers, readout, match, is_specific,
        driver_col=driver_col,
    )
    _, _, sales_bars_df_r, _, _, _, _ = viz_utils.contribution_pie_bar_data(
        bd_data, detail_data, sel_time_r, metric_cols, outliers, readout, match, is_specific,
        driver_col=driver_col,
    )
    if sel_time_l != sel_time_r:
        sales_bars_df_l["Period"] = "Current"
        sales_bars_df_r["Period"] = "Previous"
        sales_bars_df = pd.concat([sales_bars_df_l, sales_bars_df_r])
    else:
        sales_bars_df = sales_bars_df_l

    if sales_bars_df.empty:
        return None, None

    sales_bars_df["Value_MM"] = (sales_bars_df["Value"] / 1_000_000).round(2)
    sales_bars = (
        alt.Chart(sales_bars_df)
        .mark_bar()
        .encode(
            y=alt.Y("group:N", title=None, sort="-x"),
            x=alt.X("Value_MM:Q", title="Sales (MM)", stack=None),
            yOffset=alt.YOffset("Period:N"),
            color=alt.Color("Period:N", title="Period"),
            tooltip=[
                alt.Tooltip("group:N", title="Driver"),
                alt.Tooltip("Period:N", title="Period"),
                alt.Tooltip("Value_MM:Q", title="Sales (MM)", format=".2f"),
            ],
        )
        .properties(
            height=max(600, 25 * sales_bars_df["group"].nunique()),
            padding={"left": 10, "right": 30, "top": 10, "bottom": 10},
        )
    )
    sales_bars_df_tmp = sales_bars_df.copy()
    if (sales_bars_df_tmp['group'] == sales_bars_df_tmp[driver_col]).all():
        sales_bars_df_tmp = sales_bars_df_tmp.drop(columns=['group'])
    sales_bars_df_tmp = sales_bars_df_tmp.drop('Value', axis=1)

    if len(sales_bars_df) >= 3:
        sales_bars = sales_bars.configure(background="transparent").configure_view(fill="transparent")
    else:
        sales_bars = None
        # sales_bars_df_tmp = None

    return sales_bars, sales_bars_df_tmp


def _render_sales_bars_data_for_insights(
    client_code, model_group_id, sorted_times, selected_levels, detail_data,
    sel_time_l, sel_time_r, sel_time_dict, metric_cols, outliers, readout, match, is_specific,
    driver_col='Business Driver'
):
    bd_data = viz_utils.get_contribution_bd_data(
        client_code, model_group_id, sorted_times, selected_levels, readout=readout,
    )
    _, _, sales_bars_df_l, _, _, _, _ = viz_utils.contribution_pie_bar_data(
        bd_data, detail_data, sel_time_l, metric_cols, outliers, readout, match, is_specific,
        driver_col=driver_col,
    )
    _, _, sales_bars_df_r, _, _, _, _ = viz_utils.contribution_pie_bar_data(
        bd_data, detail_data, sel_time_r, metric_cols, outliers, readout, match, is_specific,
        driver_col=driver_col,
    )
    if sel_time_l != sel_time_r:
        sales_bars_df_l["Period"] = "Current"
        sales_bars_df_r["Period"] = "Previous"
        sales_bars_df = pd.concat([sales_bars_df_l, sales_bars_df_r])
    else:
        sales_bars_df = sales_bars_df_l

    if sales_bars_df.empty:
        return None, None

    sales_bars_df["Value_MM"] = (sales_bars_df["Value"] / 1_000_000).round(2)

    sales_bars_df_tmp = sales_bars_df.copy()
    if (sales_bars_df_tmp['group'] == sales_bars_df_tmp[driver_col]).all():
        sales_bars_df_tmp = sales_bars_df_tmp.drop(columns=['group'])
    sales_bars_df_tmp = sales_bars_df_tmp.drop('Value', axis=1)

    return sales_bars_df_tmp


def _sales_trend_data_for_insights(df, driver_col):
    """Melt sales value columns to long format (Driver/Time/Value/TimeLabel).

    Mirrors viz_utils.contribution_trend_data (trend mode): keep sales value columns
    (no '%', no 'share'), reshape to long, and attach display time labels.
    """
    df = viz_utils.rename_month_names(df)
    time_pat = r"(Q[1-4]|H[1-2]|M(?:1[0-2]|[1-9])) \d{4}|\d{4}"
    metric_cols = [
        c for c in df.columns
        if "sales" in c.lower() and "%" not in c and "share" not in c.lower() and re.search(time_pat, c)
    ]
    metric_change_cols = [
        c for c in df.columns
        if "sales" in c.lower() and "%" in c and "share" not in c.lower() and re.search(time_pat, c)
    ]
    if not metric_cols + metric_change_cols:
        return pd.DataFrame()

    df_plot = viz_utils.get_data_for_plot(df, metric_cols+metric_change_cols, driver_col=driver_col)
    df_plot["Time"] = df_plot["Time"].astype(str)
    sorted_times = sorted(
        df_plot["Time"].unique(),
        key=lambda x: (int(x.split()[1]), int(x.split()[0][1:])) if len(x.split()) > 1 else (x, 0),
    )
    df_plot = viz_utils._add_time_labels(df_plot, sorted_times)
    return df_plot


def _render_sales_bars(
    client_code, model_group_id, sorted_times, selected_levels, detail_data,
    sel_time_l, sel_time_r, sel_time_dict, metric_cols, outliers, readout, match, is_specific,
    driver_col='Business Driver'
):
    sales_bars, sales_bars_df_tmp = _render_sales_bars_data(
        client_code, model_group_id, sorted_times, selected_levels, detail_data,
        sel_time_l, sel_time_r, sel_time_dict, metric_cols, outliers, readout,
        match, is_specific, driver_col,
    )
    if sales_bars:
        text = (
            f"Sales Amount Contribution by Marketing driver — "
            f"{next((k for k, v in sel_time_dict.items() if v == sel_time_l), None)} "
            f"vs {next((k for k, v in sel_time_dict.items() if v == sel_time_r), None)}"
        )
        st.markdown(
            f"<h3 style='font-size:{16}px; font-weight:600; margin-bottom:6px;'>{text}</h3>",
            unsafe_allow_html=True,
        )
        if sales_bars_df_tmp is not None and not sales_bars_df_tmp.empty:
            with st.expander("Show data table"):
                st.dataframe(sales_bars_df_tmp, use_container_width=True)
        if len(sales_bars_df_tmp) >= 3:
            sales_bars = sales_bars.configure(background="transparent").configure_view(fill="transparent")
            st.altair_chart(sales_bars, use_container_width=True)


def _render_fair_share_scatter_data(
    contribution_share_data, sel_time_l, sel_time_r, m_cols, outliers,
    driver_col="Business Driver"
):
    if contribution_share_data.empty:
        return None, None

    plot_df_l, line_df, max_axis, _ = viz_utils.contribution_fair_share_data(
        contribution_share_data, sel_time_l, m_cols, outliers, driver_col=driver_col
    )
    plot_df_r, _, _, _ = viz_utils.contribution_fair_share_data(
        contribution_share_data, sel_time_r, m_cols, outliers, driver_col=driver_col
    )
    if sel_time_l != sel_time_r:
        plot_df_l["Period"] = "Current"
        plot_df_r["Period"] = "Previous"
        plot_df = pd.concat([plot_df_l, plot_df_r], ignore_index=True)
    else:
        plot_df = plot_df_l.copy()
        plot_df["Period"] = sel_time_l

    fair_line = (
        alt.Chart(line_df)
        .mark_line(strokeDash=[6, 6])
        .encode(x="x:Q", y="y:Q")
    )
    labels_charts = []
    if "Period" in plot_df.columns:
        for period in plot_df["Period"].unique():
            subset = plot_df[plot_df["Period"] == period]
            top3 = (
                subset[subset["Contribution Share"] > subset["Spend Share"]]
                .sort_values("delta", ascending=False)
                .head(3)
            )
            if not top3.empty:
                labels_charts.append(
                    alt.Chart(top3)
                    .mark_text(align="left", dx=5, dy=-5, fontSize=12, fontWeight="bold")
                    .encode(
                        x="Spend Share:Q",
                        y="Contribution Share:Q",
                        text=alt.Text(f"{driver_col}:N"),
                        color=alt.value("black"),
                    )
                )
    else:
        top3 = (
            plot_df[plot_df["Contribution Share"] > plot_df["Spend Share"]]
            .sort_values("delta", ascending=False)
            .head(3)
        )
        if not top3.empty:
            labels_charts.append(
                alt.Chart(top3)
                .mark_text(align="left", dx=5, dy=-5, fontSize=12, fontWeight="bold")
                .encode(
                    x="Spend Share:Q",
                    y="Contribution Share:Q",
                    text=alt.Text(f"{driver_col}:N"),
                    color=alt.value("black"),
                )
            )

    scatter = (
        alt.Chart(plot_df)
        .mark_point(filled=True, size=110, opacity=0.9)
        .encode(
            x=alt.X(
                "Spend Share:Q",
                title="Spend Share (%)",
                scale=alt.Scale(domain=[0, max_axis], clamp=True),
            ),
            y=alt.Y(
                "Contribution Share:Q",
                title="Contribution Share (%)",
                scale=alt.Scale(domain=[0, max_axis], clamp=True),
            ),
            shape=alt.Shape(
                "Period:N",
                scale=alt.Scale(
                    domain=["Current", "Previous"],
                    range=["circle", "triangle"],
                ),
                title="Period",
            ),
            color=alt.Color("delta:Q", title="Δ (Contrib − Spend)"),
            tooltip=[
                alt.Tooltip(f"{driver_col}:N", title="Driver"),
                alt.Tooltip("Period:N", title="Period"),
                alt.Tooltip("Spend Share:Q", title="Spend Share", format=".1f"),
                alt.Tooltip("Contribution Share:Q", title="Contribution Share", format=".1f"),
                alt.Tooltip("delta:Q", title="Δ", format=".1f"),
            ],
        )
    )
    chart = scatter + fair_line
    for lbl in labels_charts:
        chart = chart + lbl
    chart = (
        chart.properties(height=420)
        .configure(background="transparent")
        .configure_view(fill="transparent")
    )
    if len(plot_df) < 3:
        plot_df = None
        chart = None
    plot_df_format = readout_utils.add_unit_sign(plot_df,1,period_suffixes=["% Change"])
    plot_df_format['delta'] = plot_df_format['delta'].apply(viz_utils.fmt_pct)
    # plot_df_format = viz_utils.replace_with_labels(plot_df, ['Contribution Share', 'Spend Share', 'delta'])
    return plot_df_format, chart


def _render_fair_share_scatter_data_for_insights(
    contribution_share_data, sel_time_l, sel_time_r, m_cols, outliers,
    driver_col="Business Driver", client_code=None, model_group_id=None
):
    if contribution_share_data.empty:
        return None, None

    plot_df_l, line_df, max_axis, _ = viz_utils.contribution_fair_share_data(
        contribution_share_data, sel_time_l, m_cols, outliers, driver_col=driver_col
    )
    plot_df_r, _, _, _ = viz_utils.contribution_fair_share_data(
        contribution_share_data, sel_time_r, m_cols, outliers, driver_col=driver_col
    )
    if sel_time_l != sel_time_r:
        plot_df_l["Period"] = "Current"
        plot_df_r["Period"] = "Previous"
        plot_df = pd.concat([plot_df_l, plot_df_r], ignore_index=True)
    else:
        plot_df = plot_df_l.copy()
        plot_df["Period"] = sel_time_l

    plot_df_format = readout_utils.add_unit_sign(plot_df,1,period_suffixes=["% Change"], client_code=client_code, model_group_id=model_group_id)
    plot_df_format['delta'] = plot_df_format['delta'].apply(viz_utils.fmt_pct)

    return plot_df


def _render_fair_share_scatter(
    contribution_share_data, sel_time_l, sel_time_r, m_cols, outliers,
    driver_col="Business Driver"
):
    """Draw a scatter: spend share vs contribution share."""
    plot_df, chart = _render_fair_share_scatter_data(
        contribution_share_data, sel_time_l, sel_time_r, m_cols, outliers, driver_col
    )
    if not chart:
        return
    if plot_df is not None and not plot_df.empty:
        with st.expander("Show data table"):
            st.dataframe(plot_df, use_container_width=True)
    st.markdown(
        "<h3 style='font-size:16px; font-weight:600; margin-bottom:6px;'>"
        "Fair Share: Contribution vs. Spend</h3>",
        unsafe_allow_html=True,
    )
    st.altair_chart(chart, use_container_width=True)


# ---------------------------------------------------------------------------
# Contribution snapshot — build / render / orchestrator
# ---------------------------------------------------------------------------

def contribution_insight_pie_bar(total_str, cont_str, sales_str):
    if total_str and cont_str:
        instruct = (
            f"\n                    - Overall contribution by business driver: {total_str}"
            f"\n                    - Drill down contribution by marketing driver: {cont_str}"
        )
    else:
        instruct = ""
    if sales_str:
        sales_instruct = f"Sales by marketing driver: {sales_str}"
    else:
        sales_instruct = ""
    return instruct, sales_instruct


def build_contribution_snapshot_chart_configs(
    client_code, model_group_id, detail_data, cont_share_data, sorted_time,
    selected_levels, metric_cols, m_cols, outliers, is_specific, readout, match,
    sel_time_left, sel_time_right, prompt_inst, driver_col='Business Driver'
):
    """
    Pure data preparation: builds chart_configs dict from the *_data helpers.
    Corresponds to the logic inside get_contribution_pie_insight.

    Returns chart_configs dict.
    """
    chart_configs = {}
    if not sorted_time:
        return chart_configs
    select_time = viz_utils.get_time_align_dict(sorted_time)

    pie_inst = prompt_inst[prompt_inst['intentionName'] == "Contribution Pie"]['instruction'].values[0]
    bar_inst = prompt_inst[prompt_inst['intentionName'] == "Contribution Bar"]['instruction'].values[0]
    fair_scatter_inst = \
    prompt_inst[prompt_inst['intentionName'] == "Contribution Share - Spend Share Scatter"]['instruction'].values[0]

    if len(sorted_time) > 1:
        _, pie_data_l = _render_pie_data(client_code, model_group_id, sorted_time, selected_levels, detail_data, sel_time_left, select_time, metric_cols, outliers, readout, match, is_specific, driver_col)
        _, pie_data_r = _render_pie_data(client_code, model_group_id, sorted_time, selected_levels, detail_data, sel_time_right, select_time, metric_cols, outliers, readout, match, is_specific, driver_col)
        _, _, bar_data_cont = _render_cont_bars_data(client_code, model_group_id, sorted_time, selected_levels, detail_data,
                                      sel_time_left, sel_time_right, select_time, metric_cols, outliers, readout,
                                      match, is_specific, driver_col)
        _, bar_data_sales = _render_sales_bars_data(client_code, model_group_id, sorted_time, selected_levels, detail_data, sel_time_left, sel_time_right, select_time, metric_cols, outliers, readout, match, is_specific, driver_col)


        # inst_total = total_str_l + '\n\n' + total_str_r
        # inst, _ = contribution_insight_pie_bar(inst_total, inst_cont, None)
        # _, inst_sales = contribution_insight_pie_bar(None, None, sales_str)
        #
        # contrib_insight_id = f"contribution_{sel_time_left}_{sel_time_right}_bar"
        # instruct = f"""Provide a clear, concise, business-style insight summary.
        #                {inst}"""
        # chart_configs[contrib_insight_id] = {"query": "You are given a dataset about contribution.",
        #                       "instruction": instruct}

        pie_data_l['time'] = sel_time_left
        pie_data_r['time'] = sel_time_right
        contrib_pie_insight_id = f"contribution_{sel_time_left}_{sel_time_right}_pie"

        pie_data = pd.concat([pie_data_l, pie_data_r])
        chart_configs[contrib_pie_insight_id] = {"query": "You are given a dataset about contribution. Please add '%' to all contribution numbers.",
                                                 "instruction": pie_inst,
                                                 "data": pie_data.to_json(orient="records")}

        contrib_bar_insight_id = f"contribution_{sel_time_left}_{sel_time_right}_bar"

        chart_configs[contrib_bar_insight_id] = {"query": "You are given a dataset about contribution. Please add '%' to all contribution numbers.",
                                                 "instruction": bar_inst,
                                                 "data": bar_data_cont.to_json(orient="records")}

        sales_insight_id = f"sales_{sel_time_left}_{sel_time_right}_bar"
        # instruct_sales = f"""Provide a clear, concise, business-style insight summary.
        #                     {inst_sales}"""
        # chart_configs[sales_insight_id] = {"query": "You are given a dataset about sales.",
        #                                    "instruction" : instruct_sales}
        chart_configs[sales_insight_id] = {"query": "You are given a dataset about sales.",
                                                 "instruction": bar_inst,
                                                 "data": bar_data_sales.to_json(orient="records")}

        # is_specific = False
        if not is_specific:
            scatter_data, _ = _render_fair_share_scatter_data(cont_share_data, sel_time_left, sel_time_right, m_cols, outliers, driver_col)
            fair_insight_id = f"contribution_fair_share_{sel_time_left}_{sel_time_right}"

            # instruct_share = f"""Provide a clear, concise, business-style insight summary.
            #                     {scatter_str}"""
            # chart_configs[fair_insight_id] = {"query": "You are given a dataset about contribution share vs spend share.",
            #                                   "instruction": instruct_share}
            chart_configs[fair_insight_id] = {"query": "You are given a dataset about contribution share vs spend share. Please add '%' to all contribution numbers.",
                                              "instruction": fair_scatter_inst,
                                              "data": scatter_data.to_json(orient="records")}
    else:
        sel_time = select_time[sel_time_left]
        _, pie_data = _render_pie_data(client_code, model_group_id, sorted_time, selected_levels, detail_data, sel_time,
                             select_time, metric_cols, outliers, readout, match, is_specific, driver_col)
        _, _, bar_data_cont = _render_cont_bars_data(client_code, model_group_id, sorted_time, selected_levels, detail_data,
                                          sel_time, sel_time, select_time, metric_cols, outliers, readout,
                                          match, is_specific, driver_col)
        # inst, _ = contribution_insight_pie_bar(total_str, cont_str, None)
        contrib_pie_insight_id = f"contribution_{sel_time}_pie"
        contrib_bar_insight_id = f"contribution_{sel_time}_bar"
        # instruct = f"""Provide a clear, concise, business-style insight summary.
        #                             {inst}"""
        # chart_configs[contrib_insight_id] = {"query": "You are given a dataset about contribution.",
        #                                      "instruction": instruct}
        chart_configs[contrib_pie_insight_id] = {"query": "You are given a dataset about contribution. Please add '%' to all contribution numbers.",
                                                 "instruction": pie_inst,
                                                 "data": pie_data.to_json(orient="records")}
        chart_configs[contrib_bar_insight_id] = {"query": "You are given a dataset about contribution. Please add '%' to all contribution numbers.",
                                                 "instruction": bar_inst,
                                                 "data": bar_data_cont.to_json(orient="records")}


        _, bar_data_sales = _render_sales_bars_data(client_code, model_group_id, sorted_time, selected_levels, detail_data,
                                         sel_time, sel_time, select_time, metric_cols, outliers, readout, match, is_specific,
                                         driver_col)
        # _, sales_inst = contribution_insight_pie_bar(None, None, sales_str)
        sales_insight_id = f"sales_{sel_time}_bar"
        # instruct_sales = f"""Provide a clear, concise, business-style insight summary.
        #                     {sales_inst}"""
        # chart_configs[sales_insight_id] = {"query": "You are given a dataset about sales.",
        #                                    "instruction": instruct_sales}
        chart_configs[sales_insight_id] = {"query": "You are given a dataset about sales.",
                                           "instruction": bar_inst,
                                           "data": bar_data_sales.to_json(orient="records")}

        if not is_specific:
            scatter_data, _ = _render_fair_share_scatter_data(cont_share_data, sel_time, sel_time, m_cols, outliers, driver_col)
            fair_insight_id = f"contribution_fair_share_{sel_time}"
            # instruct_share = f"""Provide a clear, concise, business-style insight summary.
            #                            Data:
            #                            {table_str}
            #                            """
            # chart_configs[fair_insight_id] = {"query": "You are given a dataset about contribution share vs spend share.",
            #                                   "instruction": instruct_share}
            chart_configs[fair_insight_id] = {
                "query": "You are given a dataset about contribution share vs spend share. Please add '%' to all contribution numbers.",
                "instruction": fair_scatter_inst,
                "data": scatter_data.to_json(orient="records")}
    return chart_configs


def render_contribution_snapshot(
    client_code, model_group_id, detail_data, cont_share_data, sorted_time,
    selected_levels, metric_cols, m_cols, outliers, is_specific, readout, match,
    driver_col='Business Driver', insights=None
):
    """
    Pure Streamlit rendering for the contribution snapshot section.
    Accepts pre-computed insights dict from generate_chart_insights_parallel_cached.
    """
    if not sorted_time:
        return
    select_time = viz_utils.get_time_align_dict(sorted_time)

    if len(sorted_time) > 1:
        pcol1, pcol2 = st.columns([1, 1])
        with pcol1:
            sel_time_left_label = st.selectbox(
                "Current period:", select_time.keys(), index=0, key="pie_left_time"
            )
            sel_time_left = select_time[sel_time_left_label]
        with pcol2:
            keys_list = list(select_time.keys())
            left_idx = keys_list.index(sel_time_left_label)
            right_default_index = min(left_idx + 1, len(keys_list) - 1)
            sel_time_right_label = st.selectbox(
                "Previous period:", keys_list, index=right_default_index, key="pie_right_time"
            )
            sel_time_right = select_time[sel_time_right_label]

        col_left, col_mid, col_right = st.columns([1, 1.2, 1.2])
        with col_left:
            _render_pie(
                client_code, model_group_id, sorted_time, selected_levels, detail_data,
                sel_time_left, select_time, metric_cols, outliers, readout, match, is_specific, driver_col,
            )
            _render_pie(
                client_code, model_group_id, sorted_time, selected_levels, detail_data,
                sel_time_right, select_time, metric_cols, outliers, readout, match, is_specific, driver_col,
            )
        with col_mid:
            _render_cont_bars(
                client_code, model_group_id, sorted_time, selected_levels, detail_data,
                sel_time_left, sel_time_right, select_time, metric_cols, outliers,
                readout, match, is_specific, driver_col,
            )
        with col_right:
            _render_sales_bars(
                client_code, model_group_id, sorted_time, selected_levels, detail_data,
                sel_time_left, sel_time_right, select_time, metric_cols, outliers,
                readout, match, is_specific, driver_col,
            )

        if insights:
            _, icol1, icol2 = st.columns([1, 1.2, 1.2])
            contrib_insight_id = f"contribution_{sel_time_left}_{sel_time_right}_bar"
            with icol1:
                st.session_state.insight[contrib_insight_id] = insights.get(contrib_insight_id, '')
                saved = st.session_state.insight.get(contrib_insight_id)
                if saved:
                    render_chart_insight_box("Contribution Insight", saved)
            sales_insight_id = f"sales_{sel_time_left}_{sel_time_right}_bar"
            with icol2:
                st.session_state.insight[sales_insight_id] = insights.get(sales_insight_id, '')
                saved = st.session_state.insight.get(sales_insight_id)
                if saved:
                    render_chart_insight_box("Sales Insight", saved)

        if not is_specific:
            _render_fair_share_scatter(
                cont_share_data, sel_time_left, sel_time_right, m_cols, outliers, driver_col
            )
            if insights:
                fair_insight_id = f"contribution_fair_share_{sel_time_left}_{sel_time_right}"
                st.session_state.insight[fair_insight_id] = insights.get(fair_insight_id, '')
                saved = st.session_state.insight.get(fair_insight_id)
                if saved:
                    render_chart_insight_box("Fair Share Insight", saved)
    else:
        sel_time = st.selectbox(
            "Select a period:", sorted_time, index=0, key="pie_single_time"
        )
        sel_time = select_time[sel_time]

        _render_pie(
            client_code, model_group_id, sorted_time, selected_levels, detail_data,
            sel_time, select_time, metric_cols, outliers, readout, match, is_specific, driver_col,
        )
        _render_cont_bars(
            client_code, model_group_id, sorted_time, selected_levels, detail_data,
            sel_time, sel_time, select_time, metric_cols, outliers,
            readout, match, is_specific, driver_col,
        )
        if insights:
            contrib_insight_id = f"contribution_{sel_time}_bar"
            st.session_state.insight[contrib_insight_id] = insights.get(contrib_insight_id, '')
            saved = st.session_state.insight.get(contrib_insight_id)
            if saved:
                render_chart_insight_box("Contribution Insight", saved)

        _render_sales_bars(
            client_code, model_group_id, sorted_time, selected_levels, detail_data,
            sel_time, sel_time, select_time, metric_cols, outliers,
            readout, match, is_specific, driver_col,
        )
        if insights:
            sales_insight_id = f"sales_{sel_time}_bar"
            st.session_state.insight[sales_insight_id] = insights.get(sales_insight_id, '')
            saved = st.session_state.insight.get(sales_insight_id)
            if saved:
                render_chart_insight_box("Sales Insight", saved)

        if not is_specific:
            _render_fair_share_scatter(
                cont_share_data, sel_time, sel_time, m_cols, outliers, driver_col
            )
            if insights:
                fair_insight_id = f"contribution_fair_share_{sel_time}"
                st.session_state.insight[fair_insight_id] = insights.get(fair_insight_id, '')
                saved = st.session_state.insight.get(fair_insight_id)
                if saved:
                    render_chart_insight_box("Fair Share Insight", saved)


def get_contribution_pie(
    client_code, model_group_id, detail_data, cont_share_data, sorted_time,
    selected_levels, metric_cols, m_cols, outliers, is_specific, readout, match,
    driver_col='Business Driver', llm_vendor='deepseek'
):
    """
    Orchestrator: build → generate insights in parallel → render.

    Returns (chart_configs, insights).
    """
    if not sorted_time:
        return {}, {}

    select_time = viz_utils.get_time_align_dict(sorted_time)

    # Determine default time selections (mirrors render logic)
    if len(sorted_time) > 1:
        keys_list = list(select_time.keys())
        sel_time_left = select_time[keys_list[0]]
        right_default_index = min(1, len(keys_list) - 1)
        sel_time_right = select_time[keys_list[right_default_index]]
    else:
        sel_time_left = select_time[list(select_time.keys())[0]]
        sel_time_right = sel_time_left

    chart_configs = build_contribution_snapshot_chart_configs(
        client_code, model_group_id, detail_data, cont_share_data, sorted_time,
        selected_levels, metric_cols, m_cols, outliers, is_specific, readout, match,
        sel_time_left, sel_time_right, driver_col,
    )

    with st.spinner("Your visuals and insights are on the way..."):
        start = time.time()
        insights = generate_chart_insights_parallel_cached(chart_configs, llm_vendor)
        elapsed = time.time() - start
    st.write(f"Insight generation time: {elapsed:.2f} seconds")

    render_contribution_snapshot(
        client_code, model_group_id, detail_data, cont_share_data, sorted_time,
        selected_levels, metric_cols, m_cols, outliers, is_specific, readout, match,
        driver_col, insights,
    )

    return chart_configs, insights


def build_contribution_snapshot_chart_configs_for_insights(
    client_code, model_group_id, detail_data, cont_share_data, sorted_time,
    selected_levels, metric_cols, m_cols, outliers, is_specific, readout, match,
    sel_time_left, sel_time_right, prompt_inst, driver_col='Business Driver'
):
    """
    builds chart_configs dict from the *_data helpers.
    Corresponds to the logic inside get_contribution_pie_insight.

    Returns chart_configs dict.
    """
    chart_configs = {}
    if not sorted_time:
        return chart_configs
    select_time = viz_utils.get_time_align_dict(sorted_time)

    pie_inst = prompt_inst[prompt_inst['intentionName'] == "Contribution Pie"]['instruction'].values[0]
    bar_inst = prompt_inst[prompt_inst['intentionName'] == "Contribution Bar"]['instruction'].values[0]
    fair_scatter_inst = \
    prompt_inst[prompt_inst['intentionName'] == "Contribution Share - Spend Share Scatter"]['instruction'].values[0]

    if len(sorted_time) > 1:
        pie_data_l = _render_pie_data_for_insights(client_code, model_group_id, sorted_time, selected_levels, detail_data, sel_time_left, select_time, metric_cols, outliers, readout, match, is_specific, driver_col)
        pie_data_r = _render_pie_data_for_insights(client_code, model_group_id, sorted_time, selected_levels, detail_data, sel_time_right, select_time, metric_cols, outliers, readout, match, is_specific, driver_col)
        bar_data_cont = _render_cont_bars_data_for_insights(client_code, model_group_id, sorted_time, selected_levels, detail_data,
                                      sel_time_left, sel_time_right, select_time, metric_cols, outliers, readout,
                                      match, is_specific, driver_col)
        bar_data_sales = _render_sales_bars_data_for_insights(client_code, model_group_id, sorted_time, selected_levels, detail_data, sel_time_left, sel_time_right, select_time, metric_cols, outliers, readout, match, is_specific, driver_col)


        # inst_total = total_str_l + '\n\n' + total_str_r
        # inst, _ = contribution_insight_pie_bar(inst_total, inst_cont, None)
        # _, inst_sales = contribution_insight_pie_bar(None, None, sales_str)
        #
        # contrib_insight_id = f"contribution_{sel_time_left}_{sel_time_right}_bar"
        # instruct = f"""Provide a clear, concise, business-style insight summary.
        #                {inst}"""
        # chart_configs[contrib_insight_id] = {"query": "You are given a dataset about contribution.",
        #                       "instruction": instruct}

        pie_data_l['time'] = sel_time_left
        pie_data_r['time'] = sel_time_right
        contrib_pie_insight_id = f"contribution_{sel_time_left}_{sel_time_right}_pie"

        chart_configs[contrib_pie_insight_id] = {"query": "You are given a dataset about contribution. Please add '%' to all contribution numbers.",
                                                 "instruction": pie_inst,
                                                 "data": pd.concat([pie_data_l, pie_data_r]).to_json(orient="records")}

        contrib_bar_insight_id = f"contribution_{sel_time_left}_{sel_time_right}_bar"

        chart_configs[contrib_bar_insight_id] = {"query": "You are given a dataset about contribution. Please add '%' to all contribution numbers.",
                                                 "instruction": bar_inst,
                                                 "data": bar_data_cont.to_json(orient="records")}

        sales_insight_id = f"sales_{sel_time_left}_{sel_time_right}_bar"
        # instruct_sales = f"""Provide a clear, concise, business-style insight summary.
        #                     {inst_sales}"""
        # chart_configs[sales_insight_id] = {"query": "You are given a dataset about sales.",
        #                                    "instruction" : instruct_sales}
        chart_configs[sales_insight_id] = {"query": "You are given a dataset about sales.",
                                                 "instruction": bar_inst,
                                                 "data": bar_data_sales.to_json(orient="records")}

        if not is_specific:
            scatter_data = _render_fair_share_scatter_data_for_insights(cont_share_data, sel_time_left, sel_time_right, m_cols, outliers, driver_col, client_code=client_code, model_group_id=model_group_id)
            fair_insight_id = f"contribution_fair_share_{sel_time_left}_{sel_time_right}"

            # instruct_share = f"""Provide a clear, concise, business-style insight summary.
            #                     {scatter_str}"""
            # chart_configs[fair_insight_id] = {"query": "You are given a dataset about contribution share vs spend share.",
            #                                   "instruction": instruct_share}
            chart_configs[fair_insight_id] = {"query": "You are given a dataset about contribution share vs spend share. Please add '%' to all contribution numbers.",
                                              "instruction": fair_scatter_inst,
                                              "data": scatter_data.to_json(orient="records")}
    else:
        sel_time = select_time[sel_time_left]
        pie_data = _render_pie_data_for_insights(client_code, model_group_id, sorted_time, selected_levels, detail_data, sel_time,
                             select_time, metric_cols, outliers, readout, match, is_specific, driver_col)
        bar_data_cont = _render_cont_bars_data_for_insights(client_code, model_group_id, sorted_time, selected_levels, detail_data,
                                          sel_time, sel_time, select_time, metric_cols, outliers, readout,
                                          match, is_specific, driver_col)
        # inst, _ = contribution_insight_pie_bar(total_str, cont_str, None)
        contrib_pie_insight_id = f"contribution_{sel_time}_pie"
        contrib_bar_insight_id = f"contribution_{sel_time}_bar"
        # instruct = f"""Provide a clear, concise, business-style insight summary.
        #                             {inst}"""
        # chart_configs[contrib_insight_id] = {"query": "You are given a dataset about contribution.",
        #                                      "instruction": instruct}
        chart_configs[contrib_pie_insight_id] = {"query": "You are given a dataset about contribution. Please add '%' to all contribution numbers.",
                                                 "instruction": pie_inst,
                                                 "data": pie_data.to_json(orient="records")}
        chart_configs[contrib_bar_insight_id] = {"query": "You are given a dataset about contribution. Please add '%' to all contribution numbers.",
                                                 "instruction": bar_inst,
                                                 "data": bar_data_cont.to_json(orient="records")}


        bar_data_sales = _render_sales_bars_data_for_insights(client_code, model_group_id, sorted_time, selected_levels, detail_data,
                                         sel_time, sel_time, select_time, metric_cols, outliers, readout, match, is_specific,
                                         driver_col)
        # _, sales_inst = contribution_insight_pie_bar(None, None, sales_str)
        sales_insight_id = f"sales_{sel_time}_bar"
        # instruct_sales = f"""Provide a clear, concise, business-style insight summary.
        #                     {sales_inst}"""
        # chart_configs[sales_insight_id] = {"query": "You are given a dataset about sales.",
        #                                    "instruction": instruct_sales}
        chart_configs[sales_insight_id] = {"query": "You are given a dataset about sales.",
                                           "instruction": bar_inst,
                                           "data": bar_data_sales.to_json(orient="records")}

        if not is_specific:
            scatter_data = _render_fair_share_scatter_data_for_insights(cont_share_data, sel_time, sel_time, m_cols, outliers, driver_col,
                                                                        client_code=client_code, model_group_id=model_group_id)
            fair_insight_id = f"contribution_fair_share_{sel_time}"
            # instruct_share = f"""Provide a clear, concise, business-style insight summary.
            #                            Data:
            #                            {table_str}
            #                            """
            # chart_configs[fair_insight_id] = {"query": "You are given a dataset about contribution share vs spend share.",
            #                                   "instruction": instruct_share}
            chart_configs[fair_insight_id] = {
                "query": "You are given a dataset about contribution share vs spend share. Please add '%' to all contribution numbers.",
                "instruction": fair_scatter_inst,
                "data": scatter_data.to_json(orient="records")}
    return chart_configs

# ---------------------------------------------------------------------------
# Contribution trend helpers
# ---------------------------------------------------------------------------

def _contribution_trend_level_chart(df_level, title, sorted_labels):
    if df_level.empty:
        return None, None
    latest_time = sorted_labels[-1]
    df_latest = df_level[df_level["TimeLabel"] == latest_time]
    if not df_latest.empty:
        tactics_order = df_latest.sort_values("Value", ascending=False)["Marketing Driver"].tolist()
    else:
        tactics_order = df_level["Marketing Driver"].unique().tolist()

    reversed_time_order = sorted_labels[::-1]
    df_level = df_level.drop(['Metric', 'Time', 'ValueFormatted'], axis=1)

    base = alt.Chart(df_level).encode(
        x=alt.X(
            "Marketing Driver:N",
            title="Marketing Driver",
            sort=tactics_order,
            axis=alt.Axis(labelAngle=0),
        ),
        xOffset=alt.XOffset("TimeLabel:N", sort=reversed_time_order),
    )
    bar = base.mark_bar(opacity=0.8).encode(
        y=alt.Y("Value:Q", title="Contribution"),
        color=alt.Color(
            "TimeLabel:N",
            sort=reversed_time_order,
            legend=alt.Legend(title="Time"),
            scale=alt.Scale(scheme="blues"),
        ),
        tooltip=[
            alt.Tooltip("Marketing Driver:N", title="Marketing Driver"),
            alt.Tooltip("TimeLabel:N", title="Time"),
            alt.Tooltip("Value:Q", title="Contribution", format=".1f"),
        ],
    )
    chart = alt.layer(bar).properties(
        height=300,
        title=title,
        padding={"left": 10, "right": 30, "top": 10, "bottom": 10},
    )
    return chart, df_level


def _contribution_trend_level_chart_for_insights(df_level, title, sorted_labels):
    if df_level.empty:
        return None, None
    latest_time = sorted_labels[-1]
    df_latest = df_level[df_level["TimeLabel"] == latest_time]
    if not df_latest.empty:
        tactics_order = df_latest.sort_values("Value", ascending=False)["Marketing Driver"].tolist()
    else:
        tactics_order = df_level["Marketing Driver"].unique().tolist()

    reversed_time_order = sorted_labels[::-1]
    df_level = df_level.drop(['Metric', 'Time', 'ValueFormatted'], axis=1)
    return df_level


def contribution_change_table_str(df_wide):
    """Table-text summary of the contribution-vs-spend wide frame. '' when empty."""
    if df_wide.empty:
        return ""
    _, table_str, _, _ = readout_utils.table_to_text(pd.DataFrame(), df_wide, take_head=False)
    return table_str


def contribution_change_chart(df_wide):
    """Altair scatter of contribution share vs spend share % change. None when empty."""
    if df_wide.empty:
        return None
    contrib_col = next(c for c in df_wide.columns if "contribution share" in c.lower())
    spend_col = next(c for c in df_wide.columns if "spend share" in c.lower())
    x_min, x_max = df_wide[spend_col].min(), df_wide[spend_col].max()
    y_min, y_max = df_wide[contrib_col].min(), df_wide[contrib_col].max()
    min_val = min(x_min, y_min)
    max_val = max(x_max, y_max)
    diag_range = [min_val - 0.1 * abs(min_val), max_val + 0.1 * abs(max_val)]
    diag_line = (
        alt.Chart(pd.DataFrame({"val": diag_range}))
        .mark_line(strokeDash=[5, 5], color="blue")
        .encode(x='val:Q', y='val:Q')
    )
    vline = (
        alt.Chart(pd.DataFrame({"x": [0]}))
        .mark_rule(strokeDash=[4, 2], color="gray")
        .encode(x='x:Q')
    )
    hline = (
        alt.Chart(pd.DataFrame({"y": [0]}))
        .mark_rule(strokeDash=[4, 2], color="gray")
        .encode(y='y:Q')
    )
    scatter = alt.Chart(df_wide).mark_circle(size=120).encode(
        x=alt.X(f"{spend_col}:Q", title="% Change in Spend Share", scale=alt.Scale(domain=diag_range)),
        y=alt.Y(f"{contrib_col}:Q", title="% Change in Contribution Share", scale=alt.Scale(domain=diag_range)),
        color=alt.Color("Business Driver:N", legend=alt.Legend(title="Driver")),
        tooltip=[
            "Business Driver:N",
            alt.Tooltip(f"{spend_col}:Q", title="Spend Share % Change", format=".1f"),
            alt.Tooltip(f"{contrib_col}:Q", title="Contribution Share % Change", format=".1f"),
        ],
    )
    chart = (diag_line + vline + hline + scatter).properties(
        width=700,
        height=500,
        title=f"Contribution Share vs Spend Share % Change in {df_wide['Time'].unique()[0]}",
        padding={"left": 10, "right": 30, "top": 10, "bottom": 10},
    )
    return chart


# ---------------------------------------------------------------------------
# Contribution trend — build / render / orchestrator
# ---------------------------------------------------------------------------

def build_contribution_trend_chart_configs(
    br_detail_level, pivot_biz_table, data, selected_levels, is_specific,
    readout, match, prompt_inst,  **kwargs
):
    """
    Derives df_plot, builds chart configs dict and
    collects pre-computed charts / data for the render step.

    Returns
    -------
    chart_configs  : dict  (may be empty if no chart data)
    ch_expander    : dict  {group_key: {"chart": ..., "df_level": ...}}
    mode_result    : dict  {
                       "mode": str,
                       "chart": altair chart or None  (change mode only),
                     }
    """
    mode = kwargs.get('mode', 'trend')
    ner_filter_dict = readout.ner_filters
    data_source = pivot_biz_table if br_detail_level == 'Aggregation View' else data
    for level_col in selected_levels:
        data_source = data_source[data_source[level_col] == selected_levels[level_col]]
    df_plot, _ = viz_utils.contribution_trend_data(
        data_source, selected_levels, ner_filter_dict, is_specific, readout, match, mode
    )
    sorted_times = sorted(
        df_plot["Time"].unique(),
        key=lambda x: (int(x.split()[1]), int(x.split()[0][1:])) if len(x.split()) > 1 else (x, 0),
    )
    select_time = viz_utils.get_time_align_dict(sorted_times)
    time_label_dict = {v: k for k, v in select_time.items()}
    sorted_labels = [time_label_dict[k] for k in sorted_times if k in time_label_dict]

    chart_configs = {}
    scatter_change_inst = \
    prompt_inst[prompt_inst['intentionName'] == "Contribution Share - Spend Share % Change Scatter"][
        'instruction'].values[0]
    trend_inst = prompt_inst[prompt_inst['intentionName'] == "contribution_trend"]['instruction'].values[0]
    ch_expander = {}
    mode_result = {"mode": mode, "chart": None}

    if mode == 'trend':
        table_for_llm = []
        has_chart = False
        if is_specific:
            for d in df_plot['group'].unique().tolist():
                sub = df_plot[df_plot['group'] == d]
                ch, df_level = _contribution_trend_level_chart(
                    sub, f"Contribution trend — {d.title()}", sorted_labels
                )
                if ch is not None:
                    has_chart = True
                    ch_expander[d] = {"chart": ch, "df_level": df_level}
                    table_str = _contribution_trend_table_str(df_level)
                    if table_str:
                        table_for_llm.append(f"{d} data\n{table_str}")
        else:
            levels = ["high", "moderate", "low"]
            for lvl in levels:
                sub = df_plot[df_plot["contribution level"].str.lower() == lvl]
                ch, df_level = _contribution_trend_level_chart(
                    sub, f"Contribution trend — {lvl.title()}", sorted_labels
                )
                if ch is not None:
                    has_chart = True
                    ch_expander[lvl] = {"chart": ch, "df_level": df_level}
                    table_str = _contribution_trend_table_str(df_level)
                    if table_str:
                        table_for_llm.append(f"{lvl.title()} data\n{table_str}")

        if has_chart:
            d = df_plot.drop(['Metric', 'Time', 'Value'], axis=1).to_json(orient="records")
            trend_key = f"contribution trend by level"
            # instruct = f"""Provide a clear, concise, business-style insight summary.
            #                            Data: {table_for_llm}"""
            chart_configs = {trend_key: {"query": "You are given a dataset about contribution trend.",
                                         "instruction": trend_inst,
                                         "data": d}}
    else:
        chart = contribution_change_chart(df_plot)
        if chart:
            fair_share_key = f"contribution_change_scatter"
            # instruct = f"""Provide a clear, concise, business-style insight summary.
            #                            Data: {table_str}"""
            # chart_configs = {fair_share_key: {"query": "You are given a dataset about contribution change.",
            #                                   "instruction": instruct}}
            # contrib_col = next((c for c in df_plot.columns if "contribution share" in c.lower()), None)
            # spend_col = next((c for c in df_plot.columns if "spend share" in c.lower()), None)
            # df_plot = viz_utils.replace_with_labels(df_plot, [contrib_col, spend_col])
            df_plot = readout_utils.add_unit_sign(df_plot, period_suffixes=["% Change"])
            chart_configs = {fair_share_key: {"query": "You are given a dataset about contribution change.",
                                              "instruction": scatter_change_inst,
                                              "data": df_plot.to_json(orient="records")}}
            mode_result["chart"] = chart

    return chart_configs, ch_expander, mode_result


def build_contribution_trend_chart_configs_for_insights(
    br_detail_level, pivot_biz_table, data, selected_levels, is_specific,
    readout, match, prompt_inst, mode, client_code, model_group_id
):
    """
    derives df_plot, builds chart configs dict and
    collects pre-computed data for the insights generation

    Returns
    -------
    chart_configs  : dict  (may be empty if no chart data)
    ch_expander    : dict  {group_key: {"chart": ..., "df_level": ...}}
    mode_result    : dict  {
                       "mode": str,
                       "chart": altair chart or None  (change mode only),
                     }
    """
    ner_filter_dict = readout.ner_filters
    data_source = pivot_biz_table if br_detail_level == 'Aggregation View' else data
    for level_col in selected_levels:
        data_source = data_source[data_source[level_col] == selected_levels[level_col]]
    df_plot, _ = viz_utils.contribution_trend_data(
        data_source, selected_levels, ner_filter_dict, is_specific, readout, match, mode
    )
    sorted_times = sorted(
        df_plot["Time"].unique(),
        key=lambda x: (int(x.split()[1]), int(x.split()[0][1:])) if len(x.split()) > 1 else (x, 0),
    )
    select_time = viz_utils.get_time_align_dict(sorted_times)
    time_label_dict = {v: k for k, v in select_time.items()}
    sorted_labels = [time_label_dict[k] for k in sorted_times if k in time_label_dict]

    chart_configs = {}
    scatter_change_inst = \
    prompt_inst[prompt_inst['intentionName'] == "Contribution Share - Spend Share % Change Scatter"][
        'instruction'].values[0]
    trend_inst = prompt_inst[prompt_inst['intentionName'] == "contribution_trend"]['instruction'].values[0]

    if mode == 'trend':
        table_for_llm = []
        if is_specific:
            for d in df_plot['group'].unique().tolist():
                sub = df_plot[df_plot['group'] == d]
                df_level = _contribution_trend_level_chart_for_insights(
                    sub, f"Contribution trend — {d.title()}", sorted_labels
                )
                table_str = _contribution_trend_table_str(df_level)
                if table_str:
                    table_for_llm.append(f"{d} data\n{table_str}")
        else:
            levels = ["high", "moderate", "low"]
            for lvl in levels:
                sub = df_plot[df_plot["contribution level"].str.lower() == lvl]
                df_level = _contribution_trend_level_chart_for_insights(
                    sub, f"Contribution trend — {lvl.title()}", sorted_labels
                )
                table_str = _contribution_trend_table_str(df_level)
                if table_str:
                    table_for_llm.append(f"{lvl.title()} data\n{table_str}")

        d = df_plot.drop(['Metric', 'Time', 'Value'], axis=1).to_json(orient="records")
        trend_key = f"contribution trend by level"
        # instruct = f"""Provide a clear, concise, business-style insight summary.
        #                            Data: {table_for_llm}"""
        chart_configs = {trend_key: {"query": "You are given a dataset about contribution trend.",
                                     "instruction": trend_inst,
                                     "data": d}}
    else:
        fair_share_key = f"contribution_change_scatter"
        # contrib_col = next((c for c in df_plot.columns if "contribution share" in c.lower()), None)
        # spend_col = next((c for c in df_plot.columns if "spend share" in c.lower()), None)
        # df_plot = viz_utils.replace_with_labels(df_plot, [contrib_col, spend_col])
        df_plot = readout_utils.add_unit_sign(df_plot, period_suffixes=["% Change"], client_code=client_code, model_group_id=model_group_id)
        chart_configs = {fair_share_key: {"query": "You are given a dataset about contribution change.",
                                          "instruction": scatter_change_inst,
                                          "data": df_plot.to_json(orient="records")}}

    return chart_configs


def render_contribution_trend(
    br_detail_level, pivot_biz_table, data, selected_levels, is_specific,
    readout, match, insights, ch_expander, mode_result, **kwargs
):
    """
    Pure Streamlit rendering for contribution trend.
    Accepts pre-computed insights, ch_expander, and mode_result from
    build_contribution_trend_chart_configs.
    """
    mode = mode_result.get("mode", "trend")
    if mode == 'trend':
        if ch_expander:
            trend_key = "contribution trend by level"
            for idx, v in ch_expander.items():
                ch = v['chart']
                df_level = v['df_level']
                c1, c2, c3 = st.columns([1, 3, 1])
                with c2:
                    ch = ch.configure(background="transparent").configure_view(fill="transparent")
                    with st.expander("Show data table"):
                        st.dataframe(df_level, use_container_width=True)
                    st.altair_chart(ch, use_container_width=True)
            if insights and insights.get(trend_key):
                st.session_state.insight[trend_key] = insights[trend_key]
                saved = st.session_state.insight.get(trend_key)
                if saved:
                    render_chart_insight_box("Trend Insight", saved)
    else:
        chart = mode_result.get("chart")
        if chart:
            fair_share_key = "contribution_change_scatter"
            chart = chart.configure(background="transparent").configure_view(fill="transparent")
            st.altair_chart(chart, use_container_width=True)
            if insights and insights.get(fair_share_key):
                st.session_state.insight[fair_share_key] = insights[fair_share_key]
                saved = st.session_state.insight.get(fair_share_key)
                if saved:
                    render_chart_insight_box("Contribution Change Insight", saved)


def contribution_trend(
    br_detail_level, pivot_biz_table, data, selected_levels, is_specific,
    readout, match, llm_vendor='deepseek', **kwargs
):
    """
    Orchestrator: build → generate insights in parallel → render.

    Returns (chart_configs, insights).
    """
    mode = kwargs.get('mode', 'trend')

    chart_configs, ch_expander, mode_result = build_contribution_trend_chart_configs(
        br_detail_level, pivot_biz_table, data, selected_levels, is_specific,
        readout, match, **kwargs
    )

    if chart_configs:
        with st.spinner("Your visuals and insights are on the way..."):
            start = time.time()
            insights = generate_chart_insights_parallel_cached(chart_configs, llm_vendor)
            elapsed = time.time() - start
        st.write(f"Insights generated in {elapsed:.2f} seconds.")
    else:
        insights = {}

    render_contribution_trend(
        br_detail_level, pivot_biz_table, data, selected_levels, is_specific,
        readout, match, insights, ch_expander, mode_result, **kwargs
    )

    return chart_configs, insights