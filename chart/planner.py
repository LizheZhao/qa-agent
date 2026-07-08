import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import src.model.readout_utils as readout_utils
import viz_utils as viz_utils
from chart.common import _section, table_element, render_chart_insight_box
from streamlit_elements import elements, mui
import time


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _planner_spend_bar(
    data, spend_level, x_axis_upper,
    bar_col1="Spend - Historical", bar_col2="Spend - Forecast"
):
    with st.expander("Show data table"):
        expander_data = data.pivot(
            index="Driver", columns='Metric', values='Value'
        ).reset_index()
        st.dataframe(expander_data, use_container_width=True)

    sort_order = (
        data[data["Metric"] == bar_col1][["Driver", "Value"]]
        .sort_values(by="Value", ascending=False)["Driver"]
        .tolist()
    )

    spend_scale = alt.Scale(domain=[0, x_axis_upper]) if x_axis_upper > 0 else alt.Scale()

    bars = alt.Chart(data).mark_bar(opacity=0.85).encode(
        y=alt.X("Driver:N", sort=sort_order, axis=alt.Axis(labelAngle=0, title=None)),
        yOffset=alt.X("Metric:N", sort=[bar_col1, bar_col2]),
        x=alt.Y("Value:Q", title="Spend(MM)", scale=spend_scale),
        color=alt.Color("Metric:N", sort=[bar_col1, bar_col2]),
        tooltip=["Driver", "Metric", alt.Tooltip("Value:Q", format=",.2f")],
    )

    chart = bars.properties(
        height=800,
        title={
            "text": f"Historical Spend vs. Forecast Spend - {spend_level}",
            "subtitle": [f"Blue = {bar_col1}(MM), Yellow = {bar_col2}(MM) (sorted by {bar_col1} desc)"],
            "subtitleColor": "gray",
            "subtitleFontSize": 12,
            "anchor": "start",
        },
        padding={"left": 5, "right": 20, "top": 10, "bottom": 10},
    )
    chart = chart.configure(background="transparent")
    return chart


def metric_card_planner(key, title, hist_value, fcst_value, growth, width):
    if growth is None:
        delta_value = None
    else:
        try:
            delta_value = float(str(growth).strip().replace("%", ""))
        except Exception:
            delta_value = None

    if delta_value is None:
        arrow = ""
        color = "text.secondary"
        delta_text = ""
    else:
        arrow = "▲" if delta_value >= 0 else "▼"
        color = "green" if delta_value >= 0 else "red"
        delta_text = f"{abs(delta_value):.1f}%"

    with mui.Paper(
        key=key,
        elevation=1,
        sx={
            "p": "6px 15px",
            "borderRadius": 2,
            "display": "flex",
            "flexDirection": "column",
            "gap": "2px",
            "width": width,
        },
    ):
        mui.Typography(
            title + "",
            variant="subtitle1",
            color="text.secondary",
            fontWeight=700,
        )
        mui.Typography(
            f"Historical: {hist_value}",
            variant="caption",
            color="text.secondary",
        )
        mui.Typography(str(fcst_value), variant="h5", fontWeight=600)
        mui.Typography(
            f"{arrow} {delta_text}".strip(),
            variant="body1",
            sx={"color": color},
        )


def planner_change_fair_share(data, x_col, y_col, title):
    """Scatter chart comparing spend % vs a KPI % change, with labels for under-performers."""
    if not data.empty:
        if len(data) < 2:
            st.markdown(f"**{title}**")
            st.table(data[['Driver', x_col, y_col]].reset_index(drop=True))
        else:
            underperf = data[(data[x_col] > 0) & (data[y_col] < 0)]

            base = alt.Chart(data).mark_circle(size=80, opacity=0.7).encode(
                x=alt.X(x_col, title=x_col),
                y=alt.Y(y_col, title=y_col),
                tooltip=["Driver", x_col, y_col],
            )
            labels = alt.Chart(underperf).mark_text(
                align="left", dx=5, dy=-5, fontSize=10, color="black"
            ).encode(
                x=x_col,
                y=y_col,
                text="Driver",
            )
            diag = (
                alt.Chart(pd.DataFrame({"val": [-30, 30]}))
                .mark_line(strokeDash=[5, 5], color="blue")
                .encode(x="val:Q", y="val:Q")
            )
            vline = (
                alt.Chart(pd.DataFrame({"x": [0]}))
                .mark_rule(strokeDash=[4, 2], color="gray")
                .encode(x="x:Q")
            )
            hline = (
                alt.Chart(pd.DataFrame({"y": [0]}))
                .mark_rule(strokeDash=[4, 2], color="gray")
                .encode(y="y:Q")
            )
            chart = (base + labels + diag + vline + hline).properties(
                title=title,
                height=300,
                padding={"left": 10, "right": 30, "top": 10, "bottom": 10},
            )
            chart = chart.configure(background="transparent")
            st.altair_chart(chart, use_container_width=True)


# ---------------------------------------------------------------------------
# Public functions (moved as-is from st_utils.py)
# ---------------------------------------------------------------------------

def planner_overall_metric(client_code, model_group_id, overall_data):
    records = viz_utils.get_planner_overall_data(client_code, model_group_id, overall_data)
    with elements("planner_overall_cards"):
        with mui.Stack(direction='row', spacing=2, sx={
            "justifyContent": "center",
            "alignItems": "center"}):
            for rec in records:
                metric_card_planner(
                    key=rec['KPI'],
                    title=rec['KPI'],
                    hist_value=rec['historical_fmt'],
                    fcst_value= rec['forecast_fmt'],
                    growth=rec['growth'],
                    width = 400,
                )


def get_planner_spend_level_plot(data, levels):
    planner_data = viz_utils.get_planner_data(data, levels)
    if planner_data.empty:
        return
    df_long, _ = viz_utils.planner_spend_level_bar_data(planner_data)
    max_spend = -1
    for l in levels:
        data = df_long[df_long['Spend Level'] == l]
        chart = _planner_spend_bar(
            data, l, x_axis_upper=max_spend,
            bar_col1="Spend - Historical", bar_col2="Spend - Forecast",
        )
        st.altair_chart(chart, use_container_width=True)


def spend_kpi_scatter_insight(data,  levels):
    df = viz_utils.get_planner_data(data, levels)
    if df.empty:
        return ''
   # Pair A: Spend % vs Margin % Change
    margin_col = next((c for c in ['Margin % Change', 'KPI Value % Change'] if c in df.columns))
    eff_col = next((c for c in ['Margin ROI % Change', 'Cost Per Acquisition % Change', 'Marginal ROI % Change'] if c in df.columns))
    df_left, left_nochg, left_nomeas, _ = viz_utils.split_for_pair(df, margin_col)
    # Pair B: Spend % vs Margin ROI % Change
    df_right, right_nochg, right_nomeas, _ = viz_utils.split_for_pair(df, eff_col)

    df_ins = pd.merge(df_left, df_right, on=['Driver', 'Spend % Change'], how='inner')
    if not df_ins.empty:
        _, table_str, _, _ = readout_utils.table_to_text(pd.DataFrame(), df_ins, take_head=False)
        instruct = (
            f"From the given data,\n"
            f"- Identify drivers whose margin contribution % exceeds spend increase % and margin ROI continues to grow.\n"
            f"- Identify drivers whose margin contribution % is lower than spend increase % and margin ROI slightly drops.\n"
            f"- Identify drivers whose margin impact % is smaller than spend decrease % and efficiency improves.\n"
            f"Then, generate a business insight summary describing these drivers spend–margin efficiency patterns.\n"
            f"Data: {table_str}"
        )
        return instruct
    else:
        return ''


def spend_kpi_scatter_insight_for_insights(data,  levels):
    config = {}
    df = viz_utils.get_planner_data(data, levels)
    if df.empty:
        return config
   # Pair A: Spend % vs Margin % Change
    margin_col = next((c for c in ['Margin % Change', 'KPI Value % Change'] if c in df.columns))
    eff_col = next((c for c in ['Margin ROI % Change', 'Cost Per Acquisition % Change', 'Marginal ROI % Change'] if c in df.columns))
    df_left, left_nochg, left_nomeas, _ = viz_utils.split_for_pair(df, margin_col)
    # Pair B: Spend % vs Margin ROI % Change
    df_right, right_nochg, right_nomeas, _ = viz_utils.split_for_pair(df, eff_col)

    df_ins = pd.merge(df_left, df_right, on=['Driver', 'Spend % Change'], how='inner')

    if df_ins.empty:
        return config

    _, table_str, _, _ = readout_utils.table_to_text(pd.DataFrame(), df_ins, take_head=False)

    config = {
        "query": "You are given a dataset containing spend and KPIs % change by driver.",
        "instruction": f"From the given data,\n"
                       f"- Identify drivers whose margin contribution % exceeds spend increase % and margin ROI continues to grow.\n"
                       f"- Identify drivers whose margin contribution % is lower than spend increase % and margin ROI slightly drops.\n"
                       f"- Identify drivers whose margin impact % is smaller than spend decrease % and efficiency improves.\n"
                       f"Then, generate a business insight summary describing these drivers spend–margin efficiency patterns.\n",
        "data": df_ins.to_json(orient="records"),
    }
    return config


def spend_kpi_scatter(data, levels):
    df = viz_utils.get_planner_data(data, levels)
    if df.empty:
        return
   # Pair A: Spend % vs Margin % Change
    margin_col = next((c for c in ['Margin % Change', 'KPI Value % Change'] if c in df.columns))
    eff_col = next((c for c in ['Margin ROI % Change', 'Cost Per Acquisition % Change', 'Marginal ROI % Change'] if c in df.columns))

    df_left, left_nochg, left_nomeas, _ = viz_utils.split_for_pair(df, margin_col)
    # Pair B: Spend % vs Margin ROI % Change
    df_right, right_nochg, right_nomeas, _ = viz_utils.split_for_pair(df, eff_col)
    left_col1, left_col2 = st.columns([5,1])

    with left_col1:
        with st.expander("Show data table"):
            st.dataframe(df_left, use_container_width=True)
        planner_change_fair_share(df_left, "Spend % Change", margin_col,
            "Driver performance: Spend % vs Margin % Change")
    with left_col2:
        st.markdown("<div style='height: 120px'></div>", unsafe_allow_html=True)
        if left_nomeas:
            st.markdown("**Drivers with no measurable change (<0.1%)**")
            st.markdown("\n".join(f"- {d}" for d in left_nomeas))
        if left_nochg:
            st.markdown("**Drivers with no change**")
            st.markdown("\n".join(f"- {d}" for d in left_nochg))

    right_col1, right_col2 = st.columns([5, 1])
    with right_col1:
        with st.expander("Show data table"):
            st.dataframe(df_right, use_container_width=True)
        planner_change_fair_share(df_right, "Spend % Change", eff_col,
                                  "Driver performance: Spend % vs Margin ROI % Change")
    with right_col2:
        st.markdown("<div style='height: 120px'></div>", unsafe_allow_html=True)
        if right_nomeas:
            st.markdown("**Drivers with no measurable change (<0.1%)**")
            st.markdown("\n".join(f"- {d}" for d in right_nomeas))
        if right_nochg:
            st.markdown("**Drivers with no change**")
            st.markdown("\n".join(f"- {d}" for d in right_nochg))


def plot_football_field_altair(
    df, sort_lowest_start_top=True, title="Drivers' Diminishing Returns Landscape"
):
    endzone = 100
    pad = 50
    hist_col = "historical"
    forecast_col = "forecast"
    start_col = "start"
    end_col = "end"
    name_col = "driver"
    cap_value = 300

    df = df.sort_values(start_col).reset_index(drop=True) if sort_lowest_start_top else df.reset_index(drop=True)
    df = df.copy()
    df["y"] = list(range(len(df)))[::-1] if sort_lowest_start_top else list(range(len(df)))
    df["start_capped"] = df[start_col].clip(-cap_value, cap_value)
    df["end_capped"] = df[end_col].clip(-cap_value, cap_value)
    df["start_flag"] = df[start_col].apply(
        lambda x: "<-300" if x < -cap_value else (">300" if x > cap_value else "")
    )
    df["end_flag"] = df[end_col].apply(
        lambda x: "<-300" if x < -cap_value else (">300" if x > cap_value else "")
    )
    df["arrow_shape"] = df.apply(
        lambda r: "triangle-right" if r["end_capped"] >= r["start_capped"] else "triangle-left",
        axis=1,
    )

    points = alt.Chart(df).mark_circle(size=80, color="lightskyblue").encode(
        x=alt.X(
            "start_capped:Q", title="",
            scale=alt.Scale(domain=[-(endzone + pad), endzone + pad]),
        ),
        y=alt.Y("y:O", axis=None),
        tooltip=[
            alt.Tooltip(name_col, title="Driver"),
            alt.Tooltip(hist_col, title="Historical"),
            alt.Tooltip(forecast_col, title="Forecast"),
        ],
    )
    lines = alt.Chart(df).mark_rule(color="lightskyblue", strokeWidth=2).encode(
        x="start_capped",
        x2="end_capped",
        y="y:O",
        tooltip=[
            alt.Tooltip(name_col, title="Driver"),
            alt.Tooltip(hist_col, title="Historical"),
            alt.Tooltip(forecast_col, title="Forecast"),
        ],
    )
    arrowheads = alt.Chart(df).mark_point(size=80, color="lightskyblue").encode(
        x="end_capped",
        y="y:O",
        shape=alt.Shape("arrow_shape:N", scale=None),
        tooltip=[],
    )
    df["end_annotation"] = df.apply(
        lambda r: "Driver ROI curve suggests unsustainable low returns"
        if r[end_col] > cap_value
        else (
            "Driver ROI curve extends far below typical spend range"
            if r[end_col] < -cap_value
            else ""
        ),
        axis=1,
    )
    annotations = alt.Chart(df[df["end_annotation"] != ""]).mark_text(
        dx=15, fontSize=12, color="red", align="left"
    ).encode(
        x="end_capped:Q",
        y="y:O",
        text="end_annotation",
        tooltip=[],
    )
    df["label_x"] = df.apply(
        lambda r: r["start_capped"] - 5
        if r["end_capped"] >= r["start_capped"]
        else r["start_capped"] + 5,
        axis=1,
    )
    labels = alt.Chart(df).mark_text(dy=-5, fontSize=11, color="black", align="right").encode(
        x="label_x:Q",
        y="y:O",
        text=name_col,
        tooltip=[],
    )
    ref_data = pd.DataFrame({
        "x": [-endzone, 0, endzone],
        "label": ["Marginal ROI = 1", "Optima", "Marginal ROI = 1"],
    })
    ref_lines = alt.Chart(ref_data).mark_rule(strokeDash=[4, 2], color="red").encode(x="x:Q")
    ref_labels = alt.Chart(ref_data).mark_text(
        dy=-180, fontSize=11, color="red"
    ).encode(x="x:Q", text="label")

    chart = (points + lines + arrowheads + labels + annotations + ref_lines + ref_labels).properties(
        width=800,
        height=max(400, 25 * len(df)),
        title=alt.TitleParams(text=title, anchor="start", fontSize=20),
        padding={"left": 10, "right": 30, "top": 20, "bottom": 10},
    )
    chart = chart.configure(background="transparent")
    return chart


def plot_diminishing_return_instruct(client_code, model_group_id, data, levels):
    """Insight instruction for the diminishing-return view. '' when empty. No st.* calls."""
    planner_data = viz_utils.get_planner_data(data, levels)
    df, _ = viz_utils.driver_curves_data(client_code, model_group_id, planner_data)
    if df.empty:
        return ''
    _, table_str, _, _ = readout_utils.table_to_text(pd.DataFrame(), df, take_head=False)
    return (
        f"Classify each driver's historical and forecast spend positions relative to the optimal spend "
        f"using the following logic:\n"
        f"- < -105%: Spend too low (margin ROI < 1).\n"
        f"- -105% to -10%: Below optimal — opportunity to increase spend.\n"
        f"- -10% to 10%: Near optimal — returns-maximizing.\n"
        f"- 10% to 105%: Above optimal — diminishing returns.\n"
        f"- > 105%: Too high (margin ROI < 1).\n"
        f"Compare start (historical) and end (forecast) spend positions to assess whether spend is "
        f"moving toward or away from optimal.\n"
        f"Summarize the insight in natural business language rather than strict classifications.\n"
        f"Focus on:\n"
        f"- How spend is positioned relative to the optimal level (too high, below, near optimal, etc.).\n"
        f"- The trend or movement (improving toward, drifting away, stable, etc.).\n"
        f"- Optional short guidance (e.g., 'consider moderating spend', 'reallocation could be explored', etc.).\n"
        f"Data:\n{table_str}"
    )


def plot_diminishing_return_instruct_for_insights(client_code, model_group_id, data, levels):
    """Insight instruction for the diminishing-return view. '' when empty. No st.* calls."""
    planner_data = viz_utils.get_planner_data(data, levels)
    df, _ = viz_utils.driver_curves_data(client_code, model_group_id, planner_data)
    config = {}
    if df.empty:
        return config
    _, table_str, _, _ = readout_utils.table_to_text(pd.DataFrame(), df, take_head=False)

    config = {
        "query": "You are given a dataset containing historical and forecast spend by driver.",
        "instruction": f"Classify each driver's historical and forecast spend positions relative to the optimal spend "
                        f"using the following logic:\n"
                        f"- < -105%: Spend too low (margin ROI < 1).\n"
                        f"- -105% to -10%: Below optimal — opportunity to increase spend.\n"
                        f"- -10% to 10%: Near optimal — returns-maximizing.\n"
                        f"- 10% to 105%: Above optimal — diminishing returns.\n"
                        f"- > 105%: Too high (margin ROI < 1).\n"
                        f"Compare start (historical) and end (forecast) spend positions to assess whether spend is "
                        f"moving toward or away from optimal.\n"
                        f"Summarize the insight in natural business language rather than strict classifications.\n"
                        f"Focus on:\n"
                        f"- How spend is positioned relative to the optimal level (too high, below, near optimal, etc.).\n"
                        f"- The trend or movement (improving toward, drifting away, stable, etc.).\n"
                        f"- Optional short guidance (e.g., 'consider moderating spend', 'reallocation could be explored', etc.).\n",
        "data": df.to_json(orient="records"),
    }
    return config


def plot_diminishing_return_by_spend_level(client_code, model_group_id, data, levels):
    """Returns (df, chart).  Use plot_diminishing_return_instruct(...) for the insight text."""
    planner_data = viz_utils.get_planner_data(data, levels)
    df, _ = viz_utils.driver_curves_data(client_code, model_group_id, planner_data)
    if df.empty:
        return pd.DataFrame(), None
    chart = plot_football_field_altair(df)
    return df, chart


def get_spend_share_principle_plot(data):
    if data.empty:
        return

    fig = go.Figure()

    with st.expander("Show data table"):
        expander_data = data[
            [
                "marketingChannel",
                "maintainSpendShareRangeLeft",
                "medianPoint",
                "maintainSpendShareRangeRight",
                "historical_spend_share",
                "forecast_spend_share",
            ]
        ]
        st.dataframe(expander_data, use_container_width=True)

    for idx, row in data.iterrows():
        sample = [
            row["maintainSpendShareRangeLeft"],
            row["maintainSpendShareRangeLeft"],
            row["medianPoint"],
            row["maintainSpendShareRangeRight"],
            row["maintainSpendShareRangeRight"],
        ]
        cdata = [[
            row["maintainSpendShareRangeLeft"],
            row["medianPoint"],
            row["maintainSpendShareRangeRight"],
        ]]
        fig.add_trace(
            go.Box(
                y=sample,
                x=[row["marketingChannel"]] * len(sample),
                orientation="v",
                width=0.4,
                boxpoints=False,
                hoverinfo="skip",
                showlegend=False,
                marker_color="#8FB8E0",
                fillcolor="#BBDDF2",
            )
        )
        fig.add_trace(
            go.Scatter(
                y=[row["medianPoint"]],
                x=[row["marketingChannel"]],
                mode="markers",
                marker=dict(size=12, opacity=0),
                showlegend=False,
                hovertemplate=(
                    "Marketing Channel: %{x}<br>"
                    "Lower Bound: %{customdata[0]:.2f}%<br>"
                    "Median: %{customdata[1]:.2f}%<br>"
                    "Upper Bound: %{customdata[2]:.2f}%<extra></extra>"
                ),
                customdata=cdata,
            )
        )

    fig.add_trace(
        go.Scatter(
            x=data["marketingChannel"],
            y=data["historical_spend_share"],
            mode="markers",
            name="Historical",
            marker=dict(symbol="circle", size=10, color="#6C757D"),
            hovertemplate="Marketing Channel: %{x}<br>Historical Spend: %{y:.2f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=data["marketingChannel"],
            y=data["forecast_spend_share"],
            mode="markers",
            name="Forecast",
            marker=dict(symbol="diamond", size=10, color="#FF8C42"),
            hovertemplate="Marketing Channel: %{x}<br>Forecast Spend: %{y:.2f}%<extra></extra>",
        )
    )

    fig.update_layout(
        height=550,
        title=dict(
            text=(
                "Spend Share Principle by Marketing Channel<br>"
                "<span style='font-weight:400; font-size:0.9em;'>"
                "Hover the Median Line to see Lower Bound / Median / Upper Bound of Spend Share Principle."
                "</span>"
            )
        ),
        yaxis_title="Spend Share %",
        xaxis_title="Marketing Channel",
        xaxis=dict(type="category"),
        hovermode="closest",
    )
    st.plotly_chart(fig)


def planner_pct_bar_instruct(client_code, model_group_id, m_df, levels, core_dim, overall_data, planner_data_all, mode):
    """Insight instruction for the planner pct-bar view. No st.* calls."""
    agg, _, _, _, _ = viz_utils.planner_pct_bar_data(client_code, model_group_id, m_df, levels, core_dim, overall_data, planner_data_all, mode)
    _, table_str, _, _ = readout_utils.table_to_text(pd.DataFrame(), agg)
    return (
        f"Based on the data:\n"
        f"Report how many drivers show spend increase, no change, and spend decrease, "
        f"along with their % of total drivers.\n"
        f"For increase and decrease groups, check if any spend band accounts for 60%+ of total spend "
        f"in that category.\n"
        f"Summarize the overall budget change pattern – is spend concentrated in a few bands "
        f"(e.g., 'No change') or more evenly distributed?\n"
        f"Write the output as a clear business-style insight summary.\n"
        f"Data: {table_str}"
    )


def planner_pct_bar_instruct_for_insights(client_code, model_group_id, m_df, levels, core_dim, overall_data, planner_data_all, mode):
    """Insight instruction for the planner pct-bar view. No st.* calls."""
    agg, _, _, _, _ = viz_utils.planner_pct_bar_data(client_code, model_group_id, m_df, levels, core_dim, overall_data, planner_data_all, mode)
    _, table_str, _, _ = readout_utils.table_to_text(pd.DataFrame(), agg)
    if agg.empty:
        return {}
    config = {"query": "You are given a dataset grouped by Spend Change Group.",
              "instruction": f"Based on the data:\n"
                             f"Report how many drivers show spend increase, no change, and spend decrease, "
                             f"along with their % of total drivers.\n"
                             f"For increase and decrease groups, check if any spend band accounts for 60%+ of total spend "
                             f"in that category.\n"
                             f"Summarize the overall budget change pattern – is spend concentrated in a few bands "
                             f"(e.g., 'No change') or more evenly distributed?\n"
                             f"Write the output as a clear business-style insight summary.\n",
              "data":agg.to_json(orient="records"),
    }
    return config


def planner_pct_bar(client_code, model_group_id, m_df, levels, core_dim, overall_data, planner_data_all, mode):
    """Returns (agg_df, chart).  Use planner_pct_bar_instruct(...) for the insight text.  No st.* calls."""
    agg, line_df, labels, _, _ = viz_utils.planner_pct_bar_data(client_code, model_group_id, m_df, levels, core_dim, overall_data, planner_data_all, mode)

    if mode == "pct_change":
        chart_title = "Spend % Change Groups vs Spend/Margin Distribution"
        x_title = "Spend % Change Group"
    else:
        chart_title = "Spend Amount Groups vs Spend/Margin Distribution"
        x_title = "Spend Amount Group"

    bar = alt.Chart(agg).mark_bar(color="#CCCCCC").encode(
        x=alt.X(
            "Spend Change Group:N",
            sort=labels,
            axis=alt.Axis(labelAngle=-30),
            title=x_title,
        ),
        y=alt.Y("driver_pct:Q", title="% of driver counts"),
        tooltip=["driver_count", "driver_pct", "spend_forecast", "margin_forecast"],
    )
    line = alt.Chart(line_df).mark_line(point=True).encode(
        x=alt.X("Spend Change Group:N", sort=labels),
        y=alt.Y("Percentage:Q", title="% of forecast spend/margin"),
        color=alt.Color(
            "Metric:N",
            title="Metric",
            scale=alt.Scale(
                domain=["spend_pct", "margin_pct"],
                range=["#1f77b4", "#2ca02c"],
            ),
        ),
    )
    chart = (bar + line).properties(
        height=450,
        title=chart_title,
        padding={"left": 5, "right": 20, "top": 20, "bottom": 10},
    )
    chart = chart.configure(background="transparent").configure_view(stroke=None, fill="transparent")

    return agg, chart


def planner_spend_change_all_instruct(data):
    """Insight instruction for the spend-incremental view. No st.* calls."""
    df, _ = viz_utils.planner_spend_change_all_data(data)
    _, table_str, _, _ = readout_utils.table_to_text(pd.DataFrame(), df, take_head=False)
    return (
        f"From the data: List the top 3 drivers with the largest spend increases and spend decreases, "
        f"showing for each:\n"
        f"- % of spend change they represent,\n"
        f"- their contribution to margin change,\n"
        f"- and change in margin ROI.\n"
        f"For each spend level, summarize the pattern of spend change.\n"
        f"Provide a clear, business-style insight summary.\n"
        f"Data: {table_str}"
    )


def planner_spend_change_all_instruct_for_insights(data):
    """Insight instruction for the spend-incremental view. No st.* calls."""
    df, _ = viz_utils.planner_spend_change_all_data(data)
    _, table_str, _, _ = readout_utils.table_to_text(pd.DataFrame(), df, take_head=False)
    if df.empty:
        return {}
    config = {
        "query": "You are given a dataset containing spend % change by driver.",
        "instruction": f"From the data: List the top 3 drivers with the largest spend increases and spend decreases, "
                       f"showing for each:\n"
                       f"- % of spend change they represent,\n"
                       f"- their contribution to margin change,\n"
                       f"- and change in margin ROI.\n"
                       f"For each spend level, summarize the pattern of spend change.\n"
                       f"Provide a clear, business-style insight summary.\n",
        "data": df.to_json(orient="records"),
    }
    return config


def planner_spend_change_all(data):
    """Returns (df, chart).  Use planner_spend_change_all_instruct(data) for the insight text.  No st.* calls."""
    df, _ = viz_utils.planner_spend_change_all_data(data)
    incremental_cols = [col for col in df.columns if "incremental" in col.lower()]
    tooltip_cols = ["Driver", "Spend % Change"] + incremental_cols

    bar = (
        alt.Chart(df)
        .mark_bar(color="#4B9CD3")
        .encode(
            y=alt.Y("Spend Incremental (MM):Q", title="Spend Incremental (MM)"),
            x=alt.X(
                "Driver:N",
                sort='-y',
                title="Driver",
                axis=alt.Axis(labelAngle=-80, labelFontSize=10),
            ),
            tooltip=tooltip_cols,
        )
        .properties(
            height=450,
            title="Spend Incremental by Driver",
            padding={"left": 5, "right": 20, "top": 10, "bottom": 10},
        )
    )
    bar = bar.configure(background="transparent").configure_view(stroke=None, fill="transparent")

    return df, bar


def render_planner_filters(planner_data, scenario, planner_data_sub):
    # col1, col2 = st.columns(2)
    # with col1:
    #     scenario_options = planner_data['Scenario Description'].unique().tolist()
    #     selected_scenario = st.selectbox(
    #         "Scenario",
    #         options=scenario_options,
    #         index = 0
    #     )
    # planner_data = planner_data[planner_data['Scenario Description'] == selected_scenario]

    # with col2:
    if 'KPI' not in planner_data.columns:
        return planner_data, ""
    planner_data = planner_data[planner_data['Scenario ID'] == scenario]
    available_kpis = planner_data_sub['KPI'].unique().tolist()
    selected_kpi = st.selectbox(
        "KPI",
        options=available_kpis,
        index = 0
    )
    planner_data = planner_data[planner_data['KPI'] == selected_kpi]
    planner_data = planner_data.dropna(axis=1, how='all')

    return planner_data, selected_kpi
