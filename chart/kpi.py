"""
chart/kpi.py
KPI summary cards and pretext trend chart — pure Streamlit renderers, no LLM calls.
"""

import streamlit as st
import altair as alt
import viz_utils as viz_utils
from streamlit_elements import elements, mui


# ---------------------------------------------------------------------------
# Shared card component (also used by orchestrator_page.py)
# ---------------------------------------------------------------------------

def metric_card(key, title, value, delta, width):
    if delta is None:
        delta_value = None
    else:
        try:
            delta_value = float(str(delta).strip().replace("%", ""))
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
        mui.Typography(title, variant="subtitle2", color="text.secondary")
        mui.Typography(str(value), variant="h5", fontWeight=700)
        mui.Typography(
            f"{arrow} {delta_text}".strip(),
            variant="body2",
            sx={"color": color},
        )


# ---------------------------------------------------------------------------
# KPI cards
# ---------------------------------------------------------------------------

def most_recent_kpi_card_new(intention, pretext_table,
                              selected_levels, time, direction='row',client_code=None, model_group_id=None):
    records, _ = viz_utils.most_recent_kpi_card_data(pretext_table, selected_levels,client_code, model_group_id)
    if intention in ['margin roi', 'performance', 'contribution']:
        spend_record = viz_utils.get_total_spend_data(client_code, model_group_id, selected_levels, time)
        records.append(spend_record)
    with elements("most_recent_kpi_cards"):
        with mui.Stack(direction=direction, spacing=2, sx={
            "justifyContent": "center",
            "alignItems": "center"}):
            for rec in records:
                metric_card(
                    key=rec['driver'],
                    title=f"{rec['driver']} – {rec['kpi_col']}",
                    value=rec['value_fmt'],
                    delta=rec['delta_fmt'],
                    width=400 if direction == 'row' else 300,
                )


# ---------------------------------------------------------------------------
# Pretext trend chart
# ---------------------------------------------------------------------------

def get_pretext_trend_new(intention, pretext_data,
                           metric_labels, metric_cols, sorted_times, selected_levels,client_code=None, model_group_id=None):
    overall_df_plot, _ = viz_utils.pretext_trend_data( pretext_data, metric_cols, sorted_times, selected_levels,client_code, model_group_id)
    overall_spend = viz_utils.pretext_spend_trend_data(client_code, model_group_id, sorted_times, selected_levels)
    metric = metric_labels[0]
    select_time = viz_utils.get_time_align_dict(sorted_times)
    time_label_dict = {v: k for k, v in select_time.items()}
    sorted_labels = [time_label_dict[k] for k in sorted_times if k in time_label_dict]
    title_metric = metric.title()
    if "roi" in metric.lower():
        title_metric = metric.lower().replace("roi", "ROI").title().replace("Roi", "ROI")

    x = alt.X("TimeLabel:N", title="Time", axis=alt.Axis(labelAngle=0), sort=sorted_labels)
    y = alt.Y("Value:Q", title=title_metric, axis=alt.Axis(tickCount=5, grid=True))
    color = alt.Color("Driver:N", sort=overall_df_plot['Driver'].unique().tolist(),
                      scale=alt.Scale(scheme="tableau10"))
    tooltip = [
        alt.Tooltip("Driver", title="Business Driver"),
        alt.Tooltip("Time", title="Time"),
        alt.Tooltip("ValueFormatted", title=title_metric),
    ]

    line_chart = (
        alt.Chart(overall_df_plot)
        .mark_line(point={"filled": True, "size": 80, "shape": 'square'}, interpolate="monotone", strokeWidth=2)
        .encode(x=x, y=y, color=color, tooltip=tooltip)
    )

    if intention in ['margin roi', 'performance']:
        overall_spend["Metric"] = "Spend"
        spend_bar = (
            alt.Chart(overall_spend)
            .mark_bar(opacity=0.3)
            .encode(
                x=x,
                y=alt.Y("Value:Q", title="Spend (MM)",
                        axis=alt.Axis(orient="right", tickCount=5, grid=False)),
                tooltip=[
                    alt.Tooltip("Driver", title="Business Driver"),
                    alt.Tooltip("Time", title="Time"),
                    alt.Tooltip("ValueFormatted", title="Spend (MM)"),
                ],
                fill=alt.Fill("Metric:N",
                              scale=alt.Scale(domain=["Spend"], range=["#999999"]),
                              legend=alt.Legend(title="")),
            )
        )
        chart = alt.layer(spend_bar, line_chart).resolve_scale(y='independent')
    else:
        chart = line_chart

    chart = chart.properties(
        width=500,
        height=250,
        title={
            "text": f"Overall {title_metric} Trend",
            "subtitle": ["Business Drivers performance over time"],
            "subtitleColor": "gray",
            "subtitleFontSize": 12,
            "anchor": "start",
        },
        padding={"left": 10, "right": 30, "top": 10, "bottom": 10},
    )
    chart = chart.configure(background="transparent").configure_view(fill="transparent")

    tactics_idx = overall_df_plot.columns.get_loc('Driver')
    expander_data = overall_df_plot.pivot(
        index=list(overall_df_plot.columns[:tactics_idx + 1]),
        columns='Metric',
        values='Value',
    ).reset_index()

    with st.expander("Show data table"):
        st.dataframe(expander_data, use_container_width=True)
    st.altair_chart(chart, use_container_width=True)
