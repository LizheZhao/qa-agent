"""
chart/common.py
Shared utilities for the chart package: insight rendering helpers, the parallel
cached insight generator, and the section/table helpers that all chart modules
import from one place.
"""

import streamlit as st
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict
from streamlit_elements import elements, mui
import src.model.readout_utils as readout_utils
from src.model.readout import response_generate_chart


# ---------------------------------------------------------------------------
# LLM insight caching
# ---------------------------------------------------------------------------

def get_cached_insight(insight_id: str, system_prompt: str, instruct: str, llm_vendor: str) -> str:
    """Cached LLM call keyed by insight_id."""
    llm_gen = response_generate_chart(system_prompt, instruct, llm_vendor)
    if "</think>" in llm_gen:
        _, insight = llm_gen.split("</think>", 1)
        return insight.strip()
    return llm_gen.strip()


def generate_chart_insights_parallel_cached(
    chart_configs: Dict[str, Dict[str, str]],
    llm_vendor: str,
    max_workers: int | None = None,
) -> Dict[str, str]:
    """
    Run cached LLM insight generation in parallel, keyed by chart_id.

    chart_configs:
        {
          "chart_id": {
              "query":       "...",   # system prompt
              "instruction": "...",   # analytical focus
              "data":        "...",   # JSON string of the data
          },
          ...
        }

    Returns:
        {"chart_id": "insight text ...", ...}
    """
    if not chart_configs:
        return {}

    if max_workers is None:
        max_workers = min(8, len(chart_configs))

    chart_insights: Dict[str, str] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_chart_id = {}
        for chart_id, cfg in chart_configs.items():
            system_prompt = cfg["query"]
            instruct = (
                f"\n    Data: {cfg['data']}\n"
                f"    Analysis focus: {cfg['instruction']}\n    "
            )
            future = executor.submit(
                get_cached_insight,
                chart_id,
                system_prompt,
                instruct,
                llm_vendor,
            )
            future_to_chart_id[future] = chart_id

        for future in as_completed(future_to_chart_id):
            chart_id = future_to_chart_id[future]
            try:
                chart_insights[chart_id] = future.result()
            except Exception as exc:
                chart_insights[chart_id] = f"Insight generation failed: {exc}"

    return chart_insights


# ---------------------------------------------------------------------------
# Streamlit rendering helpers
# ---------------------------------------------------------------------------

def render_chart_insight_box(title: str, insight_text: str, height_px: int = 260):
    box = st.container(height=height_px)
    with box:
        st.markdown(f"##### {title}")
        st.markdown(insight_text)


def _section(title: str):
    st.markdown(f'<div class="sec-title">{title}</div>', unsafe_allow_html=True)
    return st.container()


def get_chart_set_inst(chart_configs: dict) -> str:
    charts_block = (
        "Analyze each data table thoroughly, then synthesize findings across all tables.\n"
        "Provide a detailed and structured answer.\n"
    )
    for chart_name, config in chart_configs.items():
        charts_block += f"""
    ### {chart_name}
    Data: {config['data']}
    Analysis focus: {config['instruction']}
    """
    return charts_block


def get_insights_set_inst(insights_configs: dict) -> str:
    block = (
        "Summarize the following chart-level insights into a cohesive narrative.\n"
        "Provide a structured answer.\n"
    )
    for title, insights in insights_configs.items():
        block += f"""
    ### {title}
    Analysis focus: {insights}
    """
    return block


def table_element(df: pd.DataFrame, title: str, key: str):
    if_hide_footer = len(df) <= 5
    columns = [{"field": col, "headerName": col, "flex": 1} for col in df.columns]
    rows = [{"id": i, **row.to_dict()} for i, (_, row) in enumerate(df.iterrows())]

    with elements(key):
        mui.Typography(
            title,
            variant="subtitle1",
            sx={
                "fontWeight": 600,
                "marginBottom": "6px",
                "textAlign": "left",
            },
        )
        mui.DataGrid(
            rows=rows,
            columns=columns,
            pageSize=5,
            rowsPerPageOptions=[5],
            autoHeight=True,
            disableColumnMenu=True,
            hideFooter=if_hide_footer,
            headerHeight=36,
            sx={
                "borderRadius": 2,
                "border": "1px solid #E0E0E0",
                "boxShadow": "0 2px 8px rgba(15, 23, 42, 0.08)",
                "& .MuiDataGrid-columnHeaders": {
                    "backgroundColor": "#F5F5F5",
                    "fontWeight": 600,
                },
                "& .MuiDataGrid-cell": {
                    "fontSize": "0.85rem",
                },
            },
        )
