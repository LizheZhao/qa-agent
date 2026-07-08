import os
import sys
import numpy as np
import pandas as pd
import streamlit as st


def reset_graph_state():
    st.session_state.query = None
    st.session_state.thread_id = None
    st.session_state.graph_result = None
    st.session_state.awaiting_clarification = False
    st.session_state.clarification_fields = None
    st.session_state.ner_results = None
    st.session_state.field_status = None
    st.session_state.final_filters = None
    st.session_state.client_code = None
    st.session_state.model_group_id = None


def new_thread_id() -> str:
    THREAD_COUNTER_KEY = "_thread_counter"
    st.session_state[THREAD_COUNTER_KEY] += 1
    return f"thread-{st.session_state[THREAD_COUNTER_KEY]}"


def render_filter_card(field_name: str, value, status: str):
    """Render a single filter field with its verified/auto-confirmed tag."""
    display_value = ", ".join(value) if isinstance(value, list) else str(value)
    tag_class = "tag-verified" if status == "verified" else "tag-auto"
    tag_label = "✓ verified" if status == "verified" else "✓ auto-confirmed"

    st.markdown(
        f"""
        <div class="filter-card">
            <div>
                <strong>{field_name.replace('_', ' ').title()}</strong>:&nbsp; {display_value}
            </div>
            <span class="{tag_class}">{tag_label}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
