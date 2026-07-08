"""Agent orchestrator page (multi-page Streamlit).

Same workflow as orchestrator_page.py, but the whole run is driven by one LangGraph
agent graph (src/agents): plan -> per-sub-query agents -> re-plan on failure -> finalize.
This page only renders graph state and forwards clarification answers back into the
graph with Command(resume=...). orchestrator_page.py is intentionally untouched.

The graph checkpointer lives in this browser session, so refreshing the page discards
an in-flight run.
"""
import os

import streamlit as st
from langgraph.types import Command

from src.agents import create_agent_app
from src.utils import get_available_client_and_model_group
from src.render.streamlit_utils import new_thread_id
from src.integrations.vendor import anthropic_override, resolve_vendor

st.set_page_config(page_title="Ask Genome - Agents", layout="wide")

SESSION_DEFAULTS = {
    "agent_query": None,
    "agent_thread": None,
    "agent_await": False,
    "agent_fields": None,
    "agent_ner_current": None,
    "agent_interrupt_meta": None,
    "_thread_counter": 0,
}
for _key, _default in SESSION_DEFAULTS.items():
    if _key not in st.session_state:
        st.session_state[_key] = _default

if st.session_state.get("agent_graph") is None:
    st.session_state.agent_graph = create_agent_app()

# --------------------------------------------------------------------------- sidebar
with st.sidebar:
    client_model_group_options = get_available_client_and_model_group()
    selected = st.selectbox("Client Code", options=client_model_group_options.keys(),
                            key="agent_client_model_group")
    client_code, model_group_id = client_model_group_options[selected]
    os.environ["CLIENT_CODE"] = client_code
    os.environ["MODEL_GROUP_ID"] = str(model_group_id)
    st.markdown(f"Client Code: `{client_code}`")
    st.markdown(f"Model Group ID: `{model_group_id}`")

    st.divider()
    st.selectbox("LLM Vendor", ["openai", "gemini", "claude"], index=0, key="agent_llm_vendor")
    os.environ["EXTERNAL_VENDOR"] = st.session_state.agent_llm_vendor
    st.toggle("Use external llm for filter extraction", value=False, key="enterprise_ai_ner")
    st.toggle("LLM filter extraction priority", value=False, key="llm_priority")
    if anthropic_override():
        st.caption(f"Agent LLM route: `{resolve_vendor(st.session_state.agent_llm_vendor)}` "
                   "(ANTHROPIC_API_KEY set)")


def _reset_agent_state() -> None:
    for key, default in SESSION_DEFAULTS.items():
        if key != "_thread_counter":
            st.session_state[key] = default


def _reset_on_context_change(current_client_code: str, current_model_group_id: int) -> None:
    context_key = (current_client_code, current_model_group_id)
    if st.session_state.get("_agent_context") != context_key:
        st.session_state["_agent_context"] = context_key
        _reset_agent_state()


_reset_on_context_change(client_code, model_group_id)


def _graph_config() -> dict:
    return {"configurable": {
        "thread_id": st.session_state.agent_thread,
        "client_code": client_code,
        "model_group_id": int(model_group_id),
        "enterprise_ai_ner": st.session_state.enterprise_ai_ner,
        "llm_priority": st.session_state.get("llm_priority", False),
        "llm_vendor": st.session_state.agent_llm_vendor,
    }}


def _drive(graph, payload) -> None:
    """Run the graph until it interrupts for clarification or finishes."""
    config = _graph_config()
    with st.spinner("Working..."):
        graph.invoke(payload, config=config)
    snapshot = graph.get_state(config)
    if snapshot.next:
        interrupt_value = snapshot.tasks[0].interrupts[0].value
        st.session_state.agent_await = True
        st.session_state.agent_fields = interrupt_value["clarification_fields"]
        st.session_state.agent_ner_current = interrupt_value.get("ner_results", {})
        st.session_state.agent_interrupt_meta = {"index": interrupt_value["sub_query_index"],
                                                 "query": interrupt_value["query"]}
    else:
        st.session_state.agent_await = False


def _render_payload(payload: dict) -> None:
    label = f"Sub-query {payload['index'] + 1}: {payload['kind']}"
    with st.expander(label, expanded=True):
        st.caption(payload["query"])
        if payload.get("rejected"):
            st.warning(payload.get("message", "This sub-query could not be answered."))
            return
        if payload["kind"] == "analytical":
            with st.chat_message("assistant", avatar="./assets/ap-logo-margin.png"):
                st.markdown(payload.get("response", ""))
        elif payload.get("response"):
            st.write(payload["response"])
        for table in payload.get("tables", []):
            if table is not None and not table.empty:
                st.dataframe(table, use_container_width=True)


def _render_clarification_form() -> None:
    meta = st.session_state.agent_interrupt_meta or {}
    fields = st.session_state.agent_fields
    current = st.session_state.agent_ner_current or {}
    st.markdown(f"#### Sub-query {meta.get('index', 0) + 1} needs confirmation")
    st.caption(meta.get("query", ""))
    with st.form("agent_clarify"):
        for field_name, options in fields.items():
            current_value = current.get(field_name, [])
            if isinstance(current_value, str):
                current_value = [current_value]
            default = [option for option in options if option in current_value]
            st.multiselect(field_name.replace("_", " ").title(), options=options,
                           default=default, key=f"agent_clarify_{field_name}")
        submitted = st.form_submit_button("Confirm selections")
    if submitted:
        selections = {f: st.session_state[f"agent_clarify_{f}"] for f in fields}
        empty = [f for f, v in selections.items() if not v]
        if empty:
            st.error(f"Please select at least one option for: {', '.join(empty)}")
        else:
            _drive(st.session_state.agent_graph, Command(resume=selections))
            st.rerun()


# --------------------------------------------------------------------------- main flow
st.title("Ask-Genome (Agents)")
query = st.chat_input("Ask a question")
if query:
    _reset_agent_state()
    st.session_state.agent_query = query
    st.session_state.agent_thread = new_thread_id()
    _drive(st.session_state.agent_graph, {"query": query})

if st.session_state.agent_thread is not None:
    values = st.session_state.agent_graph.get_state(_graph_config()).values
    st.chat_message("user").write(st.session_state.agent_query)
    if values.get("is_multi"):
        st.caption(f"Rephrased: {values.get('rephrased_query', '')}")
        st.caption(f"{len(values.get('subqueries', []))} sub-queries"
                   + (" (sequential)" if values.get("is_sequential") else ""))
    for note in values.get("replan_notes", []):
        st.caption(note)

    for payload in values.get("payloads", []):
        _render_payload(payload)

    if st.session_state.agent_await and st.session_state.agent_fields:
        _render_clarification_form()
