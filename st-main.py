import os
import streamlit as st

from src.model.filter_generator import generate_ner_filter
from src.model.readout import stream_response, process_data, response_generate, check_truncation
from src.model.common import PromptTemplates
from src.data.data_interface import ProcessIndicator
from src.utils import get_available_client_and_model_group


st.set_page_config(
    page_title='Ask Genome',
    layout='wide',
    page_icon="./assets/ap-logo.png",
)
left, center, right = st.columns([1, 3, 1])

if "thinking_content" not in st.session_state:
    st.session_state.thinking_content = ""
if "main_content" not in st.session_state:
    st.session_state.main_content = ""

# Side panel
with st.sidebar:
    commit = os.getenv("CI_COMMIT")
    st.markdown(f"Commit: `{commit}`")

    client_model_group_options = get_available_client_and_model_group()
    selected = st.selectbox("Client Code", client_model_group_options.keys())
    client_code, model_group_id = client_model_group_options[selected]

    os.environ["CLIENT_CODE"] = client_code
    os.environ["MODEL_GROUP_ID"] = str(model_group_id)

    st.markdown(f"Client Code: `{client_code}`")
    st.markdown(f"Model Group ID: `{model_group_id}`")

# Main panel
with center:
    if query := st.chat_input():
        st.session_state.thinking_content = ""
        st.session_state.main_content = ""
        st.chat_message("user", avatar=":material/person:").write(query)
        st.chat_message("assistant", avatar="./assets/ap-logo-margin.png").write("Thinking...")

        process_indicator = ProcessIndicator.from_local(client_code, model_group_id)
        readout_data = generate_ner_filter(client_code, model_group_id, query, process_indicator)

        context_str, data, benchmark_data, benchmark_str, planner_data, pretext, pivot_biz_table, \
            principle_pretext, overall_view, readout_adj = process_data(client_code, model_group_id, readout_data,
                                                                        process_indicator)

        with st.chat_message("assistant", avatar="./assets/ap-logo-margin.png"):
            with st.expander("Thinking...", expanded=False):
                placeholder_thinking = st.empty()
            placeholder_main = st.empty()

            found_think = False

            if context_str:
                # if client_code == 'LINKEDIN':
                #     st.session_state.main_content += "Response with query: "
                if readout_adj:
                    query = 'Generate insights with given context. '

                if check_truncation(query, context_str):
                    truncation_str = "(Note: Readout has been truncated.)"
                else:
                    truncation_str = ""

                for chunk in stream_response(query, context_str):
                    if not found_think:
                        st.session_state.thinking_content += chunk
                        if "</think>" in st.session_state.thinking_content:
                            found_think = True
                            before_think, after_think = st.session_state.thinking_content.split("</think>", 1)
                            st.session_state.thinking_content = before_think
                            st.session_state.main_content += after_think
                    else:
                        st.session_state.main_content += chunk
                # if client_code == 'LINKEDIN':
                #     st.session_state.main_content += "\n\nResponse without query: "
                #     found_think = False
                #     st.session_state.thinking_content = ""
                #     for chunk in stream_response('Generate insights with given context. ', context_str):
                #         if not found_think:
                #             st.session_state.thinking_content += chunk
                #             if "</think>" in st.session_state.thinking_content:
                #                 found_think = True
                #                 before_think, after_think = st.session_state.thinking_content.split("</think>", 1)
                #                 st.session_state.thinking_content = before_think
                #                 st.session_state.main_content += after_think
                #         else:
                #             st.session_state.main_content += chunk
                placeholder_thinking.write(st.session_state.thinking_content)
                placeholder_main.write(st.session_state.main_content)
                if context_str and not planner_data.empty:
                    processed_response = (f"{st.session_state.main_content} \n\n{pretext} \n\n{principle_pretext} "
                                          f"\n\n{benchmark_str}"
                                          f"Readout: {truncation_str} {context_str}")
                else:
                    if principle_pretext:
                        insight = response_generate('Derive genome principle insights based on given information. ',
                                                    principle_pretext)
                        if "</think>" in insight:
                            before_think, after_think = insight.split("</think>", 1)
                            insight = after_think
                        processed_response = (
                            f"{pretext} \n\n{st.session_state.main_content} \n\n### ROI Genome Insights:\n\n{insight} "
                            f"\n\n{benchmark_str}"
                            f"Readout: {truncation_str} {context_str}"
                            f"Principle pretext: {principle_pretext}")
                    else:
                        insight = principle_pretext
                        processed_response = (f"{pretext} \n\n{st.session_state.main_content} \n\n"
                                              f"\n\n{benchmark_str}"
                                              f"Readout: {truncation_str} {context_str}"
                                              f"Principle pretext: {insight}")
            else:
                if pretext:
                    processed_response = pretext
                else:
                    rejection_message = readout_data.ner_filters.get("rejection_message")
                    processed_response = rejection_message if rejection_message \
                        else PromptTemplates().rejection_response

            with placeholder_main:
                st.write(processed_response)

if query:
    if len(overall_view):
        st.markdown("## Overall Table")
        st.dataframe(overall_view)

    if len(pivot_biz_table):
        st.markdown("## Pivot Biz Table")
        st.dataframe(pivot_biz_table)

    if len(data):
        st.markdown("## Pivot Table")
        st.dataframe(data)

    if len(benchmark_data):
        st.markdown("## Benchmark Data")
        st.dataframe(benchmark_data)

    if len(planner_data):
        st.markdown("## Planner Data")

        inc_data = planner_data[planner_data['Increase/Decrease'] == 'Increase'].drop(columns=['Increase/Decrease'])
        dec_data = planner_data[planner_data['Increase/Decrease'] == 'Decrease'].drop(columns=['Increase/Decrease'])

        if not inc_data.empty:
            st.markdown("### Driver(s) with Spending Increase")
            st.dataframe(inc_data)

        # Display the 'dec' data table if it exists
        if not dec_data.empty:
            st.markdown("### Driver(s) with Spending Decrease")
            st.dataframe(dec_data)
