# Vendored from ask-genome-core bb1e1637bca4:(extracted functions)
# by scripts/sync_ask_genome_core.py. Do not edit by hand; change EDITS there.


# from chart/common.py
def get_chart_set_inst(chart_configs: dict) -> str:
    charts_block = (
        "Analyze each data table thoroughly, then synthesize findings across all tables.\n"
        "Provide a structured answer.\n"
    )
    for chart_name, config in chart_configs.items():
        charts_block += f"""
    ### {chart_name}
    Data: {config['data']}
    Analysis focus: {config['instruction']}
    """
    return charts_block


# from src/model/visual_config_utils.py
def _generate_insights_prompt(chart_configs: dict, query: str, insight_topic) -> str:
    """
    Generate insights from chart datasets
    Method 1: Insights from combined raw data
    Method 2: Insights from individual chart insight
    """
    ## -- Method 1 -- ##
    charts_block = get_chart_set_inst(chart_configs)
    insight_query = (
        f"You are given several datasets about {insight_topic}. "
        f"Provide structured insights from datasets and then answer the user's question: "
        f"{query} "
        f"Write your response in markdown format."
    )
    return insight_query + "\n" + charts_block
