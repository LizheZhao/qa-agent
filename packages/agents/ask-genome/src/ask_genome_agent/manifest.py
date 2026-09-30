"""Static package metadata."""

from orchestration_core import AgentManifest

AGENT_MANIFEST = AgentManifest(
    agent_id="ask_genome",
    version="0.7.0",
    description=(
        "Answer marketing-analytics questions about a specific client's modelled data: ROI, "
        "spend, contribution, source of change, cost per, activity and response, broken down by "
        "the client's own channel hierarchy, business units, KPIs, countries and time periods. "
        "Plans a question into subqueries, resolves dependencies between them, resolves each "
        "filter (intent, metric, time range, hierarchy and tagging columns), asking the user "
        "to clarify anything genuinely ambiguous, then reads the client's data and answers in "
        "plain English from the figures it finds. Also answers budget allocation and "
        "scenario planning questions (where to move spend, which drivers to grow or cut, where "
        "returns flatten) from the client's published planner scenarios, reporting those "
        "scenarios rather than optimising anything itself. See README.md for current limits."
    ),
    capabilities=(
        "query_planning",
        "dependency_resolution",
        "question_understanding",
        "filter_resolution",
        "metric_reporting",
        "hierarchy_breakdown",
        "budget_scenario_reporting",
        "response_generation",
    ),
    tool_ids=(),
)
