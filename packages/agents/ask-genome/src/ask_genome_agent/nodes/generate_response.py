"""Turning a subquery's analysis into an answer a person can read.

Ported from pages/insights_only.py in ask-genome-core. The GPT readout postprocessing built (see
process_data_adapter.insight_input) goes to the model through upstream's own `analytics_generator`
prompt, the principle text gets a call of its own, and the pretext and benchmark go around the
replies. The prompt and each client's inserts come from the vendored prompt.yaml, so they change
when upstream's do.

A failed call falls back to the readout, which has all the figures, and is recorded on the entry.

The answer is also what a dependent subquery gets.

One difference from the page: a few requirements of our own go after upstream's (see
prompts.readout_requirements), including the resolved period, so "last year" means the right year.
"""

import calendar
import logging
from typing import Any

from orchestration_core import AgentDependencies

from ask_genome_agent.prompts import readout_requirements
from ask_genome_agent.state import AskGenomeState
from ask_genome_agent.vendor.ask_genome_core.model import common as upstream_common
from ask_genome_agent.vendor.ask_genome_core.model import readout_utils

logger = logging.getLogger(__name__)

_TOOL = "analytics_generator"
_PRINCIPLE_QUERY = "Summarize insights."


async def generate_response(
    state: AskGenomeState, dependencies: AgentDependencies
) -> dict[str, Any]:
    subquery = state["plan"].subqueries[state["active_index"]]
    ner_state = dict(state.get("ner_state", {}))
    entry = dict(ner_state.get(subquery.id) or {})
    if not entry:
        return {}

    pretext = (entry.get("fixtext") or "").strip()
    if not entry.get("answerable"):
        # Nothing for the model to read, so the pretext alone, as the page does.
        if not pretext:
            return {}
        entry["answer"] = pretext
        ner_state[subquery.id] = entry
        return {"ner_state": ner_state, "last_subquery_result": pretext}

    planner = bool(entry.get("is_planner_answer"))
    source = entry.get("insight_context") or entry.get("response_context") or ""
    query = entry.get("insight_query") or subquery.query
    # Scenarios name their own period, so for them the asked one only relabels them.
    period = "" if planner else asked_period(entry.get("ner_results") or {})

    try:
        prose = await insight_call(query, source, dependencies, period=period)
    except Exception as error:
        logger.warning("response generation failed for %s: %s", subquery.id, error)
        entry["response_error"] = f"{type(error).__name__}: {error}"
        prose = (entry.get("readout") or entry.get("response_context") or "").strip()

    principle = (entry.get("principle_pretext") or "").strip()
    principle_insights = ""
    if principle:
        try:
            principle_insights = await insight_call(_PRINCIPLE_QUERY, principle, dependencies)
        except Exception as error:
            logger.warning("principle summary failed for %s: %s", subquery.id, error)
            entry["principle_error"] = f"{type(error).__name__}: {error}"
            principle_insights = principle

    entry["answer"] = assemble_answer(
        prose,
        pretext,
        principle_insights,
        entry.get("benchmark_text") or "",
        planner=planner,
    )
    ner_state[subquery.id] = entry
    # Dependents resolve "its" against what the user was told.
    return {"ner_state": ner_state, "last_subquery_result": entry["answer"]}


async def insight_call(
    query: str, readout: str, dependencies: AgentDependencies, *, period: str = ""
) -> str:
    """Ported from readout.py:response_generate_insight, with the call through the orchestrator's
    model. One user message holding the whole prompt, as upstream sends it. Our requirements go in
    through custom_instructions, so upstream's defaults stay intact."""

    templates = upstream_common.PromptTemplates()
    prompt = templates.build_readout_prompt(
        query,
        readout_utils.convert_display_format(readout),
        custom_instructions=readout_requirements(templates.get_default_requirements(_TOOL), period),
        tool=_TOOL,
    )
    result = await dependencies.model.ainvoke([("user", prompt)])
    text = result.content if isinstance(result.content, str) else str(result.content)
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    if not text.strip():
        raise ValueError("the model returned an empty response")
    return text.strip()


def assemble_answer(
    prose: str, pretext: str, principle: str, benchmark: str, *, planner: bool
) -> str:
    """Ported from the assembly in pages/insights_only.py, without its display labels.

    The pretext usually leads as the headline figure. For a planner answer it is the
    reference-only notice, so it comes last as a caveat instead.
    """

    if planner:
        parts = [prose, principle, benchmark, pretext]
    else:
        parts = [pretext, prose, principle, benchmark]
    return "\n\n".join(p.strip() for p in parts if p and p.strip())


def asked_period(ner_results: dict[str, Any]) -> str:
    """The period the question resolved to, in words, or empty if it named none."""

    start, end = str(ner_results.get("start", "")), str(ner_results.get("end", ""))
    if not (len(start) == len(end) == 6 and start.isdigit() and end.isdigit()):
        return ""
    if start[:4] == end[:4] and start[4:] == "01" and end[4:] == "12":
        return start[:4]

    def month(value: str) -> str:
        return f"{calendar.month_abbr[int(value[4:])]} {value[:4]}"

    return month(start) if start == end else f"{month(start)} to {month(end)}"
