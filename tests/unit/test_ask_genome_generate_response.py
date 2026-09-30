"""Response generation: what the model is given, the order the answer comes back in, and that a
failed call still reports the figures."""

from __future__ import annotations

from typing import Any

import pytest
from ask_genome_agent.contracts import AskGenomePlan, Subquery, SubqueryKind
from ask_genome_agent.nodes.aggregate_results import aggregate_results
from ask_genome_agent.nodes.generate_response import (
    asked_period,
    assemble_answer,
    generate_response,
)
from ask_genome_agent.prompts import readout_prompt_template
from orchestration_core import AgentDependencies

from tests.fakes import ScriptedChatModel

pytestmark = pytest.mark.unit

_QUERY = "What was the margin ROI for paid search last year?"


def _state(**entry: Any) -> dict[str, Any]:
    subquery = Subquery(id=0, query=_QUERY, kind=SubqueryKind.analytical, coarse_intent="roi")
    plan = AskGenomePlan(
        original_query=_QUERY,
        rephrased_query=_QUERY,
        is_multi=False,
        is_sequential=False,
        subqueries=(subquery,),
    )
    base: dict[str, Any] = {
        "answerable": True,
        "insight_query": "",
        "insight_context": "",
        "response_context": "paid search gross roi is 2.4 in 2025.",
        "readout": "paid search gross roi is 2.4 in 2025.",
        "fixtext": "",
        "principle_pretext": None,
        "benchmark_text": "",
        "is_planner_answer": False,
        "table_recipe": {"client_code": "LINKEDIN", "model_group_id": 12},
    }
    base.update(entry)
    return {"plan": plan, "active_index": 0, "ner_state": {0: base}}


def _deps(model: ScriptedChatModel) -> AgentDependencies:
    return AgentDependencies(model=model, tools=())


def _answer(result: dict[str, Any]) -> str:
    return str(result["ner_state"][0]["answer"])


def _prompt(model: ScriptedChatModel, index: int = 0) -> str:
    (message,) = model.calls[index]["messages"]
    return str(message.content)


async def test_the_model_reads_the_context_and_the_question() -> None:
    """Upstream's analytics_generator prompt, sent as one user message, as the page sends it."""

    model = ScriptedChatModel().queue_text("Paid search returned 2.4 in 2025.")
    result = await generate_response(_state(), _deps(model))

    assert _answer(result) == "Paid search returned 2.4 in 2025."
    prompt = _prompt(model)
    assert _QUERY in prompt
    assert "paid search gross roi is 2.4 in 2025." in prompt
    assert "Response requirements:" in prompt


async def test_the_gpt_readout_and_its_query_are_preferred() -> None:
    """When postprocessing built a GPT readout, the model reads that rather than the tables."""

    model = ScriptedChatModel().queue_text("The answer.")
    state = _state(
        insight_query="Generate insights with given context.",
        insight_context="GPT readout: roi rose.",
    )
    await generate_response(state, _deps(model))

    prompt = _prompt(model)
    assert "Generate insights with given context." in prompt
    assert "GPT readout: roi rose." in prompt
    assert "gross roi is 2.4" not in prompt


async def test_thinking_is_stripped_from_the_reply() -> None:
    model = ScriptedChatModel().queue_text("<think>working</think>\nThe answer.")
    result = await generate_response(_state(), _deps(model))
    assert _answer(result) == "The answer."


async def test_the_pretext_leads_an_ordinary_answer() -> None:
    """The headline figure first, then the answer that explains it."""

    model = ScriptedChatModel().queue_text("The answer.")
    result = await generate_response(_state(fixtext="Overall ROI is 3.1."), _deps(model))
    assert _answer(result) == "Overall ROI is 3.1.\n\nThe answer."


async def test_a_planner_answer_leads_and_the_notice_comes_last() -> None:
    """The reference-only notice is a caveat on the recommendation, not an introduction to it."""

    model = ScriptedChatModel().queue_text("Move spend into audio.").queue_text("Summary.")
    notice = "This tool provides reference insights based on published scenarios."
    state = _state(
        fixtext=notice,
        is_planner_answer=True,
        principle_pretext="Genome says.",
        benchmark_text="Bench.",
    )
    result = await generate_response(state, _deps(model))
    assert _answer(result) == f"Move spend into audio.\n\nSummary.\n\nBench.\n\n{notice}"


async def test_principles_are_summarised_and_follow_the_answer() -> None:
    """The principle text gets its own "Summarize insights." call, as the page makes."""

    model = ScriptedChatModel().queue_text("The answer.").queue_text("Principle summary.")
    state = _state(fixtext="Pre.", principle_pretext="Genome says.", benchmark_text="Bench.")
    result = await generate_response(state, _deps(model))

    assert _answer(result) == "Pre.\n\nThe answer.\n\nPrinciple summary.\n\nBench."
    second = _prompt(model, 1)
    assert "Summarize insights." in second
    assert "Genome says." in second


async def test_a_failed_principle_summary_falls_back_to_the_principle_text() -> None:
    model = ScriptedChatModel().queue_text("The answer.").queue_error(RuntimeError("timeout"))
    result = await generate_response(_state(principle_pretext="Genome says."), _deps(model))
    entry = result["ner_state"][0]
    assert entry["answer"] == "The answer.\n\nGenome says."
    assert entry["principle_error"] == "RuntimeError: timeout"


async def test_an_absent_principle_leaves_no_gap() -> None:
    """None means nothing produced it. It must not print as 'None' or leave a blank paragraph."""

    model = ScriptedChatModel().queue_text("The answer.")
    result = await generate_response(_state(principle_pretext=None), _deps(model))
    assert _answer(result) == "The answer."


async def test_nothing_to_read_but_a_pretext_reports_the_pretext() -> None:
    """As the source does. A model given nothing to read would make something up."""

    model = ScriptedChatModel()
    result = await generate_response(
        _state(answerable=False, fixtext="No scenario matched."), _deps(model)
    )
    assert _answer(result) == "No scenario matched."
    assert model.calls == ()


async def test_nothing_at_all_changes_nothing() -> None:
    model = ScriptedChatModel()
    assert await generate_response(_state(answerable=False), _deps(model)) == {}
    assert model.calls == ()


async def test_a_failed_call_still_reports_the_figures() -> None:
    """The readout has every number, so it beats a failed turn. Recorded, so it is not mistaken
    for a normal answer."""

    model = ScriptedChatModel().queue_error(RuntimeError("gateway timeout"))
    result = await generate_response(_state(), _deps(model))
    entry = result["ner_state"][0]
    assert entry["answer"] == "paid search gross roi is 2.4 in 2025."
    assert entry["response_error"] == "RuntimeError: gateway timeout"


async def test_an_empty_reply_is_treated_as_a_failure() -> None:
    model = ScriptedChatModel().queue_text("   ")
    result = await generate_response(_state(), _deps(model))
    entry = result["ner_state"][0]
    assert entry["answer"] == "paid search gross roi is 2.4 in 2025."
    assert "empty response" in entry["response_error"]


async def test_the_client_s_prompt_inserts_reach_the_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """They come from upstream's CLIENT_SPECIFIC_PROMPT, keyed by the deployment's client."""

    monkeypatch.setenv("ASK_GENOME_CLIENT_CODE", "FTR")
    monkeypatch.setenv("ASK_GENOME_MODEL_GROUP_ID", "1")
    model = ScriptedChatModel().queue_text("The answer.")
    await generate_response(_state(), _deps(model))
    assert "Interpret 'efficiency' as 'Cost Per KPI'." in _prompt(model)


def test_the_graph_metadata_shows_the_prompt_the_node_renders() -> None:
    template = readout_prompt_template()
    assert "{{ query }}" in template or "{{query}}" in template
    assert "Response requirements:" in template


def test_assembly_drops_empty_parts() -> None:
    assert assemble_answer("A", "", "", "", planner=False) == "A"
    assert assemble_answer("A", "  ", "B", "", planner=True) == "A\n\nB"


async def test_the_answer_is_what_the_user_sees() -> None:
    """Not the readout, which is written for the model."""

    state = _state()
    state["ner_state"][0]["answer"] = "Paid search returned 2.4 in 2025."
    state["unanswered"] = {}
    result = await aggregate_results(state)  # type: ignore[arg-type]
    content = result["messages"][0].content
    assert "Paid search returned 2.4 in 2025." in content
    assert "gross roi is 2.4" not in content


async def test_without_an_answer_the_readout_still_reaches_the_user() -> None:
    state = _state()
    state["unanswered"] = {}
    result = await aggregate_results(state)  # type: ignore[arg-type]
    assert "paid search gross roi is 2.4 in 2025." in result["messages"][0].content


def test_the_graph_answers_between_postprocessing_and_writing_context() -> None:
    """After run_postprocess, where the readout comes from, and before write_context."""

    from ask_genome_agent.graph import create_graph

    graph = create_graph(_deps(ScriptedChatModel())).compile().get_graph()
    edges = {(edge.source, edge.target) for edge in graph.edges}
    assert ("run_postprocess", "generate_response") in edges
    assert ("generate_response", "write_context") in edges
    assert ("run_postprocess", "write_context") not in edges


async def test_the_model_is_told_the_period_the_question_resolved_to() -> None:
    """Live, it read "last year" as 2024 on 2026-09-25, though the filter had settled on 2025."""

    model = ScriptedChatModel().queue_text("The answer.")
    state = _state(ner_results={"start": "202501", "end": "202512"})
    await generate_response(state, _deps(model))
    assert "time reference resolves to 2025. Answer for 2025" in _prompt(model)


async def test_no_period_line_where_the_question_named_none() -> None:
    model = ScriptedChatModel().queue_text("The answer.")
    state = _state(ner_results={"start": "irrelevant", "end": "irrelevant"})
    await generate_response(state, _deps(model))
    assert "time reference resolves to" not in _prompt(model)


@pytest.mark.parametrize(
    ("start", "end", "expected"),
    [
        ("202501", "202512", "2025"),
        ("202504", "202506", "Apr 2025 to Jun 2025"),
        ("202503", "202503", "Mar 2025"),
        ("202410", "202503", "Oct 2024 to Mar 2025"),
        ("irrelevant", "irrelevant", ""),
        ("", "", ""),
    ],
)
def test_asked_period_reads_as_words(start: str, end: str, expected: str) -> None:
    assert asked_period({"start": start, "end": end}) == expected


async def test_a_planner_answer_is_not_told_the_asked_period() -> None:
    """Live, "What will ROI be next quarter?" resolved to Jul-Sep 2026, and the model relabelled
    FY25 Q4 scenarios as projections for it. Scenarios carry their own period."""

    model = ScriptedChatModel().queue_text("The answer.")
    state = _state(is_planner_answer=True, ner_results={"start": "202607", "end": "202609"})
    await generate_response(state, _deps(model))
    assert "time reference resolves to" not in _prompt(model)


# ------------------------------------- what a dependent subquery gets


async def test_a_dependent_subquery_is_given_what_the_user_was_told() -> None:
    """Live, part 2 got the start of the readout, not part 1's answer, and refused "its ROI"."""

    from ask_genome_agent.nodes.write_context import write_context

    model = ScriptedChatModel().queue_text("OTT/CTV had the highest spend in 2025.")
    state = _state(fixtext="Overall spend was 190M.")
    state["context"] = {}
    state["last_subquery_result"] = "Paid Social led LMS UK in 2024. (answered at level m)"
    update = await generate_response(state, _deps(model))

    answer = _answer(update)
    assert update["last_subquery_result"] == answer
    written = write_context({**state, **update})  # type: ignore[arg-type]
    assert written["context"][0] == answer
    assert "OTT/CTV" in written["context"][0]
    assert "Paid Social" not in written["context"][0]


async def test_after_a_failed_call_the_dependent_gets_the_readout_the_user_saw() -> None:
    model = ScriptedChatModel().queue_error(RuntimeError("gateway timeout"))
    update = await generate_response(_state(), _deps(model))
    assert update["last_subquery_result"] == "paid search gross roi is 2.4 in 2025."


async def test_a_pretext_only_answer_is_what_the_dependent_gets() -> None:
    model = ScriptedChatModel()
    update = await generate_response(
        _state(answerable=False, fixtext="No scenario matched."), _deps(model)
    )
    assert update["last_subquery_result"] == "No scenario matched."


async def test_without_an_answer_the_postprocess_summary_is_left_alone() -> None:
    """Nothing was said to the user, so there's nothing better to pass on."""

    update = await generate_response(_state(answerable=False), _deps(ScriptedChatModel()))
    assert "last_subquery_result" not in update


async def test_the_resolver_reads_the_answer_and_is_not_told_to_want_its_metric() -> None:
    """Live, upstream's REJECT rule refused "What was its ROI?" because the context had no ROI."""

    from ask_genome_agent.nodes.resolve_dependency import resolve_dependency

    first = Subquery(
        id=0,
        query="Which channel had the highest spend last year?",
        kind=SubqueryKind.analytical,
        coarse_intent="spending",
    )
    second = Subquery(
        id=1,
        query="What was its ROI last year?",
        kind=SubqueryKind.analytical,
        coarse_intent="roi",
        depends_on=0,
    )
    plan = AskGenomePlan(
        original_query="q",
        rephrased_query="q",
        is_multi=True,
        is_sequential=True,
        subqueries=(first, second),
    )
    model = ScriptedChatModel().queue_tool_call(
        "ResolveOutcome",
        {"subqueries": ["What was the ROI of OTT/CTV last year?"], "resolved": True},
    )
    state = {
        "plan": plan,
        "active_index": 1,
        "context": {0: "OTT/CTV had the highest spend in 2025."},
    }
    update = await resolve_dependency(state, _deps(model))  # type: ignore[arg-type]

    assert update["resolution"].kind == "rephrased"
    system, user = model.calls[0]["messages"]
    assert "do not reject because the context lacks the metric" in " ".join(system.content.split())
    assert "OTT/CTV had the highest spend in 2025." in user.content


# ------------------------------------- our additions to upstream's requirements


async def test_upstream_s_requirements_stay_whole_and_the_additions_follow() -> None:
    """Replacing the defaults would drop all of upstream's rules."""

    from ask_genome_agent.vendor.ask_genome_core.model.common import PromptTemplates

    model = ScriptedChatModel().queue_text("The answer.")
    await generate_response(_state(), _deps(model))
    prompt = _prompt(model)
    defaults = PromptTemplates().get_default_requirements("analytics_generator")
    assert defaults in prompt
    assert prompt.index(defaults) < prompt.index("The figures in the context carry no currency")


@pytest.mark.parametrize(
    "rule",
    [
        "Do not add currency symbols",
        'name that scope with them (for example "Premium, UK")',
        "check it against every relevant row",
        "only after comparing all the relevant rows",
    ],
)
async def test_each_addition_reaches_the_model(rule: str) -> None:
    model = ScriptedChatModel().queue_text("The answer.")
    await generate_response(_state(), _deps(model))
    assert rule in _prompt(model)


async def test_the_principle_summary_gets_the_additions_but_no_period() -> None:
    model = ScriptedChatModel().queue_text("The answer.").queue_text("Summary.")
    state = _state(
        principle_pretext="Genome says.", ner_results={"start": "202501", "end": "202512"}
    )
    await generate_response(state, _deps(model))
    second = _prompt(model, 1)
    assert "Do not add currency symbols" in second
    assert "time reference resolves to" not in second
