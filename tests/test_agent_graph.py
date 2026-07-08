"""Unit tests for the agent orchestrator graph (src/agents).

All external seams (planner, coverage, NER, readout, LLM calls) are monkeypatched with
canned returns, so these tests exercise the graph wiring only: routing, the record loop,
clarification interrupt/resume, dependency resolution and re-plan behavior.
"""
from types import SimpleNamespace

import pandas as pd
from langgraph.types import Command

import src.agents.orchestrator_graph as orchestrator_graph
import src.agents.coverage_agent as coverage_agent
import src.agents.dependency_agent as dependency_agent
import src.agents.analytical_agent as analytical_agent
import src.agents.replan as replan
import src.data.data_interface as data_interface
import src.model.filter_generator_refactor as filter_generator_refactor
from src.agents import create_agent_app
from src.model.planner import (
    SubQuerySpec, SubQueryAnnotation, EnrichedSubQuery, PlannerPlan, PlannerResult,
)

CONFIG = {"configurable": {"thread_id": "t1", "client_code": "TEST", "model_group_id": 1,
                           "llm_vendor": "openai"}}


def make_plan(specs: list) -> PlannerResult:
    enriched = [EnrichedSubQuery(spec=s, annotation=SubQueryAnnotation()) for s in specs]
    plan = PlannerPlan(original_query="q", rephrased_query="rephrased q",
                       is_multi=len(specs) > 1, subqueries=specs)
    return PlannerResult(plan=plan, subqueries=enriched,
                         is_sequential=any(s.depends_on is not None for s in specs))


def patch_plan(monkeypatch, specs: list) -> None:
    monkeypatch.setattr(orchestrator_graph, "generate_plan", lambda *a, **k: make_plan(specs))


def patch_happy_analytical(monkeypatch, clarification_fields: dict = None) -> None:
    """Analytical pipeline succeeds; NER optionally asks for clarification."""
    def fake_extract(*args, **kwargs):
        return {}, {"spacy": "filter"}, {"intention": "roi"}, dict(clarification_fields or {})

    def fake_process_data(*args, **kwargs):
        table, empty = pd.DataFrame({"a": [1]}), pd.DataFrame()
        return ("readout context", table, empty, "", empty, empty, "pretext text",
                table, empty, empty, {}, "", empty, "")

    monkeypatch.setattr(filter_generator_refactor, "extract_ner_filter", fake_extract)
    monkeypatch.setattr(analytical_agent, "ProcessIndicator",
                        SimpleNamespace(from_local=lambda *a: None))
    monkeypatch.setattr(analytical_agent, "generate_readoutdata",
                        lambda **kwargs: SimpleNamespace(ner_filters={}))
    monkeypatch.setattr(analytical_agent, "process_data", fake_process_data)
    monkeypatch.setattr(analytical_agent, "_prep_visual_context",
                        lambda *a, **k: "visual context")
    monkeypatch.setattr(analytical_agent, "response_generate_with_denial",
                        lambda *a, **k: {"is_denial": False, "denial_reason": None,
                                         "answer": "the insight"})


def patch_coverage(monkeypatch) -> None:
    loader = SimpleNamespace(from_local=lambda *a: SimpleNamespace(
        df_bi=pd.DataFrame(), df_br=pd.DataFrame(), core_filters={}))
    monkeypatch.setattr(coverage_agent, "NerData", loader)
    monkeypatch.setattr(coverage_agent, "FeasibilityConfig", loader)
    monkeypatch.setattr(coverage_agent, "SpacyConfig", loader)
    monkeypatch.setattr(coverage_agent, "answer_coverage",
                        lambda *a, **k: {"response": "3 quarters", "table": None,
                                         "facts": {"count": 3}})


def run(app, payload):
    app.invoke(payload, config=CONFIG)
    return app.get_state(CONFIG)


def test_coverage_query(monkeypatch):
    patch_plan(monkeypatch, [SubQuerySpec(id=0, query="how many quarters", kind="coverage")])
    patch_coverage(monkeypatch)
    snapshot = run(create_agent_app(), {"query": "how many quarters"})

    assert not snapshot.next
    payloads = snapshot.values["payloads"]
    assert len(payloads) == 1 and payloads[0]["kind"] == "coverage"
    assert payloads[0]["response"] == "3 quarters"
    assert '"count": 3' in snapshot.values["contexts"][0]


def test_clarification_interrupt_and_resume(monkeypatch):
    patch_plan(monkeypatch, [SubQuerySpec(id=0, query="roi of tv", kind="analytical")])
    patch_happy_analytical(monkeypatch, clarification_fields={"brand": ["a", "b"]})
    app = create_agent_app()

    snapshot = run(app, {"query": "roi of tv"})
    assert snapshot.next  # paused at clarify
    interrupt_value = snapshot.tasks[0].interrupts[0].value
    assert interrupt_value["clarification_fields"] == {"brand": ["a", "b"]}
    assert interrupt_value["sub_query_index"] == 0

    snapshot = run(app, Command(resume={"brand": ["a"]}))
    assert not snapshot.next
    payloads = snapshot.values["payloads"]
    assert len(payloads) == 1 and not payloads[0].get("rejected")
    assert snapshot.values["ner_results"] == {}  # scratch cleared by record
    assert "the insight" in payloads[0]["response"]


def test_dependency_rephrase(monkeypatch):
    patch_plan(monkeypatch, [
        SubQuerySpec(id=0, query="how many quarters", kind="coverage"),
        SubQuerySpec(id=1, query="roi for each", kind="analytical", depends_on=0)])
    patch_coverage(monkeypatch)
    patch_happy_analytical(monkeypatch)
    monkeypatch.setattr(dependency_agent, "resolve_from_context",
                        lambda *a: (["roi of q1"], True, ""))
    snapshot = run(create_agent_app(), {"query": "q"})

    payloads = snapshot.values["payloads"]
    assert len(payloads) == 2
    assert payloads[1]["query"] == "roi of q1"  # rephrased query was used


def test_dependency_split(monkeypatch):
    patch_plan(monkeypatch, [
        SubQuerySpec(id=0, query="how many quarters", kind="coverage"),
        SubQuerySpec(id=1, query="roi for each", kind="analytical", depends_on=0)])
    patch_coverage(monkeypatch)
    patch_happy_analytical(monkeypatch)
    monkeypatch.setattr(dependency_agent, "resolve_from_context",
                        lambda *a: (["roi of q1", "roi of q2"], True, ""))
    snapshot = run(create_agent_app(), {"query": "q"})

    payloads = snapshot.values["payloads"]
    assert len(payloads) == 3  # coverage + two spliced analytical sub-queries
    assert [p["query"] for p in payloads[1:]] == ["roi of q1", "roi of q2"]
    assert len(snapshot.values["subqueries"]) == 3


def test_dependency_reject(monkeypatch):
    patch_plan(monkeypatch, [
        SubQuerySpec(id=0, query="how many quarters", kind="coverage"),
        SubQuerySpec(id=1, query="roi for each", kind="analytical", depends_on=0)])
    patch_coverage(monkeypatch)
    monkeypatch.setattr(dependency_agent, "resolve_from_context",
                        lambda *a: ([], False, "no items"))
    snapshot = run(create_agent_app(), {"query": "q"})

    payloads = snapshot.values["payloads"]
    assert payloads[1]["rejected"]
    assert payloads[1]["message"] == "Could not derive this sub-query from sub-query 1: no items"


def test_empty_data_replan_drop(monkeypatch):
    patch_plan(monkeypatch, [SubQuerySpec(id=0, query="roi of tv", kind="analytical")])
    patch_happy_analytical(monkeypatch)

    def empty_process_data(*args, **kwargs):
        empty = pd.DataFrame()
        return ("", empty, empty, "", empty, empty, "no data for tv",
                empty, empty, empty, {}, "", empty, "")

    monkeypatch.setattr(analytical_agent, "process_data", empty_process_data)
    monkeypatch.setattr(replan, "agent_tool_call",
                        lambda *a, **k: {"tool": "drop_current", "args": {}})
    snapshot = run(create_agent_app(), {"query": "roi of tv"})

    payloads = snapshot.values["payloads"]
    assert len(payloads) == 1 and payloads[0]["rejected"]
    assert payloads[0]["message"] == "no data for tv"


def test_replan_revise_remaining(monkeypatch):
    patch_plan(monkeypatch, [SubQuerySpec(id=0, query="roi of tv", kind="analytical")])
    patch_happy_analytical(monkeypatch)

    process_data_calls = {"count": 0}
    happy = analytical_agent.process_data

    def failing_then_happy(*args, **kwargs):
        process_data_calls["count"] += 1
        if process_data_calls["count"] == 1:
            empty = pd.DataFrame()
            return ("", empty, empty, "", empty, empty, "no data for tv",
                    empty, empty, empty, {}, "", empty, "")
        return happy(*args, **kwargs)

    monkeypatch.setattr(analytical_agent, "process_data", failing_then_happy)
    monkeypatch.setattr(replan, "agent_tool_call",
                        lambda *a, **k: {"tool": "revise_remaining", "args": {
                            "revised_subqueries": [{"query": "roi of tv overall"}],
                            "reason": "broaden the filter"}})
    # _validate_revision imports NerDataConfig locally, so patch it at its home module
    monkeypatch.setattr(data_interface, "NerDataConfig",
                        SimpleNamespace(from_local=lambda *a: SimpleNamespace(metric_info={})))
    snapshot = run(create_agent_app(), {"query": "roi of tv"})

    payloads = snapshot.values["payloads"]
    assert len(payloads) == 1 and not payloads[0].get("rejected")
    assert payloads[0]["query"] == "roi of tv overall"
    assert snapshot.values["replan_notes"] == ["Plan revised: broaden the filter"]
    assert snapshot.values["replan_count"] == 1
