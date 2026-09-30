"""Ask Genome graph that converts a raw query into a validated, dependency-resolved AskGenomePlan.

1. Uses one agent and one graph. Dependency-resolution nodes are part of the same graph instead
   of being a separate agent.
2. A separate Dependent Subquery Handler agent is not used because non-entry agents do not
   receive an AgentInvoker in the current deployment.
3. Dependency-resolution logic runs through an internal cursor loop and shares the graph's state
   directly.
4. All nodes and routing edges are fully wired and visible in the diagnostics UI.
5. _PLAN_ONLY is a temporary safety flag that stops execution after assemble_plan. It is
   currently off.
6. Question Understanding is implemented for analytical subqueries. An ambiguous field pauses the
   graph with a typed ClarificationRequest and resumes in place; coordinate_clarification asks one
   queued question at a time. Coverage subqueries still raise NotImplementedError.
7. Data filtering runs after finalize_filter: resolve_data_source picks the dataset and metrics
   for the intention, apply_data_filters applies the filter to that data, run_postprocess turns
   the filtered rows into a readout, and generate_response turns the readout into an answer. An
   out-of-scope intention skips filtering rather than raising.
"""

from collections.abc import Hashable
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph
from orchestration_core import (
    NODE_KIND_METADATA_KEY,
    PROMPT_TEMPLATE_ID_METADATA_KEY,
    PROMPT_TEMPLATE_METADATA_KEY,
    AgentDependencies,
    GraphNodeKind,
)

from ask_genome_agent.contracts import Subquery, SubqueryKind
from ask_genome_agent.nodes.advance_cursor import advance_cursor
from ask_genome_agent.nodes.aggregate_results import aggregate_results
from ask_genome_agent.nodes.annotate_feasibility import annotate_feasibility
from ask_genome_agent.nodes.answer_coverage import answer_coverage
from ask_genome_agent.nodes.apply_resolution import apply_resolution
from ask_genome_agent.nodes.assemble_plan import assemble_plan
from ask_genome_agent.nodes.decompose_and_classify import decompose_and_classify
from ask_genome_agent.nodes.extract_query import extract_query
from ask_genome_agent.nodes.generate_response import generate_response
from ask_genome_agent.nodes.postprocess.run_postprocess import run_postprocess
from ask_genome_agent.nodes.question_understanding.apply_data_filters import apply_data_filters
from ask_genome_agent.nodes.question_understanding.build_clarification_list import (
    build_clarification_list,
)
from ask_genome_agent.nodes.question_understanding.classify_fields import classify_fields
from ask_genome_agent.nodes.question_understanding.coordinate_clarification import (
    after_clarification,
    coordinate_clarification,
    should_clarify,
)
from ask_genome_agent.nodes.question_understanding.extract_entities import extract_entities
from ask_genome_agent.nodes.question_understanding.fill_remaining_fields import (
    fill_remaining_fields,
)
from ask_genome_agent.nodes.question_understanding.finalize_filter import finalize_filter
from ask_genome_agent.nodes.question_understanding.reconcile_intent import reconcile_intent
from ask_genome_agent.nodes.question_understanding.resolve_data_source import (
    after_data_source,
    resolve_data_source,
)
from ask_genome_agent.nodes.resolve_dependency import resolve_dependency
from ask_genome_agent.nodes.write_context import write_context
from ask_genome_agent.prompts import (
    FIELD_EXTRACTION_PROMPT,
    PLANNER_PROMPT,
    RESOLVE_FROM_CONTEXT_PROMPT,
    coverage_spec_template,
    readout_prompt_template,
)
from ask_genome_agent.state import AskGenomeState

_NEXT_PATH_MAP: dict[Hashable, str] = {
    "extract_entities": "extract_entities",
    "answer_coverage": "answer_coverage",
    "resolve_dependency": "resolve_dependency",
    "aggregate_results": "aggregate_results",
}

_PLAN_ONLY = False


def _start(subquery: Subquery) -> Literal["extract_entities", "answer_coverage"]:
    """Coverage skips question understanding."""

    return "answer_coverage" if subquery.kind == SubqueryKind.coverage else "extract_entities"


def _route_next(
    state: AskGenomeState,
) -> Literal["extract_entities", "answer_coverage", "resolve_dependency", "aggregate_results"]:
    if _PLAN_ONLY:
        return "aggregate_results"
    plan = state["plan"]
    idx = state["active_index"]
    if idx >= len(plan.subqueries):
        return "aggregate_results"
    if plan.subqueries[idx].depends_on is None:
        return _start(plan.subqueries[idx])
    return "resolve_dependency"


def _route_after_resolution(
    state: AskGenomeState,
) -> Literal[
    "extract_entities",
    "answer_coverage",
    "resolve_dependency",
    "aggregate_results",
    "write_context",
]:
    resolution = state["resolution"]
    # Only ever reached right after apply_resolution. See the comment there.
    assert resolution is not None
    if resolution.kind == "split":
        return _route_next(state)
    if resolution.kind == "rejected":
        return "write_context"
    return _start(state["plan"].subqueries[state["active_index"]])


def create_graph(dependencies: AgentDependencies) -> StateGraph[AskGenomeState]:
    async def _decompose_and_classify(state: AskGenomeState) -> dict[str, Any]:
        return await decompose_and_classify(state, dependencies)

    async def _resolve_dependency(state: AskGenomeState) -> dict[str, Any]:
        return await resolve_dependency(state, dependencies)

    async def _fill_remaining_fields(state: AskGenomeState) -> dict[str, Any]:
        return await fill_remaining_fields(state, dependencies)

    async def _generate_response(state: AskGenomeState) -> dict[str, Any]:
        return await generate_response(state, dependencies)

    async def _answer_coverage(state: AskGenomeState) -> dict[str, Any]:
        return await answer_coverage(state, dependencies)

    builder = StateGraph(AskGenomeState)
    builder.add_node(
        "extract_query",
        extract_query,
        metadata={NODE_KIND_METADATA_KEY: GraphNodeKind.DETERMINISTIC.value},
    )
    builder.add_node(
        "decompose_and_classify",
        _decompose_and_classify,
        metadata={
            NODE_KIND_METADATA_KEY: GraphNodeKind.MODEL.value,
            PROMPT_TEMPLATE_ID_METADATA_KEY: "ask_genome.decompose_and_classify.system",
            PROMPT_TEMPLATE_METADATA_KEY: PLANNER_PROMPT,
        },
    )
    builder.add_node(
        "annotate_feasibility",
        annotate_feasibility,
        metadata={NODE_KIND_METADATA_KEY: GraphNodeKind.DETERMINISTIC.value},
    )
    builder.add_node(
        "assemble_plan",
        assemble_plan,
        metadata={NODE_KIND_METADATA_KEY: GraphNodeKind.DETERMINISTIC.value},
    )
    builder.add_node(
        "resolve_dependency",
        _resolve_dependency,
        metadata={
            NODE_KIND_METADATA_KEY: GraphNodeKind.MODEL.value,
            PROMPT_TEMPLATE_ID_METADATA_KEY: "ask_genome.resolve_dependency.system",
            PROMPT_TEMPLATE_METADATA_KEY: RESOLVE_FROM_CONTEXT_PROMPT,
        },
    )
    builder.add_node(
        "apply_resolution",
        apply_resolution,
        metadata={NODE_KIND_METADATA_KEY: GraphNodeKind.DETERMINISTIC.value},
    )
    builder.add_node(
        "answer_coverage",
        _answer_coverage,
        metadata={
            NODE_KIND_METADATA_KEY: GraphNodeKind.MODEL.value,
            PROMPT_TEMPLATE_ID_METADATA_KEY: "ask_genome.answer_coverage.spec",
            PROMPT_TEMPLATE_METADATA_KEY: coverage_spec_template(),
        },
    )
    builder.add_node(
        "extract_entities",
        extract_entities,
        metadata={NODE_KIND_METADATA_KEY: GraphNodeKind.DETERMINISTIC.value},
    )
    builder.add_node(
        "classify_fields",
        classify_fields,
        metadata={NODE_KIND_METADATA_KEY: GraphNodeKind.DETERMINISTIC.value},
    )
    builder.add_node(
        "reconcile_intent",
        reconcile_intent,
        metadata={NODE_KIND_METADATA_KEY: GraphNodeKind.DETERMINISTIC.value},
    )
    builder.add_node(
        "fill_remaining_fields",
        _fill_remaining_fields,
        metadata={
            NODE_KIND_METADATA_KEY: GraphNodeKind.MODEL.value,
            PROMPT_TEMPLATE_ID_METADATA_KEY: "ask_genome.fill_remaining_fields.system",
            PROMPT_TEMPLATE_METADATA_KEY: FIELD_EXTRACTION_PROMPT,
        },
    )
    builder.add_node(
        "build_clarification_list",
        build_clarification_list,
        metadata={NODE_KIND_METADATA_KEY: GraphNodeKind.DETERMINISTIC.value},
    )
    builder.add_node(
        "coordinate_clarification",
        coordinate_clarification,
        metadata={NODE_KIND_METADATA_KEY: GraphNodeKind.DETERMINISTIC.value},
    )
    builder.add_node(
        "finalize_filter",
        finalize_filter,
        metadata={NODE_KIND_METADATA_KEY: GraphNodeKind.DETERMINISTIC.value},
    )
    builder.add_node(
        "resolve_data_source",
        resolve_data_source,
        metadata={NODE_KIND_METADATA_KEY: GraphNodeKind.DETERMINISTIC.value},
    )
    builder.add_node(
        "apply_data_filters",
        apply_data_filters,
        metadata={NODE_KIND_METADATA_KEY: GraphNodeKind.DETERMINISTIC.value},
    )
    builder.add_node(
        "run_postprocess",
        run_postprocess,
        metadata={NODE_KIND_METADATA_KEY: GraphNodeKind.DETERMINISTIC.value},
    )
    builder.add_node(
        "generate_response",
        _generate_response,
        metadata={
            NODE_KIND_METADATA_KEY: GraphNodeKind.MODEL.value,
            PROMPT_TEMPLATE_ID_METADATA_KEY: "ask_genome.generate_response.analytics_generator",
            PROMPT_TEMPLATE_METADATA_KEY: readout_prompt_template(),
        },
    )
    builder.add_node(
        "write_context",
        write_context,
        metadata={NODE_KIND_METADATA_KEY: GraphNodeKind.DETERMINISTIC.value},
    )
    builder.add_node(
        "advance_cursor",
        advance_cursor,
        metadata={NODE_KIND_METADATA_KEY: GraphNodeKind.DETERMINISTIC.value},
    )
    builder.add_node(
        "aggregate_results",
        aggregate_results,
        metadata={NODE_KIND_METADATA_KEY: GraphNodeKind.DETERMINISTIC.value},
    )

    builder.add_edge(START, "extract_query")
    builder.add_edge("extract_query", "decompose_and_classify")
    builder.add_edge("decompose_and_classify", "annotate_feasibility")
    builder.add_edge("annotate_feasibility", "assemble_plan")

    builder.add_conditional_edges("assemble_plan", _route_next, path_map=_NEXT_PATH_MAP)
    builder.add_conditional_edges("advance_cursor", _route_next, path_map=_NEXT_PATH_MAP)

    builder.add_edge("resolve_dependency", "apply_resolution")
    builder.add_conditional_edges(
        "apply_resolution",
        _route_after_resolution,
        path_map={**_NEXT_PATH_MAP, "write_context": "write_context"},
    )

    builder.add_edge("extract_entities", "classify_fields")
    builder.add_edge("classify_fields", "reconcile_intent")
    builder.add_edge("reconcile_intent", "fill_remaining_fields")
    builder.add_edge("fill_remaining_fields", "build_clarification_list")
    builder.add_conditional_edges(
        "build_clarification_list",
        should_clarify,
        path_map={
            "coordinate_clarification": "coordinate_clarification",
            "finalize_filter": "finalize_filter",
        },
    )
    builder.add_conditional_edges(
        "coordinate_clarification",
        after_clarification,
        path_map={
            "coordinate_clarification": "coordinate_clarification",
            "finalize_filter": "finalize_filter",
        },
    )
    builder.add_edge("finalize_filter", "resolve_data_source")
    builder.add_conditional_edges(
        "resolve_data_source",
        after_data_source,
        path_map={"apply_data_filters": "apply_data_filters", "write_context": "write_context"},
    )
    builder.add_edge("apply_data_filters", "run_postprocess")
    builder.add_edge("run_postprocess", "generate_response")
    builder.add_edge("generate_response", "write_context")
    builder.add_edge("answer_coverage", "write_context")
    builder.add_edge("write_context", "advance_cursor")
    builder.add_edge("aggregate_results", END)
    return builder
