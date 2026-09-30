import time
from typing import Any, Literal, Optional
from pydantic import BaseModel, create_model

from src.integrations.llm_external import call_enterprise_ai_api, call_enterprise_ai_structured


class NEROutput(BaseModel):
    answer: dict[str, Any]
    need_clarification: bool
    clarification_reason: str = ""


class DenialOutput(BaseModel):
    # defaults, not required fields: a model that returns only status still validates
    status: str = "answered"
    denial_reason: str | None = None


COVERAGE_OPERATIONS = (
    "count_periods", "time_range", "list_values", "metric_availability", "existence", "co_occurrence"
)


class CoverageSpec(BaseModel):
    dimensions: list[str] = []
    values: list[str] = []
    metric: Optional[str] = None
    period_type: Optional[str] = None
    operation: Literal[COVERAGE_OPERATIONS]
    needs_join: bool = False


class ResolveOutput(BaseModel):
    subqueries: list[str] = []
    resolved: bool = False
    reason: str = ""


def make_planner_model(intent_catalog: list[str]):
    intent_literal = Literal[tuple(intent_catalog)] if intent_catalog else str
    sub_query = create_model(
        "PlannerSubQuery",
        id=(int, ...),
        query=(str, ...),
        kind=(Literal["coverage", "analytical"], ...),
        coarse_intent=(Optional[intent_literal], None),
        depends_on=(Optional[int], None),
    )
    plan = create_model(
        "PlannerPlanModel",
        rephrased_query=(str, ...),
        is_multi=(bool, ...),
        subqueries=(list[sub_query], ...),
    )
    return plan


def _timed_structured_call(system_prompt: str, user_prompt: str, schema_model,
                           reasoning_effort: str, max_output_tokens: int):
    """(validated pydantic instance, elapsed seconds) - the shape planner/coverage expect.

    Reasoning tokens count against max_output_tokens on the responses api, so the budgets are
    set well above the answer size."""
    t0 = time.perf_counter()
    parsed = call_enterprise_ai_structured(system_prompt, user_prompt, schema_model,
                                           reasoning_effort=reasoning_effort,
                                           max_output_tokens=max_output_tokens)
    return parsed, time.perf_counter() - t0


def planner_llm_call(system_prompt: str, user_prompt: str, intent_catalog: list,
                     reasoning_effort: str = "low"):
    return _timed_structured_call(system_prompt, user_prompt, make_planner_model(intent_catalog),
                                  reasoning_effort, 8192)


def coverage_spec_call(system_prompt: str, user_prompt: str, reasoning_effort: str = "low"):
    return _timed_structured_call(system_prompt, user_prompt, CoverageSpec, reasoning_effort, 4096)


def resolve_from_context_call(system_prompt: str, user_prompt: str, reasoning_effort: str = "low"):
    return _timed_structured_call(system_prompt, user_prompt, ResolveOutput, reasoning_effort, 8192)


def text_llm_call(system_prompt: str, user_prompt: str, reasoning_effort: str = "none"):
    t0 = time.perf_counter()
    response = call_enterprise_ai_api("openai", system_prompt, user_prompt,
                                      reasoning_effort=reasoning_effort,
                                      extra_body={"max_output_tokens": 2048})
    return response, time.perf_counter() - t0
