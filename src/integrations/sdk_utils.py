import os
import sys
import time
import pandas as pd
import numpy as np
from typing import Any, Dict, Literal, Optional, Union
from langchain_core.output_parsers.json import JsonOutputParser
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel, create_model, field_validator

# anthropic has no native reasoning_effort; map it to an extended-thinking token budget
_ANTHROPIC_THINKING_BUDGET = {"low": 1024, "medium": 4096, "high": 16384}


class NEROutput(BaseModel):
    answer: dict[str, Any]
    need_clarification: bool
    clarification_reason: str = ""


class ResponseOutput(BaseModel):
    is_denial: bool = False
    denial_reason: str | None = None
    answer: str = ""


def make_ner_model(field_names: list[str]):
    field_type = Union[Literal["irrelevant", "all"], list]
    fields = {name: (field_type, "irrelevant") for name in field_names}
    fields["need_clarification"] = (bool, False)

    def _non_empty(cls, v):
        if isinstance(v, list) and not v:
            raise ValueError("list must be non-empty")
        return v

    validators = {"_check": field_validator(*field_names, mode="after")(_non_empty)}
    return create_model("DynamicNER", __validators__=validators, **fields)


def llm_call(system_prompt: str, user_prompt: str, fields: list, reasoning_effort: str = "medium"):
    budget = _ANTHROPIC_THINKING_BUDGET.get(reasoning_effort)
    kwargs = {"model": "claude-sonnet-4-6", "api_key": os.environ["ANTHROPIC_API_KEY"]}
    if budget:
        # extended thinking requires temperature=1 and max_tokens > budget_tokens
        kwargs.update(temperature=0.5, max_tokens=budget + 4096,
                      thinking={"type": "enabled", "budget_tokens": budget})
    llm = ChatAnthropic(**kwargs)
    ner_output_model = make_ner_model(fields)
    structured_llm = llm.with_structured_output(ner_output_model, include_raw=True)
    t0 = time.perf_counter()
    result = structured_llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])
    elapsed = time.perf_counter() - t0
    if result["parsed"] is None:
        raise ValueError(f"parse failed: {result['parsing_error']}")
    return result["parsed"], elapsed


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


def _anthropic_kwargs(reasoning_effort: str, temperature: float, default_max_tokens: int = 1024,
                      enable_thinking: bool = True) -> dict:
    # thinking + with_structured_output (forced tool calling) is unsupported by langchain_anthropic;
    # structured-output callers pass enable_thinking=False.
    budget = _ANTHROPIC_THINKING_BUDGET.get(reasoning_effort) if enable_thinking else None
    kwargs = {"model": "claude-sonnet-4-6", "api_key": os.environ["ANTHROPIC_API_KEY"],
              "max_tokens": default_max_tokens}
    if budget:
        # extended thinking requires temperature=1 and max_tokens > budget_tokens
        kwargs.update(temperature=1, max_tokens=budget + 4096,
                      thinking={"type": "enabled", "budget_tokens": budget})
    else:
        kwargs.update(temperature=temperature)
    return kwargs


def planner_llm_call(system_prompt: str, user_prompt: str, intent_catalog: list,
                     reasoning_effort: str = "medium"):
    llm = ChatAnthropic(**_anthropic_kwargs(reasoning_effort, temperature=0.3,
                                            default_max_tokens=4096, enable_thinking=False))
    structured_llm = llm.with_structured_output(make_planner_model(intent_catalog), include_raw=True)
    t0 = time.perf_counter()
    result = structured_llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])
    elapsed = time.perf_counter() - t0
    if result["parsed"] is None:
        raise ValueError(f"parse failed: {result['parsing_error']}")
    return result["parsed"], elapsed


def coverage_spec_call(system_prompt: str, user_prompt: str, reasoning_effort: str = "medium"):
    llm = ChatAnthropic(**_anthropic_kwargs(reasoning_effort, temperature=0.3,
                                            default_max_tokens=2048, enable_thinking=False))
    structured_llm = llm.with_structured_output(CoverageSpec, include_raw=True)
    t0 = time.perf_counter()
    result = structured_llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])
    elapsed = time.perf_counter() - t0
    if result["parsed"] is None:
        raise ValueError(f"parse failed: {result['parsing_error']}")
    return result["parsed"], elapsed


def resolve_from_context_call(system_prompt: str, user_prompt: str, reasoning_effort: str = "medium"):
    llm = ChatAnthropic(**_anthropic_kwargs(reasoning_effort, temperature=0.3,
                                            default_max_tokens=4096, enable_thinking=False))
    structured_llm = llm.with_structured_output(ResolveOutput, include_raw=True)
    t0 = time.perf_counter()
    result = structured_llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])
    elapsed = time.perf_counter() - t0
    if result["parsed"] is None:
        raise ValueError(f"parse failed: {result['parsing_error']}")
    return result["parsed"], elapsed


def text_llm_call(system_prompt: str, user_prompt: str, reasoning_effort: str = "none"):
    llm = ChatAnthropic(**_anthropic_kwargs(reasoning_effort, temperature=0.3))
    t0 = time.perf_counter()
    result = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])
    elapsed = time.perf_counter() - t0
    return result.content, elapsed


def response_llm_call(system_prompt: str, user_prompt: str, reasoning_effort: str = "medium"):
    budget = _ANTHROPIC_THINKING_BUDGET.get(reasoning_effort)
    kwargs = {"model": "claude-sonnet-4-6", "api_key": os.environ["ANTHROPIC_API_KEY"]}
    if budget:
        # extended thinking requires temperature=1 and max_tokens > budget_tokens
        kwargs.update(temperature=1, max_tokens=budget + 4096,
                      thinking={"type": "enabled", "budget_tokens": budget})
    llm = ChatAnthropic(**kwargs)
    structured_llm = llm.with_structured_output(ResponseOutput, include_raw=True)
    t0 = time.perf_counter()
    result = structured_llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])
    elapsed = time.perf_counter() - t0
    if result["parsed"] is None:
        raise ValueError(f"parse failed: {result['parsing_error']}")
    return result["parsed"], elapsed
