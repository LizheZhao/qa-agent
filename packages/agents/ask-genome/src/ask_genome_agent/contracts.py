"""Defines the Pydantic schemas/contracts used by the Planner and Dependency Resolver.

1. Uses two separate LLM-facing schemas:
    build_planner_output_model — strictly validates kind, client-specific coarse_intent, and
    depends_on.
    ResolveOutcome — uses simple string fields because dependency resolution only rewrites or
    splits subqueries; it does not reclassify them.
2. Every Subquery, including newly generated dependent subqueries, is created through the same
   validated Pydantic model.
3. Spliced dependent subqueries preserve the source behavior:
    kind = analytical
    coarse_intent = None
    depends_on = None
"""

from __future__ import annotations

import re
from collections.abc import Collection, Sequence
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, create_model, model_validator


class SubqueryKind(str, Enum):
    coverage = "coverage"
    analytical = "analytical"


class FeasibilityVerdict(BaseModel):
    """Informational only; never gates a subquery outright."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: Literal["feasible", "infeasible"]
    reasons: tuple[str, ...] = Field(default_factory=tuple)


class Subquery(BaseModel):
    """Canonical internal representation for every subquery, original or spliced."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: int
    query: str
    kind: SubqueryKind
    coarse_intent: str | None = None
    depends_on: int | None = None
    feasibility: FeasibilityVerdict | None = None
    spacy_hints: dict[str, Any] | None = None
    """The spaCy entity-extraction hints computed for this subquery's own text during
    annotate_feasibility.py. Persisted here (rather than discarded after the feasibility check)
    so Question Understanding can reuse them instead of recomputing the identical spaCy pass."""


class AskGenomePlan(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    original_query: str
    rephrased_query: str
    is_multi: bool
    is_sequential: bool
    subqueries: tuple[Subquery, ...]

    @model_validator(mode="after")
    def _validate_invariants(self) -> AskGenomePlan:
        """Cross-field invariants the LLM's structured-output schema doesn't itself enforce.

        Only runs on direct construction (AskGenomePlan(...) in assemble_plan.py) -- pydantic
        does not re-run validators on model_copy(), which is how apply_resolution.py's spliced
        subqueries (kind=analytical, coarse_intent=None per class docstring point 3 above) are
        allowed to exist without tripping the analytical-needs-intent check below.
        """

        if not self.subqueries:
            raise ValueError("plan must contain at least one subquery")

        ids = [s.id for s in self.subqueries]
        if sorted(ids) != list(range(len(self.subqueries))):
            raise ValueError(
                f"subquery ids must be 0..{len(self.subqueries) - 1} with no duplicates or "
                f"gaps, got {ids}"
            )

        id_set = set(ids)
        for s in self.subqueries:
            if s.depends_on is not None:
                if s.depends_on not in id_set:
                    raise ValueError(f"subquery {s.id} depends_on missing id {s.depends_on}")
                if s.depends_on >= s.id:
                    raise ValueError(
                        f"subquery {s.id} depends_on {s.depends_on} is not a lower id "
                        "(forward/self dependency)"
                    )
            if s.kind == SubqueryKind.analytical and s.coarse_intent is None:
                raise ValueError(f"subquery {s.id} is analytical but has no coarse_intent")
            if s.kind == SubqueryKind.coverage and s.coarse_intent is not None:
                raise ValueError(
                    f"subquery {s.id} is coverage but has coarse_intent {s.coarse_intent!r}"
                )
        return self


def build_planner_output_model(intent_catalog: Sequence[str]) -> type[BaseModel]:
    """Per-client dynamic schema: coarse_intent can only be one of this client's real intents."""

    intent_type: Any = Literal[tuple(intent_catalog)] if intent_catalog else str

    planner_subquery = create_model(
        "PlannerSubqueryOutput",
        __config__=ConfigDict(extra="forbid"),
        id=(
            int,
            Field(
                ...,
                description=(
                    "0-based index in dependency order; a sub-query's dependency must have a "
                    "lower id."
                ),
            ),
        ),
        query=(str, Field(..., description="The rephrased atomic sub-query, self-contained.")),
        kind=(
            Literal["coverage", "analytical"],
            Field(
                ...,
                description=(
                    "'coverage' if the sub-query asks what data EXISTS (how many "
                    "periods/quarters, which values exist, a date range, whether a "
                    "value/metric is present, or which X co-occur with which Y). 'analytical' "
                    "if it asks to analyze or compute a metric (performance, roi, spend, "
                    "change, ...). A question like 'what metrics are available for X' is "
                    "coverage, not analytical, even though it mentions a metric -- it asks "
                    "what exists, not to compute anything. 'How many' is analytical when it "
                    "asks for a metric's value ('how many impressions did digital video get', "
                    "'how many conversions last year'); it is coverage only when it counts "
                    "what the data holds ('how many quarters of paid social do we have')."
                ),
            ),
        ),
        coarse_intent=(
            intent_type | None,
            Field(
                None,
                description=(
                    "For analytical sub-queries, the single best matching intention from the "
                    "client's catalog; null for coverage sub-queries."
                ),
            ),
        ),
        depends_on=(
            int | None,
            Field(
                None,
                description=(
                    "The id of the sub-query whose answer or scope this one needs first (e.g. "
                    "a computed entity like 'the top tactic', or an inherited filter); null if "
                    "independent. A coverage sub-query that a later analytical sub-query "
                    "trends or analyzes is a common example of a dependency."
                ),
            ),
        ),
    )
    return create_model(
        "PlannerPlanOutput",
        __config__=ConfigDict(extra="forbid"),
        rephrased_query=(
            str,
            Field(..., description="A clear, professional rewrite of the whole query."),
        ),
        is_multi=(
            bool,
            Field(
                ...,
                description=(
                    "True if the query contains more than one distinct question/intention."
                ),
            ),
        ),
        subqueries=(list[planner_subquery], ...),  # type: ignore[valid-type]
    )


HIERARCHY_SENTINELS = ("irrelevant", "all", "specific")


def sanitize_field_name(name: str) -> str:
    """Make a questionnaire field name usable as a tool-call parameter name.

    Tool/function-calling parameter names must be identifier-safe. Confirmed live against the
    real gateway: a field named 'business unit focused marketing' makes it reject the entire tool
    definition with HTTP 500, which took the whole halo dimension down with it -- including
    'business_unit_for_kpi', which resolves correctly on its own. Isolated by testing the same
    call with underscores ('business_unit_for_kpi' -> 'lms'), with spaces (HTTP 500), and
    sanitized ('business_unit_focused_marketing' -> 'lms').
    """

    cleaned = re.sub(r"[^0-9a-zA-Z_]", "_", name).strip("_")
    return cleaned or "field"


def build_field_extraction_model(
    fields: Sequence[str], hierarchy_fields: Collection[str] = (), *, optional: bool = False
) -> type[BaseModel]:
    """Per-dimension dynamic schema for Question Understanding's field-fill LLM calls, mirroring
    build_planner_output_model's per-client dynamic schema pattern above.

    Fields named in `hierarchy_fields` are a constrained three-way classification rather than a
    value to extract, matching the source's get_answer_format for custom level/tagging columns
    (filter_generator.py:55-57): the LLM returns 'irrelevant'/'all'/'specific', and 'specific' is
    later resolved to concrete values from the spaCy captures. Asking for the value outright is
    what made these fields fail to resolve at all.

    `optional=True` lets a field come back unanswered (null), for callers where "this field was
    not addressed" is a real and distinct outcome that must not be recorded as an answer.
    Everywhere else an answer is mandatory, because a field the model silently omits is a field
    that vanishes from the filter.
    """

    default: Any = None if optional else ...
    unanswered_note = (
        " Answer null if the reply does not address this field at all -- do not guess."
        if optional
        else ' Always answer; use "irrelevant" rather than omitting the field.'
    )

    field_defs: dict[str, Any] = {}
    for original_name in fields:
        # The schema is keyed by the sanitized name; _fill_dimension maps answers back.
        name = sanitize_field_name(original_name)
        if name in field_defs:
            raise ValueError(
                f"questionnaire fields collide after sanitizing to {name!r}: {list(fields)}"
            )
        if original_name in hierarchy_fields:
            hierarchy_type: Any = Literal[HIERARCHY_SENTINELS]
            field_defs[name] = (
                hierarchy_type | None if optional else hierarchy_type,
                Field(
                    default,
                    description=(
                        f"Whether the question refers to '{original_name}': 'irrelevant' if "
                        "it isn't mentioned at all, 'all' if referred to in general "
                        "(by/each/for all/different, or comparing one against the rest), "
                        "'specific' if particular named values are given." + unanswered_note
                    ),
                ),
            )
        else:
            field_defs[name] = (
                list[str] | str | None if optional else list[str] | str,
                Field(
                    default,
                    description=(
                        f"For '{original_name}', answer with exactly one of: \"irrelevant\" "
                        'if the question does not mention it, "all" if it refers to every '
                        "value in general, or the specific matching value(s) named in the "
                        "question. Use a plain string for one value or a real JSON array for "
                        "several -- never a string containing an array, e.g. never "
                        '"[\\"202504\\"]".' + unanswered_note
                    ),
                ),
            )
    return create_model(
        "FieldExtractionOutput", __config__=ConfigDict(extra="forbid"), **field_defs
    )


class ResolveOutcome(BaseModel):
    """Thin schema: dependency resolution only ever returns text, never a reclassification."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    subqueries: tuple[str, ...] = Field(
        default_factory=tuple,
        description=(
            "One self-contained sub-query per REPHRASE/SPLIT result; empty when rejected."
        ),
    )
    resolved: bool = Field(
        default=False,
        description="False if the previous context does not contain what the follow-up needs.",
    )
    reason: str = Field(default="", description="Brief reason, required when resolved is false.")


class CoverageSpec(BaseModel):
    """Ported from ask-genome-core's CoverageSpec. Frozen here, so answer_coverage updates it
    with model_copy instead of in place."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    dimensions: list[str] = Field(default_factory=list)
    values: list[str] = Field(default_factory=list)
    metric: str | None = None
    period_type: str | None = None
    operation: Literal[
        "count_periods",
        "time_range",
        "list_values",
        "metric_availability",
        "existence",
        "co_occurrence",
    ]
    needs_join: bool = False


class DependencyResolution(BaseModel):
    """What the Dependent Subquery Handler decided for exactly one dependent subquery.

    ``raw_text`` carries unassigned strings for "rephrased"/"split" -- id assignment and list
    surgery happen afterward, in ``apply_resolution``, which needs the whole plan's id space.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["unchanged", "rejected", "rephrased", "split"]
    raw_text: tuple[str, ...] = Field(default_factory=tuple)
    reason: str | None = None
