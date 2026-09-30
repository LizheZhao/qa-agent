"""Turns the fields build_clarification_list flagged into ClarificationRequest objects.

The runtime asks one question at a time: each request carries a single field, and the rest wait in
a ClarificationQueue. The invocation boundary rejects more than one live interrupt, so the older
shape, which listed several fields in one message, cannot survive.

Two contract limits shape the output. option_id must be opaque-safe, so the sanitized name is the
id and the real value is the label, the same split build_field_extraction_model uses for tool
parameters. And options are capped at MAX_CLARIFICATION_OPTIONS.

Four of LINKEDIN's fields offer more values than that cap: intention at 11, start and end at 16,
detailed_kpi at 13. Dropping "out-of-scope" brings intention to exactly 10, and it was never a
choice a user makes; it is what the system concludes, and the response union already carries
cancel and free text.

A field with more values than that offers its likeliest ten and says how many more exist. That
works because free text is not optional: PendingClarification fixes free_text_allowed to True, so
every question accepts a typed answer whatever its options say. Ranking therefore matters more
than completeness, and the values spaCy matched for the field, or that appear in the question
itself, go first. LINKEDIN's start and end reach 16 values and KCC's sub brand reaches 35, so no
cap setting would have covered both.
"""

from __future__ import annotations

import logging
from collections.abc import Collection, Mapping, Sequence

from orchestration_core import (
    MAX_CLARIFICATION_OPTIONS,
    ClarificationOption,
    ClarificationRequest,
    order_clarifications,
)

# Not re-exported from the package root, unlike the option cap.
from orchestration_core.clarification import (
    MAX_OPTION_LABEL_CHARACTERS,
    MAX_QUESTION_CHARACTERS,
)

from ask_genome_agent.contracts import sanitize_field_name

logger = logging.getLogger(__name__)

AGENT_ID = "ask_genome"
DEFAULT_REASON_CODE = "incomplete_extraction"

# Everything is multi-select except intention, matching the source's _format_user_selections:
# it takes selected[0] for intention and leaves every other field as the list it arrived as.
# A question can only have one intention, while a filter can legitimately span several countries
# or business units.
_SINGLE_SELECT_FIELDS = frozenset({"intention"})

# Not a user-selectable answer: it is what the system concludes, not what the user meant.
# Removing it is also what brings intention inside the option cap.
_NON_SELECTABLE = frozenset({"out-of-scope"})


def task_id_for(subquery_id: int) -> str:
    return f"subquery-{subquery_id}"


def clarification_id_for(subquery_id: int, field: str) -> str:
    return f"sq{subquery_id}-{sanitize_field_name(field)}"


def _options_for(
    values: Sequence[str], likely: Collection[str] = ()
) -> tuple[tuple[ClarificationOption, ...], int]:
    """Selectable options for one field, most likely first, and how many were left out.

    Free text is always accepted, so a field with more values than the cap does not have to be
    skipped: it offers the best candidates and the user types anything else. That makes the order
    load-bearing rather than cosmetic, which is why the values spaCy matched and the values named
    in the question come first.

    Sanitizing ids can collide for a client holding both "sub brand" and "sub_brand", and the
    contract requires distinct ids, so the first value wins rather than the whole request failing.
    """

    ranked = [v for v in values if str(v) in likely] + [v for v in values if str(v) not in likely]

    options: list[ClarificationOption] = []
    seen: set[str] = set()
    for value in ranked:
        text = str(value)
        if text in _NON_SELECTABLE:
            continue
        option_id = sanitize_field_name(text)
        if option_id in seen:
            continue
        seen.add(option_id)
        options.append(
            ClarificationOption(option_id=option_id, label=text[:MAX_OPTION_LABEL_CHARACTERS])
        )
    omitted = max(0, len(options) - MAX_CLARIFICATION_OPTIONS)
    return tuple(options[:MAX_CLARIFICATION_OPTIONS]), omitted


def _question_for(field: str, omitted: int) -> str:
    question = f"Which {field.replace('_', ' ')} did you mean?"
    if omitted:
        question += f" These are the likeliest; {omitted} more exist, so you can type another."
    return question[:MAX_QUESTION_CHARACTERS]


def build_clarification_requests(
    clarification_fields: Mapping[str, Sequence[str]],
    *,
    subquery_id: int,
    freshness_token: str,
    reason_codes: Mapping[str, str] | None = None,
    likely_values: Mapping[str, Collection[str]] | None = None,
) -> tuple[ClarificationRequest, ...]:
    """One deterministically ordered request per flagged field.

    `freshness_token` identifies the extraction state the questions were asked about; the runtime
    refuses a resume whose token no longer matches the checkpoint. `likely_values` ranks a field's
    options so the ones that survive the cap are the plausible ones.

    A field with nothing selectable produces no request, since the contract requires at least one
    option. It stays in clarification_fields and aggregate_results discloses it.
    """

    reason_codes = reason_codes or {}
    likely_values = likely_values or {}
    requests: list[ClarificationRequest] = []

    for field, values in clarification_fields.items():
        options, omitted = _options_for(values, likely_values.get(field, ()))
        if not options:
            logger.info("not asking about %r: nothing selectable to offer", field)
            continue
        if omitted:
            logger.info("offering %d of %d values for %r", len(options), len(values), field)
        requests.append(
            ClarificationRequest(
                clarification_id=clarification_id_for(subquery_id, field),
                producer_agent_id=AGENT_ID,
                task_id=task_id_for(subquery_id),
                question=_question_for(field, omitted),
                reason_code=reason_codes.get(field, DEFAULT_REASON_CODE),
                selection_mode="single" if field in _SINGLE_SELECT_FIELDS else "multiple",
                options=options,
                freshness_token=freshness_token,
            )
        )

    return order_clarifications(requests, plan_task_ids=(task_id_for(subquery_id),))
