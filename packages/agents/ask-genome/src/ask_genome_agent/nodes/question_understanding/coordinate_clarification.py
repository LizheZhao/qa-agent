"""Asks one clarification, pauses the graph, and applies the answer when it resumes.

Replaces the end-the-turn pattern this file grew out of. The graph now stops at the interrupt
continues in place, so `ner_state` survives: the spaCy captures, the BERT probabilities and every
field already resolved are exactly as they were. Nothing re-plans, so the dimension and time range
from the original question cannot be lost in a rephrase, which is what the previous design kept
doing.

Only one live interrupt is allowed per invocation, so several flagged fields become several
requests: the head is asked and the rest wait in ClarificationQueue. Answering the head promotes
the next before anything else runs.

LangGraph restarts an interrupted node from its beginning on resume, so everything above the
interrupt call must be free of side effects. Reading the queue is; applying the answer sits after
it deliberately.
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.types import interrupt
from orchestration_core import (
    QUEUED_CLARIFICATIONS_CHANNEL,
    ClarificationQueue,
    ClarificationRequest,
)

from ask_genome_agent.nodes.question_understanding.support.clarification_requests import (
    clarification_id_for,
)
from ask_genome_agent.state import AskGenomeState

logger = logging.getLogger(__name__)

_CANCELLED = "cancelled"


def field_for(request: ClarificationRequest, candidates: list[str], subquery_id: int) -> str | None:
    """Which questionnaire field a request was asked about.

    The clarification id is derived from the sanitized field name, so the real name is recovered
    by matching rather than by carrying a second mapping through checkpointed state.
    """

    for field in candidates:
        if clarification_id_for(subquery_id, field) == request.clarification_id:
            return field
    return None


def answer_value(request: ClarificationRequest, reply: Any) -> str | list[str] | None:
    """The filter value a validated response selects, or None when nothing was chosen.

    Option ids are sanitized, so the label is the real column value and is what the filter needs.
    Free text is taken at face value: the user typed a value the option list did not offer.
    """

    if not isinstance(reply, dict):
        return str(reply)
    form = reply.get("form")
    if form == "options":
        labels = {option.option_id: option.label for option in request.options}
        chosen = [labels.get(str(oid), str(oid)) for oid in reply.get("option_ids", ())]
        if not chosen:
            return None
        return chosen[0] if len(chosen) == 1 else chosen
    if form == "free_text":
        return str(reply.get("text", "")).strip() or None
    if form == "cancel":
        return None
    return None


async def coordinate_clarification(state: AskGenomeState) -> dict[str, Any]:
    subquery = state["plan"].subqueries[state["active_index"]]
    queue = ClarificationQueue(pending=tuple(state.get(QUEUED_CLARIFICATIONS_CHANNEL, ())))
    active = queue.active
    if active is None:
        return {}

    # Nothing above this line may have side effects: the node reruns from here on resume.
    reply = interrupt(active.model_dump(mode="json"))

    remaining = queue.answered(active.clarification_id)
    ner_state = dict(state.get("ner_state", {}))
    entry = dict(ner_state[subquery.id])
    clarification_fields = dict(entry.get("clarification_fields", {}))

    field = field_for(active, list(clarification_fields), subquery.id)
    value = answer_value(active, reply)

    if field is None:
        logger.warning("resumed clarification %r matches no field", active.clarification_id)
    elif value is None:
        # Cancelled, or an empty response. The field stays in clarification_fields so
        # aggregate_results discloses it, rather than recording an answer the user never gave.
        logger.info("clarification %r left %r unresolved", active.clarification_id, field)
    else:
        ner_results = dict(entry.get("ner_results", {}))
        field_status = dict(entry.get("field_status", {}))
        ner_results[field] = value
        field_status[field] = "user-confirmed"
        entry["ner_results"] = ner_results
        entry["field_status"] = field_status
        clarification_fields.pop(field, None)

    entry["clarification_fields"] = clarification_fields
    ner_state[subquery.id] = entry
    return {QUEUED_CLARIFICATIONS_CHANNEL: remaining.pending, "ner_state": ner_state}


def should_clarify(state: AskGenomeState) -> str:
    """Whether anything is left to ask.

    There is no round cap any more. The queue is finite, the runtime permits one active pause per
    session, and a clarification expires on its own, so the loop cannot run away.
    """

    return (
        "coordinate_clarification"
        if state.get(QUEUED_CLARIFICATIONS_CHANNEL)
        else "finalize_filter"
    )


def after_clarification(state: AskGenomeState) -> str:
    """Ask the next queued question before letting the filter finalize."""

    return should_clarify(state)
