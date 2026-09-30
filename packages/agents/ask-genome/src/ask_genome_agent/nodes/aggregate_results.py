"""The graph's terminal node. Every agent here must end with an AIMessage for the router.

Each subquery reports its generated answer, or failing that its readout, which is terse but
carries the figures. One that produced neither falls back to what filtering found. The plan
itself stays in graph state, where the diagnostics UI shows it.

Fields the user cancelled out of, or was never asked about, are listed so the answer is not built
on a guess they never saw. A source_conflict is surfaced for the same reason: the metric and the
dimension come from datasets that do not go together, so the filter will likely return nothing.
"""

from typing import Any

from langchain_core.messages import AIMessage

from ask_genome_agent.state import AskGenomeState

# The readout fallback is sized for a prompt, not a chat message.
_MAX_READOUT = 4_000


def _label_for(entry: dict[str, Any], subquery: Any) -> str:
    """How to name one part of a multi-part question.

    The resolved intention is the client's own wording, so it is what to say. An out-of-scope or
    unresolved subquery has none, and the planner's coarse intent is the next best thing.
    """

    source = entry.get("data_source") or {}
    return str(source.get("intention") or subquery.coarse_intent or "").strip()


def _data_outcome(entry: dict[str, Any], *, label: str = "") -> str:
    """What filtering actually found, in the user's terms.

    An out-of-scope intention and an empty result are different failures and are worth saying
    differently: one means the question cannot be asked of this data at all, the other means it
    can but nothing matched. A non-empty rejection alongside rows is also real: one dimension was
    dropped, so the rows are narrower than what was asked for.

    `label` says which part of a multi-part question this is. Without it, two subqueries that
    match the same number of rows print the same sentence twice and read as a bug.
    """

    if entry.get("out_of_scope"):
        subject = f"'{label}'" if label else "That"
        return f"{subject} isn't something I can answer from this client's data."

    answer = (entry.get("answer") or "").strip() or (entry.get("readout") or "").strip()[
        :_MAX_READOUT
    ]
    if answer:
        return f"For {label}: {answer}" if label else answer

    if "row_count" not in entry:
        return ""

    if entry["row_count"] == 0:
        found = "No data matched that combination."
    else:
        found = f"Found {entry['row_count']} matching rows."
    if label:
        found = f"For {label}, {found[0].lower()}{found[1:]}"

    rejection = entry.get("rejection_message", "")
    return f"{found} {rejection}".strip() if rejection else found


def _refused(subquery: Any, reason: str) -> str:
    """One part of a question that was never run.

    It depends on an earlier part, and that part produced a filter rather than an answer, so
    there was nothing to resolve the reference against. Refusing is right; reporting only the
    other part's row count, as if the question had been answered, is not.
    """

    asked = subquery.query.rstrip("?").strip()
    # The reason comes from resolve_dependency's model and may already be a full sentence.
    return f"I couldn't answer '{asked}': {reason.strip().rstrip('.')}."


async def aggregate_results(state: AskGenomeState) -> dict[str, Any]:
    plan = state["plan"]
    count = len(plan.subqueries)
    summary = f"I extracted {count} subquer{'y' if count == 1 else 'ies'} from your request."

    still_unresolved: set[str] = set()
    conflicts: list[dict[str, Any]] = []
    outcomes: list[str] = []
    # Plan order, not dict order, so the parts line up with how the question was asked. One
    # subquery needs no label: there is nothing to tell it apart from.
    unanswered = state.get("unanswered", {})
    for subquery in plan.subqueries:
        if subquery.id in unanswered:
            outcomes.append(_refused(subquery, unanswered[subquery.id]))
            continue
        entry = state.get("ner_state", {}).get(subquery.id)
        if entry is None:
            continue
        still_unresolved.update(entry.get("clarification_fields", {}).keys())
        conflict = entry.get("source_conflict")
        if conflict:
            conflicts.append(conflict)
        outcome = _data_outcome(entry, label=_label_for(entry, subquery) if count > 1 else "")
        if outcome:
            outcomes.append(outcome)

    for outcome in outcomes:
        summary += f" {outcome}"

    if still_unresolved:
        fields = ", ".join(sorted(still_unresolved))
        summary += (
            f" I wasn't able to fully clarify everything, so this is my best interpretation."
            f" Still unsure about: {fields}."
        )

    for conflict in conflicts:
        dimensions = ", ".join(str(d) for d in conflict.get("dimensions", [])) or "that dimension"
        # Intention names come from the client's own map_dict.csv, so they are their vocabulary
        # rather than internal keys. They name a kind of analysis, not a metric, so word it that
        # way: alongside a metric like 'margin roi' they otherwise read as one.
        summary += (
            f" Note that '{conflict['intention']}' does not appear to be available for"
            f" {dimensions}, so this may return nothing. The analysis available for"
            f" {dimensions} is '{conflict['alternative_intention']}', if that is what you meant."
        )

    return {"messages": [AIMessage(content=summary)]}
