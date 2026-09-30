"""
Deterministic: write_context.py node updates the context store in state after a subquery
has been processed.
It records the result of the subquery in the context store, keyed by the subquery's ID, and
resets the resolution and last_subquery_result fields in state for the next iteration.
"""

from typing import Any

from ask_genome_agent.state import AskGenomeState


def write_context(state: AskGenomeState) -> dict[str, Any]:
    subquery = state["plan"].subqueries[state["active_index"]]
    resolution = state.get("resolution")
    rejected = resolution is not None and resolution.kind == "rejected"

    # Postprocessing can decline -- an intention it does not implement, a client strategy it does
    # not have, a recipe that no longer describes its table. Without this the subquery looks
    # answered: a refusal would be written into context as though it were a result, and a stale
    # recipe would leave the filtering step's row count standing as the analytical outcome.
    entry = state.get("ner_state", {}).get(subquery.id) or {}
    postprocess_error = entry.get("postprocess_error")

    failed = rejected or bool(postprocess_error)
    result = "" if failed else state.get("last_subquery_result", "")
    context = dict(state["context"])
    context[subquery.id] = result

    # A rejected subquery is the only one that leaves no ner_state entry, and `resolution` is
    # cleared on the next line, so without recording it here the final message reports one
    # answer to a two-part question and never says the other part was refused.
    unanswered = dict(state.get("unanswered", {}))
    if rejected:
        unanswered[subquery.id] = (resolution.reason if resolution else "") or "unresolved"
    elif postprocess_error:
        unanswered[subquery.id] = str(postprocess_error)

    return {
        "context": context,
        "unanswered": unanswered,
        "resolution": None,
        "last_subquery_result": "",
    }
