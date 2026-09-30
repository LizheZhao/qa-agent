"""Final step of Question Understanding: tag the resolved filter and hand it on.

Clarification itself lives in coordinate_clarification. This module used to hold it, back when a
question ended the turn and the answer arrived as a new one; that pattern existed only because the
orchestration framework had no way to pause a run, and it is gone now that it does.
"""

from typing import Any

from ask_genome_agent.state import AskGenomeState


def finalize_filter(state: AskGenomeState) -> dict[str, Any]:
    subquery = state["plan"].subqueries[state["active_index"]]
    ner_state = dict(state["ner_state"])
    entry = dict(ner_state[subquery.id])

    ner_results = entry["ner_results"]
    # Keep the bert/llm/user-confirmed provenance; only fields with no recorded origin become
    # "auto-confirmed". Overwriting all of them erased the signal an uncertainty flagger would
    # reason over.
    field_status = dict(entry.get("field_status", {}))
    for name in ner_results:
        field_status.setdefault(name, "auto-confirmed")
    entry["field_status"] = field_status
    ner_state[subquery.id] = entry

    summary = ", ".join(f"{k}={v}" for k, v in sorted(ner_results.items()))
    return {"ner_state": ner_state, "last_subquery_result": summary}
