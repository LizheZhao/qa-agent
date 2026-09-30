"""Deterministic: advance the cursor by exactly one position making it strictly sequential to
execute one subquery at a time.
"""

from typing import Any

from ask_genome_agent.state import AskGenomeState


def advance_cursor(state: AskGenomeState) -> dict[str, Any]:
    return {"active_index": state["active_index"] + 1}
