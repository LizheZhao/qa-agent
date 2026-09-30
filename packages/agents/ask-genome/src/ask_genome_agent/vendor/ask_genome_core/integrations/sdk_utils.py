# Vendored from ask-genome-core bb1e1637bca4:(stub)
# by scripts/sync_ask_genome_core.py. Do not edit by hand; change EDITS there.
"""Stands in for ask-genome-core's src/integrations/sdk_utils.py.

Nothing vendored calls these; they are imported by functions the agent does not reach. Model calls
in the orchestrator go through its own gateway, so reaching one of these is a bug to report.
"""

from typing import Any


def _unavailable(*_args: Any, **_kwargs: Any) -> Any:
    raise NotImplementedError("src/integrations/sdk_utils.py is not vendored; use the orchestrator's model instead")


coverage_spec_call = _unavailable
text_llm_call = _unavailable
