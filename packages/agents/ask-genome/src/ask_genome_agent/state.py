"""Private conversation state for the Ask Genome agent.

Standard reduced message state plus the working fields one invocation threads through its plan
and its subquery loop. Client configuration is not in here: nothing in the request carries a
per-request client, so config.py loads it lazily inside the nodes that need it.
"""

from typing import Any, NotRequired

from langgraph.graph import MessagesState
from orchestration_core import ClarificationRequest, ExecutionContext

from ask_genome_agent.contracts import AskGenomePlan, DependencyResolution


class AskGenomeState(MessagesState):
    execution_context: ExecutionContext

    # Set once, then held for the rest of the run.
    query: NotRequired[str]
    plan_output: NotRequired[dict[str, Any]]
    plan: NotRequired[AskGenomePlan]

    # Keyed by Subquery.id, never list position: ids survive a splice and positions do not.
    active_index: NotRequired[int]
    context: NotRequired[dict[int, str]]
    resolution: NotRequired[DependencyResolution | None]
    last_subquery_result: NotRequired[str]

    # Subqueries refused for want of a resolvable predecessor. They leave no ner_state entry, so
    # without this the final message would never mention them.
    unanswered: NotRequired[dict[int, str]]

    # Question Understanding's per-subquery working state, accumulated in chain order:
    #   extract_entities           -> extended_query, detected_pair, spacy_filter
    #   classify_fields            -> predictions, probabilities
    #   reconcile_intent           -> predictions, clarification_list, source_conflict
    #   fill_remaining_fields      -> ner_results, field_status
    #   build_clarification_list   -> clarification_fields
    #   resolve_data_source        -> data_source, out_of_scope
    #   apply_data_filters         -> applied_filter, records, records_truncated, row_count,
    #                                 rejection_message, time_tag, table_recipe, table_id,
    #                                 data_levels, latest_time, latest_period_type
    # `records` is a bounded sample and `row_count` the true total; `table_recipe` rebuilds the
    # whole result on demand. `data_levels` names the three granularities the rows report at, and
    # the rows carry `record_level`, which is ask-genome-core's three dataframes held as one.
    # test_ask_genome_plan_wiring.py pins this contract by running the chain for real, which the
    # per-node tests cannot.
    ner_state: NotRequired[dict[int, dict[str, Any]]]

    # Clarifications still to ask, head first, since only one interrupt may be live at a time.
    queued_clarifications: NotRequired[tuple[ClarificationRequest, ...]]
