"""Question Understanding: resolves one analytical subquery's filter (platform, metric, time
range, etc.) via spaCy + BERT + LLM extraction and an optional user-clarification loop.

Node chain (see graph.py for wiring): extract_entities -> classify_fields -> reconcile_intent ->
fill_remaining_fields -> build_clarification_list -> should_clarify, which routes to either
finalize_filter or coordinate_clarification. coordinate_clarification pauses the graph with one
ClarificationRequest, applies the answer on resume, and loops while any remain queued.

finalize_filter then hands off to resolve_data_source and apply_data_filters, which choose the
dataset the intention reads and apply the filter to it. Those nine nodes together are what
executes one subquery. The chain used to be a single unimplemented node called "execute_subquery",
which is where that name survived until it was renamed to match its file.

Stops at filtered rows. Computing a value from them and phrasing an answer is response
generation, and is not implemented.
"""
