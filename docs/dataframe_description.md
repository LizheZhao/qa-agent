# Data frame description step (ask-genome)

## Decision log

1) Ask: a data frame description step for the ask-genome agent.
2) Chosen form: a graph step. Chosen purpose: prompt context for the LLM.
3) The description may also use `level_type`, the hierarchy columns and the taggings.
4) Correction: `generate_response` must not use it. It reads only the readout, because the readout
   already carries the analysis added to the constructed tables.
5) Intended consumer: follow-up handling. The description should help the plan reconciliator decide
   whether a follow-up is
   a) answerable from the current data with no new query
   b) in need of a new query plus the current data
   c) unrelated, so a new session or graph is triggered

## What is built

`describe_dataframe(frame, level_type, data_levels)` in
`packages/agents/ask-genome/src/ask_genome_agent/nodes/question_understanding/support/describe.py`
returns plain text covering:

- row and column counts
- time range and period types
- metrics
- value min, max and missing count
- reporting levels and rows per level
- each hierarchy column (`business_driver` down to `measure`) with distinct counts and sample values
- each `level_type` group (`general_level`, `general_tagging`, `halo_level`, `halo_tagging`) with
  distinct counts and sample values

`apply_data_filters.py` calls it on the filtered frame and stores the result as
`ner_state[subquery.id]["dataframe_description"]`. Nothing reads it yet.

It is not a separate graph node. The filtered frame exists only inside `apply_data_filters`, and a
separate node would have to rebuild it from the table recipe, which reruns the narrowing on a result
that can reach about 96k rows. Making it a node later means paying that rebuild.

## Current query split and dependency flow

```
START
  |
extract_query
  |
decompose_and_classify   (LLM: splits the raw query into subqueries, sets kind,
  |                       coarse_intent and depends_on, all in one call)
annotate_feasibility     (spaCy hints + feasibility check, analytical only)
  |
assemble_plan            (builds AskGenomePlan, active_index=0, context={})
  |
  v
_route_next  <-------------------------------------------------+
  | no more subqueries ------------------------> aggregate_results --> END
  | depends_on is None:                                          ^
  |    coverage   --> answer_coverage ---------+                 |
  |    analytical --> extract_entities         |                 |
  |                    ... question understanding                |
  |                    ... resolve_data_source / apply_data_filters
  |                    ... run_postprocess --> generate_response |
  |                         |                  |                 |
  |                         +---------+--------+                 |
  |                                   v                          |
  |                             write_context                    |
  |                       (stores result under subquery.id)      |
  |                                   |                          |
  |                             advance_cursor (index + 1) ------+
  |
  | depends_on is set:
  v
resolve_dependency       (LLM sees: context[depends_on] + the follow-up text)
  |  no stored context -> rejected, no LLM call
  v
apply_resolution         (edits the plan in place)
  |
_route_after_resolution
     rephrased --> restart this same subquery (depends_on cleared, text rewritten)
     split     --> subquery replaced by N independent ones, then _route_next
     rejected  --> write_context (empty result, reason saved in `unanswered`)
```

How the split happens:

1) `decompose_and_classify` makes the first split, with `depends_on` pointing at the predecessor id.
2) `resolve_dependency` can split again at runtime.
   a) One text back: the subquery is rewritten to be self-contained and `depends_on` is cleared
      (`rephrased`).
   b) Several texts back: `apply_resolution` replaces the subquery with N new ones (`split`). Each
      gets a fresh id (max id plus 1), `kind=analytical` and no dependency, and `is_sequential` is
      recomputed.
   c) Not resolvable: `rejected`, with the reason recorded in `unanswered`.

A dependent subquery sees only `context[depends_on]`, which is the predecessor's
`last_subquery_result` (the user-facing answer, or the readout summary line). It never sees the
data, filter, levels or taggings.

## Limits relevant to the follow-up idea

1) `context` is created fresh in `assemble_plan` on each invocation. Nothing handles a follow-up that
   arrives as a new user turn.
2) There is no "answer from current data" outcome. The outcomes are rephrase, split or reject, and a
   rephrase always runs a fresh data query.
3) No code decides that a follow-up is unrelated and should start a new graph.

## Open items

1) Confirm the mapping of the three follow-up outcomes onto `resolve_dependency` and
   `apply_resolution`, and where the new-session case hooks in (probably the router).
2) Choose the prompt that receives the description. Recommendation: `resolve_dependency`, as a
   "current data" block, with a third outcome "answerable from current data".
3) Check whether `context` and `ner_state` persist across turns. Unverified.
4) Not yet run: `mypy` aborted on a duplicate-module error in `src/integrations/observability.py`,
   and the ask-genome tests do not collect locally because `langchain_core.language_models` is
   missing from the venv. No unit test exists for `describe_dataframe`. `ruff check` passes.
