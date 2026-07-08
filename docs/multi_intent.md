# Multi-intent workflow

How a single user query that carries more than one intention is detected, split, and answered. Entry point: `generate_ner_filter_considering_multi_intent(clientCode, modelgroupId, query)` in [src/model/filter_generator.py:1854](../src/model/filter_generator.py). Both `st-main.py` and `main.py` call it.

## Three separated stages

Detection, prediction, and execution are deliberately split into distinct functions so detection can run with no pipeline cost.

### 1) Detection + prediction — `detect_and_separate_multi_intent` (filter_generator.py:1791)

Pure: no NER, no `process_data`, no `response_generate` side effects. Safe to call from eval code.

a) One LLM call with `MULTI_INTENT_SEPARATION_PROMPT` ([config/prompt.yaml:124](../config/prompt.yaml)).
b) `postprocess_multi_intent_detection` validates and unpacks the JSON payload `{is_multi, is_sequential, sub-queries}`. Missing keys raise `ValueError`.
c) If multi, run the BERT classifier (`single_request`) per sub-query to predict each sub-intention, after acronym expansion.
d) **Collapse multi→single when all sub-intents agree** — a "split" that resolves to one intention pays no downstream pipeline cost.

Returns `(single, sequential, subqueries, sub_multi_preds)`.

### 2) Execution — orchestrator branches three ways

NER (`generate_ner_filter`) runs symmetrically across all cases; the cached `multitask_pred` is passed in so the BERT classifier is not re-run.

| Branch | Condition | Behavior |
|---|---|---|
| single | not multi (or collapsed) | one `generate_ner_filter` on the full query |
| parallel | multi, not sequential | independent `generate_ner_filter` per sub-query; sub-queries are mutually independent |
| sequential | multi, sequential | each sub-query is rewritten with the *prior* sub-query's full LLM answer via `FILL_SUBQUERY_WITH_PREV_RESPONSE_PROMPT` ([config/prompt.yaml:203](../config/prompt.yaml)) before NER |

The sequential branch **must** run `process_data` + `response_generate` inline (to thread the prior answer forward), so it returns those responses for the caller to reuse.

### 3) Return shape

```python
(single, sequential, readout_data_list, filled_subqueries, sub_responses)
```

`sub_responses` is populated **only** for the sequential branch. single and parallel return `[]`, and the caller computes responses itself.

## How responses are combined

There is no LLM-level synthesis across sub-queries. They are kept separate.

- **main.py** — reuses `sub_responses` when present (sequential), else runs `process_data` + `response_generate` per readout, then string-joins: `"\n\n".join(f"[{q}]\n{resp}")`.
- **st-main.py** — `single` takes the original single-shot rendering path (L67–157). The multi branch (L158–250) loops over `zip(filled_subqueries, readout_data_list)`, rendering each sub-query under its own `### Sub-query {i+1}` header with its own tables (overall, pivot biz, pivot, benchmark, planner), reusing `sub_responses[i]` for sequential and computing inline for parallel.

## Sequential threading

```
sub-query 1 ──NER──► process_data ──► response_generate ──► response 1
                                                               │
                  FILL_SUBQUERY_WITH_PREV_RESPONSE_PROMPT ◄────┘
                                  │
                                  ▼
sub-query 2 (filled) ──NER──► process_data ──► response_generate ──► response 2 ─► ...
```

If a sub-query's `ner_filters["data"]` is `None`, its threaded response is set to `""` so the next fill step does not break.

## Contracts to keep in sync

The orchestrator and both callers depend on the exact 5-tuple and on `sub_responses` being empty for non-sequential cases. When editing any of the three stages, preserve those contracts.
