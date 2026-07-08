# Planner layer (v1) - change log

Branch: `multi_turn` (PRs target `feature`). Adds a planner that runs **before** NER to
decompose a query, route coverage vs analytical sub-queries, and annotate coarse feasibility.

## Why

Three gaps on `multi_turn`: multi-part questions were not decomposed; coverage/metadata
questions ("how many quarters did we run paid social", "which countries have which brands")
had no BERT intention and did not fit the metric-synthesis pipeline; structural feasibility
was only discovered post-fetch. The planner closes these by understanding the query up front
while reusing the existing NER / clarification / readout machinery.

## Flow

```
user query
  -> PLANNER (1 Anthropic structured-output call) + spacy pre-pass + feasibility lookup   [no BERT here]
       emits subqueries[ {id, query, kind, coarse_intent, depends_on} ]
  -> ORCHESTRATOR PAGE loops sub-queries in order:
       coverage   -> coverage branch (config lookup or live groupby), bypass NER/readout
       analytical -> [if depends_on: resolve_from_context(predecessor) -> reject / rephrase(1) / split(N)]
                     clarification graph (BERT intent reconciles vs coarse_intent)
                     -> generate_readoutdata -> process_data
                     -> _prep_visual_context + response_generate_with_denial   (mirrors insights_only)
  -> render each finished sub-query in its own expander (query text + insights + tables inside)
```

No DAG (linear `depends_on`). Feasibility is annotate-only (NER decides). No downshift in v1.

## Sequential dependency: unified context resolution

A sub-query with `depends_on` set is resolved at runtime from its predecessor's context by a
single LLM step (`resolve_from_context`) that decides among three outcomes (no separate flag):

- **reject** - empty list + `resolved=false` (predecessor gave nothing usable) -> dependency-rejection message
- **rephrase** - exactly one self-contained sub-query (the classic sequential case)
- **split** - N self-contained analytical sub-queries, one per relevant item in the context
  (e.g. one trend sub-query per metric). The count is data-dependent and decided by the LLM.

On split, the orchestrator fans the dependent out into N analytical `EnrichedSubQuery` (fresh ids,
`depends_on=None`, `coarse_intent=None`) spliced into the plan; each then runs the normal analytical
pipeline and gets its own BERT intent (so roi->margin roi, spend->spending, ...). This is what makes
"what metrics are available for sqo, and how are they trended" work: the trend count is only known
after the coverage step.

## LLM routing (Anthropic when key is set)

When `ANTHROPIC_API_KEY` is set, response/NER generation uses the langchain Anthropic chat package
via `call_anthropic_chat` (in `llm_external.py`); otherwise the existing `call_api` endpoint path is
used. `call_api` and the in-house `generate_text` path are unchanged, so `insights_only` / `st-main`
keep working (they get Anthropic when the key is present, the endpoint otherwise).

- `generate_filter_response` (NER) uses `with_structured_output(NEROutput)`.
- `generate_analysis_response` (response/denial, used by `response_generate_with_denial`) uses
  `with_structured_output(ResponseOutput)`.
- `call_anthropic_chat` returns a JSON string of the validated object, so the existing parsers
  (`JsonOutputParser` for NER, `_parse_denial_response` for responses) keep working unchanged.
- `NEROutput` gained `clarification_reason` so the NER consumer's `{answer, need_clarification,
  clarification_reason}` shape is satisfied.

## Files

### New
| File | Purpose |
|---|---|
| `src/model/planner.py` | Dataclasses (`PlannerPlan`, `SubQuerySpec`, `SubQueryAnnotation`, `SubQueryRuntime`, `EnrichedSubQuery`, `PlannerResult`); `generate_plan` (plan + annotate, Streamlit-free); `spacy_hints`; `resolve_referenced_terms`; `annotate_feasibility`; `resolve_from_context` |
| `src/model/coverage.py` | `answer_coverage`: LLM coverage-spec extraction, `core_filters` value->column resolution, config lookup vs live `groupby`, deterministic fact lookup + LLM phrasing |
| `scripts/build_feasibility_config.py` | Offline builder for `feasibility_config.json` (run at data refresh) |
| `pages/orchestrator_page.py` | New Streamlit page driving the planner flow (per-sub-query clarification state machine, routing, resolve/split fan-out, dependency rejection). `st-main.py` untouched |

### Modified
| File | Change |
|---|---|
| `config/prompt.yaml` | Added `PLANNER_PROMPT`, `RESOLVE_FROM_CONTEXT_PROMPT` (reject/rephrase/split), `COVERAGE_SPEC_PROMPT`, `COVERAGE_PHRASING_PROMPT` |
| `src/integrations/sdk_utils.py` | `make_planner_model` (intent constrained to catalog), `planner_llm_call`, `CoverageSpec` + `coverage_spec_call`, `ResolveOutput` + `resolve_from_context_call`, `text_llm_call`, `_anthropic_kwargs`; `NEROutput` gains `clarification_reason` |
| `src/integrations/llm_external.py` | `call_anthropic_chat` (+ `_anthropic_enabled`); `generate_filter_response` / `generate_analysis_response` use it with `NEROutput` / `ResponseOutput` when the key is set. `call_api` untouched |
| `src/data/local.py` | `feasibility_config` path key + `get_feasibility_config` loader |
| `src/data/data_interface.py` | `FeasibilityConfig` dataclass (`from_local`, stub `from_database`, `validate`) |
| `src/model/filter_generator_refactor.py` | `extract_ner_filter` gains `coarse_intent` param; on planner/BERT intent mismatch adds `intention` to clarification offering both candidates |
| `src/model/filter_clarification_graph.py` | `extract_ner_node` threads `coarse_intent` from graph config into NER |

## Feasibility config

`feasibility_config.json` per client/model group, built offline. Per data_source (`bi`/`br`):
`metrics`, `metric_time_ranges[metric][period_type] = {start, end, count}`, and
`dimensions[col] = {category, values[val] = {period_coverage, metrics_present}}`. Columns
enumerated are only those `questionnaire.csv` classifies as level/tagging (`level_type`), so
no value cross-product is stored; joint co-occurrence is answered live via `groupby`. Data
values are stored lower-cased.

## Feasibility / coverage value resolution via core_dimension_filter

spacy matches return stemmed/lemmatized terms keyed by spacy labels (`custom_level`,
`custom_tagging`, `halo_tagging`, `core_dimension`, ...), which are neither column names nor
guaranteed equal to stored data values. To check values reliably, `resolve_referenced_terms`
maps each matched term through `core_dimension_filter` (the same structure `categorize_filters`
/ `_get_mask` use downstream) to real `{col: [values]}`, stripping meta keys (`AKA`, `org_term`,
`source`, `halo_to_ignore`, `detail`). All comparisons are lower-cased.

- `planner.annotate_feasibility`: intent->metric availability in the data_source, plus
  referenced entities resolved to `(col, values)` and checked against the config (flagged
  infeasible only when entirely absent). Annotate-only; NER makes the final call.
- `coverage.answer_coverage`: for single-dimension value ops it resolves the referenced value
  to its real column via `core_filters`, so "what metrics are available for sqo" maps `sqo` to
  its column and reads `metrics_present`.

## Design decisions

- Planner LLM does: rephrase, split, classify coverage/analytical, assign `coarse_intent`,
  set `depends_on`. It does NOT judge feasibility (code lookup) or run BERT.
- BERT intent runs per sub-query in NER and reconciles with `coarse_intent`; mismatch -> intent
  clarification offering both candidates.
- spacy runs once in the planner as a split + feasibility signal.
- Single-intent queries emit exactly one sub-query (`id` 0).
- Planner output structure is three layers: spec (LLM) / annotation (code) / runtime (orchestrator).
- Dependent resolution is a single mechanism keyed by `depends_on` (no flag): reject / rephrase / split.
- Analytical insight step mirrors `insights_only` (`_prep_visual_context` + `response_generate_with_denial`,
  `</think>` strip, principle insights, the `Pretext/Insights/Principle/Benchmark` chunk).
- When `ANTHROPIC_API_KEY` is set, NER/response generation uses Anthropic structured output; else endpoints.
- Each finished sub-query renders in its own `st.expander` (label `Sub-query N: <kind>`), with the
  sub-query text, insights/response, and tables inside. Default `expanded=True`. The in-progress
  clarification form stays outside expanders (transient).

## Verified (static)

- All changed files compile and pass `flake8` (only pre-existing F401s remain in `sdk_utils.py` /
  `llm_external.py`; my additions introduce none).
- `sdk_utils` and `coverage` import cleanly.
- Dynamic planner model accepts valid intents and rejects invalid ones (forces LLM retry).
- Coverage lookup helpers return correct results for count / list / existence / co-occurrence.
- `core_filters` resolution maps stemmed terms to real `(col, value)` (lower-cased) and the
  disjoint check flags absence.
- `ResolveOutput` round-trips for split (N) / rephrase (1) / reject ([], false).

## Not yet runtime-verified (needs DATA_DIR + ANTHROPIC_API_KEY)

- `scripts/build_feasibility_config.py` execution.
- All LLM calls (`planner_llm_call`, `coverage_spec_call`, `resolve_from_context_call`, phrasing,
  `call_anthropic_chat`).
- Per-sub-query `generate_readoutdata` / `process_data`, the analytical insight assembly, and the
  page end-to-end including clarification interrupts and the resolve/split fan-out + rerun loop.

## Deferred to v2

DAG / typed dependency edges; silent downshift; value cross-product / precomputed co-occurrence;
planner hard-reject to skip the pipeline; parallel execution of independent sub-queries; the rich
chart rendering on the page (v1 renders the text answer plus key data tables).

## How to verify

```bash
# 1. Build the feasibility config for a configured client
python scripts/build_feasibility_config.py --client-code <CODE> --model-group-id <ID>

# 2. Lint
flake8 .

# 3. Run the app and open the Planner page from the sidebar
streamlit run ./st-main.py --server.port 6999 --browser.gatherUsageStats False
```

On the page, exercise: single analytical; parallel multi-part; sequential ("roi change for the two
most efficient tactics" -> resolves to one); coverage ("how many quarters did we run paid social");
co-occurrence ("which countries have which brands"); and dynamic split ("what metrics are available
for sqo and how are they trended" -> coverage then N trend sub-queries, one per metric).
