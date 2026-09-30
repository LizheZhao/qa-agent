# Ask Genome Agent

A separately versioned LangGraph agent that converts a conversational request into a validated Ask Genome plan, resolves each analytical subquery into a client-specific filter, applies that filter to the client's data, and answers from the figures it finds.

The filtered rows are built at three granularities, the client's `result_lookup` chooses which one answers, and postprocessing produces a readout and the display tables behind it. From those tables it builds ask-genome-core's GPT readout, the longer per-metric context `pages/insights_only.py` gives the model. `generate_response` sends that through upstream's `analytics_generator` prompt and assembles the answer around what comes back: the headline figure, a summary of the ROI Genome principles and the benchmark. If the model call fails, the readout is reported instead, so the figures still arrive.

Budget and scenario questions are answered too, from the client's published planner scenarios: which drivers to grow or cut, and where returns flatten. The agent reports those scenarios rather than optimising anything itself, and says so in the answer.

Postprocessing is ask-genome-core's own `process_data`, carried as-is rather than re-ported, because that logic still changes with client requests. Its per-client behaviour, ROI Genome principles included, comes with it.

## Layout

One node per file, and the file is named after the node `graph.py` registers. If you are reading a file in `nodes/`, it is a graph step.

Everything else lives in a `support/` package beside the nodes that use it:

* `nodes/support/` — `spacy_hints`, shared by planning and Question Understanding.
* `nodes/question_understanding/support/` — `clarification_requests` (builds the typed requests), `time_periods` (builds the time prompt), `time_filter` (resolves time against the data).
* `nodes/postprocess/support/` — `contract` (what postprocessing takes and returns), `process_data_adapter` (runs the vendored `process_data` and builds what the model reads).

`vendor/ask_genome_core/` is ask-genome-core's own code, written by a sync script rather than by hand; see [The vendored copy](#the-vendored-copy).

Nothing under `support/` is a node, and `graph.py`'s `add_node` block remains the index of what actually runs.

## Flow

### Planning

`extract_query` → `decompose_and_classify` → `annotate_feasibility` → `assemble_plan`

This stage:

* Splits complex questions into subqueries.
* Classifies each as analytical or coverage.
* Assigns intent and dependencies.
* Adds feasibility information using client configuration and spaCy hints.

### Coverage

`assemble_plan` → `answer_coverage` → `write_context`

A coverage subquery asks what data exists ("how many quarters of UK data do we have", "which countries have which KPIs") and skips question understanding. The node follows upstream's function of the same name in `src/model/coverage.py`: a model turns the question into a `CoverageSpec`, the facts are looked up by upstream's own vendored helpers (the feasibility config for one dimension, a live groupby over the client data for values that co-occur), and a model phrases them in a sentence or two. The numbers come from the lookup, never the model. A dependent subquery receives the sentence and the facts behind it, as upstream's orchestrator page passes them.

The feasibility config only enumerates some dimensions (for LINKEDIN/12: country, KPI, detailed KPI and the two business-unit fields), so a coverage question about a channel finds nothing to look up. Upstream then reports it as no data; here the facts say coverage is not recorded for that dimension, so the answer does not claim paid social was never run.

### Question Understanding

`extract_entities` → `classify_fields` → `reconcile_intent` → `fill_remaining_fields` → `build_clarification_list` → `should_clarify`

From there, the graph routes to either:

* `finalize_filter`, or
* `coordinate_clarification`, which loops back to itself until no questions remain and then routes to `finalize_filter`.

A single filter dictionary is progressively refined throughout the flow.

`extract_entities` reuses the spaCy hints already generated during feasibility checking. It was registered as `execute_subquery` until this chain was built, so older traces show that name.

`classify_fields` calls the BERT classifier asynchronously. Remaining questionnaire dimensions are then filled using structured LLM output, with up to three retries. If extraction still fails, the field moves to clarification instead of crashing the request.

### Dependencies

Subqueries are processed sequentially.

A dependent subquery receives the completed context from its predecessor and can be rewritten, rejected, or split.

Because the system does not yet calculate actual analytical values, dependencies that require a predecessor's **computed result** are rejected. Dependencies that only inherit filters can still be handled.

A rejected subquery never executes, so it leaves no `ner_state` entry. Its reason is recorded in `unanswered` and `aggregate_results` states it, rather than the final message quietly reporting one answer to a two-part question.

### Data filtering

`finalize_filter` → `resolve_data_source` → `apply_data_filters` → `write_context`

`resolve_data_source` maps the intention to the dataset it reads, the metric column values that count, and the data levels. The dataset is chosen per intention, not per client: `margin roi` reads `bi` while `source of change` reads `br`, which is why both frames are loaded. An out-of-scope intention short-circuits to `write_context` instead of raising.

`apply_data_filters` translates the model's answers into values that exist in the client's columns, then intersects them with the data one field at a time. A filter that would empty the result is skipped rather than applied, and the smallest conflicting combination is named, so the rejection message says which pair disagreed instead of returning an empty table.

`records` holds a 50-row sample, not the result. Graph state is checkpointed into one MongoDB document capped at 16MB, and a LINKEDIN spend question matches around 96,000 rows. `row_count` carries the real total.

### Postprocessing

`apply_data_filters` → `run_postprocess` → `generate_response` → `write_context`

`run_postprocess` rebuilds the table from the recipe and hands it to `nodes/postprocess/support/process_data_adapter.py`, which splits it into the three level frames ask-genome-core expects and calls its `process_data`. That function builds each granularity, applies the client's `result_lookup` to whichever produced rows, and returns the readout and display tables; the adapter maps those onto `PostprocessResult`.

The node reads the filter the rebuild resolved, not the recipe's own: the recipe stores the filter as it stood before the spaCy merge, so `core_dimension` and the tagging fields postprocessing branches on are absent from it. Four more fields (`time`, `period_type`, `level` and `time_tag`) are derived after filtering, because what the answer covers is a property of the rows, not of the question.

No DataFrame crosses back into graph state, for the same reason `records` is a sample. What returns is the readout, the tables as text, the selection metadata, any warnings, and what the model is asked and reads.

#### What the model reads

`insight_input` in the adapter follows the page code in `pages/insights_only.py`, which is inline Streamlit code rather than a function and so is the one part not vendored:

* A planner question reads its scenarios as JSON (`build_planner_specific_context`, else `build_planner_general_context`), preceded by the matching rows of the client's `insight_instruction.csv`. Where no JSON can be built it reads the readout and scenario tables.
* Anything else reads `build_readout`. If that raises, it reads the readout and an `insight_fallback` warning is recorded.
* The model is only called when there is a detail, aggregate or planner table, the page's own condition. Otherwise the pretext alone is the answer.

`generate_response` renders upstream's `EXTERNAL_READOUT_PROMPT` through the vendored `PromptTemplates` as one user message, as upstream sends it, so the prompt and each client's `CLIENT_SPECIFIC_PROMPT` inserts change with a sync. The principle text gets a second "Summarize insights." call. Two differences from the page. Upstream's response requirements go in whole, through the page's own `custom_instructions` hook, with a few of the agent's after them (`prompts.readout_requirements`): the period the question resolved to, no currency the data does not give, the scope named when figures cover only some business units, no "consistently" or "across the portfolio" where rows contradict it, and rankings taken from all the rows. Each answers a failure in the 29-question sweep. And the model is asked the subquery rather than the whole turn as typed, since a turn can hold several questions. Graph state keeps up to 600,000 characters of the GPT readout; the largest seen on LINKEDIN/12 is about 254,000.

#### The vendored copy

`vendor/ask_genome_core/` is `process_data`'s module closure, written by `scripts/sync_ask_genome_core.py` from a commit recorded in `vendor/ask_genome_core/SOURCE.json`. Picking up upstream changes is a re-run at a newer commit:

```bash
uv run python scripts/sync_ask_genome_core.py --source /path/to/ask-genome-core --commit <sha>
uv run python scripts/sync_ask_genome_core.py --source /path/to/ask-genome-core --check
```

Files are copied unchanged apart from the recorded edits: `src.` imports point at the vendored package, the client and data directory are read from `ASK_GENOME_*` (bare `CLIENT_CODE` belongs to the enterprise LLM gateway), two dead `sklearn` imports are dropped, and the dispatch table `readout.py` reads at import is resolved beside the copy instead of the working directory. The two LLM client modules are stubs; nothing vendored calls them, since the orchestrator makes the model calls itself. `config/prompt.yaml` is copied so the prompts are upstream's. The copy is excluded from ruff and mypy, since its style and typing are upstream's.

`ask-genome` pins pandas below 3 because ask-genome-core deploys on pandas 2.3 and is not yet pandas 3 compatible. Lift the pin with its Python 3.12 upgrade.

A level that raises anything but "No valid data." fails the whole call, as it does upstream.

## Clarification Design

### The graph pauses and resumes in place

Clarification uses LangGraph `interrupt()`, on top of the pause/resume support the orchestration framework gained in `feature/graph-interrupt-resume`.

`build_clarification_list` turns each flagged field into a typed `ClarificationRequest` from `orchestration_core` and puts them on the shared `queued_clarifications` channel. `coordinate_clarification` takes the head of that queue, calls `interrupt()` with it, and the run suspends. When the user answers, execution continues inside that same node with `ner_state` exactly as it was, and the node routes back to itself until the queue is empty.

This replaces an earlier design where a clarification ended the turn and the answer arrived as a new one. That pattern existed only because the framework could not pause a run. It cost context: the answer went through the planner first, which rephrased it, and the dimension and time range from the original question were regularly lost.

Only one interrupt may be live per invocation, so several flagged fields become several requests asked one at a time rather than one combined question.

There is no round cap. The queue is finite, the runtime allows one active pause per session, and an unanswered clarification expires on its own.

### How an answer lands on the filter

A response is one of three forms: `options`, `free_text`, or `cancel`.

`coordinate_clarification` maps the answer back to its field, writes it into `ner_results`, and marks it:

`field_status = "user-confirmed"`

Those fields are then skipped by both:

* `fill_remaining_fields`
* `build_clarification_list`

Option ids are sanitized for transport, so the option's **label** is the real column value and is what reaches the filter. Free text is taken at face value, since the user typed a value the list did not offer.

Cancel records nothing. The field stays in `clarification_fields` so `aggregate_results` discloses it, rather than storing an answer the user never gave.

Preserving `user-confirmed` matters beyond provenance. In the halo check, spaCy may normalize `lms` to `lm` while the actual column value remains `lms`; without the guard the two values never intersect and the same question repeats indefinitely.

### Option lists are ranked and capped

`MAX_CLARIFICATION_OPTIONS` in `orchestration_core` is 10, and several fields exceed it: `start` and `end` have 16 values for LINKEDIN, `sub brand` has 35 for KCC. Raising the cap does not fix this in general.

Instead, values that the extraction already considers likely are ranked first, the list is truncated to 10, and the question says how many more exist. Every request also carries `free_text_allowed`, which is fixed `True` on the contract, so a truncated value is still reachable by typing it.

`intention` is the exception that fits: it is single-select, and with `out-of-scope` treated as non-selectable it lands at exactly 10.

## Time Resolution

Time fields are resolved relative to today's date.

The time dimension uses `DYNAMIC_TIME_PROMPT`, matching the specialized behavior in `ask-genome-core`.

It includes:

* Today's date.
* 18 worked examples.
* Dynamically generated time periods using `time_periods.generate_time_periods`.

The questionnaire's `prompt` column is treated as a template and populated with:

`cols`, `predefined_list`, `question`, `example`, `answer_format`, `context`, and `date`.

Previously, the raw template was passed directly to the model, including text such as:

`Today is {date}.`

That allowed the model to effectively invent its own reference date.

Few-shot examples are the only piece not fully ported. Selecting them requires the embedding-based `construct_learning_examples` flow, so `example` currently renders as `(no examples available)`.

## Clarification Triggers

Clarification is currently raised by:

* Incomplete LLM extraction.
* Halo mismatch.
* Response-modifier keywords such as "diminishing return", which force the intention to `planner`.

Confidence, specificity, and additional time checks remain disabled because they are commented out in the source. Re-enabling them would introduce new behavior rather than faithfully porting the existing system.

A dataset conflict between the intention and the matched dimension no longer raises clarification. See "Dataset conflicts are reported, not substituted" below.

### What gets offered

`get_available_values` ends with `sorted(vals) + additional_options.get(field, ["all", "irrelevant"])`. That default is right for tagging columns, where both are real answers, but it also reached `intention` and offered 13 choices for a client with 10 intentions plus `out-of-scope`. Neither extra is answerable: there is no "every intention at once", and an analytical question always has one. Raised by Lizhe after seeing the clarification message in the demo.

`intention` now has an explicit empty entry in `additional_options`, so it offers the client's real intentions plus `out-of-scope` and nothing else. Tagging columns are unchanged. `out-of-scope` is then dropped again when the request is built, for the reason given in "Option lists are ranked and capped".

`ner_state` still retains useful context such as:

* Raw BERT probabilities.
* spaCy captures.
* Field provenance.
* Final clarification flags.

Following discussions with Lizhe, the longer-term plan is to replace the rule-based clarification logic with an LLM uncertainty evaluator that reasons across the complete filter at once.

## Important Differences from ask-genome-core

### Dataset conflicts are reported, not substituted

When the intention and the client-specific dimensions spaCy matched imply different datasets, `validate_intention` leaves the intention alone and records a `source_conflict` on the subquery's `ner_state` entry. `aggregate_results` then tells the user, in their own terms.

The source replaces the user's intention with that dataset's default and asks them to "clarify intention". Agreed in review that this is wrong twice over, and confirmed live on "what was the margin roi for print last quarter":

* The substitution is unconditional, so the filter committed to `source of change` even where the user never agreed to it. The final message admitted being unsure about the intention while the filter had already changed it.
* A dataset incompatibility is not evidence that the user was ambiguous. They clearly wanted margin ROI. The real finding is that margin ROI may not be available for print, which no answer from the user can change.

The conflict is recorded structurally (`intention`, `intention_source`, `dimensions`, `dimension_source`, `alternative_intention`), so a later design can offer a real choice rather than impose one. What to ask the user, if anything, is open design work.

Note that spaCy and BERT are independent signals here. Neither corrects the other; this node compares them and can report a conflict.

### Stable subquery IDs

Results are keyed by `Subquery.id` rather than list position.

This prevents dependencies from accidentally pointing to the wrong predecessor if a plan is modified or a subquery is split.

### Retry failures become clarification

If structured extraction exhausts its retries, the field moves to clarification instead of raising an exception.

The source implementation logs that it is adding the field to clarification but then re-raises the exception.

### Time-period month fix

The month branch in `generate_time_periods` subtracts once rather than twice.

The source subtracts during alignment and again inside the loop, causing “last month” to resolve to the month before last.

### Confirmed fields are never re-asked

Once a user confirms a field during clarification, it remains authoritative for the rest of the flow.

## Resolved Issue: Hierarchy Fields

A major live issue was caused by questionnaire field names containing spaces.

Examples:

* LINKEDIN/12: `business unit focused marketing`
* KCC/8: `sub brand`

Structured-output tool parameters must use identifier-safe names. Passing these field names directly caused the gateway to reject the complete tool definition with HTTP 500.

As a result, every field in that dimension failed and repeatedly triggered clarification.

This was isolated by testing the same extraction with:

1. Underscore-based names: worked.
2. Names containing spaces: HTTP 500.
3. Sanitized names: worked.

`build_field_extraction_model` now uses `sanitize_field_name()` for the LLM schema and maps the response back to the real questionnaire field names.

The extraction code also now logs each retry and the final error instead of silently swallowing the exception.

This was verified live with both LINKEDIN/12 and KCC/8.

For `custom_cols`, the expected result at this stage is only:

`specific`, `all`, or `irrelevant`

Resolving `specific` into the actual client value belongs to the later data-filtering stage.

## Not Implemented Yet

### Partial failures

A level that raises inside `process_data` fails the turn, as it does upstream. That is to be settled when a real case needs it.

A dependent subquery is given the answer the user saw for the one before it, so "its" resolves against what was actually said. Only where no answer was written does it get the postprocess summary instead. Upstream passes the readout instead, and its resolution prompt refuses when the context lacks the metric the follow-up asks about; ours rejects only when the reference itself cannot be identified, since the metric is fetched after the rewrite.

### Parts of data filtering

The stage applies the filter and produces real rows. Four pieces of `ask-genome-core` are deliberately not ported yet, none of which change the shape of the result:

* **Time defaulting** (`_get_time_period_answer_dict`). Dates that Question Understanding resolved filter correctly. A question with no time reference answers `irrelevant` for `start`/`end`, and that currently means no time filter at all rather than "the latest available period". Confirmed live: a question without a time reference matched 254,338 rows. This is the largest remaining gap.
* **Further narrowing** (`map_keyword_in_query`, `combine_filters`, `_get_specific_values`). Why `specific` currently widens to every real value instead of narrowing to the spaCy captures.
* **Two client-specific correction rules** (`_ner_reconciliation`, `_apply_filtering_logic`).
* **The hierarchy split** (`get_report_levels` / `_dedup_df` / `_get_level_answer`), which partitions the one filtered frame by granularity.

### Few-shot examples

`df_examples` is loaded but is not yet used for per-dimension example selection.

## Configuration

Client configuration is loaded once per process from `ASK_GENOME_*` environment variables.

This avoids the shared `os.environ` mutation problem in `ask-genome-core`, but it means one running deployment currently supports one client configuration. Switching clients requires restarting the worker.

The `ASK_GENOME_` prefix is important because the application already uses `CLIENT_CODE` for enterprise LLM gateway routing. Sharing the same variable previously caused incorrect gateway routing.

For now, these variables must be exported before starting the worker, even though the repository documentation suggests `.env` should load automatically. That contract still needs to be reconciled.

## Tests

There are two test layers.

### Decision-level tests

`test_ask_genome_question_understanding.py`

These cover individual porting decisions, behavior differences, and bugs found during live testing using manually constructed `ner_state`.

### Wiring tests

`test_ask_genome_plan_wiring.py`

These run the Question Understanding chain sequentially without manually injecting intermediate `ner_state` values.

This is important because all Question Understanding nodes communicate through:

`ner_state[subquery.id]`

A node-level unit test can remain green even when a producer stops writing a key expected by the next node.

That already happened with `spacy_hints`: a test manually constructed a `Subquery` containing them while `assemble_plan` had silently stopped passing them through.

Every handoff was mutation-tested by intentionally renaming the key a producer writes and verifying that the wiring tests failed. That covers eight keys (`extended_query`, `spacy_filter`, `predictions`, `probabilities`, `clarification_list`, `ner_results`, `field_status`, `clarification_fields`), plus `user_confirmed`, which is covered by two behavioural mutations instead.

Five of those eight were previously invisible to the per-node tests, which stayed fully green while the pipeline was broken:

* `extract_entities → extended_query`
* `extract_entities → spacy_filter`
* `classify_fields → predictions`
* `classify_fields → probabilities`
* `reconcile_intent → clarification_list`

New handoff checks should go into the wiring tests. New behavioral or porting decisions should go into the per-node tests.

One limitation remains: mocked models such as `ScriptedChatModel` accept invalid field names that the real LLM gateway rejects. Bugs such as the hierarchy-field issue can therefore only be reproduced through live integration testing.

## Notes

### Feasibility is intentionally coarse

A subquery marked as feasible only means that some relevant concept exists in the client's configured data.

It does not guarantee that every exact value or entity mentioned in the user's question exists.

### Validation

Planning, dependency resolution, Question Understanding, and hierarchy-field fixes have been tested against real client configurations, including LINKEDIN and KCC, using live gateway calls.