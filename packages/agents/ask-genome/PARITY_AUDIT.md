# Ask Genome parity audit (stage 1)

Audit of the agent reconstruction (`src/ask_genome_agent/`) against the original workflow (`src/ask_genome_agent/vendor/ask_genome_core/`). Read only: no repo code was changed. Nothing was executed, so import and packaging findings are predicted from static reading.

## Method

Each of 11 scopes got an inventory agent, a finder and an independent adversarial reviewer. The reviewer derived its own mapping before reading the finder report. Five high-impact findings (F1, F3, F4, F5, F9) and the `src.*` imports behind F2 were re-checked against the code by the orchestrator.

Scopes: S1 planning, S2 dependency loop and aggregation, S3 coverage, S4 extraction, S5 clarification, S6a data source and filter apply, S6b narrowing, S6c time filter, S7 postprocess, S8 generate response, S9 wiring, state, config and vendor.

Per-scope working files (`P0_*`, `F_*`, `R_*`) live in the session scratchpad, not in the repo.

## Facts that shaped the audit

a) The original Streamlit pages exist at the repo root (`pages/orchestrator_page.py`, `pages/insights_only.py`, `pages/agent_orchestrator.py`). They are older snapshots. Page-only behavior was diffed against them.
b) Much of the port follows the older repo-root `src/` snapshot, not the vendored copy. This applies to `dedup_levels`, `_minimal_conflict`, filter ordering, time-before-narrowing and `narrowing.py`.
c) The vendored tree at HEAD is not the sync script output. It holds raw upstream files (`src.*` imports, real LLM clients) that contradict `vendor/ask_genome_core/SOURCE.json`.
d) README claims are often stale. `narrowing.py` and `time_filter.py` exist and port much of what the README lists as not ported.

## Parity status vocabulary

| Status | Meaning |
|---|---|
| equivalent | Same logic, same outputs. |
| degraded | A counterpart runs but does less or behaves differently. May be deliberate and documented or accidental drift. |
| missing | No counterpart in the agent. |
| added | The agent has something the original never had. |

## 1. Traceability matrix

Paths for AGENT are under `ask_genome_agent/`. `PAGE` is `pages/orchestrator_page.py` and `IO` is `pages/insights_only.py`. `vendor/` is `vendor/ask_genome_core/model/`. `FGR` is `filter_generator_refactor.py`, `FG` is `filter_generator.py`, `DF` is `data_filtering.py`, `FCG` is `filter_clarification_graph.py`.

| AGENT node (file:line) | ORIGINAL (file:line) | Relationship | Parity | Conf |
|---|---|---|---|---|
| `extract_query` nodes/extract_query.py:22-31 | PAGE:301-306 | 1:1 | equivalent | high |
| `decompose_and_classify` nodes/decompose_and_classify.py:27-34; contracts.py:105-182; prompts.py:7-19 | vendor/planner.py:152-172; prompt.yaml:205-270 | replaced, prompt rewritten | degraded | high |
| `annotate_feasibility` nodes/annotate_feasibility.py:52-103 | vendor/planner.py:113-149 | 1:1 minus metric expansion | degraded | high |
| `spacy_hints` nodes/support/spacy_hints.py:71-219 | vendor/FG:859-1076, 1423-1430; vendor/planner.py:106-110 | 1:1 minus `METRIC_GROUP` | degraded | high |
| `assemble_plan` nodes/assemble_plan.py:15-44; contracts.py:66-102 | vendor/planner.py:168-182; PAGE:41-52 | 1:1 plus validator | equivalent, validator added | high |
| `resolve_dependency` nodes/resolve_dependency.py:23-60 | vendor/planner.py:185-196; PAGE:158-177 | 1:1, documented prompt change | degraded | high |
| `apply_resolution` nodes/apply_resolution.py:14-51 | PAGE:147-177 | 1:1 | degraded | high |
| `write_context`, `advance_cursor` nodes/write_context.py:13-44 | PAGE:120-130 | split, keyed by id | equivalent | high |
| `aggregate_results` nodes/aggregate_results.py:82-129 | PAGE:102-117, 316 | merged | degraded | medium |
| `answer_coverage` nodes/answer_coverage.py:28-116 | vendor/coverage.py:94-130; PAGE:133-144 | 1:1 | equivalent, `table` dropped | high |
| `extract_entities` nodes/question_understanding/extract_entities.py:35-59 | vendor/FGR:387-404 | split, hints reused | degraded | high |
| `classify_fields` .../classify_fields.py:26-70 | vendor/FGR:395-397, 206-221 | 1:1 | equivalent | high |
| `reconcile_intent` .../reconcile_intent.py:20-115 | vendor/FG:569-597, 1455-1471; FGR:407-413 | 1:1, documented deviation | degraded, one addition | high |
| `fill_remaining_fields` .../fill_remaining_fields.py:62-311 | vendor/FGR:63-171, 419-441; FG:320-554 | split | degraded | high |
| `time_periods` .../support/time_periods.py:56-233 | vendor/FG:144-317 | 1:1 plus month fix | equivalent | high |
| `build_clarification_list` .../build_clarification_list.py:73-194 | vendor/FGR:174-345, 467-486 | 1:1 partial | degraded | high |
| `coordinate_clarification` .../coordinate_clarification.py:53-131 | vendor/FCG:63-113 | merged, one field per pause | degraded | high |
| `finalize_filter` .../finalize_filter.py:13-29 | vendor/FCG:92-113 | split | equivalent | medium |
| `clarification_requests` .../support/clarification_requests.py | none | new-no-original | added | high |
| `resolve_data_source` .../resolve_data_source.py:33-125 | vendor/FG:401-460; FGR:499-516 | 1:1 | equivalent | high |
| `apply_data_filters` .../apply_data_filters.py:61-150 | vendor/FGR:666-783 | split | degraded | high |
| `filtering` .../support/filtering.py:42-280 | vendor/DF:341-504; FGR:348-375, 535-549 | merged | degraded | high |
| `levels` .../support/levels.py:36-158 | vendor/DF:827-972 | 1:1 except dedup | degraded | high |
| `narrowing` .../support/narrowing.py:116-877 | vendor/DF:975-1064, 1104; FG:1120-1571 | 1:1 of older snapshot | degraded | high |
| `time_filter` .../support/time_filter.py:43-412 | vendor/DF:138-302, 1067-1101; FG:1574-1642; FGR:558-663 | split | degraded | high |
| `table` .../support/table.py | none | recipe replay | added | high |
| `run_postprocess` and adapter nodes/postprocess/run_postprocess.py:39-178; process_data_adapter.py:51-304 | vendor/readout.py:2010-2034; IO:424-565 | split | degraded | medium |
| `generate_response` nodes/generate_response.py:44-138 | IO:563-625; vendor/readout.py:2118-2143; common.py:169-202 | split | degraded | medium |
| `graph.py` edges graph.py:267-313 | PAGE:299-325; FCG:116-132 | wiring | equivalent | high |

### Orphaned originals

a) Planning and extraction: `METRIC_GROUP` branch (`vendor/FG:911-940`), `_expand_metric_groups` use in feasibility (`vendor/planner.py:131-137`), `construct_response_structure` enums (`vendor/FG:463-554`), few-shot and embedding scoring (`vendor/FG:82-141`), the `intention == "none"` LLM rule (`vendor/FGR:427`), `llm_priority` and enterprise paths.
b) Clarification: specificity loop (`vendor/FGR:317-324`), `enable_clarification` gate (`vendor/FGR:467-470`), `coarse_intent` reconcile (`vendor/FGR:478-484`), `_to_native` (`vendor/FCG:142-153`), `ProcessIndicator`.
c) Filtering and time: `merge_captured_metrics`, `_scope_filters_to_diag`, `_main_metric_only_in_diag` (`vendor/FG:1187-1295`), `order_filter_indicators` (`vendor/FGR:364-375`), `_get_custom_time_answer_dict`, `calendar_series`, `is_custom`, `custom_period_order` (`vendor/DF:45-134`), `_select_calendar`, `_calendar_indicators`, `_metric_narrowed_frame`, `_resolve_time_filter`, `_empty_time_filter` (`vendor/FGR:552-637`), the `halo_tagging` back-fill (`vendor/FGR:769-771`), `filter_conflicts`, `NerFilters.validate`.
d) Response: `denial_classify` (`vendor/readout.py:2146-2177`).
e) Intentionally orphaned: offline spaCy builders, page-only UI, cache and RAG tabs.

### Agent elements with no original

`unanswered`, `source_conflict`, id-keyed context, the plan validator, `NarrowingConfig`, `build_ner_filter`, `narrow_and_apply`, the `table.py` recipe replay, the `run_postprocess` summarisation and truncation, `readout_requirements`, `asked_period`, coverage `_untracked` and phrasing fallback, the aggregate "I extracted N subqueries" prefix, `_PLAN_ONLY`.

## 2. Findings

Severity order: behavior regression, lost determinism, lost error handling, unnecessary addition, organization.

### Behavior regression

| # | Finding | AGENT | ORIGINAL | Impact | Minimal fix |
|---|---|---|---|---|---|
| F1 | A rephrased dependent subquery keeps stale `spacy_hints` and `feasibility`. Confirmed by four reviewers and re-verified. | apply_resolution.py:24-28; extract_entities.py:47-51 | PAGE:168-174, 257; vendor/FGR:401-404 | The entity injected by the rephrase is missing from the filters, so the result is broader than the question. | Add `"spacy_hints": None, "feasibility": None` to the update dict. |
| F2 | The vendored tree contradicts `SOURCE.json`: `src.*` imports, bare `CLIENT_CODE`, `DATA_DIR` and `LLM_MAX_INPUT_TOKENS` reads, cwd-relative `prompt.yaml`. | graph.py:37,43 import it at module load | vendor/readout.py:12-19; coverage.py:15-16; common.py:74,93; readout_utils.py:287,4063 | Predicted: `create_graph` fails to import, or runs against stale root helpers. Prompt render errors are swallowed at generate_response.py:60 and client prompt inserts are never injected. | Re-run `scripts/sync_ask_genome_core.py` at the `SOURCE.json` commit, then add the `--check` gate to CI. |
| F3 | With no main metric, no `record_level` is added, so all three levels get the whole table. | narrowing.py:177-180; process_data_adapter.py:256-260 | vendor/FGR:745-756 returns empty frames | `KeyError` at readout_utils.py:4321, or a garbage readout. | Return empty frames in `split_levels` when `record_level` is missing. |
| F4 | An out-of-scope intention leaves the `finalize_filter` summary as the subquery context. | finalize_filter.py:28-29; resolve_data_source.py:89-94; write_context.py:25-28 | PAGE:191-194 writes `""` | A dependent subquery resolves against a field dump. | Clear `last_subquery_result` on the out-of-scope branch. |
| F5 | `intention == "none"` is no longer sent to the LLM. | fill_remaining_fields.py:62-72 | vendor/FGR:427 | A confident BERT "none" becomes out-of-scope instead of being re-classified. | Add `predictions.get("intention") == "none"` to the field set. |
| F6 | Planner versus BERT intent reconciliation is missing. | build_clarification_list.py:147-194 | vendor/FGR:478-484 | A disagreement is never put to the user. | Offer `[coarse, resolved]` in `build_clarification_list`. |
| F7 | Clarification order is discarded. | build_clarification_list.py:175 overridden by clarification_requests.py:150 and orchestration_core/clarification.py:241-251 | vendor/FGR:334-345 | Questions are asked alphabetically by id, not intention first. | Preserve the incoming order when building requests. |
| F8 | "all" and "irrelevant" sit last and are cut by the 10-option cap. "overall" is no longer removed. | build_clarification_list.py:130-143; clarification_requests.py:82-98 | vendor/FGR:188-200; PAGE:281 | A field with more than 10 values cannot pick the sentinels by option. | Reserve slots for the sentinels, drop "overall", rank start/end newest first. |
| F9 | `dedup_levels` is the older algorithm: no level columns in the key, no rollup test, unstable sort. | levels.py:123-145 | vendor/DF:915-955 | Row sets and order differ from upstream. | Port `_dedup_df` verbatim with `kind="stable"`. |
| F10 | Time resolves on the whole frame before narrowing, with no frame ladder, metric-narrowed probe or custom calendar. | apply_data_filters.py:78-90 | vendor/FGR:558-663, 721-725 | Windows land on periods the tactic does not report. | Resolve after narrowing in the order metric+terms, terms, full. |
| F11 | Time filter gaps: no end-of-coverage clip, re-pick falls back to the coarsest type, a `report_yoy` fallback is live. | time_filter.py:325-327, 341-359, 257 | vendor/DF:208-211, 222-236, 217 (commented out) | Wrong period type or window near data edges. | Clip the end, compute availability from the sliced priority, drop or document the yoy fallback. |
| F12 | `_rewrite_intention` writes metric group codes where upstream writes KPI names. spaCy-captured metric terms are not merged into `main_metric`. | narrowing.py:775-787, 341-342, 482-484 | vendor/FG:1474-1502, 1143-1207 | After a "sales" rewrite the metric filter matches nothing. | Expand through the metric config lookup and port `merge_captured_metrics`. |
| F13 | The specificity trigger and the `enable_clarification` gate are dropped. The agent docstring wrongly calls the trigger commented out. | build_clarification_list.py:3-7, 147-194; local.py:282-310 | vendor/FGR:317-324, 467-470 | Over or under-asking per client. The gate defaults to False upstream and `questionnaire.csv` is not in the repo. | Decide the policy, then port both together or document the omission. |
| F14 | Denial classification was removed from the answer step. | generate_response.py:58-63 | IO:568-600; vendor/readout.py:2146-2177 | Novelty and unanswerable questions get a normal answer. | Add a fail-open denial check or document the deviation. |
| F15 | The refusal text for empty results is lost: the rejection message and no-tactic message are never shown. | aggregate_results.py:94-96; run_postprocess.py:73-76; narrowing.py:144-170 | PAGE:200-207; vendor/FGR:757-760 | Users see jargon such as "no level produced a readout". | Prefer `rejection_message` over the raw reason and set the no-tactic text on an empty frame. |
| F16 | `PLANNER_PROMPT` was rewritten: eight worked examples and the closing Note are gone. The header still says "copied". | prompts.py:7-19 | prompt.yaml:205-270 | Planner splits, intents and dependencies can drift. | Restore the examples and Note. |
| F17 | Feasibility compares raw metric group codes to KPI names. | annotate_feasibility.py:89-92 | vendor/planner.py:131-137 | Informational `infeasible` verdicts are wrong. | Expand through `metric_configs`. |
| F18 | Filter application order has no tiers. | filtering.py:143-155 | vendor/FGR:364-375, 730-735 | Time versus tactic precedence changes on a conflict. | Port `order_filter_indicators`. |
| F19 | The `METRIC_GROUP` spaCy key is dropped. | spacy_hints.py:94-119 | vendor/FG:911-940 | Conditional on client patterns. | Add the branch and the key. |

### Lost determinism

a) Unconstrained LLM answers: the agent accepts any string where upstream enumerates valid intentions, column values and time shapes (contracts.py:250-264 vs vendor/FG:468-533). Fix: `Literal` enums for intention and listed values in the prompt.
b) Time dimension fields are not forced to start/end (fill_remaining_fields.py:282 vs vendor/FGR:67-68). Conditional on the client time row.
c) No temperature, system instruction or token caps (generate_response.py:101 vs vendor/llm_external.py:269-280).
d) Raw `trend` overlay (narrowing.py:210-211). Low confidence.
e) Planner answer order puts the pretext last (generate_response.py:120 vs IO:613-618). Fix: `[prose, pretext, principle, benchmark]`.

### Lost error handling

a) Time stage has no exception ladder or empty guards. `[]` start/end, bad strings or empty period types raise (time_filter.py:367-371, 410; narrowing.py:872).
b) A zero-match window is logged as a conflict and start/end are logged twice (filtering.py:179-216). Offering "all" for start/end can reach `int("all")` (time_filter.py:367-371).
c) `how_many == []` raises an uncaught `IndexError` (filtering.py:102-106).
d) `_minimal_conflict` lost its bounds (filtering.py:126-140).
e) Postprocess recomputes `metric_configs` with the vendored loader, which fails without `metric_formatting.json`, where the agent loader handles it (process_data_adapter.py:62 vs local.py:196-248).
f) `insight_context_truncated` is written but never read (run_postprocess.py:133).
g) A non-string model content block is stringified in the coverage phrasing step (answer_coverage.py:113).
h) `AskGenomePlan._validate_invariants` raises on malformed LLM plans where the original continued (contracts.py:66-102). Partly intentional, low severity.

### Unnecessary additions

a) `reconcile_intent.py:111` seeds a clarification for modifier keywords that the original discards.
b) Dependent coverage subqueries go through resolution (graph.py:100-102).
c) `insight_input` has a silent broad `except` (process_data_adapter.py:179).
d) A second `table_to_text` pass (process_data_adapter.py:80-82) makes extra tokenizer calls and nothing reads its output.
e) Dead code: `user_confirmed` plumbing (read in two nodes, written nowhere), unreachable `no_lookup_match`, empty-recipe and `PostprocessRequest.source` branches, the `unchanged` resolution kind, `_PLAN_ONLY`, unused `split_by_level`.
f) The aggregate "I extracted N subqueries" prefix.

### Organization

a) Stale docs: README lines 284-291 say narrowing and time defaulting are not ported, but both exist. Wrong docstrings in graph.py:12-14, narrowing.py:20-22, build_clarification_list.py:3-7, prompts.py:1-3 and process_data_adapter.py:129.
b) `time_filter.py:140` uses `calendar_type="custom"` for the standard non-fiscal calendar, the opposite of upstream.
c) Parity tests baseline against an external checkout, so drift passes by construction. Some tests stub the code they claim to pin.
d) `pyproject.toml` lacks dependencies the raw vendored files import. `__version__` is `0.1.0` against `0.7.0` elsewhere.
e) Two `metric_configs` loaders and two level partitioners.

### Unverified leads (not findings)

`level_type` is a plain dict in the agent (local.py:304) and a `defaultdict` upstream (vendor/data/local.py:177). Removal of `_to_native` with numpy ints in `clarification_fields` may break checkpoint serialization.

## 3. Coverage report

| Scope | Inventory | Finder | Reviewer | Verdict |
|---|---|---|---|---|
| S1 planning | done | done | done | F1, F16, F17, F19 confirmed |
| S2 dependency and aggregation | done | done | done | F1, F4, F15 confirmed, others downgraded |
| S3 coverage | done | done | done | Mostly equivalent, cross-reference to F2 |
| S4 extraction | done | done | done | F5, F6 confirmed, 3 new, 1 rejected |
| S5 clarification | done | done | done | 9 confirmed, 3 new |
| S6a data source and apply | done | done | done | F3, F4, F9, F18 confirmed |
| S6b narrowing | done | done | done | F12 confirmed |
| S6c time filter | done | done | done | F10, F11 confirmed |
| S7 postprocess | done | done | done | F2, F3, F15 confirmed |
| S8 response | done | done | done | F14 confirmed |
| S9 wiring and vendor | done | done | done | F2 confirmed as the blocker |

No scope ended with zero findings.

### Not reviewed or not provable

a) Client data is absent: `questionnaire.csv`, custom-calendar clients, `core_dimensions.jsonl` patterns and `report_yoy` settings are not in the repo.
b) Live model behavior, the Docker build and import resolution were not run. F2 is predicted from static reading.
c) The root pages are older snapshots. Their newer versions are unavailable, so parity for page-only behavior is unverifiable.
d) Test files were only sampled.

### Process notes

a) Subagents ran through Agent and Workflow only. No fresh CLI sessions were used.
b) A usage limit interrupted the finders once and the eight failed ones were re-run.
c) The S6a reviewer opened the finder file in the same tool batch as other reads, after forming its analysis. Treat that scope's independence as slightly weaker.

## 4. Cleanup plan

1) F2: re-run the vendor sync at the `SOURCE.json` commit and add the `--check` CI gate. Everything else depends on the vendored code importing and rendering correctly.
2) F1: reset `spacy_hints` and `feasibility` on rephrase. One line.
3) F3 and F4: empty frames when `record_level` is missing, and clear the context on the out-of-scope branch.
4) F5, F6, F7: the `intention == "none"` rule, the coarse-intent reconcile and the clarification order.
5) F9 and F8: the stable `_dedup_df` port, then sentinel slots and "overall" removal.
6) F10, F11 and the time-stage error ladder: resolve time after narrowing, restore the clip and the re-pick, guard the empty cases.
7) F12: metric expansion in `_rewrite_intention`, then `merge_captured_metrics`. Unify the two `metric_configs` loaders here.
8) Decide F13, F14, F15, F16: these need a policy choice before any port (clarification gate and specificity, denial classification, refusal wording, planner prompt examples).
9) Hygiene: fix stale docstrings and README, remove dead code, point parity tests at the vendored baseline.

## 5. Immediate fixes (chosen by the user)

These are the fixes that will change results when users compare agent behavior with the current deterministic workflow. Items 6 to 8 and 11 came from the user's own notes and are not all audit findings.

| # | Fix | Audit refs | Notes |
|---|---|---|---|
| 1 | Clarification choices: stop the 10-option truncation and restore the "Not Applicable" choice. Add an enable-clarification indicator per questionnaire field. | F8, F13 | Upstream gate is `clarification_enabled.get(f, False)` (vendor/FGR:467-470, vendor/data/local.py:184-196). Decide the default for unlisted fields explicitly, since upstream's code and comment disagree. Option-set details: vendor/FGR:188-200, agent build_clarification_list.py:130-143 and clarification_requests.py:82-98. |
| 2 | Intention clarification: when the planner `coarse_intent` and the BERT intention differ, trigger a clarification offering both. | F6 | Upstream vendor/FGR:478-484. Agent has no consumer of `Subquery.coarse_intent` in the QU chain. Add in build_clarification_list.py after `available` is built. |
| 3 | Port `_dedup_df` as the dedup levels function. | F9 | Replace levels.py:123-145 with vendor/DF:915-955 (level columns in the key, rollup test, value rounding, stable sort). |
| 4 | No main metric: do not return the whole frame untagged. | F3, F15 | narrowing.py:177-180 skips `record_level`; process_data_adapter.py:256-260 then feeds the full table to all three levels. Return empty frames and keep the no-metric rejection text (vendor/FGR:745-756). |
| 5 | Time: restore the fallback logic. Bring back `_resolve_time_stage`, the frame ladder, the time mapping functions and custom calendar support, in the original order (after narrowing). | F10, F11, lost error handling (a) | Original: vendor/FGR:552-663, 721-725; vendor/DF:45-134, 138-302, 305-338, 1067-1101, 732-770. Agent time_filter.py and apply_data_filters.py:78-90. Include `_metric_narrowed_frame`, `_select_calendar`, `_calendar_indicators`, `_empty_time_filter`, the end-of-coverage clip and the empty guards. |
| 6 | spaCy should capture metrics in the query: `METRIC_GROUP` and `metric_captured`. | F19, F12 | spacy_hints.py:94-119 lacks the `METRIC_GROUP` branch and `metric_group` key (vendor/FG:911-940). narrowing.py:341-342, 482-484 lacks `merge_captured_metrics` and the `metric_captured` bucket (vendor/FG:1143-1207, 1334). |
| 7 | NER now uses metric groups instead of exact metric names. Update `get_metric_configs` usage in the qa agent and in the feasibility checks. | F12, F17, lost error handling (e) | `_rewrite_intention` (narrowing.py:775-787) and `annotate_feasibility` (annotate_feasibility.py:89-92) still compare raw group codes with KPI names. Unify the two `get_metric_configs` implementations (local.py:196-279 and vendor/readout_utils.py:294-357: `keepMetric` filter, defaults, missing `metric_formatting.json`). |
| 8 | Preserve the filter application order and generate the `_apply_answer_dict_with_log` log in the agent, for follow-ups and messages. | F18, lost error handling (b to d) | Order: port `order_filter_indicators` (vendor/FGR:364-375, 730-735) into filtering.py:143-155. Log: vendor/DF:435-504 plus `log_to_message` (vendor/FGR:535-549). Also fix start/end double logging, log a zero-match window as `skipped_no_value`, restore `_minimal_conflict` bounds, and keep `filter_conflicts` in the filter dict (vendor/FGR:739-741). Expose the log through state so dependent subqueries and aggregate messages can use it. |
| 9 | Add denial classification to the response, and the reject messages. | F14, F15 | Denial: vendor/readout.py:2146-2177, prompt.yaml:331-358, common.py:204-223, IO:568-600. Reject text: PAGE:200-207, IO:619-625, `PromptTemplates().rejection_response`. Agent: generate_response.py:44-63 and aggregate_results.py:94-96. |
| 10 | Orchestrator planner prompt: restore the dropped examples. | F16 | prompts.py:7-19 versus prompt.yaml:205-270. See note on item 11. |
| 11 | Prompt policy: do not put specific edge cases in general prompts (for example "its ROI", "ROI for print"). Use instructions instead. | none (new rule) | Applies to every prompt in prompts.py and to item 10. Open point: the original examples are themselves specific, so restoring them as-is conflicts with this rule. Restore the shape of the examples with generic wording, and put client or case specific guidance in per-client instructions. Existing candidates to review: the REJECT rationale in prompts.py:38-42 and the hierarchy and time prompts in prompts.py:44-122. |

Suggested order inside this set: item 7 first (metric groups feed items 4, 6 and 8), then 5, then 1 to 3, then 4, 8, 9, then the prompt items 10 and 11.

Note: the user mentioned a side chat. Only the 11 items above reached this session, so anything else from that chat is not recorded here.

## 6. Backlog (future reference)

Everything in section 2 that is not in section 5. Items marked high were behavior regressions and deserve scheduling soon.

### Behavior regression

| # | Finding | Fix |
|---|---|---|
| F1 (high) | Rephrased dependent subquery keeps stale `spacy_hints` and `feasibility` (apply_resolution.py:24-28). | Add `"spacy_hints": None, "feasibility": None` to the update dict. |
| F2 (high) | Vendored tree is not the sync output: `src.*` imports, bare env vars, cwd-relative yaml (graph.py:37,43 import them at load). | Re-run the sync at the `SOURCE.json` commit and add the `--check` CI gate. Also required before items 5 to 9 can be trusted at runtime. |
| F4 (high) | Out-of-scope intention leaks the `finalize_filter` summary as context (finalize_filter.py:28-29, write_context.py:25-28). | Clear `last_subquery_result` on the out-of-scope branch. |
| F5 (high) | `intention == "none"` not sent to the LLM (fill_remaining_fields.py:62-72). | Add the clause back. |
| F7 (high) | Clarification order discarded (clarification_requests.py:150). | Preserve incoming order. Closely related to item 1. |
| F13 (part) | Specificity trigger ("specific" or "irrelevant" with no spaCy capture, vendor/FGR:317-324) not ported. | Decide with item 1, then port or document. |

### Lost determinism

a) Unconstrained LLM answers (contracts.py:250-264): add `Literal` enums for intention and list valid values.
b) Time dimension fields not forced to start/end (fill_remaining_fields.py:282).
c) No temperature, system instruction or token caps on model calls (generate_response.py:101, answer_coverage.py:98-112).
d) Raw `trend` overlay (narrowing.py:210-211). Low confidence.
e) Planner answer order puts the pretext last (generate_response.py:120). Use `[prose, pretext, principle, benchmark]`.

### Lost error handling

a) `how_many == []` raises `IndexError` (filtering.py:102-106).
b) Offering "all" for start/end can reach `int("all")` (time_filter.py:367-371). Offer only "irrelevant".
c) `insight_context_truncated` written, never read (run_postprocess.py:133).
d) Non-string model content stringified in coverage phrasing (answer_coverage.py:113).
e) Plan validator raises on malformed LLM plans (contracts.py:66-102). Partly intentional.

### Unnecessary additions

a) Modifier-keyword clarification seeded in reconcile_intent.py:111, which the original discards.
b) Dependent coverage subqueries go through resolution (graph.py:100-102).
c) Silent broad `except` in `insight_input` (process_data_adapter.py:179).
d) Second `table_to_text` pass with extra tokenizer calls (process_data_adapter.py:80-82).
e) Dead code: `user_confirmed` plumbing, unreachable `no_lookup_match`, empty-recipe and `PostprocessRequest.source` branches, `unchanged` resolution kind, `_PLAN_ONLY`, unused `split_by_level`.
f) "I extracted N subqueries" prefix in aggregate_results.py:84-85.

### Organization

a) Stale docs: README lines 284-291, and docstrings in graph.py:12-14, narrowing.py:20-22, build_clarification_list.py:3-7, prompts.py:1-3, process_data_adapter.py:129.
b) `calendar_type="custom"` in time_filter.py:140 means the standard non-fiscal calendar, the opposite of upstream.
c) Parity tests baseline against an external checkout; some tests stub the code they claim to pin.
d) `pyproject.toml` lacks dependencies the raw vendored files import; `__version__` is `0.1.0` against `0.7.0` elsewhere.
e) Two `metric_configs` loaders and two level partitioners (item 7 covers the loaders).

### Other open items

a) `ProcessIndicator` gating (legacy preprocessing, spaCy on or off) is not ported (config.py, apply_data_filters.py:23-25).
b) Diagnostic-row handling in `combine_filters` and the planner to performance override are not ported. Inert for LINKEDIN, needed when a diagnostic client is onboarded.
c) Few-shot examples and embedding scoring for field extraction are not ported (fill_remaining_fields.py:127-131).
d) `llm_priority` and enterprise-LLM extraction paths are dropped.
e) Spliced dependents: later subqueries depending on a replaced id are always refused. Optional remap.
f) Unverified: `level_type` dict versus `defaultdict`, `_to_native` removal with numpy values in `clarification_fields`, `recursion_limit` against about 13 nodes per subquery.
