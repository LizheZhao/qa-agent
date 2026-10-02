# Phase 2 final audit: readout_new.py vs readout.py + readout_utils.py

Scope note: RECONSTRUCTED is the file src/model/readout_new.py (no directory by that name). ORIGINAL is readout.py, readout_utils.py, readout_other_category.py.
Evidence note: nothing was executed end to end. data/ and the lookup CSVs (result_lookup_bi/br, tactics_level, level_rename_display) are absent from this checkout. Everything is hand-traced; some reviewers added synthetic-frame probes. Orchestrator re-verified by hand: time_message has no reader (readout_new.py:395-396,1255,1533), _readout_month_transform is never called in readout_new (readout.py:1732,1881 only), context dict lacks "dim_cols" while readout_new.py:337 reads it, _get_granular_level_pivot takes 6 params (readout_utils.py:1461) but both versions pass 5, readout.py reads "diag_scoped" (set at filter_generator.py:1183) while readout_new.py:1553 reads "diag_tagging".
Framing: readout_new is not wired into production (only new_readout_logic_test.py:10 imports it; process_data builds legacy Postprocess at readout.py:1972,1995). All regressions are in a candidate path. The test harness never compares tuple idx 10 and 13.

## 1. Traceability matrix

### 1a. Forward (RECONSTRUCTED -> ORIGINAL)
| RECON unit | ORIGINAL | status | note |
|---|---|---|---|
| PreparedReadoutData :12 | none | added | carrier; support_df None has two meanings |
| _prepare_readout_data :24 | readout_adjust data half readout_utils:6174-6240 | equivalent | harness-verified; hidden file read via load_level_rename_display |
| _build_readout_text :64 | readout_adjust text half + _top_data/_get_readout | degraded | LINKEDIN hardcode (:123 vs utils:6223); unused params; top-N cut on BR agg without level config |
| _prepare_aggregate_new :128 / _prepare_diag_new :144 | readout.py:1264-1272 / 1286-1292 | agg equivalent; diag degraded | bodies identical to each other; diag loses own growth/sort cols |
| _build_bi_readout_str_new :160 | readout.py:1256-1259,1281-1284,1293-1298,1307-1308 | equivalent except diag cols | |
| _build_bi_tables_new :253 | readout.py:1260-1272,1299-1325 | degraded | context["dim_cols"] KeyError (COLGUS >3041); agg table own sort col (benign) |
| _build_bi_fixtext_str_new :350 | readout.py:1185-1199 | degraded | pooled frame instead of per-level data1 |
| _build_bi_principle_pretext_new :400 | readout.py:1208,1222-1226 | degraded | separator lost; pooled frame |
| _prepare_bi_supplemental_context_new :433 | none (replaces data1 for fixtext/principle/time) | added | no media_channel fallback; raises outside try |
| _build_bi_benchmark_new :470 | readout.py:1248-1254 | degraded | runs post-adjust |
| _process_bi_detail_share_new :511 | _process_bi_detail_share :1073 | degraded | slots 7,12,13 hardcoded empty; planner delegated to legacy |
| _build_br_readout_str_new :706 | readout.py:1396-1402,1414-1428 | degraded | trigger_pretext prefix, empty-str_q gate lost |
| _prepare_br_supplemental_context_new :756, _build_br_marketing_driver_notice_new :781, _build_br_fixtext_str_new :799 | readout.py:1373,1382-1384,1405-1407,1439-1468 | degraded/added | predicate changed; USB notice new; contribution fixtext revived |
| _build_br_tables_new :867 | readout.py:1429-1431,1434-1438,1469-1473 | equivalent | |
| _process_br_detail_share_new :924 | _process_br_detail_share :1331 | degraded | sales delegated (faithful) |
| PostprocessNew.__init__ :1083 | Postprocess.__init__ :1480 | equivalent | discards yaml detail-fn mapping (no present effect: all 8 clients default) |
| _process_bi_share_new :1417 | _process_bi_share :1555 | degraded | lookup keys, exception policy, pretext source, month transform |
| _process_br_share_new :1195 | _process_br_share :1744 | degraded | time_message, pretext source, month transform, unguarded block |
| _postprocess_bi/br_readout_text, _output_tables | readout.py:1698-1738 / 1842-1885 | degraded (month transform) | |
| _serialize_readout_tables :1089 | table_to_text (utils:4062) | intended divergence | views no longer truncated; HTTP tokenizer called unconditionally |

### 1b. Reverse (ORIGINAL -> RECONSTRUCTED)
- readout.py live: _process_bi_detail_share, _process_br_detail_share, Postprocess.__init__, _process_bi_share, _process_br_share are mapped above. Orphaned by design (dispatch, config, LLM): set_environment_variables, resolve_function, _get_function_name_* (5), _process_bi, _process_br, process_data, response_generate, response_generate_chart, _is_error_payload, _error_payload_text, _parse_denial_response, response_generate_insight, denial_classify, stream_response, check_truncation (dead). Lines 22-1071: 12 commented defs, no live code.
- readout_utils.py (256 top-level defs; nested defs covered under parents): S6 lines 1-2700 = 131 (88 used: 39 direct, 37 transitive, 6 inherited, 6 via planner/sales delegation; 43 orphaned: 14 upstream feeders, 29 fully orphaned = 21 dead + 8 live elsewhere). S7 lines 2700-5000 = 75 (54 reachable, only 34 via readout_new's own paths; 21 orphaned: 10 dead, 1 legacy-only, 10 pages-only). S8 lines 5000-7289 = 50 (38 reached; 12 orphaned: 3 live elsewhere, 9 dead). Per-function rows are in S6/S7/S8_inventory.md and *_reviewer.md.
- readout_other_category.py: dead (ImportError on prompt_template, no importers); 3 live-syntax defs + 34 commented. Logic lives in the COLGUS model_group_id != 3041 branches (readout.py:1178-1181,1321-1322,1366-1369; readout_new.py:576-581,335-338,964-969) and utils _read_and_process_csv_other_category.
- Unmapped RECON units (new with no original): PreparedReadoutData, supplemental-context builders, marketing-driver notice builder, USB source-of-change clause, _serialize_readout_tables wrapper.

## 2. Decoupling report: readout, table display and viz as separate tools

Target: three tools (readout generation, table display, visualization) that an outer agent decides when and how to call. Internals stay deterministic. Evidence below is verified by reading unless tagged (inferred). Nothing was executed.

### 2.1 Is readout separable from table display today? Partly.

Where it is already separate: inside the per-level detail functions the readout text (str_q, fixtext, benchmark_str, principle_pretext) is built from numeric frames and never reads the pivot tables. `_get_readout` and `_data_to_text` run on numeric `data` and `top_data` (readout.py:1256,1281,1296; readout_utils.py:1551). Text and table are siblings built from the same `data` (readout.py:1260,1270,1299). BR is the same (readout.py:1385-1438).

Where it is coupled, all in `Postprocess._process_bi_share` / `_process_br_share` and the code downstream of `process_data`:
1) Text is appended from the display tables. `_table_readout_prompt` adds `table_to_text` of the formatted agg, detail or overall view to the readout (readout.py:1692-1696,1726,1835-1836,1846,1869). Those tables have already gone through `apply_level_mapping_display` (1682), `format_table_by_metric_config` (1688) and `dedup_post` (1660,1803).
2) Text-budget truncation mutates display tables. `table_to_text` returns token-fitted frames (readout_utils.py:4084-4114) that overwrite tuple slots 1 and 7 (readout.py:1692,1695,1835).
3) Labeling runs after display formatting. `_bi_custom_instruction` and `_br_custom_instruction` take the formatted `self.detail_view` (readout.py:1706,1857) and call `_recent_roi_spend`, `_trend_roi_spend`, `_trend_roi_other`, `trend_bi`, `trend_contribution_sales` and `_get_outlier_pretext`. Distribution buckets (`classify_spend` 4686, `_metric_level` 4703, `classify_roi_level_new` 4937), trend tags (`analyze_trend` 5449) and IQR outliers (`_outlier_filter` 4547) all live here. Each re-parses strings back to floats with `_clean_numeric_after_tactics` (readout_utils.py:4824) and finds columns by display names ('Tactics', 'Business Driver', 'YOY', 'Spend'). `_outlier_filter` hardcodes an "MM" suffix (4580). Verified order in readout.py: 1688 format, 1692 table_to_text, 1706 labeling.
4) Control flow reads display tables. The lookup that picks per-level readouts keys on `df.empty` of display tables (readout.py:1594-1606 BI, 1764-1776 BR). BR swaps the readout on `detail_view.equals(agg_view)` (1817-1822) and reads `biz_driver_check` from display columns (1823-1829).
5) Shared mutable state. `self.readout`, `pretext`, `detail_view`, `agg_view`, `overall_view`, `dim_cols`, `result_combo_match` are written by text and table steps in interleaved order. `readout_adjust` (readout_utils.py:6174) returns a filtered `data` and rewritten `str_q` that the table then builds from, a shared-input coupling. `_ignore_growth_outlier` writes NaN into `data` in place (readout_utils.py:2156) upstream of both routes.
6) The insight path is downstream of display, not parallel to it. `build_readout` takes `data` and `pivot_biz_table` from `process_data` (insights_only.py:442-447,661-667; utils.py:422). It reverses formatting with `strip_sign_and_unit` (insight_generation/utils.py:153-171), finds columns by regex on "{alias} {time}", and labels high, moderate and low with `label_metric_level_with_dist` (roi.py:34-91) and trends with `label_metric_trend` (utils.py:204-254), after the display table exists. Q1 and Q3 thresholds come from `bi_metric_distribution.csv`, not from the frames. The `ReadoutData` frames are used only for `ner_filters`, scoping keys and level lookup. Two fragilities (inferred): `parse_value` strips only %, $ and comma, so M/K/B suffixed values from `add_metric_format` with UsePowers likely drop out of labels silently, and the `'%' in raw` branch at roi.py:81 can never fire after `strip_sign_and_unit`.

Answer to the three questions. a) process_data readout cannot be called without display tables, because slot 0 includes table text and slots 6, 11 and 13 depend on labeling that runs on formatted tables. b) Insight generation cannot be separated, it consumes display tables as its numeric input. c) Distribution categorization and outlier labeling run after the display table is built, in both paths. Exceptions that run before display: benchmark `categorize` (readout_utils.py:1725), `_ignore_growth_outlier` and the top-N cut of the text readout (`_top_data`, readout.py:1242).

### 2.2 Tuple slot map (process_data, 14 slots, readout.py:1740,1887)

| Slot | Content | Kind | Display dependence |
|---|---|---|---|
| 0 | readout | text | includes table_to_text of display tables (1692-1696) |
| 1 | detail_view | table | formatted, truncated by token budget |
| 2, 3 | benchmark data, string | table, text | numeric before display; BR empty |
| 4, 5 | planner_data, spend_share_principle | table, frame | BR empty |
| 6 | pretext | text | fixtext plus labeling output (1716,1859) |
| 7 | agg_view | table | formatted, truncated |
| 8 | pretext_table | frame | built by `_overall_metric_w_driver` (readout_utils.py:2874) |
| 9 | pretext_table_trend | frame | only returned frame with trend labels |
| 10 | result_combo_match | dict | selected via display-table emptiness |
| 11 | principle_pretext | text | numeric origin (1208-1226) |
| 12 | overall_view | table | formatted |
| 13 | readout_adj | bool | derived from readout string (1700,1842) |

A table tool is close to free once the coupling above is cut, since it returns slots 1, 7, 8, 9 and 12.

### 2.3 Proposed stage model

Put the seam between labeling and formatting. Today the order is compute, text, format, truncate, label, serialize. Target order:
1) Compute. Numeric per-level frames with growth, sort, benchmark and outlier-NaN already applied (existing detail-function logic, unchanged).
2) Label. Distribution buckets, trend tags and outliers added as columns or a side dict on the numeric frames. Move the logic from the `_*_custom_instruction` family onto raw numerics so no string parsing remains.
3) Select. Lookup row, `match`, `biz_driver_check`, `readout_adj` decided from numeric state, not display emptiness.
4) Package. One typed `AnalysisResult` (ReadoutData plus numeric frames, labels, match, metadata). This replaces the stringly-keyed context dict, which findings 4 and 10 show is already breaking.
5) Tools consume it. Readout tool renders text and token-fits it on a copy. Table tool applies `apply_level_mapping_display`, `format_table_by_metric_config`, month transform and renames. Viz tool builds chart configs.

Display must never feed back into stage 2 or 3. `table_to_text` becomes non-mutating, with the token budget a parameter and the tokenizer injected (readout_new.py:1089 already stops the mutation for copies but its custom instructions still take `detail_view.copy()` at 1130 and 1761).

### 2.4 Decision: what should visualization consume?

Neither option as stated works. Findings that bear on it:
a) Charts today consume the display tables. `get_data_for_plot` (viz_utils.py:101-123) regexes the time token out of headers like "ROI Q1 2024" and strips $ and %. Driver and outlier detectors index by 'Tactics' / 'Business Driver' (viz_utils.py:20,1626). `viz_config` needs `match['is_specific']` and `pivot_biz_table` non-emptiness (viz_utils.py:7). KPI cards use slots 8, 9 and 12.
b) The filtered `ReadoutData` frames are not enough. They have not been aggregated to the level, pivoted by period, grown, benchmarked or looked up. A viz agent on raw frames would re-implement `process_data`.
c) Viz on formatted display tables inherits the parse fragility in 2.1 item 6 and breaks whenever display formatting changes.

Recommendation: viz consumes the numeric, labeled `AnalysisResult` (stage 4), after labeling and before formatting. It shares labels with readout and table tools, so charts and text cannot disagree about outliers, and a display change cannot break a chart. Display-name mapping (level rename, month names) is applied by a small shared display layer both the table tool and the viz tool call. Confidence: high on rejecting both stated options, medium on exact frame shape because I did not read every chart builder body (st_utils.py was not read).

### 2.5 Agent boundary and harness

Outer agent calls three tools with typed arguments and gets typed results: `generate_readout(analysis, options) -> text slots`, `render_tables(analysis, view_spec) -> formatted frames`, `build_chart_configs(analysis, chart_spec) -> configs`. Harness rules: explicit client_code and model_group_id passed through, no env reads inside tools; token budget and tokenizer as parameters; tools are idempotent and non-mutating on the shared `AnalysisResult`; the agent can request a view that skips truncation. `answer_coverage` (coverage.py:94) is the only function-shaped tool today. The planner dispatches by `spec.kind` in a Streamlit rerun cursor (orchestrator_page.py ~500) and carries results as text in `st.session_state.planner_context`, so no tool-calling loop or typed carrier exists yet.

### 2.6 Blockers to independent or concurrent calls

a) Client selection through `os.environ` (local.py:11-18). `get_data_path` is called 34 times in readout_utils.py, so client args on loaders are decorative. Main blocker for concurrency across clients.
b) `chart/*.py` has 242 `st.` calls, and `st.session_state.insight` is written inside `build_*` (roi.py:1317-1366, breakdown.py:1164-1735). Viz cannot run headless until build and render are split. `viz_utils.py` has no streamlit import but loads files (150,168-172,1203,1682,1781,2216), so it is env-bound.
c) `Postprocess.__init__` loads six CSVs per call (readout.py:1491-1500). Safe per instance once env is fixed, wasteful per tool call.
d) `get_ag_level_mapping` reads `ner_filters["clientCode"]` and `["modelGroupId"]` (viz_utils.py:1188). Only `prod_implementation/toolkit_controller.py:564` sets them, so the st-main ROI path likely raises KeyError (not run).
e) Call-site drift beyond CLAUDE.md. `process_data` takes 5 args including `metric_configs` (readout.py:2018). `st-main.py:400` and `tests/test_insights_suite.py:254` pass 4 and would raise TypeError. `insights_only.py:447` and `orchestrator_page.py:306` pass 5. Treat those two as the current callers.
f) Second tree. `src/prod_implementation/data_reducer.py:2430` has another `process_data` with explicit `ner_results`, `map_dict` and `dbc` params, and `toolkit_controller.py:332` (`_analytics_generator`) chains it statelessly. Section 4 excluded it as out of scope. It is modified in git status and is probably the better base for the agentic layer. Owner decision needed (see section 5, step 2).

### 2.7 Extractable, seams and leave alone

Extractable as-is (pure): `_build_readout_text`; `_prepare_aggregate_new` and `_prepare_diag_new` (merge into one); fixtext and principle builders once handed a frame; table builders; about 70 pure utils.
Seams needed: a) `support_df` injection for `_prepare_readout_data`; b) tokenizer and max-token injection for `table_to_text`, serialized lazily; c) `AnalysisResult` in place of the context dict; d) a loader layer for the 28 file-reading utils; e) a display layer shared by table and viz tools.
Leave alone (behavior risk): `_get_agg_data`, `_get_readout_com`, `normalize_map_dict`, `_planner_intention_res`; planner and sales paths (delegated to legacy, correct choice); shared latent bugs in 3C.
Coupling introduced by readout_new: `PostprocessNew` overrides function_*_detail after `super().__init__`, bypassing the yaml map. Fixtext moved from per-level detail into one pooled block. It splits text from tables inside the detail functions (`_build_bi_readout_str_new` 160, `_build_bi_tables_new` 253) and adds `PreparedReadoutData`, but the Postprocess layer still consumes formatted tables, so it does not reach the seam in 2.3.

## 3. Findings (ranked)
Legend: C = confirmed by finder and reviewer (and orchestrator spot-check where noted); D = decision needed.

### 3A. Behavior regressions
1. [C, high] Supplemental/fixtext block runs outside any try and raises "No valid data." when AG/MG/M are all empty (the standard rejection path). readout_new.py:449,764,1250-1276,1517-1540 vs readout.py:1575-1585,1758-1762. insights_only.py:790 and orchestrator_page.py:308 expect an empty tuple and process_data has no try. Fix: wrap in try/except with empty defaults, or return early on all-empty frames.
2. [C, high] time_message never reaches pretext (BI and BR; sales loses its entire pretext). readout_new.py:395-396,1255,1276,1533 vs readout.py:1227-1229,1356-1358,1382-1384. Likely unfinished (57d15ac). Fix: prepend self.time_message + "\n\n" after 1533 (skip planner) and 1276; pass None to _validate_time_range for sales.
3. [C, high] Planner pretext discarded; supplemental block and genome loaders now run for planner. readout_new.py:1460,1517-1540 vs readout.py:1137-1154,1635. Fix: skip 1517-1540 when intention is planner; take pretext from the planner level dict.
4. [C, high for COLGUS 3042-3045] context["dim_cols"] never set. readout_new.py:337,647-677 vs readout.py:1322; swallowed by the catch-all so the level goes silently empty. Fix: add "dim_cols": dim_cols. (orchestrator verified)
5. [C, med-high for month periods] _readout_month_transform never called, BI and BR. readout.py:1732,1881. Readout says "month 3", tables say "March". Fix: add call at end of both text-postprocess methods. (orchestrator verified)
6. [C, conditional on CSV columns, high if present] BI lookup keys: overall_ag_only column ignored (key never matches, KeyError('detail_view') at readout_new.py:1643); diag_tagging used where legacy uses diag_scoped. readout_new.py:1553,1560-1565,1621 vs readout.py:1524-1527,1598-1606,1648. readout_new last touched 2026-09-17, before legacy commits 1cb5c0f, 886cbd2, 95c2305, 5570259. Fix: port readout.py:1598-1606 and 1648-1655. Confirm the CSV columns first.
7. [C, D] BR trigger_pretext "no table readout" prefix removed (deliberate, 614b590). readout_new.py:751-752,991,1003-1013 vs readout.py:1390,1405-1407,1432-1433,1842. readout_adj stays False, so the table and instruction branch runs where legacy skipped it and insights_only.py:588 uses the raw query instead of the generic one. Fix: uncomment if unintended, else delete dead code and record the decision.
8. [C] BR contribution overall fixtext and pretext_table revived (readout_new.py:847-861 vs readout.py:1459-1468 `pass`). Fix: delete 847-861 and unused custom_tagging_condition.
9. [C] BR marketing-driver notice predicate changed: lost core_dimension gate, data.empty trigger and business_driver_captured; new USB source-of-change text (unnecessary addition). readout_new.py:781-796 vs readout.py:1405. Fix: restore O predicate per level; remove 785-787 clause.
10. [C, medium-low] Diag frame reuses detail growth_col/sort_by_col; KeyError when diag has one period and detail two. readout_new.py:145-157,222-246,306,645 vs readout.py:1289-1298. Fix: compute inside _prepare_diag_new and pass through.
11. [C, medium, data-dependent] Pooled union frame replaces processed per-level data1 for fixtext, principle, spend range, time range and uniq_dim; lookup "pretext" column no longer selectable (cannot blank pretext per combo, removed in 57d15ac). readout_new.py:433-466,806-820,1517-1540 vs readout.py:1185-1229,1635-1641,1793-1798. Reviewers corrected: driver_tags difference is harmless; BI uniq_dim is computed before the overall filters in legacy. Single restoring fix below resolves 2, 3, 8, 9, 11 and 13.
12. [C, medium] Spending principle separator: both non-empty glues text with no "\n\n" (readout_new.py:426-429 vs readout.py:1225-1226). The new empty-string result when summary is empty is an improvement; keep it. Fix: `if summary: principle_pretext += (HEADER if not principle_pretext else "\n\n") + summary`.
13. [C, low-medium] media_channel fallback missing in union builder: KeyError when a frame has neither media_channel nor marketing_channel (readout_new.py:461-465 vs readout_utils.py:680-683).
14. [C, low-medium] BR `if not str_q: data = pd.DataFrame()` gate lost (readout.py:1408-1409; reviewers disagree on severity, S5 medium vs S2 low; reproduced on NaN-rank synthetic data). Fix: compute the pre-adjust text before readout_new.py:1017 and blank data if empty.
15. [C, conditional low-medium] BR aggregate readout applies top-N and uses agg frame as data only when no level_rename_display exists (readout_new.py:73,732-749 vs readout.py:1419-1423). Fix: apply_top flag = has_level_config.
16. [C, low-medium, level-config only] BI benchmark runs on post-adjust data and ner_dict (readout_new.py:470-508 vs readout.py:1248-1258). Fix: stash pre-adjust copies.

### 3B. Lost error handling
17. [C, medium] BI per-level catch-all (readout_new.py:1473-1493) vs legacy re-raise except "No valid data." (readout.py:1575-1585). Hides findings 4 and 10. Fix: `if str(error) != "No valid data.": raise`.
18. [C, low-medium] halo `.iloc[0]` IndexError (readout_new.py:53, utils:6199) now becomes silent level loss under the catch-all.

### 3C. Shared with legacy (not readout_new regressions, fix separately)
a) _get_granular_level_pivot arity: 6 params, 5 passed (readout.py:1278, readout_new.py:191-197): TypeError for COLGUS BI with agg data. b) _benchmark_to_text hardcodes "Tactics" while _benchmark_data emits "tactics" (utils:1693,1744): KeyError for margin roi/performance with core_dimension and benchmarkROI (executed). c) _benchmark_table_display discards drop() result (utils:1788). d) half periods unparsed in _read_and_process_csv_other_category (utils:687). e) level_trend_sort labels mismatch analyze_trend. f) _trend_roi_other KeyError without response column. g) _planner_spend_range concat before empty guard (utils:3428). h) _planner_filter single-dimension char indexing (utils:4256). i) _recent_roi_spend set() ordering: lost determinism, varies with PYTHONHASHSEED (utils:4882). j) _soc_split_table ignores non-driver dimensions (utils:3986-3998). k) select_scenarios_by_kpi regex contains. l) roi_spend_trend_analyze p<0.5 vs 0.05. m) level_filter_default case mismatch with apply_level_mapping_display. n) chart/breakdown.py and chart/roi.py (20 call sites) pass kwargs add_metric_format does not accept since 6b9ceea. o) pandas 3.0.2 pinned in tracked requirements312.txt while Dockerfiles use python:3.10-slim: chained fillna(inplace) and groupby.apply grouping-column hazards apply to the 3.12 track only (emulated, ~80% confident).

### 3D. Unnecessary additions and organization
19. Dead self.time_message until fixed; unused params in _build_readout_text (client_code, model_group_id, ner_res_dict); unused context keys (period_type, driver_tags, overall_context); duplicate _prepare_aggregate_new/_prepare_diag_new bodies; redundant .copy() in _serialize_readout_tables; commented blocks at 395-396,586-593,646,659,751-752,1003-1013; PostprocessNew.__init__ discards yaml mapping.
20. Dead code (separate cleanup): readout_other_category.py (update CLAUDE.md:177 on deletion); readout.py:22-1071 commented defs (only record of old halo rejection and HILSP brand/campaign filter); 16 caller-less readout_utils helpers (reviewer corrected the 18: _overall_metric_other_category and _get_ner_key_value_other_category are live; only _overall_metric of that group is dead); resolve_function logs via root logging at readout.py:1914.

### 3E. Decisions for the owner
a) Prefix/readout_adj removal (finding 7). b) Hardcoded "LINKEDIN" at readout_new.py:123: reviewers split three ways (exact parity f-string per S5; keep hardcode and drop client_code param per S1; legacy leaks the sentinel for other clients so new behavior is arguably a fix per S8). Behavior differs only for non-LINKEDIN clients with a level_rename_display file and more than 2 level combinations. c) Legacy typo `diag_scope` at readout.py:1649,1701 (never set; guards always true) while readout_new dropped the guards: do not "fix" during cleanup without a decision. d) Keep the pooled design (narrow patches: 1, 2, 4, 8, 9 clause, 12, 17) or restore per-level fixtext and lookup-driven pretext (recommended by S1 and S2 reviewers; resolves 2, 3, 8, 9, 11 together). e) table_to_text truncation divergence is intended per the docstring; parity only if wanted.

## 4. Coverage report
Scopes S1-S9 audited by a finder and an adversarial reviewer; none had zero findings. Every in-scope ORIGINAL function is accounted for in sections 1b and the per-scope files (S1-S5 finder/reviewer, S6-S8 inventory/reviewer, S9). Not reviewed: src/prod_implementation/readout_utils.py (a forked copy with drifted signatures; modified in git status; out of the stated ORIGINAL set); numeric old-versus-new output (no data); lookup CSV contents; pandas 3 behavior (emulated only).
Process deviations to be aware of: S6 reviewer used an AST reference scan and S7 finder a caller-scan script for orphan detection, against the no-automated-scan rule; the S7 and S6 reviewers re-derived reachability by hand first and their counts matched. Reviewers' first round failed on a network error and was rerun; S1-S5 and S9 reused their independent mapping files from the first attempt (written before reading the finders).

## 5. Ordered plan (none applied)

Order follows risk: prove parity first, fix the contract, move the seam, then patch what survives the restructure, then wire. Items that live in the Postprocess layer (findings 2, 5, 6, 7) are patched after step 5 because the restructure rewrites that layer.

1. Build the parity harness. Extend new_readout_logic_test.py to compare tuple idx 6, 10 and 13 and pretext, pass normalize_map_dict, and add label comparison (distribution bucket, trend tag, outlier text). Capture legacy golden outputs. The data and lookup CSVs are absent from this checkout, so this needs a data-bearing environment. Without it no later step is provable.
2. Owner decisions that gate design: a) which tree is the base, `src/model` (readout_new) or `src/prod_implementation` (data_reducer, toolkit_controller), see 2.6(f); b) 3E(a) prefix removal; c) 3E(d) pooled fixtext versus per-level; d) viz consumes `AnalysisResult` per 2.4; e) 3E(b) LINKEDIN hardcode. Confirm the lookup CSV columns for finding 6 here too.
3. Define `AnalysisResult` and the stage boundaries in 2.3 as types only, with no behavior change. Replace the context dict. This also fixes finding 4 (`dim_cols`) structurally.
4. Move labeling onto numeric frames. Port the `_*_custom_instruction` family, `_get_pretext_overall_trend` and `_get_outlier_pretext` to read the numeric frames instead of re-parsing display strings. Assert label parity against step 1 goldens. Watch for the M/K/B suffix drop and the `"MM"` hardcode in `_outlier_filter`.
5. Make the display side pure. `table_to_text` non-mutating with injected tokenizer and budget; lookup, `match`, `biz_driver_check` and `readout_adj` decided from numeric state (readout.py:1594-1606,1764-1776,1817-1829). Display formatting, level rename, month transform and change-column renames move into one display layer.
6. Patch findings that survive the restructure: 1 and 3 (guard supplemental block, skip for planner), 17 and 18 (restore re-raise policy), 10 (per-frame diag growth and sort cols), 12 (separator), 13 (media_channel fallback), 8 and 9 (delete contribution block and USB clause). Then the Postprocess-layer findings under the step 2 decisions: 2 (time_message), 5 (month transform, now in the display layer), 6 (overall_ag_only and diag_scoped lookup port), 7, 11, 14, 15, 16.
7. Decouple the environment. Pass client_code and model_group_id explicitly through the loader layer. Split chart `build_*` from `render_*` and remove `st.session_state.insight` writes from builders. Fix the `clientCode` / `modelGroupId` ner_filters dependency in `get_ag_level_mapping`.
8. Expose the three tools (`generate_readout`, `render_tables`, `build_chart_configs`) with typed inputs and outputs, then wire the outer agent. Replace the Streamlit rerun cursor and `planner_context` text carrier with typed results. Wire `PostprocessNew` or its successor behind an off-switch (blockers: hard-coded `Postprocess(` at readout.py:1972,1995, the hasattr gate at 1971,1994, `resolve_function` on dotted paths at 1898).
9. Separate PRs: shared bugs 3C, dead-code removal (20), chart `add_metric_format` break, stale callers in 2.6(e).

## 6. Review and suggestions (orchestrator agent, ask-genome)

Scope: re-checked sections 2.1 to 2.4 against the vendored `ask_genome_core/model/readout.py` and the orchestrator wiring (`run_postprocess.py`, `process_data_adapter.py`, `graph.py`). Static reading only, nothing executed. Vendored line numbers differ from section 2 by 8 (format at 1680, `table_to_text` at 1684, `_bi_custom_instruction` at 1698).

### 6.1 Confirmed in the orchestrator

a) The coupling in 2.1 holds. `process_data` formats the tables, then `table_to_text` appends table text to the readout and overwrites the display frames with truncated copies. Labeling runs on the formatted `detail_view` after that.
b) The orchestrator repeats the coupling. `run_postprocess` calls `process_data`, then `table_to_text` again, then `insight_input`, which calls `build_readout` on the formatted `detail` and `aggregate` frames. Insights read display tables as numeric input.
c) The graph is a fixed chain: `apply_data_filters -> run_postprocess -> generate_response`. No node lets an agent choose what to produce. `rehydrate_result(recipe)` already rebuilds a frame from a recipe, so frames never need to live in checkpointed state.

### 6.2 Revised framing: analysis versus presentation

Section 2 frames the seam as readout versus table versus viz. A better cut is analysis versus presentation. Viz overlaps with readout on aggregation, growth, benchmark and the high/moderate/low, trend and outlier labels. Table display is formatting only, so it is a view of the analysis and not a candidate agent.

Treat the readout pipeline as one deterministic `analyze` stage. Presentation consumes its result as readout text, table view or charts. This keeps section 2.4's conclusion (viz reads the labeled analysis, not the raw filter output and not display strings) and drops the claim that readout and table must be fully decoupled first.

### 6.3 Two ways to build it

1) Wrap `process_data` whole as one typed `analyze` result and move no logic. Parity-safe. Labeling still runs on formatted tables, so parse fragility (M/K/B suffixes, the "MM" hardcode in `_outlier_filter`, display-name column lookups) stays.
2) Restructure labeling onto numeric frames first (plan steps 3 to 5). Cleaner, but slow, and unprovable without the golden harness from step 1.

Recommendation: option 1 now, with one addition. Labels exist today only as prose inside the pretext (slots 6 and 11) and as slot 9. Viz needs them as data. Have the labeling functions also return a structured side dict keyed by driver (level bucket, trend tag, outlier flag), computed from the same formatted tables they use now. The change is additive, the readout text stays identical and parity is easy to check. Readout and viz then cannot disagree about outliers.

Move labeling to numeric frames later, and only if goldens show the fragility in practice. It becomes a contained refactor behind a stable interface.

### 6.4 Where viz reads from

Viz reads the `analyze` result including the structured labels. It runs after aggregation and labeling, so it sits after the table stage and not on the raw filtered data. Medium confidence on the exact frame shape. Not yet checked: how much of the chart builders' driver and outlier detection (`viz_utils.py:20,1626`) duplicates the labeling logic. Check that before building the side dict, so it carries what charts need.

### 6.5 Orchestrator agent shape

1) Add a deterministic `analyze` node after `apply_data_filters`. Store only the recipe id and bounded metadata in state. Rebuild or cache frames per tool call.
2) Add a presenter agent node with typed tools `generate_readout`, `render_tables` and `build_chart_configs`, each reading the analysis by recipe id. Tools are idempotent and do not mutate it.
3) Harness: call cap, bounded outputs, explicit `client_code` and `model_group_id` arguments, no environment reads, a typed error on failure and no retry loop.
4) Placement check against `AGENTS.md`: tools live in `packages/tools` and must not import agents. These tools are agent-coupled and in-process, so keep them inside the ask-genome package unless sharing is wanted.

### 6.6 Suggested order (replaces the order in section 5 for the agentic layer)

1) Golden parity harness (unchanged, still gating).
2) `AnalysisResult` types plus the `analyze` wrapper around `process_data`, no logic moved.
3) Structured labels side dict, with parity checks on the readout text.
4) Presenter node with `generate_readout` and `render_tables`.
5) Viz tool, after the chart build and render split (blocker 2.6b). No chart module exists in the orchestrator package yet. `visual_config_utils.py` still imports Streamlit and `src.*`.
6) Numeric-frame labeling and non-mutating `table_to_text` (plan steps 4 and 5), only if step 3 goldens show real fragility.

Section 5 steps 7 and 9 and the `readout_new.py` findings are unchanged. Do not patch `readout_new.py` regressions ahead of step 2, because the seam work rewrites that layer.

### 6.7 Open decisions

a) Base tree for the agentic layer. Recommended: `readout.py` with `readout_new.py`'s per-level text/table split folded in.
b) Whether the side dict lives in the vendored `readout_utils.py` (sync script overwrites it, see PARITY_AUDIT fact c) or in an agent-side adapter. Check how `scripts/sync_ask_genome_core.py` treats local edits before choosing.
