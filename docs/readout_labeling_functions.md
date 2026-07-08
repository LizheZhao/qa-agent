# Readout labeling functions

Reference for the labeling / classification logic in the readout-generation layer. These functions turn raw numeric dataframes into the categorical tags ("high"/"low" ROI, "consistently increasing" trends, "overperforms" benchmark, outlier flags) that feed the LLM narrative and chart insights.

Unless noted, everything below lives in [src/model/readout_utils.py](../src/model/readout_utils.py). Several classifiers are **nested closures** applied via `df.apply(...)`; the line number points at the closure, and the wrapper that calls it is in the same parent function.

## Quick map

| Category | Function | Labels produced |
|---|---|---|
| Magnitude (level) | `classify_spend`, `_quartile_filter` | low / moderate / high |
| Magnitude (level) | `classify_br_metric`, `_metric_level` | low / moderate / high |
| Magnitude (ROI) | `classify_roi_level` *(dead)* | low / moderate / high |
| Magnitude (ROI) | `classify_roi_level_new`, `_recent_roi_spend` | high outlier / excellent / good / justifiable / inefficient / low outlier |
| Change pattern | `analyze_trend` | spiking / volatile / consistently increasing / consistently decreasing / flat / mild fluctuated (increase\|decrease) |
| Change pattern | `roi_spend_trend_analyze` | increase / decrease / "" |
| Change pattern | `bi_overall_trend_analyze` *(commented out)* | increase / decrease / "" |
| Benchmark | `categorize` | overperforms / is on par / underperforms / significantly over\|underperforms |
| Outlier | `_outlier_filter` | IQR text: significantly higher/lower |
| Outlier | `_outlier_filter_roi` | reads ROI "high/low outlier" level |
| Ranking | `level_trend_sort` | sort key from level + trend ranks |
| Column tagging | `_categorize_columns` | dimension / driver / metric column split |
| Display rename | `metric_label` | human-readable metric names |

---

## 1. Magnitude / level labeling (high / moderate / low)

### `classify_spend` — spend level by quartile
- Defined inside `_quartile_filter` ([readout_utils.py:3913](../src/model/readout_utils.py)), wrapper at [readout_utils.py:3894](../src/model/readout_utils.py).
- Cutoffs on the spend column: `< Q2 → low`, `<= Q3 → moderate`, `else → high`.
- Side effect: `_quartile_filter` also **drops** rows below Q1 for non-ROI intentions with >2 rows (low-spend noise), writes `{metric} level`, then sorts high→moderate→low via `level_rank = {high:0, moderate:1, low:2}`.

### `classify_br_metric` — generic BI metric level
- Defined inside `_metric_level` ([readout_utils.py:3940](../src/model/readout_utils.py)), wrapper at [readout_utils.py:3930](../src/model/readout_utils.py).
- Cutoffs: `<= Q2 → low`, `<= Q3 → moderate`, `else → high`. Writes `{metric} level`. Used by `trend_bi` and `trend_contribution_sales` for non-ROI, non-spend metrics.

### `classify_roi_level` — simple ROI threshold *(dead code)*
- Defined inside `_recent_roi_spend` ([readout_utils.py:4137](../src/model/readout_utils.py)).
- Cutoffs: `>= 1 → high`, `>= 0.8 → moderate`, `else → low`. **Currently unused** — the call site is commented out in favor of `classify_roi_level_new`.

### `classify_roi_level_new` — spend-weighted ROI tier
- Defined inside `_recent_roi_spend` ([readout_utils.py:4160](../src/model/readout_utils.py)), applied per `left_cols` group at [readout_utils.py:4224](../src/model/readout_utils.py).
- This is the **live ROI classifier**. Logic:
  1. Needs >= 3 rows, else label is "".
  2. Normalizes each tactic's spend share against total spend (`spend / (total_spend / 1e6)`).
  3. Flags tails as `high outlier` / `low outlier` using ROI quantiles (q05/q95) widened by tactics whose spend share > 5%.
  4. Buckets the remaining ROI values into quartiles (`pd.qcut`, q=4): `inefficient → justifiable → good → excellent`.
- Output column `{roi_type} level`. Ranked downstream by `level_rank_roi = {low outlier:5, inefficient:4, justifiable:3, good:2, excellent:1, high outlier:0}` ([readout_utils.py:4233](../src/model/readout_utils.py)).

---

## 2. Change-pattern labeling (trend shape over time)

### `analyze_trend` — primary multi-period trend classifier
- [readout_utils.py:4358](../src/model/readout_utils.py).
- Input: a row of period values + `all_mean_abs` + `period_type`. Needs >= 3 non-NaN values else "".
- Uses Spearman correlation (`scipy.stats.spearmanr`) over time-ordered values plus std and max-deviation ratios.
- Labels:
  - `spiking` — `max_deviation_ratio > spike_ratio` (default 4.0; callers pass 3).
  - `volatile` — `std > 0.35 * all_mean_abs`.
  - `consistently increasing` / `consistently decreasing` — `p_value < 0.05` and `rho` sign.
  - `flat` — `std < flat_multiplier * all_mean_abs` (flat_multiplier 0.06).
  - `mild fluctuated (increase)` / `mild fluctuated (decrease)` — fallback by first-vs-last value.
- Branch order differs for `year` period_type (consistency checks before volatility). Returns `{metric} trend type`.

### `roi_spend_trend_analyze` — directional ROI/spend trend
- [readout_utils.py:4474](../src/model/readout_utils.py).
- Simpler binary direction. Needs >= 3 values. Spearman: `rho > 0` and last>first → `increase`; `rho < 0` and last<first → `decrease`; else "".
- Applied to ROI and spend value columns in `_trend_roi_spend` → `{roi_type} change type` and `spend change type`.

### `bi_overall_trend_analyze` — *(commented out)*
- [readout_utils.py:4520](../src/model/readout_utils.py). Legacy first-vs-last direction tag (increase/decrease/""). Left as institutional memory; not called.

---

## 3. Benchmark performance labeling

### `categorize` — ROI vs industry benchmark
- Nested closure in the benchmark-comparison function ([readout_utils.py:1461](../src/model/readout_utils.py)), applied as `data['respond']`.
- Input: `over/under index` (percent vs benchmark ROI index). Labels:
  - `10 < x <= 100 → overperforms`
  - `-10 < x <= 10 → is on par`
  - `-100 < x <= -10 → underperforms`
  - `x > 100 → significantly overperforms`
  - `x <= -100 → significantly underperforms`
- Rendered to prose by `_benchmark_to_text` ([readout_utils.py:1479](../src/model/readout_utils.py)).

---

## 4. Outlier labeling

### `_outlier_filter` — IQR outlier flag + filter
- [readout_utils.py:3774](../src/model/readout_utils.py).
- Standard 1.5*IQR bounds on the target metric column. Returns `(filtered_df, outlier_line)` where the text buckets tactics into "Significantly higher/lower {metric}". Also strips the flagged rows from the df.

### `_outlier_filter_roi` — outlier text from ROI level
- [readout_utils.py:3834](../src/model/readout_utils.py).
- Does not recompute bounds; reads the `high outlier` / `low outlier` labels already set by `classify_roi_level_new` and formats the same higher/lower text.

---

## 5. Ranking by label

### `level_trend_sort`
- [readout_utils.py:4545](../src/model/readout_utils.py).
- Builds a composite sort key from `level_rank = {high:3, moderate:2, low:1}` and a `trend_rank` map, so high/moderate rows outrank low and ties break on trend strength. Used by `trend_bi` and `trend_contribution_sales`.
- **Discrepancy to watch:** `trend_rank` keys are `mild fluctuation (increase/decrease)` and `sudden spike`, but `analyze_trend` emits `mild fluctuated (increase/decrease)` and `spiking`. Those labels currently map to NaN in the sort. Confirm before relying on trend-tier ordering.

---

## 6. Column / display tagging (supporting)

### `_categorize_columns`
- [readout_utils.py:3240](../src/model/readout_utils.py). Splits dataframe columns into `(dimension_cols, driver_cols, metric_cols)` — metric cols are those containing a 4-digit year; the column immediately left of a metric is the driver.

### `metric_label`
- Nested closure at [readout_utils.py:5696](../src/model/readout_utils.py). Static rename map (e.g. `marketing_revenue_roi_brand → overall marketing ROI`), falling back to `col.replace("_", " ")`.

---

## 7. Trend/level orchestrators (where labels get attached)

These are not classifiers themselves but the functions that pull a metric's value columns, run the classifiers above, and emit the labeled `target_df` + a row-wise text description (`_describe_df_by_row`).

| Function | Line | Applies | Output columns |
|---|---|---|---|
| `_recent_roi_spend` | [4078](../src/model/readout_utils.py) | `classify_roi_level_new` + spend level | `{roi} level`, `spend level` |
| `_trend_roi_spend` | [4245](../src/model/readout_utils.py) | `roi_spend_trend_analyze` | `{roi} change type`, `spend change type` |
| `_trend_roi_other` | [4295](../src/model/readout_utils.py) | (selects YOY/QOQ change cols) | response / cost-per change cols |
| `_trend_roi_other_for_insights` | [4326](../src/model/readout_utils.py) | same + activity cols | response / cost-per / activity change |
| `trend_contribution_sales` | [4570](../src/model/readout_utils.py) | `analyze_trend` + `_metric_level` + `_outlier_filter` | `{metric} level`, `{metric} trend type` |
| `trend_bi` | [4671](../src/model/readout_utils.py) | `analyze_trend` + (`_metric_level` or merged ROI level) | `{metric} level`, `{metric} trend type` |

All of these gate on `_clean_numeric_after_tactics` (parse strings to floats, drop mostly-null rows) and `_quartile_filter` (drop low-magnitude noise) before labeling.
