## Benchmark Config
- Config file: `./config/benchmark.yaml`
  - RUN_NAME: `run7`
- Config file: `./config/benchmark_mapping.yaml`
  - Benchmark will run on all queries in `{GT_FILE}`
  - To disable a specific benchmark from running, set `RUN` to 0
  - To disable a specific benchmark from checking against ground truth, set `RUN_GT_COMP` to 0

## How to run?
- Execute `test.py`
- If `./{RUN_NAME}_data_filter.pkl` exist, the script will skip and generation and use results from previous run
- To rerun on same `RUN_NAME`, delete `./{RUN_NAME}_data_filter.pkl`

## Benchmark Files
- Path: `/datascience6/data/ask-genome-benchmark/{CLIENT_CODE}/{MODEL_GROUP_ID}/{VERSION}`
- Data filter
  - `./{RUN_NAME}_data_filter.pkl`
  - `./{RUN_NAME}_data_filter.csv`
- Data filter comp: `./{RUN_NAME}_data_filter_comp.csv`
- Readout table: `./{RUN_NAME}/{QUESTION_ID}/{TABLE_CODE}.csv`
- Readout text: `./{RUN_NAME}_readout.csv`

## Data Filter
- A `DataFilter` contains `Required` Fields, `Optional` Fields
  - Missing `Required` Fields could lead to bug subsequent module
  - If the `intention` is `none`, a `Rejected` class will be returned
- Check `src/benchmark/data_filter` detailed definition
- For example, for requirements of ColgusTp's DataFilter, check 
  - `DataFilter` 
  - `DataFilterColgusTp`

## Data Table
- TABLE_CODE
  - pivot_biz_table
  - data
  - planner_data: data on planner metrics;
    - Cols: 'Historical', 'Forecast', 'Incremental', and '% change' on Spend and relevant KPIs, as well as a key 'inc_dec' indicating increase or decrease in spending

## Readout
- Required Fields
  - id: `str`
    - e.g., `{CLIENT_CODE}-1`, `{CLIENT_CODE}-2`, ...
  - query: `str`
  - context_str: `str`
  - pretext: `str`
