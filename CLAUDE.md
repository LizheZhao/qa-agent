# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

"Ask Genome" is a QA agent that answers natural-language questions about marketing mix model results (ROI, spend, media tactics) for Analytic Partners clients. A query flows through NER filter extraction, data filtering and a "readout" (context string built from client data tables), then an LLM generates the final answer.

## Commands

- Run FastAPI service: `uvicorn services:app` (endpoints under root path `/ask_genome_core`)
- Run Streamlit chat UI: `streamlit run st-main.py`
- Run a single query end to end: `python main.py` (requires `CLIENT_CODE` and `MODEL_GROUP_ID` env vars; edit the hardcoded `query` in the file)
- Run benchmarks: `python test.py` (driven by `config/benchmark_mapping.yaml`; see below)
- Lint: `flake8 .` (first `pip install .` to register the custom flake8 plugin)

There is no pytest suite; `tests/test_ner.py` is a placeholder. `test.py` and `test_ner.py` at the root are benchmark scripts, not unit tests.

Note: `main.py`, `test.py` and `test_ner.py` call `logging.config.fileConfig('logging.conf')` at import, and `docs/coding_style.md` references a `requirements.txt` and `.flake8` file. None of these files are in the repo — they must exist locally (cwd) for those entry points and for lint config to work.

## Configuration and Environment Variables

Everything is configured via environment variables, populated by import-time side effects:

- `import src` runs `set_environment_variables()` which loads `config/config.yaml` into `os.environ` (only keys not already set — real env vars win)
- `import src.benchmark` similarly loads `config/benchmark.yaml` (e.g. `RUN_NAME`)
- `src/model/readout.py` loads `config/function_mapping.yaml` the same way

`CLIENT_CODE` and `MODEL_GROUP_ID` select the active client and are mutated globally per request in `services.py` — most data loaders read them from `os.environ` rather than taking parameters, so this is process-wide state.

Other key configs:
- `config/prompt.yaml`: Jinja2 readout prompt, rejection messages and per-client prompt overrides keyed by `{CLIENT_CODE}-{MODEL_GROUP_ID}` (loaded by `PromptTemplates` in `src/model/common.py`)
- `config/metric_mapping.yaml`, `config/function_mapping.yaml`: metric and function dispatch mappings

## Architecture

Core pipeline (same flow in `main.py`, `services.py` and `st-main.py`):

1. `ProcessIndicator.from_local()` (`src/data/data_interface.py`) loads per-client feature flags (legacy preprocessing, spaCy, hallucination alerts)
2. `generate_ner_filter()` (`src/model/filter_generator.py`) extracts structured filters (intention, metrics, dimensions, trend/rank) from the query using LLM structured responses, spaCy, a remote classifier and embedding-similarity in-context example selection; returns a `ReadoutData`
3. `process_data()` (`src/model/readout.py`, the largest module) filters client dataframes, builds pivot/benchmark/planner tables and assembles the readout context string
4. `response_generate()` / `stream_response()` sends the readout prompt to the LLM

Layers:
- `src/data/`: dataclasses in `data_interface.py` with `from_local()` constructors (all `from_database()` variants are stubs). `local.py` maps logical keys to files under `{DATA_DIR}/{CLIENT_CODE}/{MODEL_GROUP_ID}/` — all client data comes from that filesystem tree (a network share in practice)
- `src/model/`: NER filtering and readout logic. Client-specific variants live in `*_other_category.py` files alongside the legacy versions (`data_filtering.py` vs `data_filtering_other_category.py`, same for readout)
- `src/integrations/`: thin clients for external self-hosted services — OpenAI-compatible LLM at `LLM_SERVICE_URL` (DeepSeek R1 distill), embedding service, classifier service, and OpenTelemetry setup (`OTEL_SDK_DISABLED: "1"` by default)
- `src/benchmark/`: benchmark pipeline plus per-client `DataFilter` validation classes (e.g. `DataFilterColgusTp`) built via `data_filter_factory()`
- `src/pipelines/`: empty stubs

The LLM emits `<think>...</think>` reasoning. Streaming consumers (`services.py` `event_stream`, `st-main.py`) split chunks on `</think>` and route them to separate think/main SSE events or UI panels — preserve this contract when touching streaming code.

`services.py` passes dataframes between the `/get_data_filter` and `/stream_response` endpoints as base64-encoded feather blobs (`df_to_b64`/`b64_to_df`).

## Benchmarks

See `docs/benchmark.md`. Each entry in `config/benchmark_mapping.yaml` has `RUN`/`RUN_GT_COMP` toggles (0/1). Results are cached as `{RUN_NAME}_data_filter.pkl` under `{BENCHMARK_DIR}/{CLIENT_CODE}/{MODEL_GROUP_ID}/{VERSION}` — delete the pkl to force a rerun with the same `RUN_NAME`.

## Coding Style

From `docs/coding_style.md`:
- flake8 enforces codes `N,E,F,C,B` (PEP 8)
- `print()` is prohibited in `src/model/*` — enforced by the custom flake8 rule `G001` in `plugins/flake8_custom_rules.py` (installed via `pip install .`); use the module-level `logger` instead
- Files matching `test-*.py` are excluded from style checks (local development)
