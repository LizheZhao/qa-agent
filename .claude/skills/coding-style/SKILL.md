---
name: coding-style
description: Coding style reference for the qa-agent repo. Read before writing or reviewing any Python code in this project so new code matches the existing conventions.
---

# qa-agent Coding Style

Distilled from the newest modules (`src/model/planner.py`, `src/model/coverage.py`, `pages/orchestrator_page.py`, `src/model/filter_clarification_graph.py`) and `docs/coding_style.md`. New code must look like it was written by the same person.

## Core rules

1. Readability first. If a logic fits in 50 lines, do not write it as 200. No speculative abstractions, no unrequested error handling, no wrapper classes around plain functions.
2. Full words for names: `client_code`, `model_group_id`, `clarification_fields`, `readout_data`. Never `cc`, `mgid`, `cf`. Established short names from the repo are fine where the surrounding code already uses them (`df`, `spec`, `cfg` for `config["configurable"]`, `esq`).
3. Simple English in docstrings and comments. Short sentences. Say what and why, not how.
4. PEP 8 enforced by flake8 codes `N,E,F,C,B`. Line length up to ~110 characters (match neighbors).
5. `print()` is prohibited in `src/model/*` (custom flake8 rule G001). Use the module-level `logger` everywhere in `src/`.

## Module layout

Order inside a file:

1. Module docstring: triple-quoted, plain English. First line is a one-sentence summary; a short paragraph may follow explaining design decisions (see `planner.py`, `coverage.py`).
2. Imports in three groups separated by blank lines: stdlib, third-party, then `from src....` project imports.
3. `logger = logging.getLogger(__name__)` after the imports.
4. Private module constants in upper case with a leading underscore: `_FILTER_META_KEYS`, `_SPACY_VALUE_KEYS`.
5. Functions and classes, private helpers prefixed with `_`.

## Functions

- snake_case names; type hints on parameters and return values using modern builtins (`list`, `dict`, `tuple`, `Optional[...]`, `dict[str, Any]`).
- Docstrings are one line, or a short block for functions with non-obvious behavior. State the return shape when it is a tuple or dict: `"""Answer a coverage sub-query. Returns {response, table, spec, facts}."""`
- Inline comments are lowercase and explain rationale, not mechanics: `# data values are stored lower-cased; lower-case the LLM-extracted values before any lookup`.
- Prefer returning early over deep nesting.

## Data and config

- Data containers are `@dataclass` with `field(default_factory=...)` and `from_local()` classmethod constructors (`src/data/data_interface.py`).
- Structured LLM output uses pydantic `BaseModel` (see `src/integrations/sdk_utils.py`).
- Prompts live in `config/prompt.yaml` as `UPPER_CASE_KEYS` with `|` block scalars; code loads them with `load_yaml_file` and fills them via `.replace("SOMETHING_PLACEHOLDER", value)`.
- Configuration is read from `os.environ`; `CLIENT_CODE` and `MODEL_GROUP_ID` are process-global.

## Streamlit pages

- Module docstring explaining the page's flow and what it deliberately does not touch.
- `SESSION_DEFAULTS` dict looped into `st.session_state` at the top.
- Section banner comments: `# --------------------------------------------------------------------------- sidebar`.
- Page-local helpers are private functions (`_render_payload`, `_reset_planner_state`).
- LangGraph interaction pattern: keep the compiled app in `st.session_state`, call `invoke`, then `get_state(config)`; check `graph_state.next` for interrupts and resume with `Command(resume=...)`.

## Quotes and formatting

- Double quotes in new code (the newest modules use them); do not requote existing code.
- f-strings for logging: `logger.info(f"coverage spec: op={spec.operation}")`.
- Multi-line call wrapping: align continuation with the opening parenthesis, as the neighbors do.

## Testing and linting

- Benchmark scripts, not pytest, are the historical test surface; new unit tests go under `tests/`.
- Lint: `pip install .` once (registers the G001 plugin), then `flake8 .`.
- Files matching `test-*.py` are excluded from style checks.
