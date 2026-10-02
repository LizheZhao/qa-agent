---
name: verify
description: Launch and drive the qa-agent agentic-orchestration service (FastAPI + LangGraph + MongoDB, React diagnostic UI at /diagnostics) to prove a change works. Use after touching api/, execution/, sessions/, observability/, diagnostics/, an agent package or the frontend, or whenever a claim about runtime behavior needs evidence.
---

# Verify qa-agent

Surface: HTTP API on Uvicorn (primary), plus the React diagnostic UI served by the same process at `/diagnostics`. Windows, PowerShell. Feature map: [features/README.md](features/README.md). Read the map before declaring a proof complete.

## Prerequisites (checked on this machine, 2026-10-02)

- `.venv` must be Python 3.12 from `uv sync --frozen --all-packages --dev`. `uv` is not on PATH; it lives at `%LOCALAPPDATA%\Packages\PythonSoftwareFoundation.Python.3.10_qbz5n2kfra8p0\LocalCache\local-packages\Python310\Scripts\uv.exe`. Set `$env:UV_PYTHON_INSTALL_DIR='C:\Users\lizhe.zhao\uv-python'` first, or the default Roaming path fails with "Missing expected target directory".
- The lockfile does not cover what the vendored ask-genome code imports. After sync, add: `uv pip install --python .venv\Scripts\python.exe "langchain-anthropic<0.4" scikit-learn openai opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp-proto-http`. Do not install an unpinned `langchain-anthropic`: it upgrades `langchain-core` to 1.x and breaks the lock.
- `.env` at the repo root with real Mongo and gateway values (already present). MongoDB 10.18.9.240:27017 is a shared dev database: reads are safe, `POST /api/chat` writes retained records there.
- `frontend/dist` needs `cd frontend && npm ci && npm run build`. On 2026-10-02 `npm ci` failed with `ERR_SSL_WRONG_VERSION_NUMBER` against registry.npmjs.org, so the UI was not driven. Without dist, `/diagnostics` returns 503 "Diagnostic UI is not built".
- Known break: the real `deployments/local.yaml` fails startup on `ask_genome_agent` (`No module named 'src.insight_generation.roi'`, imported by vendored `ask_genome_core/insight_generation/utils.py`). The helper therefore defaults to `deployment.verify.yaml`, a copy with `ask_genome` removed (verification scaffolding). Use `-Full` once the import is fixed, and to verify ask-genome.

## Launch, Doctor, Cleanup

One helper, `verify.ps1`, run from the repo root. Do not pipe its output through `tail` or `head`: the detached server keeps the pipe open and the call hangs.

```powershell
.\.claude\skills\verify\verify.ps1 launch   # starts uvicorn on 127.0.0.1:8765, waits for /health
.\.claude\skills\verify\verify.ps1 doctor   # pid alive, port owned by our process tree, /health, deployment name
.\.claude\skills\verify\verify.ps1 stop     # kills only the pid we started, and its children
```

Options: `-Port N` for a second instance (the helper refuses to start if the port is taken and never kills by name), `-Full` for the real deployment. Run `doctor` first whenever anything looks off. Startup takes 15 to 40 seconds. Logs: `evidence/server.log` and `evidence/server.err.log`. Two instances on different ports can coexist because all state is in Mongo.

## Drive

Plain HTTP with `curl` or `Invoke-RestMethod` against `http://127.0.0.1:8765`. Handles:

- `GET /health`, `GET /build`
- `POST /api/chat` body `{"message": "...", "session_id": "<optional>"}` returns `{session_id, trace_id, message, tool_calls}`; errors carry `{error, session_id, trace_id}`
- `GET /api/sessions/{id}/history`
- `POST /api/sessions/{id}/clarifications/{clarification_id}/responses` body `{submission_id, response}`
- `GET /api/diagnostics/` then one of `deployment`, `sessions`, `sessions/{id}`, `sessions/{id}/turns`, `sessions/{id}/traces`, `traces`, `traces/{id}`, `traces/{id}/spans`, `traces/{id}/spans/{span_id}`, `graph-definitions/{id}`; paged routes take `limit` (1 to 50) and `cursor`
- UI: `http://127.0.0.1:8765/diagnostics/{lab|agents|conversations|runs}`, driven with the built-in browser (`navigate`, `read_page`, `find`). Stable handles: nav `aria-label="Diagnostic workspace"`, composer textarea `aria-label="Diagnostic message"`, the "New chat" button, resize separators.

Python gates (`ruff`, `mypy`, `pytest`, `scripts/validate_deployment.py`) are tests, not proof of runtime behavior; run them separately per AGENTS.md.

## Evidence

Write proofs to `.claude/skills/verify/evidence/` (`proof-<feature>.txt` or screenshots). Proof standards:

- Exercise the real route or UI path, not test-only endpoints or in-memory adapters.
- Capture the request and the resulting state. For chat: the response, then the same `trace_id` via `/api/diagnostics/traces/{id}` and the turn via `/api/sessions/{id}/history`.
- Verify side effects alongside the visible result: committed turn count, trace `status` and `observability_status`, `committed_turn_number`. A failed turn must create no turn.
- Chat is a real gateway call and a Mongo write. Do it only when the change needs it, with one short message, and record the session id.
- Do not claim a UI proof from API output, or the reverse.

## Cleanup

Run `verify.ps1 stop`, confirm nothing listens on the port, and keep `evidence/`. Chat-created Mongo sessions persist; list their ids in the proof file and never delete them without the user's say.
