# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A single-process ("shared worker") agentic orchestration service. FastAPI serves chat, history and
diagnostic APIs; LangGraph agents run in-process; MongoDB holds durable conversation turns and
execution traces; a React/Vite diagnostic UI is served from the same Uvicorn process at
`/diagnostics`.

The deployment unit is one immutable image containing the app plus every explicitly declared agent
package. Agents are never discovered by filesystem scan: `deployments/local.yaml` (validated against
`schemas/deployment.schema.json`) declares exact package, version, graph entrypoint, manifest
entrypoint and routing eligibility, and startup compiles those graphs once.

## Engineering rules live in AGENTS.md

`AGENTS.md` (root) is the binding engineering contract, with subtree additions in `src/AGENTS.md` and
`frontend/AGENTS.md`. **Read the applicable AGENTS.md before editing.** It owns the dependency-direction
rules, persistence invariants, test-marker policy, generated-file rules and per-scope verification
gates. Do not restate or contradict it here.

## Commands

Setup (Python 3.12 + `uv` workspace, one lockfile, no per-package venvs):

```bash
uv sync --frozen --all-packages --dev
```

Run the service (needs `.env` copied from `.env.example` with real MongoDB values):

```bash
uv run uvicorn agentic_orchestration.main:app --host 127.0.0.1 --port 8000
```

Full Python gate set, same as CI (`.github/workflows/ci.yaml`):

```bash
uv run ruff format --check . && uv run ruff check --no-cache . && uv run mypy && uv run pytest && uv run python scripts/validate_deployment.py
```

Single test file or test:

```bash
uv run pytest tests/unit/test_router_graph.py -k test_name
```

Markers: `unit`, `integration` run by default. `live` (enterprise gateway credentials) and `mongo`
(retains records in configured MongoDB) are opt-in and skip without their env flags. Normal collection
must never hit the network.

Frontend (Node 24.15+ on the 24 LTS line, or 26+):

```bash
cd frontend && npm ci && npm test && npm run build
```

`npm run check:api` regenerates `src/api/schema.d.ts` from FastAPI's OpenAPI and fails on drift. Run it
whenever routes or response models change. Never hand-edit `schema.d.ts` or `frontend/dist/`.

Image and compose changes also require:

```bash
docker compose config --quiet && docker build --build-arg SOURCE_COMMIT="$(git rev-parse HEAD)" -t agentic-orchestration:local .
```

## Architecture: request path

`FastAPI route (api/) -> Executor (execution/executor.py) -> AgentRegistry -> startup-compiled router
graph -> RegistryAgentInvoker -> one eligible child graph`.

Startup assembly is one phase in `src/agentic_orchestration/lifespan.py`, and reading it is the fastest
way to understand runtime wiring. Order matters: child graphs compile first with only the tools their
manifests declare, then the restricted invoker and routing catalog are built from those children, then
the router compiles with the invoker plus catalog injected, then the merged registry freezes.

The router is the durable session owner and delegates to at most one child per turn. `routable: false`
in the manifest makes an agent unreachable by delegation.

## Architecture: the three storage concerns

Keep these separate; conflating them breaks the contracts.

1. **Conversation sessions** (`sessions/`). `sessions` metadata documents plus immutable `session_turns`.
   Per-session process-local lock serializes load, graph invoke, commit. Graph execution happens
   *outside* the transaction; a short transaction then inserts the turn and advances the session only if
   the revision still matches. A conflict returns retryable `409` without reinvoking the model or tools.
   Failed invocations create no turn, so a failed first turn leaves no empty session. This is not
   LangGraph checkpointing and there is no resume. Durable document shapes, collection names and index
   names live in `sessions/schema.py`; router-era turns are schema version 2.
2. **Execution traces** (`observability/`). One root trace per attempted turn, propagated through router
   and child graphs, spanning graph/node/model/tool/child-agent boundaries with bounded
   secret-redacting content capture. Persisted best effort by a bounded writer: observability failure
   must never fail an otherwise committed conversation. Production uses a null event publisher until
   authenticated SSE exists.
3. **Diagnostic reads** (`diagnostics/`). A separate `DiagnosticReadStore` protocol for cursor-paged
   cross-collection dashboard queries. Do not add dashboard reads to `SessionStore` or conversation
   writes to `DiagnosticReadStore`.

Each has a Mongo adapter plus a memory adapter. Memory adapters are injected test substitutes, not a
runtime fallback: production fails startup rather than degrading session storage.

## Packaging layout

Every installable project is `<root>/pyproject.toml` + `src/<unique_import_package>/`. Distribution
names use hyphens, import packages use underscores. Agents cannot share one `src/` because each would
export colliding top-level `graph`, `state`, `manifest`, `prompts` modules into the shared interpreter.

- `src/agentic_orchestration/` app; `src/integrations/` external connection construction only (takes
  explicit values, never imports `Settings`)
- `packages/orchestration-core/` framework-facing contracts (`AgentManifest`, `AgentDependencies`,
  `AgentDescriptor`, invocation, observability, tools)
- `packages/enterprise-llm/` LangChain `BaseChatModel` gateway with Claude/OpenAI request shapes
- `packages/agents/<name>/` one distribution per agent, each with `manifest.py`, `graph.py`
  (`create_graph(deps) -> StateGraph`, returned uncompiled), `state.py`, `prompts.py`, optional
  `contracts.py` and `nodes/`
- `packages/tools/src/orchestration_tools/<family>/` all tools in one app-coupled distribution
- `packages/agents/_template`, `packages/tools/_template` non-runnable scaffolds; never installed

Version synchronization is a real gate: an architectural package change must keep its
`pyproject.toml` version, exported manifest version, `deployments/local.yaml` declaration and
`uv.lock` in agreement, or `scripts/validate_deployment.py` fails.

## Frontend structure

`src/app/` composition only (providers, navigation, error boundary, shell). `src/features/<domain>/`
owns its components, requests, feature types, tests and CSS. `src/components/` only for things two or
more features use. Renderer-neutral trace transformation stays in `features/traces/projection.ts` and
layout in `canvas-model.ts`, importing no React and no browser globals. Tabs are Lab, Agents,
Conversations, Runs under the `/diagnostics/` Vite base.

## Reference docs

`ARCHITECTURE.md` for boundaries and durability semantics. `docs/repository-layout.md`,
`docs/routing-agent.md`, `docs/execution-observability.md`, `docs/diagnostic-ui.md`,
`docs/enterprise-llm.md`, and the `docs/adding-a{n-agent,-subagent,-subgraph,-tool}.md` how-tos.
`README.md` lists what is deliberately *not* implemented (summaries, auth/tenancy, streaming,
checkpoint-resume, distributed queues, remote agents).
