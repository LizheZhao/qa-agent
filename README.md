# Agentic Orchestration

An in-process agentic orchestration worker with durable conversation history. Compatible agent
distributions are pinned into one immutable shared-worker image and compiled once at startup.

## What is implemented

- Python 3.12 `uv` workspace and one exact `uv.lock` dependency closure.
- Separately versioned `agentic-orchestration`, `orchestration-core`, and `enterprise-llm` packages.
- Schema-backed deployment composition with installed-version and declared-entrypoint checks.
- FastAPI liveness and safe build metadata endpoints.
- A LangChain `BaseChatModel` enterprise gateway with Claude/OpenAI request shapes, native structured
  tool calling, normalized messages/responses/usage/errors, and sanitization.
- A separately versioned routing entry agent with typed single-child delegation and a routable
  marketing-science specialist using the standard LangGraph model/tool loop.
- A registered internal `calculate_sum` tool and non-streaming chat/history APIs.
- MongoDB-backed sessions with immutable successful turns, complete internal message representation,
  per-turn routing affinity, per-process serialization, revision guards, and transactional
  commit-on-success semantics.
- Versioned compiled graph definitions and ordered attempted-turn tracing across graph, node, model,
  tool, and child-agent boundaries, with safe bounded content capture. The MongoDB observability
  adapter is exercised by opt-in scripted and live-gateway service-level execution tests.
- An internal React/TypeScript diagnostic UI with a multi-turn Lab, paged conversation and run
  browsing, active-agent graph catalog, executed/compiled graph views, and span inspection.
- Non-runnable agent and tool templates, documentation, and a reproducible shared-worker container.

Not implemented: generated conversation summaries, conversation turn strata, the business data-to-insight agent,
LangGraph checkpoint/resume behavior,
authentication or enforced tenant/user ownership, large session artifacts,
distributed work queues, RPC/remote agents, dynamic installation, response streaming, or generalized
platform features.

## Quick start

Install [uv](https://docs.astral.sh/uv/), then from the repository root:

```bash
uv python install 3.12
uv sync --frozen --all-packages --dev
uv run python scripts/validate_deployment.py
cd frontend
npm ci
npm run build
cd ..
uv run uvicorn agentic_orchestration.main:app --host 127.0.0.1 --port 8000
```

The diagnostic frontend requires Node.js 24.15 or newer on the Node 24 LTS line (or Node 26+). Its
committed lockfile makes `npm ci` the reproducible install command.
After the build, browse to `http://127.0.0.1:8000/diagnostics`. The API still starts when no frontend
build exists and returns build instructions at that route.

Copy `.env.example` to the gitignored `.env` and replace its placeholders first. The application
loads `.env` automatically; real environment variables take precedence. Runtime startup requires
`MONGODB_HOST`, `MONGODB_PORT`, `MONGODB_USER`, `MONGODB_PWD`, and `MONGODB_DB`. MongoDB must support
transactions. Startup verifies access and idempotently creates the `sessions`, `session_turns`,
`execution_traces`, `execution_spans`, and `graph_definitions` collections plus
their query and uniqueness indexes. Session initialization is critical; observability initialization
reports a degraded condition without widening the chat service's availability boundary.

Verify in another shell:

```bash
curl --fail http://localhost:8000/health
curl --fail http://localhost:8000/build
curl -sS -H 'Content-Type: application/json' \
  -d '{"message":"Reply with a short greeting."}' http://localhost:8000/api/chat
```

`POST /api/chat` without a session ID allocates a server-generated ID before execution. A successful
first turn creates the durable session; a traced failure returns the allocated session ID and trace ID for
diagnostics without manufacturing a session. Reuse a committed session ID for later turns and read
`GET /api/sessions/{session_id}/history`. Supplying an unknown
ID does not create a session and returns `404`. History is durable across worker restarts and replicas;
the external history response projects only each turn's user and final-assistant messages, while the
durable turn retains tool decisions, tool results, call IDs, content blocks, and model metadata.

Conversation persistence is not LangGraph checkpointing. Failed graph invocations do not create turns,
and a database revision conflict returns a retryable `409` without reinvoking the graph.

Run every local quality gate:

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run pytest -m unit
uv run pytest -m integration
uv run pytest
uv run python scripts/validate_deployment.py
uv run python scripts/generate_dependency_projections.py --check
docker compose config --quiet
```

These gates build no Docker image. Local and CI verification are image-free while the ds6 build
suspension is in force. An image is produced only by an explicitly requested release build under the
[Docker build and retention policy](docs/docker-retention-policy.md). Do not run `docker build` to
satisfy a review.

Live gateway smoke tests are opt-in and described in [docs/enterprise-llm.md](docs/enterprise-llm.md).
Deterministic graph/API tests use a scripted model and injected memory session store under `tests`; no
fake gateway or process-memory fallback is installed in the worker runtime.

The entry-agent routing contract and its durability requirements are described in
[docs/routing-agent.md](docs/routing-agent.md).

The distribution roots, `src` directories, import-package namespaces, agent modules, and shared tools
layout are explained in [docs/repository-layout.md](docs/repository-layout.md).

The approved UI-driven design for durable execution snapshots, compiled graph overlays, node inspection,
turn strata, and best-effort live observation is documented in
[docs/execution-observability.md](docs/execution-observability.md).

The implementation-ready design for the repository's internal React/TypeScript diagnostic UI,
cursor-paged sessions, trace hydration, graph catalog, and FastAPI static serving is documented in
[docs/diagnostic-ui.md](docs/diagnostic-ui.md).
