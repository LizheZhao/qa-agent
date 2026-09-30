# Engineering contract

This file applies repository-wide. A nested `AGENTS.md` adds rules for its subtree.

## Workspace and package boundaries

- Use Python 3.12 and the root `uv` workspace. Keep the single dependency closure in `uv.lock`; do not
  add per-package virtual environments or lockfiles.
- Each installable Python distribution has its own `pyproject.toml` and `src/<unique_import_package>/`.
  Distribution names use hyphens and import packages use underscores.
- Agents remain separately versioned distributions under `packages/agents/`. An agent package owns its
  manifest, state, prompts, and graph. Define prompt text/builders in `prompts.py`; `graph.py` may import
  and attach those definitions to nodes.
- `orchestration-core` owns framework-facing contracts, `enterprise-llm` owns enterprise model
  transport, and tool families live under `packages/tools/src/orchestration_tools/<family>/`.
- Preserve dependency direction: application -> core/gateway; agents -> core. Core must not import
  agents, gateway, or application internals; tools must not import agents; agents must not import
  application internals; gateway code must not own prompts or routing. Tests may import any component.
- Deployment manifests, not package/directory scanning, select installed agents and entrypoints. Do not
  add runtime package installation or discovery by filesystem scan.

## Cross-cutting runtime invariants

- Conversation persistence belongs to `agentic_orchestration.sessions`. Agents receive rehydrated
  messages and do not connect to MongoDB. Domain data access goes through registered tools and service
  clients rather than session storage.
- Changes to durable session documents belong in `sessions/schema.py`. Preserve compatibility with
  stored documents or provide an explicit migration, and test the compatibility path that changed.
- MongoDB is the runtime durability boundary. Memory stores are injected test adapters, not a runtime
  fallback.
- Preserve the validation policy of the contract being changed. Controlled API and persistence models
  currently reject unknown fields; do not silently relax them as a side effect of unrelated work.
- Do not log credentials, authentication headers, raw provider payloads, or uncaptured model content.
  Diagnostic content may be exposed only through the existing bounded, secret-redacting capture and
  response contracts. Keep module imports free of network access.

## Tests, versions, and generated files

- Mark isolated deterministic tests `unit` and in-process component tests `integration`. `live` and
  `mongo` tests are opt-in and must skip when their environment flags or credentials are absent.
- Normal test collection and `uv run pytest` must not contact external services. Use
  `ScriptedChatModel`, memory adapters, and dependency injection; do not add test-detection branches to
  production code.
- Add tests for the observable guarantee changed, especially successful-turn immutability, revision
  conflicts, commit-on-success, ordering, and sanitized failures when the change touches those areas.
- Do not hand-edit `frontend/src/api/schema.d.ts`, generated build metadata, or `frontend/dist/`.
  Regenerate them with the repository commands.
- Dependency changes require `uv lock`. When an architectural package changes, keep its `pyproject.toml`
  version, exported manifest version (for agents), deployment declaration, and lockfile synchronized.
  Update documentation only when the documented architecture or public contract changes.

## Verification

Use focused tests while iterating. For Python source, Python package metadata, scripts, or tests, run
the repository's CI quality gates:

```bash
uv run ruff format --check .
uv run ruff check --no-cache .
uv run mypy
uv run pytest
uv run python scripts/validate_deployment.py
```

For `compose.yaml`, `Dockerfile`, dependency/packaging, deployment, or image-content changes, also run:

```bash
docker compose config --quiet
uv run python scripts/generate_dependency_projections.py --check
```

Do not run `docker build`, `docker compose build`, or `docker compose up` for any change. Builds on the
ds6 host are suspended, no approved external builder exists, and every retained checkpoint image
contributed to the incident that imposed the suspension. The image-content gate is deferred to an
explicitly requested release build; state in your report that it has not run.

Frontend changes use `frontend/AGENTS.md`. Documentation-only changes require `git diff --check`; run a
documented command if the edit changes that command. Credentialed `live`/`mongo` tests are required only
when the task targets those integrations. Report any skipped applicable gate and its reason.
