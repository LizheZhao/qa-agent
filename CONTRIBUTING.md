# Contributing

Use Python 3.12 and `uv`; do not use a second environment or lockfile. Setup:

```bash
uv python install 3.12
uv sync --frozen --all-packages --dev
```

Before submitting changes, run exactly:

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

None of these gates builds a Docker image, and you must not add one. Builds on the ds6 host are
suspended and no approved external builder exists, so an image is produced only by an explicitly
requested release build. Say in your pull request that the image-content gate has not run.

Tests marked `live` are opt-in and must skip without secrets. Collection and normal tests must never
contact external services. Never commit `.env`, credentials, tokens, authentication headers, prompts,
or gateway responses. Avoid logging content; use the sanitizer for diagnostics. Do not introduce
import-time network access.

Runtime session persistence uses MongoDB, but deterministic tests must inject `MemorySessionStore`.
Changes to durable session documents belong in `agentic_orchestration.sessions.schema`; update the
message round-trip, revision-conflict, API, and persistence tests with any contract change. Live MongoDB
checks must use unique record IDs and remove their validation records.

Keep real agents as separately versioned packages, tools visibly under `packages/tools`, and API routes
separate from execution. Update the deployment manifest only for architectural packages and update the
lockfile with `uv lock` when dependencies change. No dynamic installation occurs at runtime.

The pull-request workflow enforces every command above. Live gateway checks are isolated in a manual
workflow protected by the `enterprise-gateway-stage` environment and its secrets.
