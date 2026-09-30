# Registered tools

`orchestration-tools` is one application-internal Python distribution installed into the shared
worker. It is not an independently selected architectural component in deployment manifests; its
exact version and dependencies are resolved in `uv.lock`. Individual tools do not become separate
distributions.

Tool implementations are visibly separate from agents under `src/orchestration_tools/<family>/`. The
`orchestration_tools` directory is the installed Python namespace; family directories organize related
operations within the single distribution. Each tool has a globally stable name, Pydantic input schema,
async implementation, LLM-facing description, normalized result/error behavior, and explicit optional
runtime dependencies. Tools call API service clients and never connect directly to MongoDB or another
domain database. The internal `arithmetic` family provides the side-effect-free `calculate_sum` tool
and is registered by the application.

Creating `packages/tools/<family>/pyproject.toml` would make every family an independently versioned
distribution, which is intentionally outside the current release-coupled tools model. See
[`docs/repository-layout.md`](../../docs/repository-layout.md) for the package-layer distinction.
