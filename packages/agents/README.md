# Agent packages

Each real agent is an independently versioned Python distribution under this directory. Copy
`_template`, rename every placeholder, implement typed behavior, and register the exact package,
version, graph entrypoint, manifest entrypoint, and routing eligibility in the deployment manifest.
Agents declare their own routing-safe capabilities and tool IDs. They may depend on
`orchestration-core` and registered tools, never application internals. The template is deliberately
non-runnable and is not installed in a deployment.

`router/` owns deployed chat sessions and selects at most one eligible child per turn.
`marketing-science/` is the first routable specialist. Both receive explicit dependencies from the
worker, own their graph state, and do not read application configuration.

Each agent directory is a distribution root. Its source lives under a unique import package so all
agents can be installed into the same interpreter without colliding on generic module names such as
`graph` or `state`:

```text
router/
├── pyproject.toml
├── README.md
└── src/router_agent/
    ├── graph.py
    ├── manifest.py
    ├── prompts.py
    └── state.py
```

See [`docs/repository-layout.md`](../../docs/repository-layout.md) for the workspace-wide convention.
