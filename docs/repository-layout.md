# Repository and Python package layout

The repository is a `uv` workspace containing several independently versioned Python distributions.
Each installable project uses the standard `src` layout:

```text
<distribution root>/
├── pyproject.toml
├── README.md                 # when the distribution needs package-specific documentation
└── src/
    └── <unique import package>/
        ├── __init__.py
        └── ...
```

These layers have different jobs:

- The distribution root is the release and dependency boundary. Its `pyproject.toml` supplies the
  installable distribution name and version.
- `src/` prevents Python from accidentally importing repository files that were not included in the
  built distribution. Tests therefore exercise the installed package shape.
- The import-package directory supplies a globally unique Python namespace in the shared worker.
  Distribution names use hyphens by convention, while Python imports use underscores.

Agent files cannot live directly under each project's `src/` directory. Every installed agent would
otherwise export colliding top-level modules such as `graph`, `state`, `manifest`, and `prompts` into
the same interpreter. A shared namespace package such as `orchestration_agents.router` would still
require namespace and agent directories and would add another layer without changing isolation.

## Current workspace

```text
agentic-orchestration/
├── pyproject.toml
├── src/
│   ├── agentic_orchestration/       # FastAPI application and runtime assembly
│   └── integrations/                # external connection construction
└── packages/
    ├── orchestration-core/
    │   ├── pyproject.toml
    │   └── src/orchestration_core/  # framework-facing contracts
    ├── enterprise-llm/
    │   ├── pyproject.toml
    │   └── src/enterprise_llm/      # enterprise model adapter
    ├── agents/
    │   ├── router/
    │   │   ├── pyproject.toml
    │   │   └── src/router_agent/
    │   └── marketing-science/
    │       ├── pyproject.toml
    │       └── src/marketing_science_agent/
    └── tools/
        ├── pyproject.toml
        └── src/orchestration_tools/
            └── arithmetic/          # one tool family in the shared tools distribution
```

Agents are separate distributions because they have independent identities, versions, manifests, and
deployment entrypoints. `orchestration-core` and `enterprise-llm` are separate distributions because
their dependency direction and release contracts differ from the application and agents.

Tools currently use one application-coupled `orchestration-tools` distribution. Tool families live
under `src/orchestration_tools/<family>/`; individual tools do not receive distribution roots or
deployment entries. If a future tool family needs an independent release lifecycle, it can become a
separate distribution through an explicit architecture change.

## Agent package contents

Agent import packages use consistent responsibility-based modules:

```text
src/<agent_import_package>/
├── __init__.py
├── contracts.py       # agent-owned typed models when needed
├── graph.py           # graph construction and transitions
├── manifest.py        # static identity, capabilities, and tool IDs
├── prompts.py         # agent-owned prompt text and prompt builders
├── state.py           # private graph state
└── py.typed
```

Not every agent needs every optional module. Prompts stay outside `graph.py`; graph code imports them
from `prompts.py`. Private nodes, subgraphs, or subagents may gain subdirectories when their size and
ownership justify them.

Generated `__pycache__`, virtual-environment, test-cache, and build directories are ignored artifacts,
not part of the intended repository structure.

## Internal diagnostic frontend

The internal engineering UI lives at the repository's top-level `frontend/` directory as
a React, TypeScript, and Vite project. It is not a Python distribution and is distinct from the future
customer-facing product frontend, which will live in another repository.

```text
agentic-orchestration/
├── frontend/
│   ├── package.json
│   ├── package-lock.json
│   ├── vite.config.ts
│   ├── src/                    # maintained diagnostic UI source
│   ├── node_modules/           # generated; ignored
│   └── dist/                   # generated browser build; ignored
├── src/
├── packages/
└── pyproject.toml
```

FastAPI serves the compiled `frontend/dist/` output at `/diagnostics`. The source layout, build
contract, diagnostic APIs, and implementation slices are defined in
[`diagnostic-ui.md`](diagnostic-ui.md).
