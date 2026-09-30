# Adding an agent

1. Copy `packages/agents/_template` to a directory named for the agent.
2. Rename `agent_template`, replace all TODOs, and rename `pyproject.toml.template` to `pyproject.toml`.
3. Define narrow typed input/result contracts, explicit dependencies, state, prompts, nodes, and tests.
4. Implement `create_graph(dependencies: AgentDependencies) -> StateGraph`; return it uncompiled.
5. Add the package to `tool.uv.workspace.members`, run `uv lock`, and add exact package/version/
   graph and manifest entrypoints plus routing eligibility under `agents` in `deployments/local.yaml` or
   another tracked manifest. The exported manifest declares the exact agent ID, version, routing-safe
   capabilities, and agent-owned tool IDs.
6. Set `entry_agent` only when the deployment has a primary agent. Run all contribution gates.

The worker imports only deployment declarations and compiles each graph once at startup. It resolves
and injects only the tools declared by each agent package. Agents may use core contracts and registered
tools, never app internals or direct database access. Do not use the template itself as installed
behavior or invent placeholder business behavior. Durable conversation history is loaded and committed
by the worker around entry-agent invocation, not by agent graph code. Keep agent-owned prompt constants
and builders in `prompts.py`; keep graph construction and transitions in `graph.py`.
