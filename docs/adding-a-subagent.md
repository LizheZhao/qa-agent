# Adding a subagent

A subagent is an independently contracted reasoning component. Create it from the agent template as a
separate Python distribution with a narrow typed request/result, explicit tool permissions, trace
identity, exact version, and declared graph and metadata entrypoints. Register it in the deployment
manifest and pin it in the shared image. The narrow in-process `AgentInvoker` mediates eligible calls;
do not import app internals or introduce RPC, queues, per-agent containers, or remote-agent abstractions
before the runtime contract requires them.
