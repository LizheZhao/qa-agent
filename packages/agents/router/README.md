# Router agent

`router-agent` is the deployment entry agent and durable owner of chat sessions. For each user turn it
makes one structured model decision: answer, clarify, reject, or delegate to one eligible child through
the injected `AgentInvoker`. Delegation is graph control flow, not a tool.

The router receives an immutable routing catalog and no business tools. Internal decisions are returned
as controlled graph state for application persistence and are not appended to conversation messages.
Its static and catalog-derived instructions are owned by `prompts.py`; `graph.py` owns only graph
construction, validation, and transitions.
