# Routing entry agent

Status: implemented contract for the `router_dev` milestone. Future behavior is identified explicitly.

## Purpose and ownership

The routing agent is the deployment entry agent and the sole owner of every chat session. It makes one
router-model call for each user turn and chooses exactly one of these outcomes; a delegated child may
then make its own model calls:

- answer ordinary conversation itself;
- reject or redirect an unsupported request;
- delegate to one eligible agent; or
- ask a routing clarification when it cannot safely choose an agent.

Routing occurs per turn. A previous route is a preference for contextual follow-ups, not a permanent
session assignment. The router remains the durable `agent_id` even when a child agent handles a turn.
Fan-out, parallel delegation, multi-agent planning, and synthesis are outside this milestone.

Users cannot select an agent through the normal chat request. A future authenticated diagnostic UI may
configure a validated routing override through a separate session-management contract. An override
constrains the router; it does not change session ownership or bypass agent eligibility checks.

## Initial behavior and target agents

The router always makes its own model call because it owns both general conversation and traffic
selection. Until another eligible agent exists, it may answer directly or delegate only to the sole
eligible target; it must not invent undeployed destinations.

The intended early routing boundaries are:

- The router handles greetings, ordinary conversation, capability questions, unsupported requests,
  and routing clarifications.
- The `marketing-science-agent` handles substantive explanatory questions about marketing science.
- A future data-to-insight agent handles business diagnostic requests that may require several
  clarification turns before it can form a viable data request, issue an authorized query through a
  tool, and translate the result into insights.

The router decides only whether a request belongs with the data-to-insight agent. SQL scoping and its
clarification conversation belong to that agent. A short reply to one of its questions, such as
`last quarter` or `US only`, should normally remain affiliated with that agent.

## Runtime boundary

The router is an independently versioned agent package. It may depend on `orchestration-core`, but it
must not import application internals, the runtime registry, or another agent package. The application
injects an `AgentInvoker` and an immutable eligible-agent catalog when it constructs the router graph.

Conceptually, the invoker contract is:

```python
class AgentInvoker(Protocol):
    async def invoke(
        self,
        agent_id: str,
        request: ConversationRequest,
        context: InvocationContext,
    ) -> ConversationResult: ...
```

The concrete contract may use more specific names, but it must provide typed request and result
boundaries. Passing arbitrary graph state dictionaries across the public agent boundary is not the
long-term contract.

The application-owned implementation validates the target against the catalog and enforces timeout,
cancellation, maximum depth, no self-invocation, and normalized errors. It also creates parent/child
trace identity for execution observability. Child graphs remain startup-compiled;
delegation never imports, installs, or compiles an agent during a request.

Startup therefore assembles the immutable registry in two phases:

1. Load and validate all deployment declarations and package metadata.
2. Compile non-entry agents with their own dependencies and tools.
3. Construct a restricted invoker over those compiled graphs.
4. Compile the router with that invoker and the eligible-agent catalog.
5. Freeze the complete registry before accepting traffic.

## Catalog and tool eligibility

Agent packages export stable identity, version, description, and routing capability metadata. The
deployment manifest selects which exact packages are present and eligible in a deployment. Startup
validates that package metadata and the deployment declaration agree, then supplies the router only
the filtered immutable catalog.

Tool eligibility belongs to each agent package, not to the router. An agent declares the tool IDs it
requires or permits; the worker resolves those IDs from the registered tool set and injects only that
agent's tools. Startup fails on missing or duplicate declarations. The router sees agent capabilities
suitable for traffic selection, not another agent's tool inventory.

Neither the router model nor user content can expand the declared set. The router itself receives no
business tools unless its own package explicitly declares a need for them. Delegation is an injected
runtime capability, not blanket access to every registered graph. A future deployment policy may
further restrict package-declared tools if an operational use case requires it.

## Router graph contract

The model produces a constrained routing outcome. A representative internal contract is:

```python
class RouteDecision(BaseModel):
    outcome: Literal["answer", "delegate", "clarify", "reject"]
    target_agent: str | None = None
    reason_code: RouteReason
    response: str | None = None
```

The router graph treats this decision as control flow. An `answer`, `clarify`, or `reject` outcome ends
with the router's response; a `delegate` outcome follows a conditional graph edge to a node that calls
the injected `AgentInvoker`. A handoff is not a registered tool, does not use a business `ToolNode`, and
does not appear in an agent's tool inventory.

The gateway's forced native function schema is a private wire-format mechanism for obtaining
`RouteDecision`. The graph validates its arguments directly with Pydantic and unwraps it before
execution; it does not turn delegation into tool semantics or persist synthetic handoff messages.

The runtime validates that a delegated target is eligible and that fields match the selected outcome.
Free-form hidden reasoning is neither requested nor stored. A reason code is operational metadata, not
an authorization decision.

If the router answers, clarifies, or rejects, that text becomes the final assistant message. If it
delegates, the selected agent receives the conversational request and its final answer becomes the
turn's final assistant message. Internal routing-model output must not appear as a separate user-visible
assistant turn or pollute later conversational context.

The router may use the full conversation to recognize follow-ups. Child-agent context is explicit in
the invocation request: initially a conversational target may receive the rehydrated history, while
future agents may receive a narrower context projection. A child must not gain access to session storage
or unrelated domain data through delegation.

## Durable route affinity

Session metadata records the router as `agent_id`. Each successful turn also records a nullable
selected-child identifier and stable routing outcome. This is durable conversational control state,
distinct from diagnostic execution snapshots and live events.

The router uses the most recent selected child as a preference when interpreting contextual follow-ups.
It may change the selection when the user changes topics. A clarification emitted by a child preserves
that affinity so the user's short answer returns to the same child unless there is strong contrary
evidence.

Route metadata is committed atomically with the successful turn. A failed router or child invocation
does not create a turn or advance affinity. Route metadata must use controlled schema fields rather than
provider response metadata or hidden synthetic chat messages.

## Failure behavior

The invoker distinguishes unknown target, forbidden target, timeout, execution failure, invalid result,
and cancellation. The router or API maps those conditions to stable sanitized behavior. It must not
silently try a different agent after a child failure because the first child may already have performed
a side effect.

The existing commit-on-success rule remains: the router call, optional child call, and final response
must all succeed before the turn and its route metadata are committed. Database revision conflicts are
returned without reinvoking either model or tools.

## Data-to-insight boundary

The planned data-to-insight agent owns its clarification loop and the conversion from business intent
to an authorized data request. It does not connect directly to a database. Query validation and
execution follow the existing domain boundary through registered tools and service clients.

Any eventual query tool should enforce authorization independently of model output, use read-only
credentials, constrain accessible schemas and operations, apply row/time/resource limits, and return a
controlled result or artifact reference. These controls are tool/service responsibilities, not routing
instructions.

## Minimum verification for this milestone

Tests should establish that:

- startup compiles children before the router and freezes the final registry;
- the durable session owner is always the router;
- the router can answer directly and invoke at most one eligible child;
- self, unknown, forbidden, and over-depth child invocations are rejected;
- the router graph exposes only one delegation path per turn;
- each agent receives only its declared tools;
- internal route decisions are not exposed as conversational messages;
- successful turns atomically retain route affinity;
- failed invocations do not advance session history or affinity; and
- a terse response following a child clarification remains routed to that child.

A broad routing-quality evaluation corpus is deferred until there are enough real destinations to make
classification quality measurable.
