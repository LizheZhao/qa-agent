# Execution observability and diagnostic UI

Status: implemented through historical diagnostic hydration, compiled/runtime graph canvases, paged
conversation and run browsing, multi-turn Lab chat, and span inspection. MongoDB persistence and
hydration have been exercised with retained scripted and live-gateway router-to-specialist turns.
Conversation turn-strata composition, live delivery, and production authorization remain deferred.
The concrete internal-UI and read-API implementation plan is maintained in
[`diagnostic-ui.md`](diagnostic-ui.md).

## Product objective

The diagnostic UI reconstructs how a conversation executed, not merely what messages it produced. A
conversation is displayed as ordered turn strata. Each turn overlays its executed runtime path on the
compiled agent graphs that were available for the recorded deployment versions.

The UI must support both live observation and historical hydration from MongoDB. It must represent:

- entry-agent and child-agent boundaries;
- graph `START` and `END` nodes;
- conditional routing and untaken destinations;
- repeated node executions and loop iterations;
- parallel fan-out in the same logical execution layer;
- nested model and tool calls;
- node inputs, prompt templates, resolved model messages, outputs, and state deltas;
- successful, failed, cancelled, and skipped work; and
- timing, usage, version, and sanitized error metadata.

Authentication, tenant ownership, and diagnostic authorization are prerequisites for production
deployment. Trace identifiers and session identifiers are never treated as authorization credentials.

## Execution is a hierarchical DAG

The stored execution model is a hierarchical directed acyclic graph of execution instances, even when
the UI renders it with a tree-like layout. A compiled graph may contain cycles, but each runtime loop
creates new node instances with increasing iteration numbers. Parallel children share a logical layer
while retaining independent timing.

```text
Conversation
├─ Turn 1
│  ├─ START
│  ├─ router.decide
│  ├─ router.delegate
│  ├─ marketing_science.model
│  └─ END
├─ Turn 2
│  ├─ START
│  ├─ router.decide
│  ├─ router.delegate
│  ├─ data_insight.clarify
│  └─ END
└─ Turn 3
   ├─ START
   ├─ router.decide
   ├─ router.delegate
   ├─ data_insight.plan                         iteration 1
   ├─ parallel superstep
   │  ├─ schema_discovery
   │  └─ metric_definition
   ├─ data_insight.plan                         iteration 2
   ├─ execute_query
   ├─ synthesize_insight
   └─ END
```

The application currently invokes a new graph execution for each user turn. An agent is not suspended
while waiting for the next user response. Conversation-level continuity is composed from turn traces
and the durable route affinity stored with successful turns.

## Compiled graph definitions

A graph definition describes possible topology independently from any execution. Definitions are
identified by agent version and a content hash covering the entrypoint, topology, and prompt
definitions. Deployment versions remain on execution traces.

A definition contains:

- agent ID, agent version, and graph entrypoint;
- stable definition-node IDs and display names;
- synthetic `START` and `END` nodes;
- node kinds such as deterministic, router, model, tool, child-agent, and subgraph;
- possible directed and conditional edges;
- conditional edge labels when statically known;
- subgraph ownership and hierarchy; and
- prompt template identities for model-calling nodes when statically known.

The conversation view does not eagerly inline every graph eligible at a router. It initially loads the
executed path and represents unselected agents as compact, collapsed destinations. The UI fetches an
unvisited graph definition only when the user expands it and caches definitions by version.

The default visual states are `unvisited`, `running`, `completed`, `failed`, `cancelled`, `skipped`, and
`selected`. Untaken compiled edges are dimmed rather than represented as runtime spans.

## Runtime trace hierarchy

One attempted user turn creates one trace, including attempts that fail before a durable session turn
is committed. The trace contains nested spans:

```text
turn
└─ graph
   ├─ node
   │  ├─ model call
   │  └─ tool call
   └─ child-agent invocation
      └─ graph
         └─ node
```

Every span has an opaque ID. Human-readable graph paths are derived labels and are not used as primary
identity. A minimum span contract contains:

```text
trace_id
span_id
parent_span_id
caused_by_span_ids
session_id
turn_number or attempted_turn_number
agent_id
agent_version
graph_definition_id
definition_node_id
span_kind
status
sequence_started
sequence_completed
superstep
iteration
started_at
completed_at
duration_ms
attributes
error
failure_detail
```

`sequence_started` and `sequence_completed` are monotonic within a trace and establish observed event
order without relying only on wall-clock timestamps. `superstep` captures the scheduler's logical
layer: LangGraph nodes scheduled in the same superstep render in the same layer. Async work hidden
inside one node remains nested inside that node unless explicitly instrumented as a child span.

Parent relationships describe containment. Causal relationships describe why work became runnable.
Both are required because parallel siblings share a parent but may have different causal predecessors.
The runtime resolves causal predecessors from compiled graph edges plus the latest actually executed
source-node instances. LangGraph `branch:to:<target>` trigger channels remain attributes but are not
interpreted as source node IDs. If compiled topology is unavailable, causality falls back to the prior
executed superstep and observability is marked degraded.

## Node inspection and content

Selecting a runtime node opens an inspector containing, where applicable:

- agent, graph, node, turn, superstep, iteration, status, and timing;
- typed input projection;
- prompt template identity and template text;
- fully resolved model messages in provider-independent form;
- model identity and safe generation configuration;
- tool schemas presented to the model;
- structured model decisions and tool calls;
- node output and state delta;
- token and usage metadata;
- sanitized public error information;
- bounded internal failure detail for failed spans, including a redacted exception chain, traceback
  frame locations, and a stable fingerprint; and
- parent, child, causal predecessor, and causal successor links.

The current v4 capture records node inputs and state deltas, resolved model messages and outputs,
prompt-template references, tool inputs and outputs, model/provider observations, provider-shaped usage
metadata, sanitized public errors, and bounded failure details. Diagnostic APIs normalize supported
observations into typed optional
response fields; the UI does not consume the storage model's open attribute dictionary directly. Model
generation configuration and the exact tool schemas presented to each model call remain future capture
fields; the UI must render absent fields honestly.

The trace should reference existing durable conversation message IDs rather than duplicate those
messages when possible. It must still store internal routing decisions, resolved prompts, intermediate
outputs, and state deltas that are intentionally absent from conversation history.

Large values such as query results and DataFrames are stored as authorized artifacts rather than
embedded in span documents. The span retains controlled metadata such as artifact ID, schema, row
count, byte size, and preview availability. Artifact authorization is independent from trace access.

Credentials, tokens, authorization headers, and raw provider envelopes remain unconditionally
excluded. Authentication does not make secret capture acceptable.

## Best-effort durable MongoDB storage

Traces are diagnostic records without a TTL. MongoDB is the historical hydration source for both
successful and failed execution attempts. Conversation storage is critical application state;
execution storage is best-effort durable state and must not expand the chat service's availability
boundary. They remain separate because they have different commit, failure, and query semantics.

The intended collections are:

### `execution_traces`

The approved v4 contract stores one root document per attempted turn containing trace identity, the
immutable session ID allocated before execution, attempted turn number, root span, execution and
observability statuses, lifecycle sequences, timestamps, deployment versions, and terminal error. The
contract also records a typed, descriptive request origin so diagnostic-UI traffic can be distinguished
from other API traffic without treating that label as authorization. The diagnostic read path promotes
readable v2 records by using whichever legacy session identity is set and an `unknown` origin, and
promotes readable v3 records without manufacturing failure detail that was never captured.

### `execution_spans`

One document per graph, node, model, tool, or child-agent execution instance. Documents are keyed by
trace and span identity and contain ordering, hierarchy, state projections, prompt/model/tool details,
usage, timing, status, and a snapshot sequence. Failed terminal snapshots can additionally contain a
bounded exception chain, frame locations without locals or source text, a stable fingerprint, and flags
that disclose redaction or truncation. Start and terminal snapshots are idempotent monotonic
upserts: an older or retried snapshot cannot replace newer stored state.

### `graph_definitions`

Content-addressed compiled graph metadata identified by agent version and a definition hash covering
topology and prompt definitions. Definitions are stored once and reused by many traces. Deployment,
application, and core versions belong to trace records rather than immutable graph identity.

### Future artifact storage

Large inputs and outputs require a separate artifact contract and possibly a store other than MongoDB.
Trace documents contain only controlled artifact references and metadata.

A successful `session_turns` document records its `trace_id`. Execution traces are written while work
is running and therefore are not part of the short conversation-turn commit transaction. A failed
first attempt can have an execution trace even though commit-on-success correctly leaves no session.
The same `session_id` remains on a trace and its spans whether or not commit succeeds;
`committed_turn_number` records whether durable conversation history advanced. This immutable identity
keeps failed-attempt lookup and indexing independent from lifecycle rewrites.
Callbacks allocate ordering and enqueue snapshots without awaiting MongoDB. A bounded per-trace writer
drains ordered batches continuously and attempts a bounded final flush. Trace persistence failure is
reported through structured operational telemetry and an explicit degraded status where that status
can still be persisted. It does not change an otherwise successful committed conversation into a
failed response.

Cancellation before a successful conversation commit terminalizes every open span and the trace as
`cancelled`, performs a bounded writer flush, and releases the active run. Commit itself is shielded so
the API can determine whether durable history advanced. If cancellation arrives after the commit, the
trace is completed and linked to that committed turn even though the response is no longer delivered.

## Planned live delivery and reconnect

The recorder already produces ephemeral lifecycle envelopes through a synchronous publisher seam.
Production currently installs the null publisher because authenticated diagnostic APIs, an SSE broker,
and subscriber buffers have not been implemented. Every future live envelope contains `trace_id` and
a monotonic sequence number. Publisher implementations must use bounded, nonblocking subscriber
buffers so network I/O never runs on the graph callback path.

Once the diagnostic transport exists, initial load or reconnection will:

1. reads the current trace and span snapshots from MongoDB;
2. reconstructs lifecycle state from their start and completion sequences;
3. opens an authenticated server-sent-events stream; and
4. applies new events idempotently while accepting that events during a disconnect may be absent.

Live delivery is best effort. Reconnection and worker changes recover the latest durable snapshot, not
an exact event log. If exact cross-worker replay becomes a demonstrated requirement, use a
purpose-built transient stream rather than duplicating lifecycle state in MongoDB. The first UI
transport is SSE; bidirectional control is not required for observation.

## Diagnostic APIs

The approved initial read surface is:

```text
GET /api/diagnostics/sessions
GET /api/diagnostics/sessions/{session_id}
GET /api/diagnostics/sessions/{session_id}/turns
GET /api/diagnostics/sessions/{session_id}/traces
GET /api/diagnostics/traces
GET /api/diagnostics/traces/{trace_id}
GET /api/diagnostics/traces/{trace_id}/spans
GET /api/diagnostics/traces/{trace_id}/spans/{span_id}
GET /api/diagnostics/graph-definitions/{definition_id}
GET /api/diagnostics/deployment
```

Session, turn, and attempt collections are cursor-paged. Slim trace-span summaries hydrate as one
initial canvas unit, while captured span detail loads on selection. Detailed
response, cursor, query, and error contracts are defined in [`diagnostic-ui.md`](diagnostic-ui.md).
List and read operations must enforce authentication, tenant/session ownership, diagnostic role
permissions, and artifact-specific authorization before production deployment. Access to diagnostic
content is auditable. UI visibility is never considered an authorization control.

### Failure-detail security boundary

Failure detail is privileged operational data, not a public chat-error contract. The span-detail API is
the only diagnostic projection that returns it; trace and span-list responses remain slim and expose
only sanitized terminal errors. Exception messages are bounded and passed through credential redaction.
Traceback capture stores filenames, function names, and line numbers, but never locals, arguments,
source lines, environment values, authentication headers, or raw provider request/response bodies.
External-provider exception messages are omitted even when diagnostic access is authenticated.

The current development deployment relies on environmental access restriction. Before this contract is
enabled in any production-accessible deployment, the orchestration endpoint must enforce authenticated
identity, tenant/session ownership, an explicit diagnostic-read permission, and access auditing. MongoDB
credentials alone do not protect data returned by HTTP. Add authorization and cross-tenant tests before
removing the development-only restriction.

## UI behavior

The conversation page uses turn strata. Within a turn, horizontal or vertical layers represent
supersteps and nested boundaries can be collapsed:

```text
Turn 3

[START] -> [router.decide] -> [data_insight]
                                  |
                                  +-- expanded child graph ----------------------+
                                      [plan #1] -> [schema] ----+                |
                                                    [metric] ---+-> [plan #2] -> END
```

Recommended interactions:

- switch between executed-path and compiled-graph overlays;
- collapse or expand agent, subgraph, node, model, and tool boundaries;
- lazily load unvisited agent definitions;
- select a node to open its inspector;
- scrub or jump between turns;
- follow parent and causal links;
- distinguish running work from persisted terminal state; and
- display parallel nodes in the same superstep layer with their actual duration overlap.

The internal application boundary, React/TypeScript/Vite layout, `/diagnostics` URL, session paging,
trace projection, subgraph accommodation, and code slices are specified in
[`diagnostic-ui.md`](diagnostic-ui.md). This document remains the source of truth for execution and
storage semantics.

## Production authorization prerequisite

Authentication and authorization will be implemented before the service receives a production
deployment. Production diagnostic access requires:

- authenticated caller identity;
- enforced tenant and session ownership;
- an explicit diagnostic-read permission;
- separate authorization for large artifacts and business data;
- access auditing;
- non-enumerable opaque identifiers in addition to authorization; and
- tests proving cross-tenant trace, span, graph, and artifact isolation.

Development may use injected identities or test adapters, but the production API must not rely on a
development-only route toggle as its security boundary.

## Implementation sequence

1. **Implemented:** define graph-definition, trace, span, content, status, and event schemas in
   `orchestration-core` without storage dependencies.
2. **Implemented:** create the root trace at the API boundary and propagate its context through the
   executor, router, child invoker, child graphs, model calls, and tool calls.
3. **Implemented:** provide an in-memory trace store and ordered recorder for deterministic tests.
4. **Implemented:** instrument graph/node boundaries and capture LangGraph supersteps and iterations.
5. **Implemented:** record nested model, tool, child-agent, and child-graph spans.
6. **Implemented and service-tested:** add buffered, monotonic MongoDB trace/span snapshot persistence,
   graph-definition storage, and UI-oriented indexes. Retained scripted and live-gateway executions
   verify router-to-specialist hydration without durable lifecycle-event documents.
7. **Implemented and service-tested:** successful session turns contain their trace IDs, while failed
   attempts reach terminal traces independently and remain hydratable from MongoDB.
8. **Implemented and tested:** topology-aware causal links distinguish independent parallel branches;
   cancellation terminalizes traces and writers without leaking active runs; missing diagnostic graph
   metadata degrades capture without failing agent execution.
9. **Implemented:** add v3 immutable session identity, typed chat failure envelopes, paged diagnostic
   reads, slim span hydration, and on-selection detail APIs specified in `diagnostic-ui.md`.
10. **Implemented:** add the internal frontend shell, multi-turn Lab, session/run workflows, structured
    trace list, inspector, graph canvas, and active compiled catalog. Turn-strata composition remains a
    later interaction layer. Add best-effort SSE only after historical hydration is stable.
11. **Implemented:** add v4 bounded, redacted failure detail to failed spans and expose it only through
    on-demand span detail.
12. Add production authorization and cross-tenant isolation gates before deployment.
