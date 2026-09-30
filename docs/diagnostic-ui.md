# Internal diagnostic UI

Status: active implementation. Observability v3, diagnostic read APIs, generated frontend contracts,
Lab, conversation/run browsing, active graph catalog, trace projection, attachment-aware inspector,
composer, and React Flow canvases are implemented. Turn-strata composition and later interaction
refinements remain.

## Purpose and boundary

This repository owns one internal engineering UI for inspecting and exercising the orchestration
service. It is not the customer-facing product frontend; that application will live in a separate
repository with its own product, release, and authorization concerns.

The diagnostic UI answers three engineering questions:

1. What conversations and execution attempts happened recently?
2. How did a selected turn move through router, agent, graph, model, and tool boundaries?
3. What topology could an agent graph execute, including paths that were not selected?

The UI includes a small chat composer because generating a new turn and inspecting its trace is a
single diagnostic workflow. It does not attempt to provide production chat UX, streaming responses,
attachments, feedback, conversation management, or customer-facing presentation.

The first canvas renders one selected turn because that is the smallest honest hydration and
projection unit. The intended conversation view remains an ordered stack of turn strata: later slices
compose already-loaded turn projections without changing the per-turn trace contract.

The naming is intentionally distinct:

- `frontend/` is the repository directory containing the diagnostic UI source;
- `/diagnostics` is the browser URL where FastAPI serves its compiled assets; and
- “diagnostic UI” is the audience and product boundary.

## Decisions

- The frontend uses React, TypeScript, and Vite.
- Source lives in the top-level `frontend/` directory.
- Vite emits browser assets into `frontend/dist/`; `dist/` and `node_modules/` are generated and
  ignored by Git.
- FastAPI serves the latest build at `/diagnostics` without requiring a second runtime process.
- `uvicorn agentic_orchestration.main:app` remains the single service command after a frontend build.
- Vite's development server is optional and may proxy `/api` to FastAPI for hot reload.
- The initial browser target is a current Chromium-based browser.
- The initial UI reads real service and MongoDB state. There is no user-selectable fixture mode;
  sanitized fixtures are still used by automated frontend tests.
- Conversation listings use cursor pagination with 10 sessions per page, newest first. Their primary
  label is a bounded preview of the first user message; opaque IDs remain secondary debugging data.
- Historical hydration is implemented before best-effort server-sent events.
- Graph rendering technology is selected when the canvas slice begins.
- Forced routing is deferred. Its eventual contract must be diagnostic-only, explicit, and auditable.
- A tab mounts on first visit and remains mounted while hidden. Switching tabs preserves loaded cursor
  pages, hydrated resources, and local selection state without refetching them.

## Information architecture

The application has four top-level tabs and one shared inspector:

- **Lab** is the only writable surface and supports new or continuing multi-turn diagnostic chat. It
  deliberately presents only conversation history and the composer; trace inspection stays in
  Conversations and Runs.
- **Agents** catalogs the active deployment's compiled graph definitions.
- **Conversations** browses committed durable history and a selected turn trace.
- **Runs** browses every router invocation, including failures with no committed conversation.

Switching tabs restores the tab's last deep link and retained in-memory state rather than carrying an
uncommitted run into the conversation view. Runs appear only in the Runs browser; the selected run's
middle panel is reserved for its trace and does not repeat the list.

### Session trace view

This view combines a paged session browser, turn-aware transcript, selected-turn execution canvas,
and node inspector. The middle workspace is split between transcript and execution: selecting an
assistant response loads that committed turn's `trace_id`, highlights the response, and updates both
the graph and structured execution list. Rapid selections are latest-request-wins so stale hydration
cannot replace the user's newer choice.

```text
┌ Conversations ───────┬ Conversation / execution ─────────────┬ Inspector ───────────┐
│ newest first         │ Turn 4                                │ node identity         │
│ 10 per page          │ user → assistant                      │ input and state       │
│                      │                                       │ prompt/messages       │
│ selected session     │ START → router.decide                 │ output/state delta    │
│                      │              ↓                        │ timing/usage/error    │
│ [load older]         │       marketing_science               │ parent/causal links   │
│                      │         ├ model                        │ versions              │
│ [new-turn composer]  │         └ tool                         │                       │
└──────────────────────┴───────────────────────────────────────┴───────────────────────┘
```

Selecting a session loads only its newest turn page. Selecting a turn loads its trace, spans, and
referenced graph definitions. Older turns and older session pages load on demand.

The Lab composer calls the existing `POST /api/chat` endpoint. A successful new turn selects its
returned session; using the composer is visibly marked as a live enterprise-gateway action that writes
durable session and observability records. Once trace creation has begun, a failed response also
returns its session and trace IDs in a controlled error envelope. Lab remains chat-only: its error
banner includes the short run ID for lookup in Runs without automatically changing tabs. Failures
before trace creation legitimately have no trace ID.

A recent-runs entry point complements the conversation list. A run is one router invocation, including
failed, cancelled, retried, or uncommitted execution; this view is required because a failed first run
has a trace and stable session ID but, under commit-on-success semantics, no durable session document.
The UI calls these “runs” because “attempt” is a persistence term, not a useful behavioral label.

After trace creation, chat failures retain their existing HTTP status and use this response shape:

```text
error:
  code: stable_machine_code
  message: sanitized_human_message
session_id: string
trace_id: string
```

Validation, unknown-session, and storage failures that occur before trace creation use the ordinary API
error contract without invented IDs.

### Graph catalog

The catalog begins with the graph definitions active in the current deployment. It presents agents,
versions, graph topology, prompt-owning nodes, and possible edges.

The Agents tab loads definitions by the immutable IDs reported by the active deployment and renders
their compiled topology. Selecting a node opens its static prompt and routing metadata in the shared
inspector; definition URLs are directly refreshable.

Historical definitions are content-addressed and loaded when a trace references them. Listing and
comparing every historical definition, plus backlinks to executions by definition ID, are later
enhancements rather than requirements of the first catalog.

### Inspector

The shared right-side inspector renders the selected graph node or runtime span. It must tolerate
absent capture fields and show why content is omitted rather than presenting an empty value as data.

For a runtime span it can show:

- span, agent, graph, definition-node, turn, superstep, and iteration identity;
- status, observed sequence order, timestamps, and duration;
- captured input, output, and state delta;
- prompt-template identity and resolved model messages;
- typed model/provider identity and provider-independent usage fields when present;
- tool identity, arguments, and result when present;
- sanitized error information;
- parent, children, causal predecessors, and causal successors; and
- deployment, application, core, and agent versions inherited from the trace.

For a compiled but unvisited node it shows only static graph metadata, including the prompt template
when available. The UI must never invent runtime input, output, timing, or status for unvisited nodes.

## URL design

The URL is durable UI state so engineers can share a selected trace or graph:

```text
/diagnostics
/diagnostics/lab
/diagnostics/runs?trace={trace_id}&span={span_id}
/diagnostics/sessions/{session_id}
/diagnostics/sessions/{session_id}?turn={turn_number}&trace={trace_id}&span={span_id}
/diagnostics/graphs
/diagnostics/graphs/{definition_id}
```

The first release may implement fewer deep-link parameters, but IDs must not live only in transient
component state. A refresh of a supported deep link restores the same selection.

FastAPI's SPA fallback applies only beneath `/diagnostics`. It must never intercept `/api`, `/health`,
`/build`, `/openapi.json`, or documentation routes.

## Frontend build and serving

Browsers do not execute the TypeScript and React source directly. Vite compiles it into `index.html`,
browser JavaScript, CSS, and versioned assets under `frontend/dist/`:

```text
frontend/
├── package.json
├── package-lock.json
├── tsconfig.json
├── vite.config.ts
├── index.html
├── src/
│   └── ...
└── dist/                         # generated; ignored by Git
    ├── index.html
    └── assets/
```

Vite is configured with `/diagnostics/` as its base path. FastAPI serves the asset directory and returns
the compiled `index.html` for supported client-side paths under `/diagnostics`.

The ordinary built workflow is:

```bash
cd frontend
npm ci
npm run build
cd ..
uv run uvicorn agentic_orchestration.main:app --host 127.0.0.1 --port 8000
```

No Node process remains after `npm run build`. Uvicorn serves the generated files alongside the API.
The generated directory need not be committed or published as a separate package.

If `frontend/dist/index.html` is absent, API startup still succeeds. Requests beneath `/diagnostics`
return a clear diagnostic response explaining that `npm run build` has not been run.

For active frontend work, `npm run dev` runs Vite with hot reload and a same-origin-style proxy to the
FastAPI port. That development workflow uses two processes; the one-Uvicorn workflow always serves the
most recent completed build.

If this internal service is later copied to another host or placed in a container, the deployment build
must run the frontend build and copy `frontend/dist/` into the runtime. This does not require a separate
frontend package or independent image.

## Frontend source layout

The implementation separates API contracts, domain projections, and presentation:

```text
frontend/src/
├── app/                           # shell, routes, providers, error boundaries
├── api/                           # typed HTTP client and generated OpenAPI types
├── features/
│   ├── sessions/                  # session pages and turn transcript
│   ├── traces/                    # hydration and runtime projection
│   ├── agents/                    # active compiled graph catalog
│   ├── inspector/                 # node and span detail
│   ├── runs/                      # recent execution workflow
│   └── chat/                      # Lab and minimal composer
├── components/                    # shared browser, status, notice, and layout components
├── shared/                        # small domain-neutral presentation helpers
├── styles/
├── main.tsx
└── vite-env.d.ts
```

OpenAPI is the API contract source of truth. TypeScript response types should be generated from the
FastAPI schema or checked against generated types rather than maintained as unrelated handwritten
copies. `npm run generate:api` exports the FastAPI schema and regenerates those checked-in aliases.

Server data and transient UI state remain distinct. Session pages, turns, traces, spans, and graph
definitions are cacheable server resources; selected turn/span, panel sizes, and expanded containers
are local UI state.

## Diagnostic read boundary

The current `SessionStore` owns critical load/commit behavior and currently hydrates an entire session.
It should not be expanded into a cross-collection dashboard abstraction. Add a separate read-only
`DiagnosticReadStore` protocol with MongoDB and deterministic memory implementations.

This store owns paged and projection-oriented queries across:

- `sessions`;
- `session_turns`;
- `execution_traces`;
- `execution_spans`; and
- `graph_definitions`.

The read store never commits conversation state, creates indexes, or reaches agents. Collection-owning
write stores remain responsible for index creation. The diagnostic store delegates existing
trace/span/definition reads to `ExecutionTraceStore` and owns only the new paged and
projection-oriented queries, preventing two implementations of the same validation logic from
drifting. Production authorization will be enforced at the diagnostic API boundary before these
queries are called.

The planned backend layout is:

```text
src/agentic_orchestration/
├── api/
│   └── diagnostics.py              # HTTP routes and controlled error mapping
└── diagnostics/
    ├── contracts.py                # response and query models
    ├── cursors.py                  # versioned opaque cursor codec
    ├── store.py                    # DiagnosticReadStore protocol
    ├── mongodb.py                  # production read adapter and projections
    ├── memory.py                   # deterministic test adapter
    └── frontend.py                 # /diagnostics static and SPA fallback setup
```

Lifespan constructs the Mongo read store from the same typed database handle, publishes it through an
API dependency, and retains the active agent-to-definition mapping produced during graph compilation.
The deterministic memory adapter composes the injected session store and execution trace store because
there is no database handle in that path. Both adapters guarantee span-summary ordering by
`sequence_started` as part of the store contract rather than relying on HTTP routes to sort.
The existing `GET /api/sessions/{session_id}/history` remains a simple conversation projection and is
not used as the diagnostic UI's paged query API.

## Initial diagnostic APIs

All endpoints are read-only except the existing chat endpoint used by the composer. Exact Pydantic
field names are finalized in code, but the following resource boundaries are approved.

### Session page

```text
GET /api/diagnostics/sessions?limit=10&cursor={opaque_cursor}
```

Response:

```text
items[]:
  session_id
  agent_id
  display_title
  revision
  status
  created_at
  updated_at
next_cursor: string | null
```

`display_title` is a whitespace-normalized, 160-character maximum projection of the first user
message. This is deterministic and requires no model call; a future generated conversation summary can
replace its value without changing the UI contract. The full session ID remains available for database
pinpointing but is not the primary browser label.

The default and initial UI limit is 10; the server enforces a small maximum such as 50. Results sort by
`updated_at DESC, _id DESC` so ties are deterministic.

The cursor is URL-safe base64url-encoded canonical JSON with a version, last `updated_at`, and last
session ID. It is opaque to the frontend, not encrypted, and not an authorization credential. Invalid
or incompatible cursors return a controlled `422`, not a MongoDB or decoder error.

MongoDB requires an index on `(updated_at DESC, _id DESC)`. The query requests `limit + 1` records to
decide whether a next cursor exists. Sessions updated while paging may move to the newest page; the UI
offers an explicit refresh rather than pretending pages form a permanent snapshot. This deliberately
optimizes for recently active sessions, not snapshot-stable traversal. The exclusive descending
predicate is equivalent to `updated_at < cursor.updated_at` or equal time with `_id < cursor.id`.

### Session resource

```text
GET /api/diagnostics/sessions/{session_id}
```

This projection supplies the header for a direct session deep link and does not depend on the resource
being present in the currently loaded session page. A stable `session_id` is allocated before an
attempt begins; a failed first attempt can therefore have an ID even when this endpoint correctly
returns `404` because no conversation turn committed.

### Turn page

```text
GET /api/diagnostics/sessions/{session_id}/turns?limit=25&cursor={opaque_cursor}
```

The first request returns the newest turns and a cursor for older turns. Each item contains the turn
number, committed time, input and final-assistant projections, routing outcome, selected agent ID, and
nullable trace ID. The UI presents an explicit `no trace recorded` state for promoted historical turns
without one. Internal tool/model messages remain accessible through trace spans rather than being
rendered as ordinary conversation messages.

Turn pagination uses immutable `turn_number` ordering. The frontend presents each loaded page in
chronological order and prepends older pages when requested. Its cursor contains a version and the next
exclusive turn-number boundary. The Mongo projection filters `generated_messages` to the final response
before applying the shared durable-turn validator. It is valid under the current validator; any future
cross-message invariant must use the complete session read or introduce a summary-specific read model.

### Session runs

```text
GET /api/diagnostics/sessions/{session_id}/traces?limit=25&cursor={opaque_cursor}
```

This lists committed and uncommitted runs for a known session so failed or cancelled executions do
not disappear merely because no successful turn references them. Attempts sort by
`attempted_turn_number DESC, started_at DESC, trace_id DESC` and are cursor-paged so retries of one turn
remain deterministic. Every trace and span uses the same immutable `session_id` assigned at request
start; `committed_turn_number` indicates whether the attempt produced durable conversation history.
This avoids a union across committed and prospective identity fields and permits one matching
session/attempt/time index. Committed summaries include a bounded preview of that run's own user input,
resolved through the existing `(session_id, turn_number)` index; opaque trace IDs remain secondary.

Pre-v3 router traces were explicitly cleared for this development deployment. Session-attempt lookup
therefore uses only immutable v3 `session_id`; legacy `prospective_session_id` is retained solely in the
direct v2 read-promotion path, not as a paged lookup key. Existing Mongo deployments may still contain
the superseded `execution_trace_session_turn` and `execution_trace_prospective_turn` indexes because
index creation does not remove them. Dropping those named indexes is an explicit operator migration and
is not performed automatically at application startup.

### Recent runs

```text
GET /api/diagnostics/traces?limit=25&cursor={opaque_cursor}
```

This newest-first page is the recovery path for failed or cancelled first attempts that have no
session document. Items are slim trace summaries containing trace and session identity, attempted and
committed turn numbers, execution and observability status, start/completion time, diagnostic origin,
and sanitized terminal error metadata. A trace selected here can be inspected without manufacturing a
durable session.

### Trace and spans

```text
GET /api/diagnostics/traces/{trace_id}
GET /api/diagnostics/traces/{trace_id}/spans
GET /api/diagnostics/traces/{trace_id}/spans/{span_id}
```

The trace endpoint returns a controlled trace projection. The spans endpoint returns a slim summary of
the latest monotonic snapshot for every span ordered by `sequence_started`; summaries contain topology,
status, ordering, timing, and capture-presence metadata but not captured payload values. The span-detail
endpoint returns the selected span's controlled input, messages, output, state delta, public error,
bounded failure detail, and typed model/tool observations. Failure detail contains redacted exception
messages and frame locations without locals, arguments, or source text. This keeps one trace as the
initial topology hydration unit without coupling
canvas load time to potentially multi-megabyte captures or introducing premature span pagination.

Diagnostic response models are the public API contract; they do not expose the storage model's open
`attributes` dictionary as the inspector contract. Known model identity, provider, usage, tool identity,
and scheduler fields receive typed optional response fields. Unknown internal attributes remain absent
until deliberately promoted into a versioned response contract.

### Graph definitions and active deployment

```text
GET /api/diagnostics/graph-definitions/{definition_id}
GET /api/diagnostics/deployment
```

The deployment response identifies the active deployment and agent versions plus the graph-definition
ID currently compiled for each agent. It also exposes the current routable-agent catalog separately
from graph topology. A router's dynamic `delegate` node does not acquire fictional compiled edges to
every eligible agent; static `destination_agent_id` applies only when a compiled node has a fixed child
destination. Definitions are cached by immutable ID in the browser.

Unknown resources return `404`. Storage unavailability returns a sanitized `503`. Diagnostic content
models forbid unexpected fields and never expose Mongo `_id` separately from controlled public IDs.

## Trace-to-canvas projection

The frontend projection layer is independent from the eventual graph rendering library. It transforms
durable contracts into a renderer-neutral model in this order:

1. Index spans by `span_id` and definitions by `definition_id`.
2. Build containment from `parent_span_id` without assuming sequence adjacency.
3. Build causal edges from `caused_by_span_ids`.
4. Group node instances by graph span, `superstep`, and `iteration`.
5. Join runtime nodes to compiled nodes through `graph_definition_id` and `definition_node_id`.
6. Add compiled but unvisited non-boundary nodes and edges only when the overlay is enabled.
7. Instantiate the definition's `START` and `END` nodes exactly once as visual boundary nodes; they
   never have runtime spans.
8. Place nodes sharing a graph and superstep in the same logical layer while retaining actual timing.

The projector must preserve opaque IDs and never construct identity from display labels. Unknown node
kinds or absent definitions render as explicit generic/unknown elements rather than failing the page.

Executed-path mode emphasizes runtime instances and causal edges. Compiled mode adds possible untaken
nodes and dims untaken paths. Model, tool, and child-agent spans render as selectable attachments to
their owning runtime node; a child-agent attachment connects onward to its child graph. Inspector
navigation covers parent, child, causal-predecessor, and causal-successor relationships.

## Subgraph accommodation

The current contracts already reserve a `subgraph` node kind and `owner_path`, but current agents do not
yet require private nested subgraphs. The UI architecture must therefore accommodate hierarchy without
inventing unsupported runtime data.

The renderer-neutral model treats every graph as a container with an optional parent reference. Current
definitions have no parent; future observability schema versions may add an owning definition/node
reference and emit a nested graph span. The UI then expands that graph inside its owning node using the
same projector recursively.

Subgraphs and child agents remain different concepts:

- a subgraph is a private workflow owned and versioned with its parent agent; and
- a child agent has its own manifest, package/version identity, invocation span, and graph definition.

The first canvas must not flatten these into one generic “nested agent” concept. Breadcrumbs and
collapse state use graph/container identity so arbitrary nesting depth can be added without redesigning
the surrounding session page.

## Historical first, live later

The first implementation hydrates MongoDB snapshots after execution. This establishes stable API,
projection, inspector, and rendering behavior before adding transient delivery.

The later SSE flow is:

1. hydrate current trace and span snapshots;
2. remember the largest observed sequence;
3. subscribe to `/api/diagnostics/traces/{trace_id}/stream`;
4. apply envelopes idempotently by trace sequence; and
5. rehydrate snapshots after reconnect or detected gaps.

SSE is a best-effort acceleration, not the durable source. MongoDB remains authoritative after reload,
disconnect, worker change, or missed event.

Until SSE exists, submitting chat displays a real request-pending state rather than implying node-level
progress. A persisted `running` trace is labelled `running — no live updates`; if neither it nor its
spans has changed within a configurable presentation threshold, the UI adds a stale warning instead of
spinning forever.
`observability_status=degraded` is always a trace-level completeness banner independent from execution
success or failure.

## Security boundary

This UI is internal but can display prompts, intermediate state, and sanitized business context. Until
authentication and tenant/session authorization exist, diagnostic routes and the service must remain
restricted to an approved development environment.

Failed-span detail additionally exposes exception types, redacted messages, traceback filenames,
functions, and line numbers. MongoDB authentication is not sufficient protection once these records are
returned through HTTP. Do not make diagnostic routes production-accessible until endpoint authentication,
tenant/session authorization, diagnostic-read permission, and auditing are enforced and tested.

The UI itself is never an authorization boundary. Before any production deployment, diagnostic APIs
must enforce authenticated identity, tenant/session ownership, diagnostic-read permission, artifact
authorization, and access auditing as described in the execution-observability design.

Opaque session, trace, and cursor values do not grant access. The frontend must not persist diagnostic
payloads in local storage or log complete API responses to the browser console.

Diagnostic-originated requests should carry an explicit, non-secret origin marker into trace metadata
so they can later be filtered from other engineering traffic. This marker is descriptive and never an
authorization signal.

## Testing strategy

### Backend

- unit-test cursor encoding, decoding, validation, and stable tie ordering;
- test 10-item session pages and correct next-cursor behavior;
- test direct session reads and recent attempts without durable session documents;
- test turn pagination and chronological presentation boundaries;
- test committed plus failed/cancelled session attempts;
- test slim span ordering, on-demand span detail, and immutable definition lookup;
- test sanitized `404`, `422`, and `503` responses;
- test SPA assets, deep-link fallback, and missing-build behavior without affecting API startup; and
- retain opt-in MongoDB integration coverage for the actual indexes and query shapes.

### Frontend

- use sanitized, versioned API fixtures captured from deterministic orchestration tests;
- test loading, empty, missing, degraded, failed, cancelled, and oversized-content states;
- test pending, stale-running, no-trace-recorded, and failed-composer navigation states;
- test pagination, refresh, selection, and stale-request cancellation;
- test transcript-to-trace and span-to-inspector navigation;
- test the renderer-neutral projector independently from canvas components;
- test deep links and browser refresh beneath `/diagnostics`; and
- add browser/visual regression tests when the graph renderer is selected.

Fixture data is test infrastructure, not a selectable UI runtime mode. Secrets and raw provider
envelopes are prohibited from fixtures.

## Code implementation sequence

### Slice 0: observability and chat contract readiness — implemented

1. Advance the observability storage contract to v3 with one immutable `session_id` on traces and spans
   from attempt start plus a typed request-origin field; use commit fields to describe whether durable
   conversation history advanced and promote readable v2 records by taking whichever legacy session
   identity is present.
2. Return a typed error envelope containing `session_id` and `trace_id` for chat failures after trace
   creation.
3. Define typed trace summary, span summary, span detail, model observation, tool observation, and
   observability-status response contracts.
4. Add indexes through their owning session and execution stores.

Slice 0 is complete when a failed first turn is independently addressable, trace/span documents do not
change identity on commit, and OpenAPI describes every field required by the initial inspector.

### Slice 1: diagnostic read APIs — implemented

1. Define Pydantic response and cursor contracts under `agentic_orchestration.diagnostics`.
2. Add the read-only store protocol plus MongoDB and memory adapters.
3. Add paged session, turn, per-session attempt, and recent-attempt queries.
4. Expose direct session, trace, slim-span, span-detail, graph-definition, and active-deployment reads.
5. Wire the read store during lifespan and cover it with unit/integration tests.

Slice 1 is complete when two sessions with identical update times paginate deterministically, a failed
first attempt is discoverable without a session document, direct deep links hydrate independently from
list pages, a selected trace can hydrate all current span summaries and definitions, invalid cursors
are controlled, and no diagnostic query can mutate session or execution collections.

### Slice 2: frontend scaffold and static serving — implemented

1. Create top-level `frontend/` with React, TypeScript, Vite, npm scripts, and lockfile.
2. Configure Vite's `/diagnostics/` base and development API proxy.
3. Add FastAPI static asset and SPA fallback behavior.
4. Add an application shell, URL state, API client, error boundary, and build smoke test.
5. Update root quick-start commands and ignore generated directories.

### Slice 3: conversation and turn workflow — implemented and visually verified

1. Implement the 10-item newest-first conversation browser with first-message display titles.
2. Implement lazy turn pages and turn-aware transcript rendering.
3. Add router-run browsing and selected-trace hydration.
4. Add the minimal chat composer and post-success or post-failure selection/refresh.

### Slice 4: trace projection and inspection — implemented

1. Implement the renderer-neutral trace projector.
2. Build a structured execution list grouped by graph, superstep, and iteration.
3. Build the full node/span inspector with on-selection detail hydration and navigation links.
4. Validate the projection against sequential, delegated, tool-loop, parallel, failed, and cancelled
   fixtures.

### Slice 5: graph canvas and active catalog — implemented

1. Evaluate the graph renderer against nested containers, parallel ranks, custom nodes, overlays,
   selection, and expected trace sizes. React Flow was selected with deterministic application-owned
   layering.
2. Render executed-path mode and compiled topology mode, including dimmed unvisited nodes.
3. Add the active graph catalog and definition deep links.
4. Preserve the recursive container boundary for future private subgraphs.

### Slice 6: conversation turn strata

1. Compose loaded per-turn projections into an ordered conversation execution view.
2. Add turn jumping/scrubbing without requiring every historical turn to be loaded.
3. Preserve independent trace status, observability completeness, and compiled overlay state per turn.
4. Validate cross-turn navigation with successful, failed, retried, and trace-less historical turns.

### Deferred

- forced router or nested-router decisions;
- live SSE and reconnect behavior;
- historical definition comparison;
- large artifact retrieval;
- generalized trace search and analytics;
- customer-facing chat features; and
- production authentication and authorization implementation.

Code writing begins with Slice 0 and does not require choosing a graph rendering library. Slice 2 can
proceed in parallel only after the initial diagnostic API response shapes are fixed, so the frontend
does not stabilize around temporary handwritten contracts.
