# Backend application contract

This subtree owns FastAPI delivery and runtime assembly. The repository-root contract also applies.

## Module boundaries

- `main.py` constructs the FastAPI application, registers routers/handlers, and configures diagnostic
  static serving. Endpoint behavior belongs in `api/`; runtime resource and registry assembly belongs in
  `lifespan.py`; graph invocation belongs in `execution/`.
- API route modules translate HTTP inputs, outputs, dependencies, and controlled errors. They must not
  import `pymongo`, issue database queries, compile graphs, or construct provider models.
- Persistence queries and writes stay in the relevant `mongodb.py` adapter. Callers depend on the
  existing `SessionStore`, `ExecutionTraceStore`, or `DiagnosticReadStore` protocol rather than a
  concrete MongoDB adapter.
- Add or widen a protocol only when a runtime caller needs implementation independence or a deterministic
  test substitute. Do not introduce protocol/adapter layers for helpers with one local implementation.
- External connection construction stays in top-level `integrations`. Its constructors accept explicit
  connection values and do not import application `Settings` or session contracts.
- Lifespan owns external resources. Register their cleanup immediately after successful construction so
  any later startup failure closes them.

## Runtime and persistence invariants

- The deployment manifest and declared entrypoints determine registry composition. Graphs compile at
  startup; child invocation is restricted to eligible agents; request handling does not mutate registry
  membership.
- Session-store initialization is required for startup. Observability initialization and persistence are
  best effort and must not turn an otherwise successful conversation into a failed one.
- Graph execution occurs before the short commit transaction. Successful turns are immutable and commit
  once with revision and `(session_id, turn_number)` uniqueness guards. Failed invocations do not create
  or modify turns. A commit conflict must not reinvoke the model or tools automatically.
- Session mutation stays behind `SessionStore`; cross-collection diagnostic queries stay behind
  `DiagnosticReadStore`. Do not add dashboard reads to the session mutation protocol or conversation
  writes to the diagnostic protocol.
- Convert expected domain/storage failures to controlled, sanitized HTTP responses. Raw exceptions,
  database details, and provider payloads must not enter response bodies.

## Tests

- Tests under `tests/unit/` use the `unit` marker; in-process API/lifespan/graph workflows under
  `tests/integration/` use `integration`.
- Deterministic tests inject the applicable memory stores and `ScriptedChatModel`. Production modules
  must not select those implementations based on a test flag.
- When changing a shared store protocol, test the affected observable behavior against every maintained
  implementation of that protocol. Add conflict, ordering, missing-data, or sanitization cases only
  when the changed contract promises those behaviors.
- Run the focused test module while iterating, then the applicable root verification commands.
