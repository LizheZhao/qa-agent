# Graph interrupt and resume

This document defines the runtime contract for pausing a child-agent graph to ask a
clarifying question and later resuming that same graph. The continuation retains completed
work and checkpointed task context; it does not submit the answer as a new chat message or
route the request again.

The mechanism is generic infrastructure. It becomes active for a production workload only
when a deployed child graph is compiled with the durable checkpointer and produces a
`ClarificationRequest`. Router-only text clarification remains a completed chat turn and does
not use this mechanism.

## User-visible behavior

- A session may have at most one active paused run.
- A clarification is answerable for 10 minutes from publication. It expires at the boundary,
  not after it.
- The question is not a committed turn while it is pending. Finishing the graph commits one
  logical turn containing the original request, the ordered clarification exchange, and the
  final answer.
- The user may select offered options, enter bounded free text, or cancel. Option selection and
  free text cannot be combined in one response.
- Cancellation ends the logical turn at the displayed question without resuming the graph. A
  later chat message starts a new routed turn.
- An unrelated chat message cannot implicitly answer a clarification. It receives a conflict
  while an unexpired run is active.
- When several tasks require clarification, the producer presents one question at a time in
  deterministic FIFO order. The remaining questions stay in checkpointed state.

## Clarification contracts

The shared contracts live in `orchestration_core.clarification`.

`ClarificationRequest` contains:

- a clarification ID, producing agent ID, and task ID;
- the question and bounded reason code;
- `selection_mode`, either `single` or `multiple`;
- between 1 and 10 ordered options with distinct IDs and labels; and
- a producer freshness token.

Questions are limited to 1,000 characters, option labels to 200 characters, option details to
500 characters, and free-text responses to 2,000 characters. Unknown fields are rejected.

The response is a discriminated union:

```json
{"form": "options", "option_ids": ["option-a"]}
```

```json
{"form": "free_text", "text": "Use the manually corrected scope."}
```

```json
{"form": "cancel"}
```

A single-choice request accepts exactly one known option ID. A multiple-choice request accepts
one or more distinct known option IDs. Free text and cancellation are always available.

## Runtime boundary

The router and its selected child are separate graph invocations. Only pause-capable child
graphs receive a checkpointer. The router itself is not checkpointed.

An initial child invocation returns one of two distinct outcomes:

- `ConversationResult` when the child completed; or
- `PausedConversation` when the child stopped at a clarification.

The separate pause type prevents callers from treating an interrupted graph as a completed
conversation with an empty answer. A paused child has its own opaque run ID, which is also its
LangGraph checkpoint thread ID. The run ID is not the session ID.

`RegistryAgentInvoker.resume` continues the child recorded by the pending run directly. It does
not invoke the router. Before applying the response it verifies that:

- the checkpoint exists and is still paused;
- the recorded child is installed and eligible;
- the graph definition and continuation contract versions still match;
- the clarification ID identifies the active interrupt; and
- the producer freshness token still matches the checkpoint.

A failed gate produces a bounded restart-required reason: `missing_checkpoint`, `not_paused`,
`graph_version`, `contract_version`, `unknown_clarification`, or `stale_producer`. The runtime
does not attempt a best-effort resume against incompatible state.

LangGraph restarts an interrupted node from its beginning. Work that must not repeat belongs in
an earlier node, behind a durable receipt, or after the interrupt. External mutations still
require their own authorization and idempotency controls.

## Multiple clarifications

`ClarificationQueue` holds the active question at its head and later questions in checkpointed
state. Producers order requests by plan position, task ID, and clarification ID rather than by
worker completion time. After an answer is applied, the producer presents the next queued
question before dispatching newly dependent work.

The invocation boundary accepts exactly one live LangGraph interrupt. More than one live
interrupt is an invalid child result; choosing between concurrent interrupts at that boundary
would make presentation depend on race order.

## HTTP API

### Start an ordinary turn

`POST /api/chat` remains the routed chat operation.

- `200` returns a completed answer.
- `202` returns a durable `PendingClarification` after a child pauses.
- `409` reports that the session already has an active, unexpired pause.

The pending response exposes the session and clarification IDs, question, reason code,
selection mode, ordered options, expiry, and trace ID. It does not expose the child run ID,
selected child, graph definition, contract version, or freshness token.

### Answer or cancel a clarification

```text
POST /api/sessions/{session_id}/clarifications/{clarification_id}/responses
```

The body contains a client-generated idempotency key and exactly one response:

```json
{
  "submission_id": "2c538f80-7e62-4c20-aeee-b68aeb4fc3e5",
  "response": {"form": "options", "option_ids": ["option-a"]}
}
```

The server resolves every continuation authority from the pending record. Clients never submit
the run ID, child ID, graph definition, contract version, or freshness token.

The endpoint returns:

- `200` when the graph finishes or cancellation commits the interrupted turn;
- `202` when the graph pauses again on the next queued clarification;
- `404` when no matching open or replayable clarification exists;
- `409` for a conflicting claim, revision conflict, idempotency-key reuse with different
  content, or a restart-required resume gate;
- `410` when the clarification has expired;
- `422` when the response violates the clarification contract;
- `500` when continuation execution fails; and
- `503` when durable session storage is unavailable.

Replaying the same `submission_id` with the same response is idempotent. Reusing it with a
different response is a conflict. A failed continuation retains its claim so that only the same
submission can retry it.

## Durable state

MongoDB is the runtime durability boundary. Memory implementations are deterministic test
adapters, not production fallbacks.

The application stores paused-run state in `pending_runs` and ordered exchange events in
`clarification_audit`. A pending run records the originating session revision, selected child,
checkpoint thread, graph and contract versions, original request, route metadata, active
clarification, claim state, expiry, and trace lineage.

Pending-run statuses are:

- `awaiting_response` — published and answerable;
- `claimed` — a response owns the run and may be executing;
- `consumed` — the continued graph completed and its turn committed;
- `canceled` — cancellation committed the question-ending turn; and
- `expired` — expiry committed the question-ending turn.

The active record carries a session key protected by a unique sparse index, enforcing at most
one `awaiting_response` or `claimed` run per session across replicas. A first-turn pause creates
no empty session document: the first durable session document is still created only when a turn
commits, and its revision remains at least 1.

Audit events record published questions and their ordered options, accepted answers,
cancellation, and expiry. Rejected malformed responses do not become conversation audit events;
their failures are represented by bounded, sanitized diagnostics.

The implementation currently guarantees session-scoped and revision-safe consumption. Tenant
and user fields remain nullable, so it does not claim cross-user authorization without a trusted
authentication boundary.

## Consistency and recovery

Checkpoint writes and session writes are separate durability operations. The following ordering
provides the consistency boundary:

1. LangGraph writes the interrupted checkpoint.
2. The session store transactionally publishes the pending pointer and its first audit event,
   guarded by the expected session revision.
3. A response is validated and atomically claimed with its `submission_id` before graph resume.
4. Completion transactionally commits the final turn, consumes the pending run, and preserves
   continuation lineage.

A clarification is returned to a client only after publication succeeds. A checkpoint left by a
failed publication is unreachable and may be cleaned later.

If a process stops after graph completion but before the final session transaction, a retry
recovers the completed output from the checkpoint and commits it without repeating model or tool
work. Checkpoint cleanup is never used to enforce expiry or prevent duplicate continuation; the
pending-run state machine provides those guarantees.

Cancellation and expiry use the same transactional close operation. A response claimed before
expiry wins and is not interrupted by the sweeper. Competing replicas may attempt the same
closure, but the compare-and-set transition commits at most one turn and one audit event.

## Checkpoint storage

`agentic_orchestration.execution.mongodb.MongoCheckpointSaver` implements the pinned
`BaseCheckpointSaver` using the same `pymongo.AsyncMongoClient` as session persistence. It is
async-only; its synchronous methods raise instead of blocking or moving database work into a
thread pool.

Checkpoints and pending writes use separate collections. Ordinary pending writes are immutable
after insertion. Reserved negative-index writes for error, interrupt, resume, and scheduling
state may be replaced on retry. `task_path` is stored as payload rather than write identity.

Checkpoint state is restricted to bounded task context, receipts, compact results, artifact
references, metrics, and queued clarifications. Raw result tables, credentials, and provider
payloads do not belong in graph state. Referenced artifacts must remain available for at least
the pause window.

The stock serializer runs with pickle fallback disabled, and the document boundary rejects
pickle payloads. The JSON reconstruction path has an explicit application-type allowlist. The
upstream msgpack extension can still reconstruct import-addressed objects, so checkpoint
collections are a trusted internal persistence boundary; this is not protection against a
maliciously modified checkpoint document.

## History projection

`GET /api/sessions/{session_id}/history` keeps committed messages in `messages` and exposes an
open question separately as `pending_clarification`. A first-turn pause therefore returns `200`
with an empty message list and the pending question, without fabricating a session document.

After closure, history reconstructs the ordered clarification exchange from the audit. For a
canceled or expired run, the exchange already ends at the last displayed question, so the
turn's identical final message is not rendered a second time.

History retrieval performs request-time expiry closure before projecting state. It is a
read-oriented API, but a request for an expired session can therefore transactionally commit the
expiry outcome.

## Expiry lifecycle

Three paths enforce or process expiry:

- response claiming rejects an answer at or after its deadline;
- ordinary chat and history close an expired run before deciding whether it is active; and
- a bounded periodic sweep closes abandoned runs in deadline order.

Only `awaiting_response` runs are eligible for expiry closure. A status-leading MongoDB index
limits the sweep to those records, and the transition rule independently refuses to close a
claimed run. A failed sweep logs the failure and continues on the next interval; response-time
validation remains authoritative.

The sweep is configured with:

- `PENDING_RUN_SWEEP_INTERVAL_SECONDS`, default `60`; set to `0` to disable periodic sweeping;
  and
- `PENDING_RUN_SWEEP_LIMIT`, default `50`, allowed range 1–500.

MongoDB TTL deletion is not used because expiry must first commit a question-ending turn and an
audit event.

## Operations and limitations

- Graph node names and checkpointed state must remain resume-compatible for at least the
  10-minute pause window. An incompatible deployment causes a restart-required response.
- The expiry sweep is process-owned by the application lifespan. Multiple replicas may run it
  safely because durable transitions settle races.
- The general diagnostic dashboard does not currently project pending runs or clarification
  audit events. Session history exposes the user-facing pending question and completed exchange.
- No deployed production child currently produces this clarification contract or receives the
  checkpointer. Producer integration and workload-specific questions are separate from this
  runtime facility.
- The facility does not provide arbitrary state editing, time travel, concurrent runs within one
  session, or recovery from every graph failure.

## Primary references

- [LangGraph interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts)
- [LangGraph persistence](https://docs.langchain.com/oss/python/langgraph/persistence)
- [LangGraph subgraphs](https://docs.langchain.com/oss/python/langgraph/use-subgraphs)
- [LangGraph backward compatibility](https://docs.langchain.com/oss/python/langgraph/backward-compatibility)
- [MongoDB checkpointer implementation](https://github.com/langchain-ai/langchain-mongodb/blob/main/libs/langgraph-checkpoint-mongodb/langgraph/checkpoint/mongodb/saver.py)
