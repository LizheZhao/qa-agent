# Chat turn

One user message goes through the router, which answers itself or delegates to at most one routable child agent. The turn commits to Mongo and one root trace is recorded.

## Sub-features

- New session (omit `session_id`) and continuing session
- Router answer versus child delegation (`marketing_science`; `ask_genome` with `-Full`)
- Retryable 409 on a revision conflict, with no model reinvoke
- Failed invocation: error body with `trace_id`, no turn, no empty session
- Pause for clarification (202 with a pending clarification)

## How to get to it (user POV)

Lab tab composer in the diagnostic UI, or `POST /api/chat`.

## Driving it with HTTP

1. Launch and doctor.
2. `POST /api/chat` with `{"message":"hello"}` via `Invoke-RestMethod -ContentType application/json`. Needs a valid gateway token in `.env`.
3. Take `session_id` and `trace_id` from the response, then check:
   - `GET /api/sessions/{session_id}/history`: the user and assistant messages appear
   - `GET /api/diagnostics/traces/{trace_id}`: `status: completed`, `committed_turn_number: 1`
   - `GET /api/diagnostics/sessions/{session_id}/turns`: one turn
4. Send a second message with the same `session_id`: revision and turn number advance.

End state that proves it: response 200, history shows the turn, trace `completed` with `observability_status: complete`.

## Gotchas

- Real LLM gateway call, tens of seconds; the gateway timeout is 180 s.
- Writes retained records to the shared Mongo database. Record the ids you create.
- `ChatRequest` forbids extra fields and blank messages (422).
