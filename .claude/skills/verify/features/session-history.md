# Session history and clarifications

Reload a session's committed conversation, and answer or cancel a clarification that paused a run.

## Sub-features

- History: messages, `storage`, `durable`, `pending_clarification`
- Clarification response: answer or cancel, idempotent by `submission_id`; expired pauses are refused with 410 and swept every `PENDING_RUN_SWEEP_INTERVAL_SECONDS`

## How to get to it (user POV)

Conversations tab transcript, the Lab clarification form, or the two routes.

## Driving it with HTTP

1. Take any `session_id` from `GET /api/diagnostics/sessions?limit=1` (read-only).
2. `GET /api/sessions/{session_id}/history`: expect 200, `durable: true`, ordered messages. Unknown id: 404.
3. For clarifications a chat message that triggers a pause is needed (see chat.md). Then `POST /api/sessions/{id}/clarifications/{clarification_id}/responses` with `{"submission_id":"<uuid>","response":{...}}`. Read the exact `ClarificationResponse` shape from `/openapi.json` first.

## Gotchas

- A first-turn pause has a pending run but no session document yet; history still returns 200.
- Reusing a `submission_id` must not create a second turn.
