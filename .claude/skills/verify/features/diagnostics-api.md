# Diagnostics read API

Cursor-paged, read-only views over sessions, turns, traces, spans and compiled graph definitions. Backs the UI.

## Sub-features

- Deployment catalog: agents, versions, `routable`, `graph_definition_id`
- Session list (newest first), session detail, turns per session
- Trace list (every attempt, including failed ones with no committed turn), trace detail, spans, span detail
- Graph definition topology by id

## How to get to it (user POV)

A developer opens the Agents, Conversations or Runs tab, which call these routes. Directly: `GET /api/diagnostics/...`.

## Driving it with HTTP

1. `verify.ps1 launch`, then `doctor`.
2. `GET /api/diagnostics/deployment`: expect `entry_agent: router` and one row per declared agent (two with the verification deployment).
3. `GET /api/diagnostics/sessions?limit=2`: expect `items` with `display_title` and `revision`, plus `next_cursor`; pass it back as `cursor` for page 2.
4. `GET /api/diagnostics/graph-definitions/<graph_definition_id from step 2>`: expect `schema_version`, `nodes`, `agent_id`.
5. `GET /api/diagnostics/traces?limit=2`, then `/traces/{trace_id}/spans`.

Proof from the first run: `../evidence/proof-readonly.txt` and `../evidence/proof-agents-feature.txt`.

## Gotchas

- `limit` above 50 and a bad cursor are 422; storage outage is 503; unknown session or graph definition is 404.
- Reads hit the shared dev Mongo, so lists contain other people's sessions.
